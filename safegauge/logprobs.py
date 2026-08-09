import copy
from typing import Dict, Optional

from transformers import AutoTokenizer

from .config import THINKING_BYPASS_PREFILL, infer_thinking_bypass_prefill
from .server_backends import create_server_backend


def _logprob_mapping_get(mapping, token_id):
    """Return the logprob entry for an observed token ID."""
    if mapping is None:
        return None
    if hasattr(mapping, "model_dump"):
        mapping = mapping.model_dump()
    if token_id in mapping:
        return mapping[token_id]
    string_id = str(token_id)
    if string_id in mapping:
        return mapping[string_id]
    return None


def _entry_value(entry, name):
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get(name)
    return getattr(entry, name, None)


def _serializable_logprob_mapping(mapping):
    """Convert one prompt-logprob position to JSON-compatible values."""
    if mapping is None:
        return None
    if hasattr(mapping, "model_dump"):
        mapping = mapping.model_dump()
    return {
        token_id: {
            "logprob": _entry_value(entry, "logprob"),
            "rank": _entry_value(entry, "rank"),
            "decoded_token": _entry_value(entry, "decoded_token"),
        }
        for token_id, entry in mapping.items()
    }


def _extract_suffix_result(
    *,
    text: str,
    suffix_logprobs,
    suffix_token_ids,
    suffix_start: int,
) -> Dict:
    """Build a backend-independent result for the semantic suffix."""
    suffix_token_ids = list(suffix_token_ids)
    if not suffix_token_ids:
        raise ValueError("assistant semantic suffix must contain at least one token")
    if suffix_logprobs is None:
        raise ValueError("backend did not return suffix logprobs")
    suffix_logprobs = list(suffix_logprobs)
    if len(suffix_logprobs) != len(suffix_token_ids):
        raise ValueError(
            "suffix logprob length does not match suffix token length: "
            f"{len(suffix_logprobs)} != {len(suffix_token_ids)}"
        )
    suffix_end = suffix_start + len(suffix_token_ids)
    observed = [
        _logprob_mapping_get(token_logprobs, token_id)
        for token_logprobs, token_id in zip(suffix_logprobs, suffix_token_ids)
    ]
    for index, (mapping, entry, token_id) in enumerate(
        zip(suffix_logprobs, observed, suffix_token_ids),
        start=suffix_start,
    ):
        if mapping is None:
            raise ValueError(
                f"missing suffix logprobs at prompt position {index}"
            )
        if entry is None:
            raise ValueError(
                f"suffix logprobs at prompt position {index} do not contain "
                "the observed suffix "
                f"token ID {token_id}"
            )
    return {
        "text": text,
        "suffix_start": suffix_start,
        "suffix_end": suffix_end,
        "suffix_token_ids": suffix_token_ids,
        "suffix_logprobs": [
            _serializable_logprob_mapping(item) for item in suffix_logprobs
        ],
        "all_logprobs": [_entry_value(item, "logprob") for item in observed],
        "all_rank": [_entry_value(item, "rank") for item in observed],
    }


def safe_token_concat(tokens1, tokens2):
    if hasattr(tokens1, "get"):
        tokens1 = tokens1["input_ids"]
        if isinstance(tokens1[0], list):
            tokens1 = tokens1[0]
    if hasattr(tokens2, "get"):
        tokens2 = tokens2["input_ids"]
        if isinstance(tokens2[0], list):
            tokens2 = tokens2[0]

    return list(tokens1) + list(tokens2)


class LogProbsExtractor:
    """Extract token log probabilities for an assistant suffix.

    Server mode: pass base_url (+ optional api_key).
    Offline mode: pass llm (vllm.LLM instance).
    """

    def __init__(
        self,
        thinking_bypass_prefill: Optional[str] = None,
        # server mode args
        base_url: Optional[str] = None,
        api_key: str = "none",
        temperature: float = 0,
        top_p: float = 0.95,
        # offline mode args
        llm=None,
        # backend selection and tokenizer override
        server_backend: str = "vllm",
        tokenizer_path: Optional[str] = None,
    ):
        """
        Server mode: pass base_url and select vLLM or SGLang with
            server_backend. Model information is auto-detected from the server.
        Offline mode: only llm is required. tokenizer is auto-loaded from the LLM.
        """
        self.model = None       # server API model id (may be alias)
        self.model_root = None  # actual model path on disk
        self.server_backend = None
        self.server_version = None
        if base_url is not None:
            self.mode = "server"
            self.base_url = base_url
            self.api_key = api_key
            self.temperature = temperature
            self.top_p = top_p
            self._backend = create_server_backend(
                server_backend,
                self.base_url,
                self.api_key,
            )
            self.server_backend = self._backend.name
            server_info = self._backend.discover()
            self.model = server_info.model
            self.model_root = server_info.model_root
            self.server_version = server_info.version
            detected_tokenizer_path = server_info.tokenizer_path
            version_text = (
                f" | version: {self.server_version}"
                if self.server_version
                else ""
            )
            print(
                "[LogProbsExtractor] Server mode: "
                f"{self.base_url} | backend: {self.server_backend} "
                f"| model: {self.model}{version_text}"
            )
        elif llm is not None:
            self.mode = "offline"
            self.llm = llm
            model_config = self.llm.llm_engine.model_config
            self.model_root = model_config.model
            detected_tokenizer_path = self.model_root
            print(f"[LogProbsExtractor] Offline mode | model: {self.model_root}")
        else:
            raise ValueError("Must provide either base_url (server mode) or llm (Offline mode)")

        if thinking_bypass_prefill is None:
            model_identity = " ".join(
                value
                for value in (
                    self.model_root,
                    self.model,
                    tokenizer_path,
                    detected_tokenizer_path,
                )
                if value
            )
            self.thinking_bypass_prefill = infer_thinking_bypass_prefill(
                model_identity,
            )
        elif thinking_bypass_prefill in THINKING_BYPASS_PREFILL:
            self.thinking_bypass_prefill = thinking_bypass_prefill
        else:
            supported = ", ".join(THINKING_BYPASS_PREFILL)
            raise ValueError(
                "Unknown thinking_bypass_prefill "
                f"'{thinking_bypass_prefill}'. Supported values: {supported}"
            )

        self.tokenizer_path = (
            tokenizer_path
            or detected_tokenizer_path
            or self.model_root
            or self.model
        )
        if not self.tokenizer_path:
            raise ValueError(
                "Cannot auto-load tokenizer from server model information; "
                "pass tokenizer_path explicitly"
            )
        print(
            "[LogProbsExtractor] Auto-loading tokenizer from: "
            f"{self.tokenizer_path}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
        )

    def _encode_assistant_text(self, text: str) -> list[int]:
        """Encode an assistant prefill fragment without adding wrapper tokens."""
        if not text:
            return []
        if self.thinking_bypass_prefill == "kimi_k3":
            # Kimi-K3's XTML bypass is represented by structural tokens. The
            # semantic suffix is trusted configuration and uses the same
            # tokenizer path, while its scores remain a separate feature span.
            return list(
                self.tokenizer.encode(text, allow_special_tokens=True)
            )
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_assistant_prefill(self, content: str) -> tuple[str, str]:
        prefix = THINKING_BYPASS_PREFILL[self.thinking_bypass_prefill]
        if prefix:
            if not content.startswith(prefix):
                raise ValueError(
                    "Final assistant prefill does not start with the configured "
                    f"thinking bypass {self.thinking_bypass_prefill!r}; use "
                    "apply_suffix() to construct it"
                )
            return prefix, content[len(prefix):]
        return "", content

    def _tokenize(self, messages):
        msgs_tokens = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True
        )
        prefix_text, suffix_text = self._split_assistant_prefill(
            messages[-1]["content"]
        )
        if not suffix_text:
            raise ValueError("assistant semantic suffix must be non-empty")
        bypass_tokens = self._encode_assistant_text(prefix_text)
        semantic_suffix_tokens = self._encode_assistant_text(suffix_text)
        assistant_prefill_tokens = safe_token_concat(
            bypass_tokens,
            semantic_suffix_tokens,
        )
        all_input_tokens = safe_token_concat(
            msgs_tokens,
            assistant_prefill_tokens,
        )
        return all_input_tokens, semantic_suffix_tokens

    def apply_suffix(self, messages: list, suffix: str) -> list:
        """
        Append a semantic suffix as an assistant prefill.

        Args:
            messages: conversation messages before the assistant response.
            suffix: semantic behavior description to probe.
        Returns:
            A copied conversation with the assistant suffix appended.
        """
        if not isinstance(suffix, str) or not suffix.strip():
            raise ValueError("suffix must be a non-empty string")

        work_msg = copy.deepcopy(messages)
        prefix = THINKING_BYPASS_PREFILL[self.thinking_bypass_prefill]
        work_msg.append({"role": "assistant", "content": prefix + suffix})
        return work_msg

    def get_logprobs(self, messages, logprobs_num=2) -> Dict:
        """
        Get log probabilities for the prefill part of the messages.

        messages: list of messages; the final message must contain the assistant suffix.
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris is"}
            ]

        Tip: use apply_suffix() to build the prefilled messages.
        """
        if not isinstance(logprobs_num, int) or logprobs_num < 0:
            raise ValueError("logprobs_num must be a non-negative integer")
        all_input_tokens, semantic_suffix_tokens = self._tokenize(messages)
        suffix_start = len(all_input_tokens) - len(semantic_suffix_tokens)

        if self.mode == "server":
            return self._get_logprobs_server(
                all_input_tokens,
                semantic_suffix_tokens,
                suffix_start,
                logprobs_num,
            )
        else:
            return self._get_logprobs_offline(
                all_input_tokens,
                semantic_suffix_tokens,
                suffix_start,
                logprobs_num,
            )

    def _get_logprobs_server(
        self,
        all_input_tokens,
        semantic_suffix_tokens,
        suffix_start,
        logprobs_num,
    ):
        scored = self._backend.score(
            all_input_tokens,
            suffix_start,
            logprobs_num,
            self.temperature,
            self.top_p,
        )
        return _extract_suffix_result(
            text=scored.text,
            suffix_logprobs=scored.positions,
            suffix_token_ids=semantic_suffix_tokens,
            suffix_start=suffix_start,
        )

    def _get_logprobs_offline(
        self,
        all_input_tokens,
        semantic_suffix_tokens,
        suffix_start,
        logprobs_num,
    ):
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            prompt_logprobs=logprobs_num,
            logprobs=logprobs_num
        )
        outputs = self.llm.generate(
            prompts=[all_input_tokens],
            sampling_params=sampling_params
        )
        output = outputs[0]
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None:
            raise ValueError("offline vLLM did not return prompt_logprobs")
        suffix_logprobs = list(prompt_logprobs)[-len(semantic_suffix_tokens):]
        return _extract_suffix_result(
            text=output.outputs[0].text,
            suffix_logprobs=suffix_logprobs,
            suffix_token_ids=semantic_suffix_tokens,
            suffix_start=suffix_start,
        )
