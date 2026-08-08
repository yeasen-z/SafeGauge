import copy
from typing import Dict, Optional

from transformers import AutoTokenizer

from .config import THINKING_BYPASS_PREFILL, infer_thinking_bypass_prefill


def _logprob_mapping_get(mapping, token_id):
    """Return the logprob entry for the observed token id from vLLM output."""
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


def _suffix_span(prompt_logprobs, suffix_token_ids) -> tuple[int, int]:
    """Return the trailing semantic-suffix span in prompt-logprob positions."""
    suffix_token_ids = list(suffix_token_ids)
    if not suffix_token_ids:
        raise ValueError("assistant semantic suffix must contain at least one token")
    if len(prompt_logprobs) < len(suffix_token_ids):
        raise ValueError(
            "prompt_logprobs is shorter than the semantic suffix: "
            f"{len(prompt_logprobs)} < {len(suffix_token_ids)}"
        )
    end = len(prompt_logprobs)
    return end - len(suffix_token_ids), end


def _extract_suffix_result(
    *,
    text: str,
    prompt_logprobs,
    suffix_token_ids,
) -> Dict:
    """Select the trailing semantic suffix from returned prompt logprobs."""
    if prompt_logprobs is None:
        raise ValueError("vLLM did not return prompt_logprobs")

    prompt_logprobs = list(prompt_logprobs)
    suffix_start, suffix_end = _suffix_span(
        prompt_logprobs,
        suffix_token_ids,
    )
    suffix_logprobs = prompt_logprobs[suffix_start:suffix_end]
    observed = [
        _logprob_mapping_get(token_logprobs, token_id)
        for token_logprobs, token_id in zip(suffix_logprobs, suffix_token_ids)
    ]
    for index, (mapping, entry, token_id) in enumerate(
        zip(suffix_logprobs, observed, suffix_token_ids),
        start=suffix_start,
    ):
        if mapping is not None and entry is None:
            raise ValueError(
                f"prompt_logprobs[{index}] does not contain observed suffix "
                f"token ID {token_id}"
            )
    return {
        "text": text,
        "suffix_start": suffix_start,
        "suffix_end": suffix_end,
        "suffix_token_ids": list(suffix_token_ids),
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
    ):
        """
        Server mode: only base_url is required. model name, tokenizer, and
            logprobs permission are auto-detected from the server.
        Offline mode: only llm is required. tokenizer is auto-loaded from the LLM.
        """
        self.model = None       # server API model id (may be alias)
        self.model_root = None  # actual model path on disk
        if base_url is not None:
            self.mode = "server"
            self.base_url = base_url
            self.api_key = api_key
            self.temperature = temperature
            self.top_p = top_p
            self._auto_detect_server_info()
            print(f"[LogProbsExtractor] Server mode: {self.base_url} | model: {self.model}")
        elif llm is not None:
            self.mode = "offline"
            self.llm = llm
            model_config = self.llm.llm_engine.model_config
            self.model_root = model_config.model
            print(f"[LogProbsExtractor] Offline mode | model: {self.model_root}")
        else:
            raise ValueError("Must provide either base_url (server mode) or llm (Offline mode)")

        if thinking_bypass_prefill is None:
            model_identity = " ".join(
                value for value in (self.model_root, self.model) if value
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

        # auto-load tokenizer
        if self.model_root:
            print(f"[LogProbsExtractor] Auto-loading tokenizer from: {self.model_root}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_root, trust_remote_code=True)
        else:
            raise ValueError("Cannot auto-load tokenizer: no model_root available.")

    def _auto_detect_server_info(self):
        """Query vLLM server to get model name, path, and check logprobs permission."""
        from openai import OpenAI
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            models = client.models.list()
            if not models.data:
                return

            server_model = models.data[0]
            self.model = server_model.id
            self.model_root = getattr(server_model, "root", None)

            # check logprobs permission
            permissions = getattr(server_model, "permission", [])
            if permissions:
                perm = permissions[0] if isinstance(permissions[0], dict) else permissions[0].__dict__
                allow_logprobs = perm.get("allow_logprobs", None)
                if allow_logprobs is False:
                    raise PermissionError(f"Server model '{self.model}' does not allow logprobs")

            print(f"[LogProbsExtractor] Auto-detected model: {self.model} | root: {self.model_root}")
        except PermissionError:
            raise
        except Exception as e:
            print(f"[LogProbsExtractor] Could not query server for model info: {e}")

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
        all_input_tokens, semantic_suffix_tokens = self._tokenize(messages)

        if self.mode == "server":
            return self._get_logprobs_server(
                all_input_tokens,
                semantic_suffix_tokens,
                logprobs_num,
            )
        else:
            return self._get_logprobs_offline(
                all_input_tokens,
                semantic_suffix_tokens,
                logprobs_num,
            )

    def _get_logprobs_server(
        self,
        all_input_tokens,
        semantic_suffix_tokens,
        logprobs_num,
    ):
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        response = client.completions.create(
            model=self.model,
            prompt=all_input_tokens,
            max_tokens=1,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_body={
                "prompt_logprobs": logprobs_num,
                "add_special_tokens": False,
            }
        )

        choice = response.choices[0]
        return _extract_suffix_result(
            text=choice.text,
            prompt_logprobs=getattr(choice, "prompt_logprobs", None),
            suffix_token_ids=semantic_suffix_tokens,
        )

    def _get_logprobs_offline(
        self,
        all_input_tokens,
        semantic_suffix_tokens,
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
        return _extract_suffix_result(
            text=output.outputs[0].text,
            prompt_logprobs=output.prompt_logprobs,
            suffix_token_ids=semantic_suffix_tokens,
        )
