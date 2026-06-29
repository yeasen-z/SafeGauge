import copy
from typing import Dict, Optional
from transformers import AutoTokenizer
from .config import REASONING_PREFIX


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


class SuffixLogProbsExtractor:
    """Extract token log probabilities for an assistant suffix.

    Server mode: pass base_url (+ optional api_key).
    Offline mode: pass llm (vllm.LLM instance).
    """

    def __init__(
        self,
        reasoning_parser: str = "none",
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
        self.reasoning_parser = reasoning_parser

        if base_url is not None:
            self.mode = "server"
            self.base_url = base_url
            self.api_key = api_key
            self.temperature = temperature
            self.top_p = top_p
            self._auto_detect_server_info()
            print(f"[SuffixLogProbsExtractor] Server mode: {self.base_url} | model: {self.model}")
        elif llm is not None:
            self.mode = "offline"
            self.llm = llm
            self.model_root = self.llm.llm_engine.model_config.model
            print(f"[SuffixLogProbsExtractor] Offline mode | model: {self.model_root}")
        else:
            raise ValueError("Must provide either base_url (server mode) or llm (Offline mode)")

        # auto-load tokenizer
        if self.model_root:
            print(f"[SuffixLogProbsExtractor] Auto-loading tokenizer from: {self.model_root}")
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

            print(f"[SuffixLogProbsExtractor] Auto-detected model: {self.model} | root: {self.model_root}")
        except PermissionError:
            raise
        except Exception as e:
            print(f"[SuffixLogProbsExtractor] Could not query server for model info: {e}")

    def _tokenize(self, messages):
        msgs_tokens = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True
        )
        suffix_tokens = self.tokenizer.encode(
            messages[-1]["content"],
            add_special_tokens=False,
        )
        all_input_tokens = safe_token_concat(msgs_tokens, suffix_tokens)
        return all_input_tokens, suffix_tokens

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
        prefix = REASONING_PREFIX.get(self.reasoning_parser, "")
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
        all_input_tokens, suffix_tokens = self._tokenize(messages)

        if self.mode == "server":
            return self._get_logprobs_server(
                all_input_tokens,
                suffix_tokens,
                logprobs_num,
            )
        else:
            return self._get_logprobs_offline(
                all_input_tokens,
                suffix_tokens,
                logprobs_num,
            )

    def _get_logprobs_server(self, all_input_tokens, suffix_tokens, logprobs_num):
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        response = client.completions.create(
            model=self.model,
            prompt=all_input_tokens,
            max_tokens=1,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_body={
                "prompt_logprobs": logprobs_num
            }
        )

        choice = response.choices[0]
        suffix_logprobs = choice.prompt_logprobs[-len(suffix_tokens):]

        return {
            "text": choice.text,
            "suffix_logprobs": suffix_logprobs,
            "all_logprobs": [list(d.values())[0]["logprob"] for d in suffix_logprobs],
            "all_rank": [list(d.values())[0]["rank"] for d in suffix_logprobs],
        }

    def _get_logprobs_offline(self, all_input_tokens, suffix_tokens, logprobs_num):
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
        output_suffix_logprobs = output.prompt_logprobs[-len(suffix_tokens):]

        all_logprobs = []
        all_rank = []
        suffix_logprobs = []

        for token_dict in output_suffix_logprobs:
            if token_dict is None:
                all_logprobs.append(None)
                all_rank.append(None)
                continue

            token_id = list(token_dict.keys())[0]
            token_info = token_dict[token_id]
            all_logprobs.append(token_info.logprob)
            all_rank.append(token_info.rank)
            suffix_logprobs.append({
                token_id: {
                    "logprob": token_info.logprob,
                    "rank": token_info.rank,
                    "decoded_token": token_info.decoded_token
                }
            })

        return {
            "text": output.outputs[0].text,
            "suffix_logprobs": suffix_logprobs,
            "all_logprobs": all_logprobs,
            "all_rank": all_rank,
        }
