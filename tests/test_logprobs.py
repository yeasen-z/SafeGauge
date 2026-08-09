import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from safegauge.config import THINKING_BYPASS_PREFILL
from safegauge.logprobs import LogProbsExtractor, _extract_suffix_result
from safegauge.server_backends import ServerModelInfo, SuffixLogprobs


def logprob(logprob_value, rank, decoded_token):
    return {
        "logprob": logprob_value,
        "rank": rank,
        "decoded_token": decoded_token,
    }


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.rendered_messages = messages
        return [10, 11]

    def encode(self, text, **kwargs):
        if text == THINKING_BYPASS_PREFILL["qwen3"]:
            return [20]
        if text == "semantic suffix":
            return [30, 31]
        raise AssertionError(f"Unexpected text passed to tokenizer: {text!r}")


class SuffixResultTests(unittest.TestCase):
    def test_result_uses_absolute_prompt_span(self):
        result = _extract_suffix_result(
            text="generated",
            suffix_logprobs=[
                {30: logprob(-0.2, 1, "semantic")},
                {31: logprob(-0.3, 2, " suffix")},
            ],
            suffix_token_ids=[30, 31],
            suffix_start=3,
        )

        self.assertEqual(result["suffix_start"], 3)
        self.assertEqual(result["suffix_end"], 5)
        self.assertEqual(result["suffix_token_ids"], [30, 31])
        self.assertEqual(result["all_logprobs"], [-0.2, -0.3])
        self.assertEqual(result["all_rank"], [1, 2])

    def test_result_rejects_wrong_length(self):
        with self.assertRaisesRegex(ValueError, "length does not match"):
            _extract_suffix_result(
                text="",
                suffix_logprobs=[{30: logprob(-0.2, 1, None)}],
                suffix_token_ids=[30, 31],
                suffix_start=3,
            )

    def test_result_rejects_wrong_observed_token(self):
        with self.assertRaisesRegex(ValueError, "token ID 30"):
            _extract_suffix_result(
                text="",
                suffix_logprobs=[{99: logprob(-0.2, 1, None)}],
                suffix_token_ids=[30],
                suffix_start=3,
            )

    def test_result_rejects_missing_position(self):
        with self.assertRaisesRegex(ValueError, "missing suffix logprobs"):
            _extract_suffix_result(
                text="",
                suffix_logprobs=[None],
                suffix_token_ids=[30],
                suffix_start=3,
            )

    def test_result_accepts_openai_json_shape(self):
        result = _extract_suffix_result(
            text="",
            suffix_logprobs=[
                {
                    "30": {
                        "logprob": -0.2,
                        "rank": 1,
                        "decoded_token": "semantic",
                    }
                }
            ],
            suffix_token_ids=[30],
            suffix_start=3,
        )
        self.assertEqual(result["all_logprobs"], [-0.2])

    def test_get_logprobs_rejects_negative_top_k(self):
        extractor = LogProbsExtractor.__new__(LogProbsExtractor)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            extractor.get_logprobs([], logprobs_num=-1)


class TokenizationTests(unittest.TestCase):
    def test_bypass_is_context_but_not_part_of_suffix_span(self):
        extractor = LogProbsExtractor.__new__(LogProbsExtractor)
        extractor.thinking_bypass_prefill = "qwen3"
        extractor.tokenizer = FakeTokenizer()
        prefix = THINKING_BYPASS_PREFILL["qwen3"]
        messages = [
            {"role": "user", "content": "input"},
            {"role": "assistant", "content": prefix + "semantic suffix"},
        ]

        all_tokens, suffix_tokens = extractor._tokenize(messages)

        self.assertEqual(all_tokens, [10, 11, 20, 30, 31])
        self.assertEqual(suffix_tokens, [30, 31])
        self.assertEqual(len(all_tokens) - len(suffix_tokens), 3)

    @patch("safegauge.logprobs.AutoTokenizer.from_pretrained")
    @patch("safegauge.logprobs.create_server_backend")
    def test_server_backend_defaults_to_vllm(
        self,
        create_backend,
        from_pretrained,
    ):
        backend = Mock()
        backend.name = "vllm"
        backend.discover.return_value = ServerModelInfo(
            model="model",
            model_root="model",
            tokenizer_path="tokenizer",
        )
        backend.score.return_value = SuffixLogprobs("", [])
        create_backend.return_value = backend

        extractor = LogProbsExtractor(base_url="http://server/v1")

        create_backend.assert_called_once_with(
            "vllm",
            "http://server/v1",
            "none",
        )
        from_pretrained.assert_called_once_with(
            "tokenizer",
            trust_remote_code=True,
        )
        self.assertEqual(extractor.server_backend, "vllm")


class OfflineVLLMTests(unittest.TestCase):
    def test_offline_path_keeps_absolute_suffix_span(self):
        prompt_logprobs = [
            None,
            {11: logprob(-0.1, 1, "context")},
            {20: logprob(-0.4, 3, "bypass")},
            {30: logprob(-0.2, 1, "semantic")},
            {31: logprob(-0.3, 2, " suffix")},
        ]
        output = SimpleNamespace(
            prompt_logprobs=prompt_logprobs,
            outputs=[SimpleNamespace(text="x")],
        )
        extractor = LogProbsExtractor.__new__(LogProbsExtractor)
        extractor.llm = SimpleNamespace(
            generate=Mock(return_value=[output]),
        )

        fake_vllm = SimpleNamespace(SamplingParams=lambda **kwargs: kwargs)
        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            result = extractor._get_logprobs_offline(
                [10, 11, 20, 30, 31],
                [30, 31],
                3,
                2,
            )

        self.assertEqual(result["suffix_start"], 3)
        self.assertEqual(result["suffix_end"], 5)
        self.assertEqual(result["all_logprobs"], [-0.2, -0.3])


if __name__ == "__main__":
    unittest.main()
