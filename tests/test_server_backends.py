import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from safegauge.logprobs import _extract_suffix_result
from safegauge.server_backends import (
    SGLangServerBackend,
    ServerRequestError,
    VLLMServerBackend,
    _normalize_sglang_positions,
    _sglang_root_url,
)


def vllm_entry(logprob, rank, text):
    return SimpleNamespace(logprob=logprob, rank=rank, decoded_token=text)


class FakeVLLMClient:
    def __init__(self, prompt_logprobs):
        model = SimpleNamespace(id="test-model", root="test-tokenizer", permission=[])
        self.models = SimpleNamespace(
            list=Mock(return_value=SimpleNamespace(data=[model]))
        )
        choice = SimpleNamespace(text="x", prompt_logprobs=prompt_logprobs)
        self.completions = SimpleNamespace(
            create=Mock(return_value=SimpleNamespace(choices=[choice]))
        )


class VLLMBackendTests(unittest.TestCase):
    def test_vllm_returns_only_trailing_suffix_positions(self):
        prompt_logprobs = [
            None,
            {11: vllm_entry(-0.1, 1, "context")},
            {20: vllm_entry(-0.4, 3, "bypass")},
            {30: vllm_entry(-0.2, 1, "semantic")},
            {31: vllm_entry(-0.3, 2, " suffix")},
        ]
        client = FakeVLLMClient(prompt_logprobs)
        backend = VLLMServerBackend("http://server/v1", "none", client=client)

        info = backend.discover()
        scored = backend.score([10, 11, 20, 30, 31], 3, 2, 0.0, 0.95)

        self.assertEqual(info.model, "test-model")
        self.assertEqual(info.tokenizer_path, "test-tokenizer")
        self.assertEqual(scored.positions, prompt_logprobs[-2:])
        request = client.completions.create.call_args.kwargs
        self.assertEqual(request["prompt"], [10, 11, 20, 30, 31])
        self.assertEqual(request["extra_body"]["prompt_logprobs"], 2)
        self.assertFalse(request["extra_body"]["add_special_tokens"])


class SGLangBackendTests(unittest.TestCase):
    def test_root_url_accepts_server_root_or_v1_url(self):
        self.assertEqual(_sglang_root_url("http://server:30000"), "http://server:30000")
        self.assertEqual(
            _sglang_root_url("http://server:30000/v1/"), "http://server:30000"
        )

    def test_normalizer_merges_observed_and_top_logprobs(self):
        meta_info = {
            "input_token_logprobs": [
                [None, 20, "bypass"],
                [-0.2, 30, "semantic"],
                [-0.3, 31, " suffix"],
            ],
            "input_top_logprobs": [
                None,
                [[-0.1, 99, "other"], [-0.2, 30, "semantic"]],
                [[-0.1, 98, "other"], [-0.2, 97, "other2"]],
            ],
        }

        positions = _normalize_sglang_positions(
            meta_info,
            [30, 31],
            lookback_token_id=20,
            top_k=2,
        )
        result = _extract_suffix_result(
            text="x",
            suffix_logprobs=positions,
            suffix_token_ids=[30, 31],
            suffix_start=3,
        )

        self.assertEqual(result["all_logprobs"], [-0.2, -0.3])
        self.assertEqual(result["all_rank"], [2, None])
        self.assertIn(31, result["suffix_logprobs"][1])

    def test_normalizer_rejects_token_misalignment(self):
        meta_info = {
            "input_token_logprobs": [
                [None, 20, None],
                [-0.2, 99, None],
            ],
            "input_top_logprobs": [None, [[-0.2, 99, None]]],
        }
        with self.assertRaisesRegex(ValueError, "alignment mismatch"):
            _normalize_sglang_positions(
                meta_info,
                [30],
                lookback_token_id=20,
                top_k=1,
            )

    def test_normalizer_rejects_missing_lookback_row(self):
        meta_info = {
            "input_token_logprobs": [[-0.2, 30, None]],
            "input_top_logprobs": [[[-0.2, 30, None]]],
        }
        with self.assertRaisesRegex(ValueError, "lookback plus suffix"):
            _normalize_sglang_positions(
                meta_info,
                [30],
                lookback_token_id=20,
                top_k=1,
            )

    def test_normalizer_rejects_invalid_lookback_row(self):
        for lookback_row in ([None, 99, None], [-0.1, 20, None]):
            with self.subTest(lookback_row=lookback_row):
                with self.assertRaisesRegex(ValueError, "lookback alignment"):
                    _normalize_sglang_positions(
                        {
                            "input_token_logprobs": [
                                lookback_row,
                                [-0.2, 30, None],
                            ],
                            "input_top_logprobs": [
                                None,
                                [[-0.2, 30, None]],
                            ],
                        },
                        [30],
                        lookback_token_id=20,
                        top_k=1,
                    )

    def test_normalizer_allows_top_k_zero(self):
        positions = _normalize_sglang_positions(
            {
                "input_token_logprobs": [
                    [None, 20, None],
                    [-0.2, 30, None],
                ]
            },
            [30],
            lookback_token_id=20,
            top_k=0,
        )
        self.assertEqual(positions[0][30]["logprob"], -0.2)
        self.assertIsNone(positions[0][30]["rank"])

    def test_normalizer_rejects_missing_suffix_logprob(self):
        with self.assertRaisesRegex(ValueError, "no logprob for suffix token"):
            _normalize_sglang_positions(
                {
                    "input_token_logprobs": [
                        [None, 20, None],
                        [None, 30, None],
                    ]
                },
                [30],
                lookback_token_id=20,
                top_k=0,
            )

    def test_score_sends_native_generate_request(self):
        backend = SGLangServerBackend("http://server:30000/v1", "secret")
        backend._request = Mock(
            return_value={
                "text": "x",
                "meta_info": {
                    "input_token_logprobs": [
                        [None, 20, None],
                        [-0.2, 30, None],
                        [-0.3, 31, None],
                    ],
                    "input_top_logprobs": [
                        None,
                        [[-0.2, 30, None]],
                        [[-0.3, 31, None]],
                    ],
                },
            }
        )

        scored = backend.score([10, 11, 20, 30, 31], 3, 1, 0.0, 0.95)

        self.assertEqual(len(scored.positions), 2)
        self.assertEqual(backend.base_url, "http://server:30000")
        self.assertEqual(backend.headers["Authorization"], "Bearer secret")
        call = backend._request.call_args
        self.assertEqual(call.args, ("/generate",))
        payload = call.kwargs["payload"]
        self.assertEqual(payload["input_ids"], [10, 11, 20, 30, 31])
        self.assertEqual(payload["logprob_start_len"], 2)
        self.assertEqual(payload["top_logprobs_num"], 1)

    def test_discovery_uses_current_endpoints(self):
        backend = SGLangServerBackend("http://server:30000", "none")

        def request(path, method="GET", payload=None):
            if path == "/model_info":
                return {
                    "model_path": "model",
                    "tokenizer_path": "tokenizer",
                }
            if path == "/server_info":
                return {"version": "0.5.x"}
            raise AssertionError(path)

        backend._request = Mock(side_effect=request)

        info = backend.discover()

        self.assertEqual(info.model, "model")
        self.assertEqual(info.tokenizer_path, "tokenizer")
        self.assertEqual(info.version, "0.5.x")
        self.assertEqual(
            [call.args[0] for call in backend._request.call_args_list],
            ["/model_info", "/server_info"],
        )

    def test_discovery_falls_back_to_legacy_endpoints(self):
        backend = SGLangServerBackend("http://server:30000", "none")

        def request(path, method="GET", payload=None):
            if path in ("/model_info", "/server_info"):
                raise ServerRequestError("not found", status_code=404)
            if path == "/get_model_info":
                return {
                    "model_path": "model",
                    "tokenizer_path": "tokenizer",
                }
            if path == "/get_server_info":
                return {"version": "0.4.x"}
            raise AssertionError(path)

        backend._request = request

        info = backend.discover()

        self.assertEqual(info.model, "model")
        self.assertEqual(info.tokenizer_path, "tokenizer")
        self.assertEqual(info.version, "0.4.x")

    def test_discovery_does_not_hide_server_errors(self):
        backend = SGLangServerBackend("http://server:30000", "none")
        backend._request = Mock(
            side_effect=ServerRequestError("server error", status_code=500)
        )

        with self.assertRaisesRegex(ServerRequestError, "server error"):
            backend.discover()

        backend._request.assert_called_once_with("/model_info")


if __name__ == "__main__":
    unittest.main()
