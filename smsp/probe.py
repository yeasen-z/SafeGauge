import json
import os

import numpy as np
import torch

from .logprobs import SuffixLogProbsExtractor
from .mlp import BinaryMlp


class SuffixRiskProbe:
    """Score a conversation using the log probabilities of a supplied suffix."""

    def __init__(
        self,
        checkpoint_path: str,
        base_url: str = None,
        api_key: str = "none",
        llm=None,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Probe checkpoint not found: {checkpoint_path}")

        self.mlp_model = BinaryMlp.load(checkpoint_path, map_location=self.device)
        self.mlp_model.eval()
        meta_path = os.path.splitext(checkpoint_path)[0] + ".meta.json"
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        reasoning_parser = self.meta.get("reasoning_parser", "none")
        if base_url is not None:
            self.logprobs_extractor = SuffixLogProbsExtractor(
                reasoning_parser=reasoning_parser,
                base_url=base_url,
                api_key=api_key,
            )
        elif llm is not None:
            self.logprobs_extractor = SuffixLogProbsExtractor(
                reasoning_parser=reasoning_parser,
                llm=llm,
            )
        else:
            raise ValueError("Must provide either base_url (server mode) or llm (offline mode)")

        self.model_name = (
            self.logprobs_extractor.model
            or self.logprobs_extractor.model_root
        )

    def _logprobs_to_features(self, logprobs: list, input_dim: int) -> torch.Tensor:
        arr = np.array(logprobs[:input_dim], dtype=np.float32)
        if len(arr) < input_dim:
            pad = np.full(input_dim - len(arr), -10.0, dtype=np.float32)
            arr = np.concatenate([arr, pad])
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)

    def probe(self, messages: list, suffix: str) -> dict:
        suffixed_messages = self.logprobs_extractor.apply_suffix(messages, suffix)
        logprobs_result = self.logprobs_extractor.get_logprobs(suffixed_messages)
        raw_logprobs = logprobs_result["all_logprobs"]

        threshold = self.meta.get("best_threshold", 0.5)
        input_dim = self.mlp_model.input_dim
        features = self._logprobs_to_features(raw_logprobs, input_dim)

        with torch.no_grad():
            logits = self.mlp_model(features)
            score = torch.sigmoid(logits).item()

        return {
            "label": "risk" if score > threshold else "safe",
            "score": round(score, 6),
            "threshold": threshold,
            "suffix": suffix,
            "logprobs": raw_logprobs[:input_dim],
        }

    def probe_batch(self, messages_list: list, suffix: str) -> list:
        results = []
        for messages in messages_list:
            try:
                results.append(self.probe(messages, suffix))
            except Exception as exc:
                results.append({
                    "label": "error",
                    "score": 0.0,
                    "suffix": suffix,
                    "error": str(exc),
                })
        return results

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "reasoning_parser": self.logprobs_extractor.reasoning_parser,
            "mode": self.logprobs_extractor.mode,
            "metadata": self.meta,
        }
