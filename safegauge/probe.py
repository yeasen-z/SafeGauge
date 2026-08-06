import json
import os
import warnings

import numpy as np
import torch

from .feature_spec import (
    CHECKPOINT_SCHEMA,
    FEATURE_SPEC_SCHEMA,
    feature_spec_hash,
    finalize_feature_spec,
    require_matching_feature_specs,
)
from .logprobs import LogProbsExtractor
from .mlp import BinaryMlp


class SafeGauge:
    """Score a conversation using the log probabilities of a supplied suffix."""

    def __init__(
        self,
        checkpoint_path: str,
        base_url: str = None,
        api_key: str = "none",
        llm=None,
        device: str = None,
        unsafe_allow_feature_mismatch: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.unsafe_allow_feature_mismatch = unsafe_allow_feature_mismatch
        self._validated_suffixes: set[str] = set()

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Probe checkpoint not found: {checkpoint_path}")

        meta_path = os.path.splitext(checkpoint_path)[0] + ".meta.json"
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = {}
        if self.meta.get("schema") != CHECKPOINT_SCHEMA:
            self._contract_failure(
                f"Unsupported checkpoint schema {self.meta.get('schema')!r}; "
                f"expected {CHECKPOINT_SCHEMA!r}"
            )
        self.feature_spec = self.meta.get("feature_spec")
        if not isinstance(self.feature_spec, dict):
            self._contract_failure(
                "Checkpoint metadata has no versioned feature_spec. Re-extract "
                "features and retrain, or explicitly enable "
                "unsafe_allow_feature_mismatch for a legacy checkpoint."
            )
        elif self.meta.get("feature_spec_hash") != feature_spec_hash(
            self.feature_spec
        ):
            self._contract_failure("Checkpoint feature_spec_hash is invalid")
        elif self.feature_spec.get("schema") != FEATURE_SPEC_SCHEMA:
            self._contract_failure(
                f"Unsupported feature spec schema "
                f"{self.feature_spec.get('schema')!r}; "
                f"expected {FEATURE_SPEC_SCHEMA!r}"
            )

        self.mlp_model = BinaryMlp.load(checkpoint_path, map_location=self.device)
        self.mlp_model.eval()

        thinking_bypass_prefill = self.meta.get("thinking_bypass_prefill")
        if isinstance(self.feature_spec, dict):
            thinking_bypass_prefill = self.feature_spec.get(
                "thinking_bypass", {}
            ).get("profile", thinking_bypass_prefill)
        if base_url is not None:
            self.logprobs_extractor = LogProbsExtractor(
                thinking_bypass_prefill=thinking_bypass_prefill,
                base_url=base_url,
                api_key=api_key,
            )
        elif llm is not None:
            self.logprobs_extractor = LogProbsExtractor(
                thinking_bypass_prefill=thinking_bypass_prefill,
                llm=llm,
            )
        else:
            raise ValueError("Must provide either base_url (server mode) or llm (offline mode)")

        self.model_name = (
            self.logprobs_extractor.model
            or self.logprobs_extractor.model_root
        )
        expected_suffix = self.meta.get("suffix")
        if isinstance(self.feature_spec, dict):
            expected_suffix = self.feature_spec.get("semantic_suffix", {}).get(
                "text", expected_suffix
            )
        if expected_suffix:
            self.validate_suffix(expected_suffix)

    def _contract_failure(self, message: str) -> None:
        if self.unsafe_allow_feature_mismatch:
            warnings.warn(
                f"UNSAFE feature mismatch override: {message}",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        raise ValueError(message)

    def validate_suffix(self, suffix: str) -> None:
        """Validate model, tokenizer, bypass, suffix, and vectorization identity."""
        if suffix in self._validated_suffixes:
            return
        if not isinstance(self.feature_spec, dict):
            self._contract_failure(
                "Cannot validate a suffix because this checkpoint has no feature_spec"
            )
            self._validated_suffixes.add(suffix)
            return
        suffix_id = self.feature_spec.get("semantic_suffix", {}).get("id")
        extraction_spec = self.logprobs_extractor.feature_spec(suffix, suffix_id)
        runtime_spec = finalize_feature_spec(
            extraction_spec,
            input_dim=self.mlp_model.input_dim,
            pad_value=float(self.meta.get("pad_value", -10.0)),
            positive_label=1,
        )
        try:
            require_matching_feature_specs(
                self.feature_spec,
                runtime_spec,
                context="checkpoint and runtime",
            )
        except ValueError as exc:
            self._contract_failure(str(exc))
        self._validated_suffixes.add(suffix)

    def _logprobs_to_features(self, logprobs: list, input_dim: int) -> torch.Tensor:
        pad_value = float(self.meta.get("pad_value", -10.0))
        arr = np.array(
            [pad_value if item is None else item for item in logprobs[:input_dim]],
            dtype=np.float32,
        )
        if len(arr) < input_dim:
            pad = np.full(input_dim - len(arr), pad_value, dtype=np.float32)
            arr = np.concatenate([arr, pad])
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)

    def probe(self, messages: list, suffix: str) -> dict:
        self.validate_suffix(suffix)
        suffixed_messages = self.logprobs_extractor.apply_suffix(messages, suffix)
        logprobs_result = self.logprobs_extractor.get_logprobs(suffixed_messages)
        raw_logprobs = logprobs_result["all_logprobs"]

        threshold = float(self.meta.get("best_threshold", 0.5))
        input_dim = self.mlp_model.input_dim
        features = self._logprobs_to_features(raw_logprobs, input_dim)

        with torch.no_grad():
            logits = self.mlp_model(features)
            score = torch.sigmoid(logits).item()

        return {
            "label": "risk" if score >= threshold else "safe",
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
                    "score": None,
                    "suffix": suffix,
                    "error": str(exc),
                })
        return results

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "thinking_bypass_prefill": (
                self.logprobs_extractor.thinking_bypass_prefill
            ),
            "mode": self.logprobs_extractor.mode,
            "feature_spec_hash": self.meta.get("feature_spec_hash"),
            "unsafe_allow_feature_mismatch": self.unsafe_allow_feature_mismatch,
            "metadata": self.meta,
        }
