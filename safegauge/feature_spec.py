"""Versioned feature-space metadata shared by extraction, training, and serving."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


FEATURE_SPEC_SCHEMA = "safegauge.feature_spec.v1"
CHECKPOINT_SCHEMA = "safegauge.binary_mlp.v2"


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible metadata deterministically."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def feature_spec_hash(feature_spec: dict[str, Any]) -> str:
    """Return the stable identity of a complete feature specification."""
    return sha256_json(feature_spec)


def finalize_feature_spec(
    extraction_spec: dict[str, Any],
    *,
    input_dim: int,
    pad_value: float,
    positive_label: int = 1,
) -> dict[str, Any]:
    """Add the classifier-side vectorization contract to an extraction spec."""
    feature_spec = copy.deepcopy(extraction_spec)
    feature_spec["vectorization"] = {
        "input_dim": int(input_dim),
        "pad_value": float(pad_value),
        "order": "semantic_suffix_token_order",
        "overflow": "truncate_right",
        "underflow": "pad_right",
    }
    feature_spec["classification"] = {
        "positive_label": int(positive_label),
        "positive_meaning": "risk",
    }
    return feature_spec


def feature_spec_differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Return concise leaf paths that differ between two JSON values."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = []
        for key in sorted(set(expected) | set(actual)):
            child_path = f"{path}.{key}" if path else key
            if key not in expected or key not in actual:
                differences.append(child_path)
                continue
            differences.extend(
                feature_spec_differences(expected[key], actual[key], child_path)
            )
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        if expected == actual:
            return []
        return [path]
    return [] if expected == actual else [path]


def require_matching_feature_specs(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    context: str,
) -> None:
    """Raise with actionable paths when feature spaces are not identical."""
    if feature_spec_hash(expected) == feature_spec_hash(actual):
        return
    paths = feature_spec_differences(expected, actual)
    summary = ", ".join(paths[:8]) or "unknown fields"
    if len(paths) > 8:
        summary += f", ... ({len(paths)} differences)"
    raise ValueError(f"Feature spec mismatch for {context}: {summary}")
