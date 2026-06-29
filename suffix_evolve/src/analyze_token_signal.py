#!/usr/bin/env python
"""Token-level attribution for a cached suffix probe."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.safetybench_smsp import arrays, ensemble_predict, load_cache  # noqa: E402
from smsp.mlp import BinaryMlp  # noqa: E402
from suffix_evolve.src.run_search import candidate_population  # noqa: E402
from suffix_evolve.src.search_metrics import metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("suffix_evolve/results/semantic_pilot_v1"),
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=Path("suffix_evolve/configs/semantic_candidates_v1.json"),
    )
    parser.add_argument(
        "--suffix-ids",
        nargs="+",
        default=["mutation_0b76a477d0", "baseline_careful_correct"],
    )
    parser.add_argument(
        "--model-root",
        default="/share/workspace/models/meta/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ordered_cache(path: Path) -> list[dict]:
    return sorted(load_cache(path).values(), key=lambda item: item["sample_id"])


def predict_from_states(X: np.ndarray, states: list[dict]) -> np.ndarray:
    predictions = []
    for state in states:
        mean = np.asarray(state["normalization_mean"])
        std = np.asarray(state["normalization_std"])
        values = torch.tensor((X - mean) / std, dtype=torch.float32)
        model = BinaryMlp(int(state["input_dim"]))
        model.load_state_dict(state["state_dict"])
        model.eval()
        with torch.no_grad():
            predictions.append(torch.sigmoid(model(values)).numpy().reshape(-1))
    return np.mean(predictions, axis=0)


def analyze(
    suffix: dict,
    run_dir: Path,
    tokenizer,
    permutations: int,
    seed: int,
) -> dict:
    suffix_id = suffix["id"]
    train = ordered_cache(run_dir / "features" / f"{suffix_id}__fit.jsonl")
    search = ordered_cache(run_dir / "features" / f"{suffix_id}__search.jsonl")
    X_train, y_train = arrays(train)
    X_search, y_search = arrays(search)
    scores, states = ensemble_predict(
        X_train, y_train, X_search, 300, 1e-2, [42, 43, 44]
    )
    base = metrics(y_search, scores)
    token_ids = tokenizer.encode(suffix["text"], add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    if len(tokens) != X_search.shape[1]:
        raise ValueError(
            f"Token/feature mismatch for {suffix_id}: {len(tokens)} vs {X_search.shape[1]}"
        )
    rng = np.random.default_rng(seed)
    rows = []
    for index, (token_id, token) in enumerate(zip(token_ids, tokens)):
        raw_auc = float(roc_auc_score(y_search, X_search[:, index]))
        drops = []
        auc_drops = []
        tpr_drops = []
        for _ in range(permutations):
            permuted = X_search.copy()
            permuted[:, index] = rng.permutation(permuted[:, index])
            changed = metrics(y_search, predict_from_states(permuted, states))
            drops.append(base.joint - changed.joint)
            auc_drops.append(base.roc_auc - changed.roc_auc)
            tpr_drops.append(base.tpr_at_fpr_005 - changed.tpr_at_fpr_005)
        positive = X_search[y_search == 1, index]
        negative = X_search[y_search == 0, index]
        rows.append(
            {
                "feature_index": index,
                "token_id": int(token_id),
                "token": token,
                "decoded": tokenizer.decode([token_id]),
                "positive_mean_logprob": float(positive.mean()),
                "negative_mean_logprob": float(negative.mean()),
                "mean_logprob_gap": float(positive.mean() - negative.mean()),
                "univariate_auc": raw_auc,
                "univariate_auc_direction_free": max(raw_auc, 1 - raw_auc),
                "permutation_joint_drop_mean": float(np.mean(drops)),
                "permutation_joint_drop_std": float(np.std(drops)),
                "permutation_auc_drop_mean": float(np.mean(auc_drops)),
                "permutation_tpr5_drop_mean": float(np.mean(tpr_drops)),
            }
        )
    return {
        "suffix_id": suffix_id,
        "text": suffix["text"],
        "fit_samples": len(train),
        "search_samples": len(search),
        "roc_auc": base.roc_auc,
        "tpr_at_fpr_0.05": base.tpr_at_fpr_005,
        "joint": base.joint,
        "tokens_by_position": rows,
        "tokens_by_permutation_importance": sorted(
            rows,
            key=lambda row: row["permutation_joint_drop_mean"],
            reverse=True,
        ),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.candidate_config.read_text(encoding="utf-8"))
    population = candidate_population(config, True)
    by_id = {item["id"]: item for item in population}
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root, trust_remote_code=True
    )
    output = {
        "method": (
            "Repeated feature permutation through the fitted three-seed MLP; "
            "positive drop means the ordered token logprob feature was useful."
        ),
        "suffixes": [
            analyze(
                by_id[suffix_id],
                args.run_dir,
                tokenizer,
                args.permutations,
                args.seed,
            )
            for suffix_id in args.suffix_ids
        ],
    }
    path = args.run_dir / "token_signal_analysis.json"
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for suffix in output["suffixes"]:
        print(f"\n{suffix['suffix_id']}: {suffix['text']}")
        for row in suffix["tokens_by_permutation_importance"][:8]:
            print(
                f"  {row['feature_index']:02d} {row['decoded']!r}: "
                f"joint_drop={row['permutation_joint_drop_mean']:+.4f}, "
                f"auc_drop={row['permutation_auc_drop_mean']:+.4f}, "
                f"tpr5_drop={row['permutation_tpr5_drop_mean']:+.4f}"
            )


if __name__ == "__main__":
    main()
