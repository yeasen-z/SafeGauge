#!/usr/bin/env python
"""Recompute the joint search objective from existing baseline feature caches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.safetybench_smsp import (  # noqa: E402
    arrays,
    ensemble_predict,
    load_cache,
)
from suffix_evolve.src.search_metrics import (  # noqa: E402
    pareto_ids,
    question_bootstrap_lcb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(
            "results/safetybench_smsp/suffix_comparison/"
            "llama31_8b_zh_answer_correctness_v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("suffix_evolve/results/baseline_audit.json"),
    )
    parser.add_argument("--bootstrap", type=int, default=500)
    return parser.parse_args()


def ordered_records(path: Path) -> list[dict]:
    records = load_cache(path)
    return sorted(records.values(), key=lambda item: item["sample_id"])


def main() -> None:
    args = parse_args()
    suffixes = json.loads(
        (args.baseline_dir / "suffixes.json").read_text(encoding="utf-8")
    )["suffixes"]
    rows = []
    for suffix in suffixes:
        suffix_id = suffix["id"]
        train = ordered_records(
            args.baseline_dir / "features" / f"{suffix_id}__train.jsonl"
        )
        validation = ordered_records(
            args.baseline_dir / "features" / f"{suffix_id}__validation.jsonl"
        )
        X_train, y_train = arrays(train)
        X_validation, y_validation = arrays(validation)
        scores, _ = ensemble_predict(
            X_train, y_train, X_validation, 300, 1e-2, [42, 43, 44]
        )
        result = question_bootstrap_lcb(
            y_validation,
            scores,
            np.asarray([item["question_id"] for item in validation]),
            repetitions=args.bootstrap,
            seed=42,
        )
        rows.append(
            {
                "suffix_id": suffix_id,
                "text": suffix["text"],
                "token_count": suffix["token_count"],
                **result,
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    output = {
        "note": "Validation-only audit; held-out test was not read.",
        "objective": "0.5*AUROC + 0.5*TPR@FPR5 with question-bootstrap LCB",
        "pareto_suffix_ids": pareto_ids(rows),
        "ranking": sorted(rows, key=lambda row: row["joint_lcb"], reverse=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
