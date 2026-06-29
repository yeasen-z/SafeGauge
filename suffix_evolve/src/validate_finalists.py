#!/usr/bin/env python
"""Confirm search finalists on validation, then optionally run test once."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smsp.helper import (  # noqa: E402
    arrays,
    candidate_metrics,
    ensemble_predict,
    load_cache,
)
from suffix_evolve.src.evolution_extract import extract_split, verify_suffix  # noqa: E402
from suffix_evolve.src.reliable_logprobs import (  # noqa: E402
    ActualTokenLogProbsExtractor,
)
from suffix_evolve.src.run_search import (  # noqa: E402
    candidate_population,
    load_zh_candidates,
)
from suffix_evolve.src.search_metrics import (  # noqa: E402
    metrics,
    paired_question_bootstrap_delta,
    pareto_ids,
    question_bootstrap_lcb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("data/safetybench/bilingual_answer_heldout_v1/config.json"),
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=Path("suffix_evolve/configs/semantic_candidates_v1.json"),
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("suffix_evolve/results/semantic_v1")
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:24509/v1")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument(
        "--finalist-ids",
        nargs="+",
        help="Optional frozen subset of search finalists to validate.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="After validation freezes one suffix, extract and score test once.",
    )
    parser.add_argument(
        "--baseline-suffix-id",
        default="baseline_correctness_statement",
        help="Suffix ID to use as the comparison baseline.",
    )
    parser.add_argument(
        "--training-source",
        choices=["full_train", "search_cache"],
        default="full_train",
        help="Use full train extraction or reuse search-stage fit/search caches.",
    )
    return parser.parse_args()


def score_split(
    train_records: list[dict],
    eval_records: list[dict],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    X_train, y_train = arrays(train_records)
    X_eval, y_eval = arrays(eval_records)
    scores, states = ensemble_predict(
        X_train, y_train, X_eval, args.epochs, args.lr, args.seeds
    )
    uncertainty = question_bootstrap_lcb(
        y_eval,
        scores,
        np.asarray([item["question_id"] for item in eval_records]),
        repetitions=args.bootstrap,
        seed=42,
    )
    return scores, {"uncertainty": uncertainty, "model_states": states}


def score_records(
    train_records: list[dict],
    eval_records: list[dict],
    args: argparse.Namespace,
) -> np.ndarray:
    X_train, y_train = arrays(train_records)
    X_eval, _ = arrays(eval_records)
    scores, _ = ensemble_predict(
        X_train, y_train, X_eval, args.epochs, args.lr, args.seeds
    )
    return scores


def load_ordered_cache(path: Path) -> list[dict]:
    return sorted(load_cache(path).values(), key=lambda item: item["sample_id"])


def main() -> None:
    args = parse_args()
    search = json.loads(
        (args.run_dir / "search_results.json").read_text(encoding="utf-8")
    )
    data_config = json.loads(args.data_config.read_text(encoding="utf-8"))
    candidate_config = json.loads(
        args.candidate_config.read_text(encoding="utf-8")
    )
    population = candidate_population(candidate_config, True)
    by_id = {item["id"]: item for item in population}
    finalist_ids = args.finalist_ids or search["finalist_suffix_ids"]
    not_search_finalists = set(finalist_ids) - set(search["finalist_suffix_ids"])
    if not_search_finalists:
        raise ValueError(
            f"Requested IDs were not frozen search finalists: {sorted(not_search_finalists)}"
        )
    missing = set(finalist_ids) - set(by_id)
    if missing:
        raise ValueError(f"Finalists missing from candidate config: {sorted(missing)}")

    extractor = ActualTokenLogProbsExtractor(
        base_url=args.base_url, api_key=args.api_key
    )
    finalists = [verify_suffix(extractor, by_id[item]) for item in finalist_ids]
    baseline_suffix = verify_suffix(extractor, by_id[args.baseline_suffix_id])
    train_frame = load_zh_candidates(data_config, "train")
    validation_frame = load_zh_candidates(data_config, "validation")
    feature_dir = args.run_dir / "features"
    rows = []
    records_by_id = {}
    scores_by_id = {}
    for suffix in finalists:
        if args.training_source == "search_cache":
            train_records = load_ordered_cache(
                feature_dir / f"{suffix['id']}__fit.jsonl"
            ) + load_ordered_cache(feature_dir / f"{suffix['id']}__search.jsonl")
        else:
            train_records = extract_split(
                train_frame,
                suffix,
                extractor,
                feature_dir / f"{suffix['id']}__train.jsonl",
                args.workers,
            )
        validation_records = extract_split(
            validation_frame,
            suffix,
            extractor,
            feature_dir / f"{suffix['id']}__validation.jsonl",
            args.workers,
        )
        validation_scores, result = score_split(
            train_records, validation_records, args
        )
        row = {
            "suffix_id": suffix["id"],
            "family": suffix["family"],
            "text": suffix["text"],
            "token_count": suffix["token_count"],
            **result["uncertainty"],
        }
        rows.append(row)
        records_by_id[suffix["id"]] = (train_records, validation_records)
        scores_by_id[suffix["id"]] = validation_scores
        print(json.dumps(row, ensure_ascii=False), flush=True)

    pareto = pareto_ids(rows)
    selected = max(rows, key=lambda row: row["joint_lcb"])
    output = {
        "selection_split": "validation",
        "test_used_for_selection": False,
        "pareto_suffix_ids": pareto,
        "selected_suffix": selected,
        "validation_candidates": sorted(
            rows, key=lambda row: row["joint_lcb"], reverse=True
        ),
    }
    selected_validation_records = records_by_id[selected["suffix_id"]][1]
    selected_validation_scores = scores_by_id[selected["suffix_id"]]
    if args.training_source == "search_cache":
        baseline_train_records = load_ordered_cache(
            feature_dir / f"{baseline_suffix['id']}__fit.jsonl"
        ) + load_ordered_cache(feature_dir / f"{baseline_suffix['id']}__search.jsonl")
    else:
        baseline_train_records = extract_split(
            train_frame,
            baseline_suffix,
            extractor,
            feature_dir / f"{baseline_suffix['id']}__train.jsonl",
            args.workers,
        )
    baseline_validation_records = extract_split(
        validation_frame,
        baseline_suffix,
        extractor,
        feature_dir / f"{baseline_suffix['id']}__validation.jsonl",
        args.workers,
    )
    baseline_validation_scores = score_records(
        baseline_train_records,
        baseline_validation_records,
        args,
    )
    validation_y = np.asarray(
        [item["label"] for item in selected_validation_records]
    )
    validation_question_ids = np.asarray(
        [item["question_id"] for item in selected_validation_records]
    )
    baseline_point = metrics(validation_y, baseline_validation_scores)
    output["baseline_validation"] = {
        "suffix_id": baseline_suffix["id"],
        "text": baseline_suffix["text"],
        "roc_auc": baseline_point.roc_auc,
        "tpr_at_fpr_0.05": baseline_point.tpr_at_fpr_005,
        "joint": baseline_point.joint,
    }
    output["paired_delta_vs_baseline_validation"] = (
        paired_question_bootstrap_delta(
            validation_y,
            selected_validation_scores,
            baseline_validation_scores,
            validation_question_ids,
            repetitions=max(args.bootstrap, 1000),
            seed=42,
        )
    )
    if args.evaluate_test:
        suffix = next(item for item in finalists if item["id"] == selected["suffix_id"])
        test_frame = load_zh_candidates(data_config, "test")
        baseline_train_validation_records = baseline_train_records + baseline_validation_records
        baseline_test_records = extract_split(
            test_frame,
            baseline_suffix,
            extractor,
            feature_dir / f"{baseline_suffix['id']}__test.jsonl",
            args.workers,
        )
        test_records = extract_split(
            test_frame,
            suffix,
            extractor,
            feature_dir / f"{suffix['id']}__test.jsonl",
            args.workers,
        )
        train_records, validation_records = records_by_id[suffix["id"]]
        dev_records = train_records + validation_records
        test_scores = score_records(dev_records, test_records, args)
        baseline_test_scores = score_records(
            baseline_train_validation_records,
            baseline_test_records,
            args,
        )
        test_y = np.asarray([item["label"] for item in test_records])
        test_question_ids = np.asarray([item["question_id"] for item in test_records])
        output["held_out_test"] = candidate_metrics(test_records, test_scores)
        output["held_out_test"]["joint"] = 0.5 * (
            output["held_out_test"]["roc_auc"]
            + output["held_out_test"]["tpr_at_fpr_0.05"]["tpr"]
        )
        baseline_test_point = metrics(test_y, baseline_test_scores)
        output["baseline_held_out_test"] = {
            "suffix_id": baseline_suffix["id"],
            "text": baseline_suffix["text"],
            "roc_auc": baseline_test_point.roc_auc,
            "tpr_at_fpr_0.05": baseline_test_point.tpr_at_fpr_005,
            "joint": baseline_test_point.joint,
        }
        output["delta_vs_baseline"] = {
            "roc_auc": output["held_out_test"]["roc_auc"] - baseline_test_point.roc_auc,
            "tpr_at_fpr_0.05": (
                output["held_out_test"]["tpr_at_fpr_0.05"]["tpr"]
                - baseline_test_point.tpr_at_fpr_005
            ),
            "joint": output["held_out_test"]["joint"] - baseline_test_point.joint,
        }
        output["paired_delta_vs_baseline_test"] = paired_question_bootstrap_delta(
            test_y,
            np.asarray(test_scores),
            np.asarray(baseline_test_scores),
            test_question_ids,
            repetitions=max(args.bootstrap, 1000),
            seed=42,
        )
    (args.run_dir / "validation_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
