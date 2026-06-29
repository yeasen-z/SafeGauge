#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.safetybench_smsp import (  # noqa: E402
    arrays,
    ensemble_predict,
)
from suffix_evolve.src.evolution_extract import extract_split, verify_suffix  # noqa: E402
from suffix_evolve.src.reliable_logprobs import (  # noqa: E402
    ActualTokenLogProbsExtractor,
)
from suffix_evolve.src.search_metrics import (  # noqa: E402
    pareto_ids,
    question_bootstrap_lcb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested black-box suffix search for SafetyBench zh."
    )
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-fraction", type=float, default=0.20)
    parser.add_argument(
        "--stage-fit-questions", type=int, nargs="+", default=[200, 600, 1200]
    )
    parser.add_argument(
        "--stage-search-questions", type=int, nargs="+", default=[100, 300, 600]
    )
    parser.add_argument(
        "--stage-keep", type=int, nargs="+", default=[20, 8, 6]
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument(
        "--candidate-ids",
        nargs="+",
        help="Optional candidate subset, primarily for reproducibility checks.",
    )
    parser.add_argument(
        "--include-generated-mutations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def candidate_population(config: dict, include_mutations: bool) -> list[dict]:
    candidates = [dict(item) for item in config["candidates"]]
    if include_mutations and "mutation_phrases" in config:
        phrases = config["mutation_phrases"]
        for opener in phrases["openers"]:
            for subject in phrases["subjects"]:
                for judgment in phrases["judgments"]:
                    text = f"{opener}{subject}{judgment}"
                    candidates.append(
                        {
                            "id": f"mutation_{stable_id(text)}",
                            "family": "compositional_mutation",
                            "text": text,
                        }
                    )
    by_text: dict[str, dict] = {}
    for candidate in candidates:
        by_text.setdefault(candidate["text"], candidate)
    result = list(by_text.values())
    ids = [item["id"] for item in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique")
    return result


def load_zh_candidates(data_config: dict, split: str) -> pd.DataFrame:
    path = ROOT / data_config["files"][split]["candidates"]["path"]
    frame = pd.read_csv(path, keep_default_na=False)
    result = frame.loc[frame["language"] == "zh"].copy()
    if result.empty:
        raise ValueError(f"No Chinese candidates in {path}")
    return result.reset_index(drop=True)


def nested_question_split(
    frame: pd.DataFrame, search_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    questions = frame.drop_duplicates("question_id")[
        ["question_id", "category", "correct_answer_index"]
    ].reset_index(drop=True)
    strata = (
        questions["category"].astype(str)
        + "::"
        + questions["correct_answer_index"].astype(str)
    )
    counts = strata.value_counts()
    strata = strata.where(strata.map(counts) >= 2, "rare")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=search_fraction, random_state=seed
    )
    fit_indexes, search_indexes = next(splitter.split(questions, strata))
    return (
        questions.iloc[fit_indexes]["question_id"].to_numpy(),
        questions.iloc[search_indexes]["question_id"].to_numpy(),
    )


def deterministic_limit(ids: np.ndarray, maximum: int | None, seed: int) -> np.ndarray:
    ids = np.asarray(ids).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    if maximum is not None:
        ids = ids[:maximum]
    return ids


def subset(frame: pd.DataFrame, question_ids: np.ndarray, split: str) -> pd.DataFrame:
    result = frame.loc[frame["question_id"].isin(question_ids)].copy()
    result["split"] = split
    return result.reset_index(drop=True)


def evaluate_candidate(
    suffix: dict,
    fit_frame: pd.DataFrame,
    search_frame: pd.DataFrame,
    extractor: ActualTokenLogProbsExtractor,
    args: argparse.Namespace,
) -> dict:
    cache_dir = args.run_dir / "features"
    fit_records = extract_split(
        fit_frame,
        suffix,
        extractor,
        cache_dir / f"{suffix['id']}__fit.jsonl",
        args.workers,
    )
    search_records = extract_split(
        search_frame,
        suffix,
        extractor,
        cache_dir / f"{suffix['id']}__search.jsonl",
        args.workers,
    )
    X_fit, y_fit = arrays(fit_records)
    X_search, y_search = arrays(search_records)
    scores, _ = ensemble_predict(
        X_fit, y_fit, X_search, args.epochs, args.lr, args.seeds
    )
    uncertainty = question_bootstrap_lcb(
        y_search,
        scores,
        np.asarray([record["question_id"] for record in search_records]),
        repetitions=args.bootstrap,
        seed=args.seed,
    )
    return {
        "suffix_id": suffix["id"],
        "family": suffix["family"],
        "text": suffix["text"],
        "token_count": suffix["token_count"],
        **uncertainty,
    }


def main() -> None:
    args = parse_args()
    data_config = json.loads(args.data_config.read_text(encoding="utf-8"))
    candidate_config = json.loads(
        args.candidate_config.read_text(encoding="utf-8")
    )
    population = candidate_population(
        candidate_config, args.include_generated_mutations
    )
    if args.candidate_ids:
        requested = set(args.candidate_ids)
        unknown = requested - {item["id"] for item in population}
        if unknown:
            raise ValueError(f"Unknown candidate IDs: {sorted(unknown)}")
        population = [item for item in population if item["id"] in requested]
    protected_ids = list(candidate_config.get("protected_ids", []))
    unknown_protected = set(protected_ids) - {item["id"] for item in population}
    if unknown_protected:
        raise ValueError(f"Unknown protected candidate IDs: {sorted(unknown_protected)}")
    train = load_zh_candidates(data_config, "train")
    fit_ids, search_ids = nested_question_split(
        train, args.search_fraction, args.seed
    )
    stage_lengths = {
        len(args.stage_fit_questions),
        len(args.stage_search_questions),
        len(args.stage_keep),
    }
    if len(stage_lengths) != 1:
        raise ValueError("All --stage-* lists must have equal lengths")
    if any(value <= 0 for value in (*args.stage_fit_questions, *args.stage_search_questions)):
        raise ValueError("Stage question budgets must be positive")
    if any(value <= 0 for value in args.stage_keep):
        raise ValueError("Stage keep counts must be positive")
    if args.stage_fit_questions != sorted(args.stage_fit_questions):
        raise ValueError("Fit question budgets must be nondecreasing")
    if args.stage_search_questions != sorted(args.stage_search_questions):
        raise ValueError("Search question budgets must be nondecreasing")
    if args.stage_keep != sorted(args.stage_keep, reverse=True):
        raise ValueError("Stage keep counts must be nonincreasing")
    fit_ids = deterministic_limit(
        fit_ids, max(args.stage_fit_questions), args.seed
    )
    search_ids = deterministic_limit(
        search_ids, max(args.stage_search_questions), args.seed + 1
    )

    plan = {
        "candidate_count": len(population),
        "stages": [
            {
                "fit_questions": fit_budget,
                "search_questions": search_budget,
                "keep": keep,
            }
            for fit_budget, search_budget, keep in zip(
                args.stage_fit_questions,
                args.stage_search_questions,
                args.stage_keep,
            )
        ],
        "validation_touched": False,
        "test_touched": False,
        "objective": "question-bootstrap LCB of 0.5*AUROC + 0.5*TPR@FPR5",
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "search_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    extractor = ActualTokenLogProbsExtractor(
        base_url=args.base_url, api_key=args.api_key
    )
    verified = [verify_suffix(extractor, item) for item in population]
    active = verified
    stage_outputs = []
    for stage_index, (fit_budget, search_budget, keep) in enumerate(
        zip(
            args.stage_fit_questions,
            args.stage_search_questions,
            args.stage_keep,
        ),
        start=1,
    ):
        fit_frame = subset(train, fit_ids[:fit_budget], "fit")
        search_frame = subset(train, search_ids[:search_budget], "search")
        rows = [
            evaluate_candidate(item, fit_frame, search_frame, extractor, args)
            for item in active
        ]
        stage_pareto = set(pareto_ids(rows))
        ranking = sorted(
            rows,
            key=lambda row: (row["joint_lcb"], row["roc_auc"]),
            reverse=True,
        )
        protected_active = [
            item for item in protected_ids if any(row["suffix_id"] == item for row in rows)
        ]
        retained_ids = list(protected_active)
        retained_ids.extend(
            row["suffix_id"]
            for row in ranking
            if row["suffix_id"] not in retained_ids
        )
        retained_ids = retained_ids[:keep]
        stage_output = {
            "stage": stage_index,
            "fit_questions": fit_budget,
            "search_questions": search_budget,
            "evaluated": len(active),
            "pareto_suffix_ids": sorted(stage_pareto),
            "retained_suffix_ids": retained_ids,
            "ranking": ranking,
        }
        stage_outputs.append(stage_output)
        (args.run_dir / f"stage_{stage_index:02d}.json").write_text(
            json.dumps(stage_output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        active = [item for item in active if item["id"] in retained_ids]
    rows = stage_outputs[-1]["ranking"]
    pareto = set(pareto_ids(rows))
    ranking = sorted(
        rows,
        key=lambda row: (row["joint_lcb"], row["roc_auc"]),
        reverse=True,
    )
    finalists = stage_outputs[-1]["retained_suffix_ids"]
    output = {
        "protocol": plan,
        "pareto_suffix_ids": sorted(pareto),
        "finalist_suffix_ids": finalists,
        "stages": stage_outputs,
        "search_ranking": ranking,
        "next_step": (
            "Extract finalists on the untouched validation split; do not inspect "
            "test until one suffix is selected."
        ),
    }
    (args.run_dir / "search_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(ranking).to_csv(args.run_dir / "search_ranking.csv", index=False)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
