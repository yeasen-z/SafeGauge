#!/usr/bin/env python
"""Re-run SMSP probes on prepared JSONL datasets and suffix sets."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


TASKS = {
    "toxic_chat": {
        "data_dir": "data/toxic_chat/official_test_heldout_v1",
        "suffix_set": "suffix_evolve/suffix_sets.json::toxic_chat.smsp_candidates_v1",
        "train": "train.jsonl",
        "validation": None,
        "test": "test.jsonl",
        "primary_suffix_id": "content_moderation_review",
    },
    "safetybench": {
        "data_dir": "data/safetybench/bilingual_answer_heldout_v1",
        "suffix_set": "suffix_evolve/suffix_sets.json::safetybench.answer_correctness_v1",
        "train": "candidates_train.jsonl",
        "validation": "candidates_validation.jsonl",
        "test": "candidates_test.jsonl",
        "primary_suffix_id": "careful_correct_conclusion",
    },
    "ragtruth": {
        "data_dir": "data/ragtruth/official_test_heldout_v1",
        "suffix_set": "suffix_evolve/suffix_sets.json::ragtruth.hallucination_detection_v1",
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
        "primary_suffix_id": "fully_supported_grounded",
    },
    "halueval": {
        "data_dir": "data/halueval/pair_grouped_heldout_v1",
        "suffix_set": "suffix_evolve/suffix_sets.json::halueval.hallucination_detection_v1",
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
        "primary_suffix_id": "unsupported_or_conflicting",
    },
    "faithbench": {
        "data_dir": "data/faithbench/unwanted_only_source_grouped_v1",
        "suffix_set": "suffix_evolve/suffix_sets.json::faithbench.hallucination_detection_v1",
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
        "primary_suffix_id": "all_summary_claims_verified",
    },
    "bump": {
        "data_dir": "data/bump/article_grouped_heldout_v1",
        "suffix_set": "suffix_evolve/suffix_sets.json::bump.faithfulness_detection_v1",
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
        "primary_suffix_id": "careful_pair_consistency",
    },
    "unknown_unknowns": {
        "data_dir": "data/unknown_unknowns/trigger_heldout_v1",
        "suffix_set": "suffix_evolve/suffix_sets.json::unknown_unknowns.hidden_behavior_detection_v1",
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
        "primary_suffix_id": "hidden_behavior_present",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:24509/v1",
        help="OpenAI-compatible vLLM URL.",
    )
    parser.add_argument("--api-key", default="none")
    parser.add_argument(
        "--reasoning-parser",
        default="qwen3",
        choices=["none", "qwen3", "deepseek_r1"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reprobe_qwen35_35b_a3b"),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASKS),
        default=sorted(TASKS),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--logprobs-num", type=int, default=2)
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Run only the predeclared primary suffix for each task.",
    )
    parser.add_argument(
        "--limit-records",
        type=int,
        help="Create temporary truncated splits for a quick connectivity run.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_suffix_config(ref: str) -> dict[str, Any]:
    if "::" not in ref:
        return load_json(ROOT / ref)
    file_name, key = ref.split("::", 1)
    config = load_json(ROOT / file_name)
    current: Any = config.get("suffix_sets", config)
    for part in key.split("."):
        current = current[part]
    if not isinstance(current, dict):
        raise ValueError(f"Suffix config reference did not resolve to an object: {ref}")
    return current


def suffixes(ref: str) -> list[dict[str, Any]]:
    config = load_suffix_config(ref)
    if "suffixes" in config:
        return config["suffixes"]
    if "suffix" in config:
        return [{"id": config["name"], "text": config["suffix"]}]
    raise ValueError(f"No suffixes found in {ref}")


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def maybe_limited_copy(source: Path, dest: Path, limit: int | None) -> Path:
    if limit is None:
        return source
    dest.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, dest.open("w", encoding="utf-8") as out:
        for index, line in enumerate(src):
            if index >= limit:
                break
            out.write(line)
    return dest


def metric_value(metrics: dict[str, Any], key: str) -> float | None:
    section = metrics.get("validation")
    if section is None:
        section = next(
            (value for name, value in metrics.items() if name.startswith("test:")),
            None,
        )
    if section is None:
        return None
    value = section.get(key)
    if isinstance(value, dict):
        return value.get("tpr")
    return value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "base_url": args.base_url,
        "reasoning_parser": args.reasoning_parser,
        "tasks": {},
    }
    for task_name in args.tasks:
        task = TASKS[task_name]
        data_dir = ROOT / task["data_dir"]
        task_dir = args.output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        train_path = maybe_limited_copy(
            data_dir / task["train"],
            task_dir / "_limited" / task["train"],
            args.limit_records,
        )
        validation_path = None
        if task["validation"] is not None:
            validation_path = maybe_limited_copy(
                data_dir / task["validation"],
                task_dir / "_limited" / task["validation"],
                args.limit_records,
            )
        test_path = maybe_limited_copy(
            data_dir / task["test"],
            task_dir / "_limited" / task["test"],
            args.limit_records,
        )
        rows = []
        task_suffixes = suffixes(task["suffix_set"])
        if args.primary_only:
            primary_id = task["primary_suffix_id"]
            task_suffixes = [item for item in task_suffixes if item["id"] == primary_id]
            if not task_suffixes:
                raise ValueError(f"Primary suffix {primary_id} not found for {task_name}")
        for suffix in task_suffixes:
            suffix_id = suffix["id"]
            suffix_text = suffix["text"]
            run_dir = task_dir / suffix_id
            feature_dir = run_dir / "features"
            train_features = feature_dir / "train.jsonl"
            validation_features = feature_dir / "validation.jsonl"
            test_features = feature_dir / "test.jsonl"
            for split_name, input_path, output_path in (
                ("train", train_path, train_features),
                ("validation", validation_path, validation_features),
                ("test", test_path, test_features),
            ):
                if input_path is None:
                    continue
                command = [
                    sys.executable,
                    "scripts/get_logprobs.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--suffix",
                    suffix_text,
                    "--suffix-id",
                    suffix_id,
                    "--base-url",
                    args.base_url,
                    "--api-key",
                    args.api_key,
                    "--reasoning-parser",
                    args.reasoning_parser,
                    "--logprobs-num",
                    str(args.logprobs_num),
                ]
                if args.overwrite_features:
                    command.append("--overwrite")
                run(command)
            train_command = [
                sys.executable,
                "scripts/train_probe.py",
                "--train",
                str(train_features),
                "--output-dir",
                str(run_dir / "probe"),
                "--weighted-bce",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
            ]
            if validation_path is not None:
                train_command.extend(["--validation", str(validation_features)])
            train_command.extend(["--test", str(test_features)])
            run(train_command)
            metrics_path = run_dir / "probe" / "metrics.json"
            metrics = load_json(metrics_path)
            row = {
                "suffix_id": suffix_id,
                "suffix": suffix_text,
                "metrics_path": str(metrics_path),
                "selection_auc": metric_value(metrics, "roc_auc"),
                "selection_tpr_at_fpr_0.05": metric_value(
                    metrics,
                    "tpr_at_fpr_0.05",
                ),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.sort(
            key=lambda item: (
                -1.0 if item["selection_auc"] is None else item["selection_auc"],
                -1.0
                if item["selection_tpr_at_fpr_0.05"] is None
                else item["selection_tpr_at_fpr_0.05"],
            ),
            reverse=True,
        )
        summary["tasks"][task_name] = rows
        (task_dir / "summary.json").write_text(
            json.dumps({"task": task_name, "results": rows}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
