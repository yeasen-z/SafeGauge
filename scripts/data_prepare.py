#!/usr/bin/env python
"""Prepare SMSP JSONL splits from CSV/JSON/JSONL records.

Output records use the common SMSP format:

{
  "id": "train-000001",
  "messages": [{"role": "user", "content": "..."}],
  "label": 1,
  "metadata": {...}
}
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_TEXT_COLUMNS = ("messages", "prompt", "text", "question", "input")


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("records"), list):
            return data["records"]
        raise ValueError("JSON input must be a list or contain a records list")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported input format: {path}")


def parse_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list) and all(
                isinstance(item, dict) and "role" in item and "content" in item
                for item in parsed
            ):
                return parsed
        return [{"role": "user", "content": value}]
    raise ValueError("messages/text field must be a string or OpenAI messages list")


def label_value(value: Any, positive_values: set[str]) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in positive_values:
        return 1
    if normalized in {"0", "false", "no", "negative", "neg", "safe", "benign"}:
        return 0
    raise ValueError(f"Cannot parse label value: {value!r}")


def choose_text_column(rows: list[dict[str, Any]], requested: str | None) -> str:
    if requested:
        return requested
    if not rows:
        raise ValueError("Input has no rows")
    keys = set(rows[0])
    for key in DEFAULT_TEXT_COLUMNS:
        if key in keys:
            return key
    raise ValueError(
        "Could not infer text column; pass --text-column or provide messages/prompt/text/question/input"
    )


def normalize_rows(
    rows: list[dict[str, Any]],
    *,
    text_column: str,
    label_column: str,
    id_column: str | None,
    split_name: str,
    positive_values: set[str],
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        if text_column not in row:
            raise KeyError(f"Missing text column {text_column!r}")
        if label_column not in row:
            raise KeyError(f"Missing label column {label_column!r}")
        sample_id = row.get(id_column) if id_column else None
        if sample_id in (None, ""):
            sample_id = f"{split_name}-{index:06d}"
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {text_column, label_column}
        }
        output.append(
            {
                "id": str(sample_id),
                "messages": parse_messages(row[text_column]),
                "label": label_value(row[label_column], positive_values),
                "metadata": metadata,
            }
        )
    return output


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def stratified_split(
    records: list[dict[str, Any]],
    *,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    by_label: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_label.setdefault(int(record["label"]), []).append(record)
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for label_records in by_label.values():
        rng.shuffle(label_records)
        n = len(label_records)
        train_end = int(n * train_ratio)
        validation_end = train_end + int(n * validation_ratio)
        splits["train"].extend(label_records[:train_end])
        splits["validation"].extend(label_records[train_end:validation_end])
        splits["test"].extend(label_records[validation_end:])
    for split_records in splits.values():
        rng.shuffle(split_records)
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SMSP JSONL data")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output JSONL path for --no-split mode. Defaults to OUTPUT_DIR/data.jsonl.",
    )
    parser.add_argument("--text-column")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--id-column")
    parser.add_argument(
        "--positive-values",
        default="1,true,yes,positive,pos,risk,attack,toxic,harmful",
        help="Comma-separated string labels treated as positive",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Write all normalized records to data.jsonl instead of train/validation/test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None and args.output_file is None:
        raise ValueError("Pass --output-dir, or pass --output-file with --no-split")
    if args.output_file is not None and not args.no_split:
        raise ValueError("--output-file is only valid with --no-split")
    rows = read_records(args.input)
    text_column = choose_text_column(rows, args.text_column)
    positive_values = {
        item.strip().lower()
        for item in args.positive_values.split(",")
        if item.strip()
    }
    records = normalize_rows(
        rows,
        text_column=text_column,
        label_column=args.label_column,
        id_column=args.id_column,
        split_name="data",
        positive_values=positive_values,
    )
    if args.no_split:
        output_path = args.output_file or args.output_dir / "data.jsonl"
        write_jsonl(output_path, records)
        files = {"data": str(output_path)}
        manifest_dir = None if args.output_file else output_path.parent
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir = args.output_dir
        splits = stratified_split(
            records,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )
        files = {}
        for split_name, split_records in splits.items():
            for index, record in enumerate(split_records):
                record["id"] = record["id"] or f"{split_name}-{index:06d}"
            file_name = f"{split_name}.jsonl"
            write_jsonl(args.output_dir / file_name, split_records)
            files[split_name] = file_name
            positives = sum(int(item["label"]) for item in split_records)
            print(f"{split_name}: {len(split_records)} records, positives={positives}")
    manifest = {
        "schema": "smsp.records.v1",
        "source": str(args.input),
        "text_column": text_column,
        "label_column": args.label_column,
        "files": files,
    }
    if manifest_dir is not None:
        (manifest_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
