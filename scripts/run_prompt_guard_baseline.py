#!/usr/bin/env python
"""Evaluate a Prompt Guard sequence classifier on prepared SMSP JSONL splits."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "id": record["id"],
            "text": record["messages"][0]["content"],
            "label": int(record["label"]),
            "metadata": record.get("metadata", {}),
        }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def best_f1_threshold(labels: list[int], scores: list[float]) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def tpr_at_fpr(labels: list[int], scores: list[float], max_fpr: float = 0.05) -> dict[str, float | None]:
    if len(set(labels)) < 2:
        return {"tpr": None, "threshold": None}
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid = np.where(fpr <= max_fpr + 1e-12)[0]
    if len(valid) == 0:
        return {"tpr": 0.0, "threshold": None}
    best = valid[int(np.argmax(tpr[valid]))]
    threshold = float(thresholds[best])
    if not np.isfinite(threshold):
        threshold = None
    return {"tpr": float(tpr[best]), "threshold": threshold}


def evaluate(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    y_score = np.asarray(scores, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y == 1)).sum())
    tn = int(((y_pred == 0) & (y == 0)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    fpr = fp / (fp + tn) if fp + tn else None
    fnr = fn / (fn + tp) if fn + tp else None
    tpr5 = tpr_at_fpr(labels, scores, 0.05)
    return {
        "samples": int(len(labels)),
        "positive": int((y == 1).sum()),
        "negative": int((y == 0).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_score)),
        "average_precision": float(average_precision_score(y, y_score)),
        "tpr_at_fpr_0.05": tpr5,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "fpr": fpr,
        "fnr": fnr,
    }


def predict(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    model: torch.nn.Module,
    batch_size: int,
    max_length: int,
    device: str,
) -> list[dict[str, Any]]:
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [item["text"] for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return {
            "ids": [item["id"] for item in batch],
            "labels": [item["label"] for item in batch],
            "metadata": [item["metadata"] for item in batch],
            "encoded": encoded,
        }

    predictions = []
    loader = DataLoader(TextDataset(records), batch_size=batch_size, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict", leave=False):
            encoded = {key: value.to(device) for key, value in batch["encoded"].items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            for sample_id, label, metadata, prob in zip(
                batch["ids"], batch["labels"], batch["metadata"], probs
            ):
                predictions.append(
                    {
                        "id": sample_id,
                        "label": int(label),
                        "score": float(prob[1]),
                        "metadata": metadata,
                    }
                )
    return predictions


def write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def by_category_metrics(
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    val_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    test_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validation:
        val_by_category[row["metadata"].get("behavior_id", "unknown")].append(row)
    for row in test:
        test_by_category[row["metadata"].get("behavior_id", "unknown")].append(row)

    rows = []
    for category in sorted(test_by_category):
        val_rows = val_by_category[category]
        test_rows = test_by_category[category]
        threshold = best_f1_threshold(
            [row["label"] for row in val_rows],
            [row["score"] for row in val_rows],
        )
        metrics = evaluate(
            [row["label"] for row in test_rows],
            [row["score"] for row in test_rows],
            threshold,
        )
        tpr5 = metrics.pop("tpr_at_fpr_0.05")
        rows.append(
            {
                "behavior_id": category,
                "behavior": test_rows[0]["metadata"].get("behavior", "unknown"),
                **metrics,
                "tpr_at_fpr_0.05": tpr5["tpr"],
                "tpr_at_fpr_0.05_threshold": tpr5["threshold"],
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Prompt Guard model directory")
    parser.add_argument("--data-dir", type=Path, required=True, help="Prepared SMSP split directory")
    parser.add_argument("--train", default="train.jsonl")
    parser.add_argument("--validation", default="validation.jsonl")
    parser.add_argument("--test", default="test.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation_path = args.data_dir / args.validation
    if not validation_path.exists():
        validation_path = args.data_dir / args.train
    validation_records = load_jsonl(validation_path)
    test_records = load_jsonl(args.data_dir / args.test)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
    model.to(device)
    model.eval()
    validation_predictions = predict(
        validation_records,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    test_predictions = predict(
        test_records,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    write_predictions(args.output_dir / "validation_predictions.jsonl", validation_predictions)
    write_predictions(args.output_dir / "test_predictions.jsonl", test_predictions)

    threshold = best_f1_threshold(
        [row["label"] for row in validation_predictions],
        [row["score"] for row in validation_predictions],
    )
    metrics = {
        "model": str(args.model),
        "data_dir": str(args.data_dir),
        "validation_path": str(validation_path),
        "test_path": str(args.data_dir / args.test),
        "score_label": "LABEL_1",
        "score_interpretation": "Prompt Guard malicious probability",
        "max_length": args.max_length,
        "validation": evaluate(
            [row["label"] for row in validation_predictions],
            [row["score"] for row in validation_predictions],
            threshold,
        ),
        "test": evaluate(
            [row["label"] for row in test_predictions],
            [row["score"] for row in test_predictions],
            threshold,
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    category_rows = by_category_metrics(validation_predictions, test_predictions)
    (args.output_dir / "test_by_behavior_category_thresholds.json").write_text(
        json.dumps(category_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "test_by_behavior_category_thresholds.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(category_rows[0].keys()))
        writer.writeheader()
        writer.writerows(category_rows)

    print(json.dumps(metrics["test"], ensure_ascii=False, indent=2))
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
