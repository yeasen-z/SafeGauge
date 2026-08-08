#!/usr/bin/env python
"""Train a SafeGauge BinaryMlp from extracted suffix logprob features."""
from __future__ import annotations

import argparse
import copy
import json
import sys
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
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safegauge.dataset import SimpleDataset  # noqa: E402
from safegauge.mlp import BinaryMlp  # noqa: E402


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


def feature_vector(record: dict[str, Any], input_dim: int, pad_value: float) -> np.ndarray:
    values = record.get("logprobs")
    if values is None:
        values = record.get("all_logprobs")
    if values is None:
        raise KeyError("Feature record must contain logprobs or all_logprobs")
    cleaned = [pad_value if value is None else float(value) for value in values]
    if len(cleaned) < input_dim:
        cleaned.extend([pad_value] * (input_dim - len(cleaned)))
    return np.asarray(cleaned[:input_dim], dtype=np.float32)


def arrays(
    records: list[dict[str, Any]],
    *,
    input_dim: int,
    pad_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([feature_vector(record, input_dim, pad_value) for record in records])
    labels = [int(record["label"]) for record in records]
    invalid_labels = sorted(set(labels) - {0, 1})
    if invalid_labels:
        raise ValueError(
            f"SafeGauge requires 0=safe and 1=risk; found labels {invalid_labels}"
        )
    y = np.asarray(labels, dtype=np.float32)
    return X, y


def infer_input_dim(records: list[dict[str, Any]], requested: int | None) -> int:
    if requested is not None:
        return requested
    lengths = []
    for record in records:
        values = record.get("logprobs") or record.get("all_logprobs")
        if values is None:
            raise KeyError("Feature record must contain logprobs or all_logprobs")
        lengths.append(len(values))
    return max(lengths)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    device: str,
    shuffle: bool,
) -> DataLoader:
    dataset = SimpleDataset(X, y, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(
    model: BinaryMlp,
    loader: DataLoader,
    device: str,
    *,
    decision_threshold: float | None = None,
) -> dict[str, Any]:
    model.eval()
    labels = []
    probs = []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device))
            batch_probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs.extend(batch_probs.tolist())
            labels.extend(y.cpu().numpy().reshape(-1).tolist())
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(probs, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Evaluation requires both positive and negative labels")
    threshold_metrics = {}
    if decision_threshold is None:
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
        best_idx = int(np.argmax(f1_scores))
        if best_idx >= len(thresholds):
            decision_threshold = float(np.nextafter(np.max(y_score), np.inf))
        else:
            decision_threshold = float(thresholds[best_idx])
        threshold_metrics = {
            "best_f1": float(f1_scores[best_idx]),
            "best_threshold": decision_threshold,
        }
    decision_threshold = float(decision_threshold)
    y_pred = (y_score >= decision_threshold).astype(np.int64)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    valid = np.where(fpr <= 0.05)[0]
    tpr_at_fpr5 = float(np.max(tpr[valid])) if len(valid) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "decision_threshold": decision_threshold,
        **threshold_metrics,
        "tpr_at_fpr_0.05": {
            "tpr": tpr_at_fpr5,
            "threshold": float(roc_thresholds[valid[np.argmax(tpr[valid])]]) if len(valid) else None,
        },
        "labels": y_true.tolist(),
        "scores": y_score.tolist(),
    }


def train_one_epoch(
    model: BinaryMlp,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
) -> float:
    model.train()
    total = 0.0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device).float().unsqueeze(1)
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SafeGauge MLP probe")
    parser.add_argument("--train", type=Path, required=True, help="Train feature JSONL")
    parser.add_argument("--validation", type=Path, help="Validation feature JSONL")
    parser.add_argument("--test", type=Path, nargs="*", default=[], help="Optional test feature JSONL files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dim", type=int)
    parser.add_argument("--pad-value", type=float, default=-10.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weighted-bce", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def strip_predictions(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"labels", "scores"}}


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_records = load_jsonl(args.train)
    validation_records = load_jsonl(args.validation) if args.validation else None
    if not train_records:
        raise ValueError(f"Feature file is empty: {args.train}")
    test_record_sets = {}
    for test_path in args.test:
        test_records = load_jsonl(test_path)
        test_record_sets[test_path] = test_records
    input_dim = infer_input_dim(train_records, args.input_dim)
    X_train, y_train = arrays(train_records, input_dim=input_dim, pad_value=args.pad_value)
    train_loader = make_loader(
        X_train,
        y_train,
        batch_size=args.batch_size,
        device=device,
        shuffle=True,
    )
    validation_loader = None
    if validation_records:
        X_val, y_val = arrays(validation_records, input_dim=input_dim, pad_value=args.pad_value)
        validation_loader = make_loader(
            X_val,
            y_val,
            batch_size=args.batch_size,
            device=device,
            shuffle=False,
        )
    model = BinaryMlp(input_dim).to(device)
    if args.weighted_bce:
        positives = float(np.sum(y_train == 1))
        negatives = float(np.sum(y_train == 0))
        if positives == 0:
            raise ValueError("weighted BCE requires at least one positive example")
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], device=device)
        )
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        row = {"epoch": epoch, "train_loss": loss}
        if validation_loader is not None:
            val_metrics = evaluate(model, validation_loader, device)
            row["validation"] = strip_predictions(val_metrics)
            metric = val_metrics["roc_auc"]
            if metric > best_metric:
                best_metric = metric
                best_state = copy.deepcopy(model.state_dict())
        else:
            best_state = copy.deepcopy(model.state_dict())
            best_metric = loss * -1.0
        history.append(row)
        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            suffix = ""
            if "validation" in row:
                suffix = f" val_auc={row['validation']['roc_auc']:.4f}"
            print(f"epoch={epoch} loss={loss:.4f}{suffix}", flush=True)
    model.load_state_dict(best_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "model.pt"
    metrics: dict[str, Any] = {
        "input_dim": input_dim,
        "train_samples": len(train_records),
        "positive_train_samples": int(np.sum(y_train == 1)),
        "history": history,
    }
    final_validation_metrics = None
    if validation_loader is not None:
        final_validation_metrics = evaluate(model, validation_loader, device)
        metrics["validation"] = strip_predictions(final_validation_metrics)
    best_threshold = (
        final_validation_metrics["best_threshold"]
        if final_validation_metrics is not None
        else 0.5
    )
    first_record = train_records[0]
    model.save(
        str(checkpoint_path),
        meta={
            "input_dim": input_dim,
            "best_threshold": best_threshold,
            "pad_value": args.pad_value,
            "suffix_id": first_record.get("suffix_id"),
            "suffix": first_record.get("suffix"),
            "suffix_token_ids": first_record.get("suffix_token_ids"),
            "thinking_bypass_prefill": first_record.get(
                "thinking_bypass_prefill",
                "none",
            ),
            "train": str(args.train),
            "validation": str(args.validation) if args.validation else None,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weighted_bce": args.weighted_bce,
            "seed": args.seed,
        },
    )
    for test_path, test_records in test_record_sets.items():
        X_test, y_test = arrays(test_records, input_dim=input_dim, pad_value=args.pad_value)
        test_loader = make_loader(
            X_test,
            y_test,
            batch_size=args.batch_size,
            device=device,
            shuffle=False,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            decision_threshold=best_threshold,
        )
        metrics[f"test:{test_path}"] = strip_predictions(test_metrics)
        prediction_path = args.output_dir / f"{test_path.stem}_predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as f:
            for record, label, score in zip(test_records, test_metrics["labels"], test_metrics["scores"]):
                f.write(
                    json.dumps(
                        {
                            "id": record.get("id"),
                            "label": int(label),
                            "score": float(score),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved {checkpoint_path}")
    print(f"saved {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
