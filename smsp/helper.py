"""Reusable helpers for SMSP experiment scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from .mlp import BinaryMlp


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Load JSONL cache records keyed by sample_id/id."""
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("sample_id", record.get("id"))
            if key is None:
                key = str(len(cache))
            cache[str(key)] = record
    return cache


def arrays(
    records: list[dict[str, Any]],
    input_dim: int | None = None,
    pad_value: float = -10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert feature records into padded X/y arrays."""
    if not records:
        raise ValueError("records must not be empty")
    values = [record.get("logprobs") or record.get("all_logprobs") for record in records]
    if any(item is None for item in values):
        raise KeyError("All records must contain logprobs or all_logprobs")
    width = input_dim or max(len(item) for item in values if item is not None)
    X = []
    for item in values:
        row = [pad_value if value is None else float(value) for value in item]
        if len(row) < width:
            row.extend([pad_value] * (width - len(row)))
        X.append(row[:width])
    y = [int(record["label"]) for record in records]
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def _normalize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mean) / std, mean, std


def _predict_state(X: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(state["normalization_mean"], dtype=np.float32)
    std = np.asarray(state["normalization_std"], dtype=np.float32)
    values = torch.tensor((X - mean) / std, dtype=torch.float32)
    model = BinaryMlp(int(state["input_dim"]))
    model.load_state_dict(state["state_dict"])
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(values)).numpy().reshape(-1)


def ensemble_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    epochs: int,
    lr: float,
    seeds: list[int],
    batch_size: int = 64,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit a small seed ensemble and return averaged probabilities."""
    X_fit, mean, std = _normalize_fit(np.asarray(X_train, dtype=np.float32))
    X_eval_norm = (np.asarray(X_eval, dtype=np.float32) - mean) / std
    y_fit = np.asarray(y_train, dtype=np.float32)
    positives = float(np.sum(y_fit == 1))
    negatives = float(np.sum(y_fit == 0))
    pos_weight = None
    if positives > 0 and negatives > 0:
        pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)
    predictions = []
    states = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = BinaryMlp(X_fit.shape[1])
        criterion = (
            torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            if pos_weight is not None
            else torch.nn.BCEWithLogitsLoss()
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        dataset = TensorDataset(
            torch.tensor(X_fit, dtype=torch.float32),
            torch.tensor(y_fit, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.train()
        for _ in range(epochs):
            for batch_X, batch_y in loader:
                logits = model(batch_X)
                loss = criterion(logits, batch_y.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(
                model(torch.tensor(X_eval_norm, dtype=torch.float32))
            ).numpy().reshape(-1)
        predictions.append(scores)
        states.append(
            {
                "input_dim": int(X_fit.shape[1]),
                "normalization_mean": mean.tolist(),
                "normalization_std": std.tolist(),
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            }
        )
    return np.mean(predictions, axis=0), states


def candidate_metrics(records: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    y = np.asarray([int(record["label"]) for record in records])
    scores = np.asarray(scores, dtype=np.float64)
    fpr, tpr, thresholds = roc_curve(y, scores)
    valid = np.where(fpr <= 0.05)[0]
    if len(valid):
        best_valid = valid[np.argmax(tpr[valid])]
        tpr5 = {
            "tpr": float(tpr[best_valid]),
            "fpr": float(fpr[best_valid]),
            "threshold": float(thresholds[best_valid]),
        }
    else:
        tpr5 = {"tpr": 0.0, "fpr": None, "threshold": None}
    best_f1 = {"f1": 0.0, "threshold": None, "precision": 0.0, "recall": 0.0}
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        current = f1_score(y, pred, zero_division=0)
        if current > best_f1["f1"]:
            best_f1 = {
                "f1": float(current),
                "threshold": float(threshold),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
            }
    output = {
        "samples": int(len(records)),
        "positive_samples": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "oracle_best_f1": best_f1,
        "tpr_at_fpr_0.05": tpr5,
    }
    if all("question_id" in record for record in records):
        correct = 0
        total = 0
        by_question: dict[Any, list[int]] = {}
        for index, record in enumerate(records):
            by_question.setdefault(record["question_id"], []).append(index)
        for indexes in by_question.values():
            best = max(indexes, key=lambda index: scores[index])
            correct += int(y[best] == 1)
            total += 1
        output["question_level"] = {
            "questions": total,
            "top1_accuracy": float(correct / total) if total else 0.0,
        }
    return output
