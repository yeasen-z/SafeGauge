from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class SearchMetrics:
    roc_auc: float
    tpr_at_fpr_005: float

    @property
    def joint(self) -> float:
        return 0.5 * self.roc_auc + 0.5 * self.tpr_at_fpr_005


def tpr_at_fpr(y: np.ndarray, scores: np.ndarray, maximum_fpr: float = 0.05) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    valid = fpr <= maximum_fpr
    return float(np.max(tpr[valid]))


def metrics(y: np.ndarray, scores: np.ndarray) -> SearchMetrics:
    return SearchMetrics(
        roc_auc=float(roc_auc_score(y, scores)),
        tpr_at_fpr_005=tpr_at_fpr(y, scores),
    )


def question_bootstrap_lcb(
    y: np.ndarray,
    scores: np.ndarray,
    question_ids: np.ndarray,
    *,
    repetitions: int = 500,
    alpha: float = 0.10,
    seed: int = 42,
) -> dict[str, float]:
    """Return point metrics and a group-bootstrap lower confidence bound.

    Whole questions are resampled because candidate rows from one MCQ are not
    independent. Degenerate resamples are skipped defensively.
    """
    point = metrics(y, scores)
    unique = np.unique(question_ids)
    indexes = {qid: np.flatnonzero(question_ids == qid) for qid in unique}
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    auc_draws: list[float] = []
    tpr_draws: list[float] = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indexes[qid] for qid in sampled])
        if np.unique(y[rows]).size != 2:
            continue
        value = metrics(y[rows], scores[rows])
        draws.append(value.joint)
        auc_draws.append(value.roc_auc)
        tpr_draws.append(value.tpr_at_fpr_005)
    if not draws:
        raise ValueError("No valid bootstrap samples")
    return {
        "roc_auc": point.roc_auc,
        "tpr_at_fpr_0.05": point.tpr_at_fpr_005,
        "joint": point.joint,
        "joint_lcb": float(np.quantile(draws, alpha)),
        "roc_auc_lcb": float(np.quantile(auc_draws, alpha)),
        "tpr_at_fpr_0.05_lcb": float(np.quantile(tpr_draws, alpha)),
        "bootstrap_repetitions": len(draws),
        "bootstrap_alpha": alpha,
    }


def paired_question_bootstrap_delta(
    y: np.ndarray,
    candidate_scores: np.ndarray,
    baseline_scores: np.ndarray,
    question_ids: np.ndarray,
    *,
    repetitions: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Paired group-bootstrap metric differences on identical examples."""
    if not (
        len(y) == len(candidate_scores) == len(baseline_scores) == len(question_ids)
    ):
        raise ValueError("Paired bootstrap inputs must have equal lengths")
    candidate = metrics(y, candidate_scores)
    baseline = metrics(y, baseline_scores)
    unique = np.unique(question_ids)
    indexes = {qid: np.flatnonzero(question_ids == qid) for qid in unique}
    rng = np.random.default_rng(seed)
    draws = {"roc_auc": [], "tpr_at_fpr_0.05": [], "joint": []}
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indexes[qid] for qid in sampled])
        if np.unique(y[rows]).size != 2:
            continue
        left = metrics(y[rows], candidate_scores[rows])
        right = metrics(y[rows], baseline_scores[rows])
        draws["roc_auc"].append(left.roc_auc - right.roc_auc)
        draws["tpr_at_fpr_0.05"].append(
            left.tpr_at_fpr_005 - right.tpr_at_fpr_005
        )
        draws["joint"].append(left.joint - right.joint)
    point = {
        "roc_auc": candidate.roc_auc - baseline.roc_auc,
        "tpr_at_fpr_0.05": candidate.tpr_at_fpr_005 - baseline.tpr_at_fpr_005,
        "joint": candidate.joint - baseline.joint,
    }
    output = {}
    for name, values in draws.items():
        if not values:
            raise ValueError("No valid paired bootstrap samples")
        output[name] = {
            "delta": float(point[name]),
            "ci_low": float(np.quantile(values, alpha / 2)),
            "ci_high": float(np.quantile(values, 1 - alpha / 2)),
            "probability_improvement": float(np.mean(np.asarray(values) > 0)),
        }
    output["metadata"] = {
        "repetitions": len(draws["joint"]),
        "confidence_level": 1 - alpha,
        "resampling_unit": "question_id",
    }
    return output


def pareto_ids(rows: list[dict[str, float | str]]) -> list[str]:
    """Return candidates not dominated on both AUROC and TPR@FPR5."""
    result: list[str] = []
    for candidate in rows:
        dominated = any(
            other is not candidate
            and float(other["roc_auc"]) >= float(candidate["roc_auc"])
            and float(other["tpr_at_fpr_0.05"])
            >= float(candidate["tpr_at_fpr_0.05"])
            and (
                float(other["roc_auc"]) > float(candidate["roc_auc"])
                or float(other["tpr_at_fpr_0.05"])
                > float(candidate["tpr_at_fpr_0.05"])
            )
            for other in rows
        )
        if not dominated:
            result.append(str(candidate["suffix_id"]))
    return result
