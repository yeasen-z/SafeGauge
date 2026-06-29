import numpy as np

from suffix_evolve.src.search_metrics import (
    metrics,
    paired_question_bootstrap_delta,
    pareto_ids,
    question_bootstrap_lcb,
)


def test_perfect_ranking_has_perfect_metrics():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    result = metrics(y, scores)
    assert result.roc_auc == 1.0
    assert result.tpr_at_fpr_005 == 1.0


def test_bootstrap_resamples_questions_and_is_reproducible():
    y = np.tile(np.array([1, 0, 0]), 20)
    scores = np.tile(np.array([0.8, 0.2, 0.1]), 20)
    question_ids = np.repeat(np.arange(20), 3)
    first = question_bootstrap_lcb(
        y, scores, question_ids, repetitions=30, seed=7
    )
    second = question_bootstrap_lcb(
        y, scores, question_ids, repetitions=30, seed=7
    )
    assert first == second
    assert first["joint_lcb"] == 1.0


def test_pareto_front():
    rows = [
        {"suffix_id": "a", "roc_auc": 0.90, "tpr_at_fpr_0.05": 0.40},
        {"suffix_id": "b", "roc_auc": 0.89, "tpr_at_fpr_0.05": 0.50},
        {"suffix_id": "c", "roc_auc": 0.88, "tpr_at_fpr_0.05": 0.30},
    ]
    assert pareto_ids(rows) == ["a", "b"]


def test_paired_bootstrap_detects_improvement():
    y = np.tile(np.array([1, 0, 0]), 30)
    candidate = np.tile(np.array([0.9, 0.2, 0.1]), 30)
    baseline = np.tile(np.array([0.4, 0.6, 0.3]), 30)
    question_ids = np.repeat(np.arange(30), 3)
    result = paired_question_bootstrap_delta(
        y, candidate, baseline, question_ids, repetitions=30
    )
    assert result["roc_auc"]["delta"] > 0
    assert result["roc_auc"]["probability_improvement"] == 1.0
