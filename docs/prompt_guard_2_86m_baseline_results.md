# Prompt Guard 2 86M Baseline Results

This file records the full baseline metrics for
`Llama-Prompt-Guard-2-86M`. The root README keeps only the compact comparison
columns: `AUROC`, `F1`, and `TPR@FPR5`.

Model path:

```text
/ssd/workspace/zms/models/Llama-Prompt-Guard-2-86M/
```

Runner:

```text
scripts/run_prompt_guard_baseline.py
```

The classifier scores each prepared SMSP record with the `LABEL_1` softmax
probability, interpreted as the malicious/prompt-attack score. `F1` uses the
validation-selected operating threshold. `TPR@FPR5` is computed separately by
sweeping the score threshold and taking the maximum true-positive rate among
ROC operating points with `FPR <= 0.05`.

## Held-Out Test Metrics

### Score Metrics

| Benchmark | Samples | Pos | Neg | AUROC | AUPRC | Accuracy | Precision | Recall | F1 | TPR@FPR5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Un-unknowns | 1610 | 760 | 850 | 0.61 | 0.57 | 0.53 | 0.50 | 0.82 | 0.63 | 0.09 |
| ToxicChat 0124 | 5083 | 362 | 4721 | 0.73 | 0.34 | 0.93 | 0.55 | 0.25 | 0.35 | 0.31 |
| SafetyBench zh | 6978 | 2288 | 4690 | 0.50 | 0.34 | 0.33 | 0.33 | 1.00 | 0.49 | 0.08 |
| RAGTruth | 2675 | 943 | 1732 | 0.52 | 0.33 | 0.48 | 0.40 | 0.90 | 0.55 | 0.02 |
| HaluEval | 3000 | 1500 | 1500 | 0.51 | 0.51 | 0.50 | 0.50 | 1.00 | 0.67 | 0.05 |
| FaithBench | 100 | 58 | 42 | 0.49 | 0.59 | 0.52 | 0.57 | 0.71 | 0.63 | 0.00 |
| BUMP | 180 | 90 | 90 | 0.50 | 0.50 | 0.50 | 0.50 | 1.00 | 0.67 | 0.00 |

### Threshold And Error Rates

| Benchmark | F1 threshold | TPR@FPR5 threshold | FPR | FNR |
|---|---:|---:|---:|---:|
| Un-unknowns | 0.0042 | 0.9756 | 0.72 | 0.18 |
| ToxicChat 0124 | 0.4833 | 0.0096 | 0.02 | 0.75 |
| SafetyBench zh | 0.0004 | 0.0311 | 1.00 | 0.00 |
| RAGTruth | 0.0043 | 0.0272 | 0.75 | 0.10 |
| HaluEval | 0.0004 | 0.0098 | 1.00 | 0.00 |
| FaithBench | 0.0009 | n/a | 0.74 | 0.29 |
| BUMP | 0.0014 | n/a | 1.00 | 0.00 |

### Confusion Matrix

| Benchmark | TP | TN | FP | FN |
|---|---:|---:|---:|---:|
| Un-unknowns | 626 | 234 | 616 | 134 |
| ToxicChat 0124 | 92 | 4646 | 75 | 270 |
| SafetyBench zh | 2288 | 0 | 4690 | 0 |
| RAGTruth | 852 | 437 | 1295 | 91 |
| HaluEval | 1500 | 0 | 1500 | 0 |
| FaithBench | 41 | 11 | 31 | 17 |
| BUMP | 90 | 0 | 90 | 0 |

## Result Files

| Benchmark | Result directory |
|---|---|
| Un-unknowns | `results/prompt_guard_2_86m_baselines/unknown_unknowns/` |
| ToxicChat 0124 | `results/prompt_guard_2_86m_baselines/toxic_chat/` |
| SafetyBench zh | `results/prompt_guard_2_86m_baselines/safetybench/` |
| RAGTruth | `results/prompt_guard_2_86m_baselines/ragtruth/` |
| HaluEval | `results/prompt_guard_2_86m_baselines/halueval/` |
| FaithBench | `results/prompt_guard_2_86m_baselines/faithbench/` |
| BUMP | `results/prompt_guard_2_86m_baselines/bump/` |
