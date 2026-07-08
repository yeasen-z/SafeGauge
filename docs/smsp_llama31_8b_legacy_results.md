# Legacy Llama-3.1-8B-Instruct Results

These are earlier SMSP benchmark runs with `Llama-3.1-8B-Instruct`. They are
kept for provenance only. The main `README.md` reports the unified
`Qwen3.5-35B-A3B` benchmark table.

| Benchmark | Method | Model | AUROC | AUPRC | F1 | TPR@FPR5 |
|---|---|---|---:|---:|---:|---:|
| JBB-Behaviors | SMSP | Llama-3.1-8B-Instruct | 0.9674 | 0.9738 | 0.9053 | 0.8600 |
| HarmBench binary | SMSP | Llama-3.1-8B-Instruct | 0.9827 | 0.9793 | 0.9445 | 0.9344 |
| ToxicChat 0124 | SMSP | Llama-3.1-8B-Instruct | 0.9717 | 0.7881 | 0.7202 | 0.8702 |
| SafetyBench zh | SMSP | Llama-3.1-8B-Instruct | 0.8992 | 0.8075 | 0.7407 | 0.5114 |
| RAGTruth | SMSP | Llama-3.1-8B-Instruct | 0.8137 | 0.7364 | 0.6610 | 0.4358 |
| HaluEval | SMSP | Llama-3.1-8B-Instruct | 0.9257 | 0.9341 | 0.8466 | 0.6913 |
| FaithBench | SMSP | Llama-3.1-8B-Instruct | 0.5846 | 0.6345 | 0.7342* | 0.0345 |
| BUMP | SMSP | Llama-3.1-8B-Instruct | 0.7485 | 0.7961 | 0.7215 | 0.4222 |

`F1` is the best threshold-selected F1 for the recorded run. `TPR@FPR5` is the
maximum true-positive rate among ROC operating points with `FPR <= 0.05`. `*`
marks a value close to the all-positive class-prior baseline.
