# Generative Guard Baselines

This document records how to run LLM-style guardrail models as additional
baselines. These baselines are separate from SMSP and from the classifier-style
`Llama-Prompt-Guard-2-86M` baseline.

## Candidate Models

| Baseline | ModelScope model | Runner template | Positive labels |
|---|---|---|---|
| Llama Guard 3 8B | [`LLM-Research/Llama-Guard-3-8B`](https://modelscope.cn/models/LLM-Research/Llama-Guard-3-8B) | `llama_guard` | `unsafe` |
| Qwen3Guard | serve the downloaded Qwen3Guard checkpoint name | `qwen3_guard` | `unsafe`, `controversial` |

The [Qwen3Guard technical report](https://arxiv.org/abs/2510.14276) describes
generative and stream variants in 0.6B, 4B, and 8B sizes. For this benchmark
table, use the generative variant when available because it matches the
full-record safety classification setup.

## Runner

```text
scripts/run_generative_guard_baseline.py
```

The runner uses an OpenAI-compatible chat-completions endpoint, so the model can
be served by vLLM after downloading from ModelScope. It writes:

```text
metrics.json
validation_predictions.jsonl
test_predictions.jsonl
```

The score is the positive-label probability from first-token logprobs when the
server exposes chat logprobs. If logprobs are not available, the score falls
back to the parsed discrete label, which makes AUROC and TPR@FPR5 less
informative.

## Example Commands

Llama Guard:

```bash
python scripts/run_generative_guard_baseline.py \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key none \
  --model Llama-Guard-3-8B \
  --template llama_guard \
  --data-dir data/unknown_unknowns/trigger_heldout_v1 \
  --output-dir results/llama_guard_3_8b_baselines/unknown_unknowns
```

Qwen3Guard:

```bash
python scripts/run_generative_guard_baseline.py \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key none \
  --model Qwen3Guard \
  --template qwen3_guard \
  --data-dir data/unknown_unknowns/trigger_heldout_v1 \
  --output-dir results/qwen3_guard_baselines/unknown_unknowns
```

Use `--limit-records 20` for a connectivity test before running the full split.
