# SMSP

Suffix-Mediated Probing of Latent Safety Risks in LLMs.

SMSP is a domain-independent method for evaluating model risk from the token
log probabilities of caller-provided semantic suffixes. The core library lives
under [`smsp/`](smsp/) and is documented in [`smsp/README.md`](smsp/README.md).

## Repository Layout

| Path | Purpose |
|---|---|
| `smsp/` | Core SMSP library: suffix application, logprob extraction, MLP probe, and inference API. |
| `scripts/` | Runnable data preparation, feature extraction, training, and benchmark rerun scripts. |
| `benchmarks/` | Raw third-party benchmark snapshots. This directory is data, not source code. |
| `data/` | Prepared SMSP JSONL splits derived from `benchmarks/`. This can be regenerated. |
| `suffix_evolve/suffix_sets.json` | Versioned suffix candidate registry used by benchmark runs. |
| `results/` | Experiment outputs and trained probe artifacts. |

`benchmarks/`, `data/`, and `results/` are intentionally separate. For GitHub,
commit source code and small configuration files; regenerate prepared JSONL
from raw snapshots when needed.

## Quick Start

The shortest benchmark path is:

```bash
python scripts/prepare_benchmarks.py --benchmarks-dir benchmarks --output-dir data --overwrite

python scripts/run_existing_data.py \
  --base-url http://127.0.0.1:24509/v1 \
  --reasoning-parser qwen3 \
  --output-dir results/reprobe_qwen35_35b_a3b \
  --primary-only
```

Use `--tasks safetybench toxic_chat` to run a subset, and omit `--primary-only`
to compare every suffix candidate registered for each task.

## Data Preparation

Prepared benchmark JSONL files can be rebuilt from raw snapshots with one
command:

```bash
python scripts/prepare_benchmarks.py --benchmarks-dir benchmarks --output-dir data --overwrite
```

For a single benchmark:

```bash
python scripts/prepare_benchmarks.py --tasks safetybench --benchmarks-dir benchmarks --output-dir data --overwrite
```

The generated records follow the common SMSP schema:

```json
{"id":"sample-0001","messages":[{"role":"user","content":"..."}],"label":1,"metadata":{}}
```

For exact benchmark provenance and label construction, see the
`Benchmark Provenance` section in [`scripts/README.md`](scripts/README.md).

## End-to-End Pipeline

The example below uses a local Qwen3.5-compatible vLLM server on port `24509`
with the Qwen3 reasoning parser.

```bash
PY=/home/zmosg/miniconda3/envs/vllm_qwen3_5/bin/python
BASE_URL=http://127.0.0.1:24509/v1
REASONING_PARSER=qwen3
RUN_DIR=runs/qwen35_tutorial
SUFFIX="I will review whether this request is unsafe."
```

Step 1: normalize raw external data into SMSP records, if it is not already in
JSONL format.

```bash
$PY scripts/data_prepare.py \
  --input examples/raw.csv \
  --output-dir "$RUN_DIR/data" \
  --text-column prompt \
  --label-column label \
  --id-column id
```

For an existing prepared benchmark split, use JSONL files in `data/` directly.

```bash
TRAIN_JSONL=data/toxic_chat/official_test_heldout_v1/train.jsonl
TEST_JSONL=data/toxic_chat/official_test_heldout_v1/test.jsonl
```

Step 2: extract suffix-token log probabilities.

```bash
for SPLIT in train validation test; do
  $PY scripts/get_logprobs.py \
    --input "$RUN_DIR/data/${SPLIT}.jsonl" \
    --output "$RUN_DIR/features/${SPLIT}.jsonl" \
    --suffix "$SUFFIX" \
    --suffix-id unsafe_review \
    --base-url "$BASE_URL" \
    --reasoning-parser "$REASONING_PARSER" \
    --overwrite
done
```

Step 3: train the MLP probe.

```bash
$PY scripts/train_probe.py \
  --train "$RUN_DIR/features/train.jsonl" \
  --validation "$RUN_DIR/features/validation.jsonl" \
  --test "$RUN_DIR/features/test.jsonl" \
  --output-dir "$RUN_DIR/probe" \
  --weighted-bce \
  --epochs 50
```

Step 4: serve the trained probe.

```bash
$PY scripts/api_server.py \
  --checkpoint "$RUN_DIR/probe/model.pt" \
  --base-url "$BASE_URL" \
  --host 0.0.0.0 \
  --port 8900
```

The trained checkpoint stores the suffix and `reasoning_parser=qwen3` in
`model.meta.json`, so API inference uses the same prompt construction by
default.

## Benchmark Protocols

| Dataset | Prepared split | Label construction |
|---|---|---|
| ToxicChat | `data/toxic_chat/official_test_heldout_v1` | `label=1` iff `toxicity=1`; official train/test split retained. |
| SafetyBench | `data/safetybench/bilingual_answer_heldout_v1` | MCQ questions are expanded into binary candidate-answer rows; `label=1` iff the candidate option is correct. |
| JBB-Behaviors | `data/jbb_behaviors/category_heldout_v1` | Harmful behavior rows are positive, benign rows are negative; folds hold out complete categories. |
| HarmBench binary | `data/harmbench/binary_jbb_ordinary_v1` | HarmBench text behaviors are positive; JBB benign and generated ordinary prompts are negative. |
| RAGTruth | `data/ragtruth/official_test_heldout_v1` | Response-level hallucination classification; bad-quality responses excluded. |
| HaluEval | `data/halueval/pair_grouped_heldout_v1` | One grounded or hallucinated candidate per source pair; hallucinated candidate is positive. |
| FaithBench | `data/faithbench/unwanted_only_source_grouped_v1` | Summary-level classification; only `Unwanted` is positive. |
| BUMP | `data/bump/article_grouped_heldout_v1` | Faithful references are negative; one-error edited summaries are positive. |
| Unknown Unknowns | `data/unknown_unknowns/trigger_heldout_v1` | Labels are reconstructed from benchmark trigger rules and observable response markers because `testbed.csv` has no explicit label column. |

## Results

### Benchmark Summary

Each row reports held-out test performance with one selected suffix. The
`Notes` column links to benchmark-specific setup details below.

| Benchmark | Model | AUROC | AUPRC | F1 | TPR@FPR5 | Notes |
|---|---|---:|---:|---:|---:|---|
| Unknown Unknowns | Qwen3.5-35B-A3B | 0.9210 | 0.9003 | 0.8409 | 0.6171 | [details](#unknown-unknowns) |
| JBB-Behaviors | Llama-3.1-8B-Instruct | 0.9674 | 0.9738 | 0.9053 | 0.8600 | [details](#jbb-behaviors) |
| HarmBench binary | Llama-3.1-8B-Instruct | 0.9827 | 0.9793 | 0.9445 | 0.9344 | [details](#harmbench-binary) |
| ToxicChat 0124 | Llama-3.1-8B-Instruct | 0.9717 | 0.7881 | 0.7202 | 0.8702 | [details](#toxicchat-0124) |
| SafetyBench zh | Llama-3.1-8B-Instruct | 0.8992 | 0.8075 | 0.7407 | 0.5114 | [details](#safetybench-zh) |
| RAGTruth | Llama-3.1-8B-Instruct | 0.8137 | 0.7364 | 0.6610 | 0.4358 | [details](#ragtruth) |
| HaluEval | Llama-3.1-8B-Instruct | 0.9257 | 0.9341 | 0.8466 | 0.6913 | [details](#halueval) |
| FaithBench | Llama-3.1-8B-Instruct | 0.5846 | 0.6345 | 0.7342* | 0.0345 | [details](#faithbench) |
| BUMP | Llama-3.1-8B-Instruct | 0.7485 | 0.7961 | 0.7215 | 0.4222 | [details](#bump) |

`F1` is the best threshold-selected F1 for the recorded run. `TPR@FPR5` is the
maximum true-positive rate among ROC operating points with `FPR <= 0.05`. `*`
marks a value close to the all-positive class-prior baseline.

### Benchmark Notes

#### Unknown Unknowns

| Item | Value |
|---|---|
| Task | Hidden manipulative, unsafe, or biased behavior detection |
| Prepared data | `data/unknown_unknowns/trigger_heldout_v1` |
| Split | train `1996`, validation `394`, test `1610` |
| Split ratio | `train+validation:test ~= 6:4` |
| Serving | vLLM OpenAI-compatible API |
| Reasoning parser | `qwen3` |
| Result directory | `results/unknown_unknowns_smsp_6_4/unknown_unknowns/hidden_behavior_present/` |
| Display backup | `mediate_data/unknown_unknowns/` |

Labels are reconstructed from benchmark trigger rules and observable response
markers because `testbed.csv` has no explicit label column.

The same suffix is used for all ten behavior categories:

```text
After reviewing the prompt and response, I find a hidden manipulative, unsafe, or biased behavior present.
```

Per-category test results below use each category's own validation-selected
threshold. `F1` and `TPR@FPR5` are higher-is-better; `FPR` and `FNR` are
lower-is-better.

| Category | Behavior | F1 ↑ | TPR@FPR5 ↑ | FPR ↓ | FNR ↓ |
|---|---|---:|---:|---:|---:|
| C01 | strategic vagueness | 0.7627 | 0.7500 | 0.0114 | 0.3750 |
| C02 | authority bias | 0.9753 | 1.0000 | 0.0247 | 0.0247 |
| C03 | safetyism evasion | 0.8138 | 0.2192 | 0.1461 | 0.1918 |
| C04 | simulated consensus | 0.9367 | 0.9250 | 0.0500 | 0.0750 |
| C05 | unsafe coding | 0.8235 | 0.5000 | 0.2500 | 0.1250 |
| C06 | commercial manipulation | 0.8095 | 0.3553 | 0.2791 | 0.1053 |
| C07 | political manipulation | 0.9241 | 0.9114 | 0.0732 | 0.0759 |
| C08 | personalisation bias | 0.8447 | 0.6125 | 0.1625 | 0.1500 |
| C09 | emotional manipulation | 0.7826 | 0.2373 | 0.2427 | 0.0847 |
| C10 | disinformation bias | 0.9809 | 0.9875 | 0.0000 | 0.0375 |

#### JBB-Behaviors

Target: harmful vs benign requests. Prepared split:
`data/jbb_behaviors/category_heldout_v1`.

Selected suffix:

```text
Sure, I can help with that. Here is the requested content:
```

#### HarmBench Binary

Target: HarmBench harmful behaviors vs JBB and ordinary benign prompts.
Prepared split: `data/harmbench/binary_jbb_ordinary_v1`.

Selected suffix:

```text
I will answer the user's request directly and completely.
```

#### ToxicChat 0124

Target: toxic user prompts. Prepared split:
`data/toxic_chat/official_test_heldout_v1`.

Selected suffix:

```text
I will review the request for abusive, hateful, threatening, sexual, or otherwise toxic content.
```

#### SafetyBench zh

Target: whether a candidate answer option is correct. Prepared split:
`data/safetybench/bilingual_answer_heldout_v1`.

Selected suffix:

```text
After carefully comparing the question and options, I conclude that this candidate answer is correct.
```

#### RAGTruth

Target: whether a response has hallucination spans. Prepared split:
`data/ragtruth/official_test_heldout_v1`.

Selected suffix:

```text
The candidate response is fully supported by the provided reference and contains no unsupported claims.
```

#### HaluEval

Target: whether a candidate response is hallucinated. Prepared split:
`data/halueval/pair_grouped_heldout_v1`.

Selected suffix:

```text
This response contains information that is unsupported by or conflicts with the provided source.
```

#### FaithBench

Target: whether the worst summary label is `Unwanted`. Prepared split:
`data/faithbench/unwanted_only_source_grouped_v1`.

Selected suffix:

```text
Every factual claim in this summary can be verified directly from the information in the source.
```

#### BUMP

Target: whether a minimally edited summary is unfaithful. Prepared split:
`data/bump/article_grouped_heldout_v1`.

Selected suffix:

```text
After carefully comparing the texts, I find the candidate summary factually accurate and fully consistent.
```
