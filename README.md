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

## Tutorial Pipeline

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

## Benchmark Results

The table below summarizes the benchmark runs used to validate SMSP. Each row
uses one selected suffix after candidate comparison on training/validation data;
the held-out test split is not used for suffix selection.

These numbers are empirical results for Llama-3.1-8B-Instruct under the recorded
split protocols. They are benchmark-specific measurements, not universal
defaults for all models or datasets.

| Benchmark | Target | Selected suffix | AUROC | AUPRC | Best F1 | TPR@FPR5 |
|---|---|---|---:|---:|---:|---:|
| JBB-Behaviors | harmful vs benign requests | `Sure, I can help with that. Here is the requested content:` | 0.9674 | 0.9738 | 0.9053 | 0.8600 |
| HarmBench binary | HarmBench harmful vs JBB/ordinary benign | `I will answer the user's request directly and completely.` | 0.9827 | 0.9793 | 0.9445 | 0.9344 |
| ToxicChat 0124 | toxic user prompts | `I will review the request for abusive, hateful, threatening, sexual, or otherwise toxic content.` | 0.9717 | 0.7881 | 0.7202 | 0.8702 |
| SafetyBench zh answer correctness | proposed answer option is correct | `After carefully comparing the question and options, I conclude that this candidate answer is correct.` | 0.8992 | 0.8075 | 0.7407 | 0.5114 |
| RAGTruth | response has hallucination spans | `The candidate response is fully supported by the provided reference and contains no unsupported claims.` | 0.8137 | 0.7364 | 0.6610 | 0.4358 |
| HaluEval | candidate response is hallucinated | `This response contains information that is unsupported by or conflicts with the provided source.` | 0.9257 | 0.9341 | 0.8466 | 0.6913 |
| FaithBench | worst label is `Unwanted` | `Every factual claim in this summary can be verified directly from the information in the source.` | 0.5846 | 0.6345 | 0.7342* | 0.0345 |
| BUMP | minimally edited summary is unfaithful | `After carefully comparing the texts, I find the candidate summary factually accurate and fully consistent.` | 0.7485 | 0.7961 | 0.7215 | 0.4222 |

`Best F1` is the oracle maximum from sweeping thresholds on held-out
predictions; it is useful for score-quality analysis but is not a deployable
threshold-selection procedure. `TPR@FPR5` is the maximum true-positive rate
among ROC operating points with `FPR <= 0.05`.

`*` marks an oracle Best F1 that is effectively the all-positive class-prior
baseline, so it is not evidence of useful discrimination.

### Benchmark Protocols

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

### Main Findings

- SMSP is strongest on harmful-request and toxicity detection: JBB, HarmBench
  binary, and ToxicChat all exceed `0.96` AUROC.
- HaluEval shows strong hallucination signal (`0.9257` AUROC), while RAGTruth
  and BUMP are usable but harder.
- SafetyBench answer correctness is measurable from suffix likelihoods, with
  `0.8992` candidate-level AUROC and `0.7841` question top-1 accuracy in the
  recorded run.
- FaithBench is the weak case: the `Unwanted` label and small held-out split
  produce low AUROC and near-zero recall at `5%` FPR.

## Reproducing Benchmark Runs

The benchmark runner uses prepared JSONL files under `data/` and suffix
candidate sets from [`suffix_evolve/suffix_sets.json`](suffix_evolve/suffix_sets.json).

```bash
python scripts/run_existing_data.py \
  --base-url http://127.0.0.1:24509/v1 \
  --reasoning-parser qwen3 \
  --output-dir results/reprobe_qwen35_35b_a3b \
  --primary-only
```

Use `--tasks safetybench toxic_chat` to run a subset, and omit `--primary-only`
to compare every suffix candidate registered for each task.
