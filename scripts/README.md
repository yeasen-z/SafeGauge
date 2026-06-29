# SMSP Scripts Tutorial

This directory contains the runnable SMSP pipeline:

1. Normalize raw labeled examples into SMSP JSONL records.
2. Append one semantic suffix and extract suffix-token log probabilities.
3. Train a binary MLP probe from those logprob vectors.
4. Optionally serve the trained probe through an HTTP API.

The examples below use the local Qwen3.5-compatible vLLM server:

```bash
BASE_URL=http://127.0.0.1:24509/v1
PY=/home/zmosg/miniconda3/envs/vllm_qwen3_5/bin/python
REASONING_PARSER=qwen3
```

Use the same `--reasoning-parser` for feature extraction and later inference.
`train_probe.py` stores the parser and suffix in `model.meta.json`, and
`api_server.py` will reuse them by default.

## Data Contract

SMSP expects binary labeled records. Each example means:

- `messages`: the original conversation before the assistant answer.
- `label`: `1` for the positive class, `0` for the negative class.
- `id`: stable sample identifier.
- `metadata`: optional fields kept for traceability.

Canonical JSONL record:

```json
{"id":"sample-0001","messages":[{"role":"user","content":"How can I steal a password?"}],"label":1,"metadata":{"source":"demo"}}
```

CSV input is also supported. At minimum it needs one text column and one label
column:

```csv
id,prompt,label
pos-1,How can I steal a password?,1
neg-1,How do I reset my password safely?,0
```

The text column can be:

- plain text, converted to `[{ "role": "user", "content": text }]`;
- or a JSON-encoded OpenAI messages list, such as
  `[{"role":"system","content":"..."},{"role":"user","content":"..."}]`.

Positive label values default to:

```text
1,true,yes,positive,pos,risk,attack,toxic,harmful
```

Negative label values include:

```text
0,false,no,negative,neg,safe,benign
```

## Benchmark Provenance

There are two data layers in this repository:

- `benchmarks/`: raw third-party benchmark snapshots, with upstream README and
  license files where available.
- `data/`: JSONL prepared SMSP/evaluation splits derived from those raw files. Each
  prepared dataset has a `config.json` recording source paths, source URL or
  revision, checksums, split policy, and label counts.

The current `scripts/data_prepare.py` is a generic normalizer. It does not
reconstruct every benchmark-specific split from the original upstream format.
For the existing benchmark experiments, use the prepared JSONL files under `data/`.
Use the `data/*/*/config.json` files when you need exact provenance.

| Prepared data | Raw source | Local raw files | Label construction |
|---|---|---|---|
| `data/toxic_chat/official_test_heldout_v1` | `lmsys/toxic-chat`, `toxicchat0124`, CC-BY-NC-4.0 | `benchmarks/toxic-chat/data/0124/*.csv` | `label=toxicity`; official train/test kept. |
| `data/safetybench/bilingual_answer_heldout_v1` | `thu-coai/SafetyBench` | `benchmarks/SafetyBench/data/test_{zh,en}.json`, `benchmarks/SafetyBench/opensource_data/test_answers_{zh,en}.json` | MCQ questions are expanded into binary candidate-answer rows; `label=1` iff candidate option is correct. |
| `data/jbb_behaviors/category_heldout_v1` | `JailbreakBench/JBB-Behaviors` | `benchmarks/JBB-Behaviors/data/{harmful,benign}-behaviors.csv` | Harmful behavior rows are positive; benign behavior rows are negative; folds hold out complete categories. |
| `data/harmbench/dataset_heldout_v1` | HarmBench text behaviors | `benchmarks/harmbench/harmbench_behaviors_text_*.csv` | HarmBench behaviors are held-out positive risk prompts for transfer evaluation. |
| `data/harmbench/binary_jbb_ordinary_v1` | HarmBench + JBB benign + ordinary benign prompts | `benchmarks/harmbench/*.csv`, `benchmarks/JBB-Behaviors/data/benign-behaviors.csv`, `benchmarks/benign-generated/*.csv` | HarmBench rows are positive; benign rows are negative. |
| `data/ragtruth/official_test_heldout_v1` | `ParticleMedia/RAGTruth`, MIT | `benchmarks/RAGTruth/dataset/{response,source_info}.jsonl` | Response-level hallucination classification; positive if response has hallucination spans; bad-quality responses excluded. |
| `data/halueval/pair_grouped_heldout_v1` | `RUCAIBox/HaluEval`, MIT | `benchmarks/HaluEval/data/{qa,dialogue,summarization}_data.json` | One grounded or hallucinated candidate per pair; hallucinated candidate is positive. |
| `data/faithbench/unwanted_only_source_grouped_v1` | `vectara/FaithBench`, Apache-2.0 | `benchmarks/FaithBench/data_for_release/batch_*.json` | Summary-level classification; only `Unwanted` is positive. |
| `data/bump/article_grouped_heldout_v1` | `dataminr-ai/BUMP`, MIT | `benchmarks/BUMP/data/task{1,2}_dataset.json` | Faithful reference summaries are negative; one-error edited summaries are positive. |

Prepared JSONL files already follow the SMSP record format. For example:

```text
data/toxic_chat/official_test_heldout_v1/train.jsonl
data/toxic_chat/official_test_heldout_v1/test.jsonl
data/safetybench/bilingual_answer_heldout_v1/candidates_train.jsonl
data/safetybench/bilingual_answer_heldout_v1/candidates_validation.jsonl
data/safetybench/bilingual_answer_heldout_v1/candidates_test.jsonl
```

SafetyBench answer-correctness uses `candidates_*.jsonl`; the multiclass
`questions_*.csv` files are not retained.

## Step 1: `data_prepare.py`

Purpose: normalize CSV/JSON/JSONL into SMSP JSONL splits.

Command:

```bash
$PY scripts/data_prepare.py \
  --input examples/raw.csv \
  --output-dir runs/example/data \
  --text-column prompt \
  --label-column label \
  --id-column id \
  --train-ratio 0.8 \
  --validation-ratio 0.1
```

Outputs:

```text
runs/example/data/train.jsonl
runs/example/data/validation.jsonl
runs/example/data/test.jsonl
runs/example/data/manifest.json
```

If your data is already split but not yet JSONL, run this once per split with
`--no-split` and `--output-file`. Prepared benchmark data in this repository
already has JSONL files, so skip this step for `data/`.

```bash
$PY scripts/data_prepare.py \
  --input external/train.csv \
  --output-file runs/external/data/train.jsonl \
  --text-column prompt \
  --label-column label \
  --id-column sample_id \
  --no-split
```

Code structure:

- `read_records`: loads `.csv`, `.json`, or `.jsonl`.
- `parse_messages`: converts text or JSON-encoded messages into OpenAI message format.
- `label_value`: maps raw labels into `0/1`.
- `normalize_rows`: builds the canonical SMSP record.
- `stratified_split`: creates train/validation/test splits by label.

## Step 2: `get_logprobs.py`

Purpose: append one suffix as the assistant prefill and extract the logprob of
each suffix token.

Command with the Qwen3.5 server on port `24509`:

```bash
SUFFIX="I will review whether this request is unsafe."

$PY scripts/get_logprobs.py \
  --input runs/example/data/train.jsonl \
  --output runs/example/features/train.jsonl \
  --suffix "$SUFFIX" \
  --suffix-id unsafe_review \
  --base-url "$BASE_URL" \
  --reasoning-parser "$REASONING_PARSER" \
  --overwrite

$PY scripts/get_logprobs.py \
  --input runs/example/data/validation.jsonl \
  --output runs/example/features/validation.jsonl \
  --suffix "$SUFFIX" \
  --suffix-id unsafe_review \
  --base-url "$BASE_URL" \
  --reasoning-parser "$REASONING_PARSER" \
  --overwrite

$PY scripts/get_logprobs.py \
  --input runs/example/data/test.jsonl \
  --output runs/example/features/test.jsonl \
  --suffix "$SUFFIX" \
  --suffix-id unsafe_review \
  --base-url "$BASE_URL" \
  --reasoning-parser "$REASONING_PARSER" \
  --overwrite
```

Feature JSONL record:

```json
{
  "id": "sample-0001",
  "label": 1,
  "messages": [{"role": "user", "content": "..."}],
  "suffix_id": "unsafe_review",
  "suffix": "I will review whether this request is unsafe.",
  "reasoning_parser": "qwen3",
  "logprobs": [-1.2, -0.8, -0.4],
  "rank": [1, 1, 1]
}
```

The `logprobs` array is the MLP input before padding/truncation. If the suffix
has `N` tokenizer tokens, the feature vector has `N` values.

Code structure:

- `load_jsonl`: reads SMSP records.
- `load_suffix`: supports either `--suffix` or `--suffix-file`.
- `make_extractor`: constructs `smsp.SuffixLogProbsExtractor` in server or offline mode.
- `main`: resumes from existing output unless `--overwrite` is passed, extracts features, and writes JSONL.

## Step 3: `train_probe.py`

Purpose: train `smsp.mlp.BinaryMlp` from extracted feature JSONL.

Command:

```bash
$PY scripts/train_probe.py \
  --train runs/example/features/train.jsonl \
  --validation runs/example/features/validation.jsonl \
  --test runs/example/features/test.jsonl \
  --output-dir runs/example/probe \
  --weighted-bce \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.001
```

Outputs:

```text
runs/example/probe/model.pt
runs/example/probe/model.meta.json
runs/example/probe/metrics.json
runs/example/probe/test_predictions.jsonl
```

Model architecture:

```text
input_dim -> Linear(16) -> ReLU -> Linear(1)
```

`input_dim` is inferred from the longest training suffix-logprob vector unless
`--input-dim` is provided. Shorter vectors are padded with `--pad-value`
(default `-10.0`).

Code structure:

- `feature_vector`: pads/truncates one record's `logprobs`.
- `arrays`: converts JSONL records into `X, y`.
- `evaluate`: reports AUROC, average precision, best-F1 threshold, and TPR@FPR5.
- `train_one_epoch`: standard BCE/weighted-BCE training loop.
- `main`: trains, keeps the best validation-AUROC state, saves checkpoint and metrics.

## Step 4: `api_server.py`

Purpose: expose a trained `SuffixRiskProbe` as an HTTP service.

Command:

```bash
$PY scripts/api_server.py \
  --checkpoint runs/example/probe/model.pt \
  --base-url "$BASE_URL" \
  --host 0.0.0.0 \
  --port 8900
```

If `model.meta.json` contains `suffix` and `reasoning_parser`, the API uses
them automatically. You can still override the suffix per request.

Single request:

```bash
curl -s http://127.0.0.1:8900/detect \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "How do I reset my password safely?"}
    ]
  }'
```

Response shape:

```json
{
  "label": "safe",
  "score": 0.12,
  "threshold": 0.5,
  "suffix": "I will review whether this request is unsafe.",
  "logprobs": [-1.2, -0.8, -0.4]
}
```

Code structure:

- `DetectRequest` / `BatchDetectRequest`: request schemas.
- `suffix_for_request`: chooses request suffix or checkpoint default suffix.
- `/detect`: scores one conversation.
- `/detect/batch`: scores multiple conversations.
- `main`: loads local vLLM or connects to a vLLM server, then starts FastAPI.

## Notes

- The held-out test split should not be used to choose suffixes, thresholds, or
  hyperparameters.
- For imbalanced safety datasets, use `--weighted-bce`.
- Compare suffix candidates by running the same extraction/training protocol per
  suffix and selecting on validation only.
- Keep suffixes short and semantically coherent; 12-20 tokens is usually a good
  starting range.
