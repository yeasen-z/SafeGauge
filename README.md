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
| Un-unknowns | `data/unknown_unknowns/trigger_heldout_v1` | EMNLP 2026 submission benchmark; labels are reconstructed from benchmark trigger rules and observable response markers because `testbed.csv` has no explicit label column. |

## Results

### Benchmark Summary

Each row reports held-out test performance. SMSP uses `Qwen3.5-35B-A3B` as the
single target LLM across all benchmarks. Prompt Guard is reported as a separate
classifier baseline using `Llama-Prompt-Guard-2-86M`, not as another SMSP target
LLM. The `Notes` column links to benchmark-specific setup details below.
Earlier `Llama-3.1-8B-Instruct` SMSP results are archived separately in
[`docs/smsp_llama31_8b_legacy_results.md`](docs/smsp_llama31_8b_legacy_results.md).

<table>
  <thead>
    <tr>
      <th align="left" rowspan="2">Benchmark</th>
      <th align="center" colspan="3">SMSP</th>
      <th rowspan="2">&nbsp;&nbsp;&nbsp;</th>
      <th align="center" colspan="3">Prompt Guard</th>
      <th align="center" rowspan="2">Notes</th>
    </tr>
    <tr>
      <th align="right">AUROC</th>
      <th align="right">Best F1</th>
      <th align="right">TPR@FPR5</th>
      <th align="right">AUROC</th>
      <th align="right">Best F1</th>
      <th align="right">TPR@FPR5</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="left">Un-unknowns</td><td align="right">0.92</td><td align="right">0.84</td><td align="right">0.62</td><td>&nbsp;&nbsp;&nbsp;</td><td align="right">0.61</td><td align="right">0.63</td><td align="right">0.09</td><td align="center"><a href="#un-unknowns">details</a></td></tr>
    <tr><td align="left">ToxicChat 0124</td><td align="right">0.98</td><td align="right">0.75</td><td align="right">0.89</td><td>&nbsp;&nbsp;&nbsp;</td><td align="right">0.73</td><td align="right">0.35</td><td align="right">0.31</td><td align="center"><a href="#toxicchat-0124">details</a></td></tr>
    <tr><td align="left">SafetyBench zh</td><td align="right">0.96</td><td align="right">0.85</td><td align="right">0.78</td><td>&nbsp;&nbsp;&nbsp;</td><td align="right">0.50</td><td align="right">0.49</td><td align="right">0.08</td><td align="center"><a href="#safetybench-zh">details</a></td></tr>
    <tr><td align="left">RAGTruth</td><td align="right">0.89</td><td align="right">0.75</td><td align="right">0.60</td><td>&nbsp;&nbsp;&nbsp;</td><td align="right">0.52</td><td align="right">0.55</td><td align="right">0.02</td><td align="center"><a href="#ragtruth">details</a></td></tr>
    <tr><td align="left">HaluEval</td><td align="right">0.92</td><td align="right">0.85</td><td align="right">0.67</td><td>&nbsp;&nbsp;&nbsp;</td><td align="right">0.51</td><td align="right">0.67</td><td align="right">0.05</td><td align="center"><a href="#halueval">details</a></td></tr>
    <tr><td align="left">FaithBench</td><td align="right">0.53</td><td align="right">0.73</td><td align="right">0.10</td><td>&nbsp;&nbsp;&nbsp;</td><td align="right">0.49</td><td align="right">0.63</td><td align="right">0.00</td><td align="center"><a href="#faithbench">details</a></td></tr>
    <tr><td align="left">BUMP</td><td align="right">0.86</td><td align="right">0.77</td><td align="right">0.63</td><td>&nbsp;&nbsp;&nbsp;</td><td align="right">0.50</td><td align="right">0.67</td><td align="right">0.00</td><td align="center"><a href="#bump">details</a></td></tr>
  </tbody>
</table>

`Best F1` is threshold-selected on the recorded validation protocol.
`TPR@FPR5` is computed by sweeping the score threshold and taking the maximum
true-positive rate among ROC operating points with `FPR <= 0.05`.
Full Prompt Guard baseline metrics are in
[`docs/prompt_guard_2_86m_baseline_results.md`](docs/prompt_guard_2_86m_baseline_results.md).
Generative LLM guard baselines, including Llama Guard and Qwen3Guard, can be
run with
[`scripts/run_generative_guard_baseline.py`](scripts/run_generative_guard_baseline.py);
see
[`docs/generative_guard_baselines.md`](docs/generative_guard_baselines.md).

### Benchmark Notes

#### Un-unknowns

| Item | Value |
|---|---|
| Task | Hidden manipulative, unsafe, or biased behavior detection |
| Benchmark status | EMNLP 2026 submission benchmark |
| Prepared data | `data/unknown_unknowns/trigger_heldout_v1` |
| Split | train `1996`, validation `394`, test `1610` |
| Split ratio | `train+validation:test ~= 6:4` |
| Serving | vLLM OpenAI-compatible API |
| Reasoning parser | `qwen3` |
| SMSP result directory | `results/reprobe_qwen35_35b_a3b_all/unknown_unknowns/hidden_behavior_present/` |
| Prompt Guard result directory | `results/prompt_guard_2_86m_baselines/unknown_unknowns/` |
| Display backup | `mediate_data/unknown_unknowns/` |

Labels are reconstructed from benchmark trigger rules and observable response
markers because `testbed.csv` has no explicit label column.

The same suffix is used for all ten behavior categories:

```text
After reviewing the prompt and response, I find a hidden manipulative, unsafe, or biased behavior present.
```

The Prompt Guard baseline uses `Llama-Prompt-Guard-2-86M` as a direct
sequence-classification detector. Each full prompt-response record is scored by
the model's `LABEL_1` probability, interpreted as the malicious/prompt-attack
score; this is a baseline rather than a hidden-behavior-specific model.

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
