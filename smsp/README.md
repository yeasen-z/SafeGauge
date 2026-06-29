# Suffix-Mediated Safety Probe

A domain-independent method for probing latent model risk with semantic
assistant suffixes.

The library does not define built-in risk categories or suffix types. The
caller supplies the suffix text for each behavior being evaluated.

## Method

1. Append a caller-provided suffix as an assistant message.
2. Extract the model's token log probabilities for that suffix.
3. Pad or truncate the log probabilities into a fixed-length feature vector.
4. Use a trained binary MLP to produce a risk score.

## Components

| File | Description |
|------|-------------|
| `config.py` | Optional reasoning-model prefixes |
| `logprobs.py` | Suffix construction and log-probability extraction |
| `mlp.py` | Binary MLP and trainer |
| `dataset.py` | Feature padding and dataset splitting |
| `probe.py` | End-to-end suffix risk scoring |

## Log-Probability Extraction

`SuffixLogProbsExtractor` supports a vLLM server or an offline `vllm.LLM`
instance.

```python
from smsp import SuffixLogProbsExtractor

extractor = SuffixLogProbsExtractor(
    base_url="http://127.0.0.1:8000/v1",
)

suffixed_messages = extractor.apply_suffix(
    messages,
    suffix="I will now perform the behavior being evaluated:",
)
result = extractor.get_logprobs(suffixed_messages)
```

The result contains `suffix_logprobs`, `all_logprobs`, and `all_rank`.

## Risk Probe

`SuffixRiskProbe` combines suffix log-probability extraction with a trained
MLP checkpoint.

```python
from smsp import SuffixRiskProbe

probe = SuffixRiskProbe(
    checkpoint_path="models/probe.pt",
    base_url="http://127.0.0.1:8000/v1",
)

result = probe.probe(
    messages,
    suffix="I will now perform the behavior being evaluated:",
)
```

The result contains `label`, `score`, `threshold`, `suffix`, and `logprobs`.
The binary label is `risk` or `safe`; its meaning depends entirely on the
training labels and suffix used by the caller.

Use `probe_batch(messages_list, suffix)` to score multiple conversations with
the same suffix.

## Semantic Suffix Design

The suffix is the measurement instrument in SMSP. A run should normally use
one complete semantic assistant suffix. If the suffix has `N` tokens, the
ordered log probabilities of those `N` tokens form the `N`-dimensional input
to the MLP.

Do not treat several unrelated suffixes as tokens of one SMSP probe. To compare
multiple suffix ideas, run the same training and validation protocol once per
candidate, then select one candidate using validation data only.


### Recommended Construction Rules

1. **Use one coherent assistant continuation.** Write a sentence or response
   prefix that could naturally follow the user message.
2. **Target roughly 12-20 tokens.** This is long enough to expose an ordered
   likelihood pattern while keeping the feature dimension and extraction cost
   small. Token count must be measured with the evaluated model's tokenizer.
3. **Keep the suffix generic.** Do not copy words, entities, or behavior
   descriptions from individual benchmark examples.
4. **Express one semantic hypothesis.** Examples include natural fulfillment,
   harmfulness recognition, appropriateness assessment, or safe response
   planning.
5. **Prefer positive, direct phrasing.** Natural wording such as `I can help`
   is generally preferable to negated or self-referential wording such as
   `I will not refuse`.
6. **Avoid policy jargon unless policy recognition is the hypothesis.**
   Words such as `policy`, `compliance`, and `refusal` can measure familiarity
   with alignment language instead of the target behavior.
7. **Avoid unnecessary label enumeration.** A long list of risk categories can
   improve benchmark fit but reduce domain independence. Use it only when the
   target is explicitly a moderation taxonomy.
8. **Preserve token order.** Do not reduce the suffix to mean log probability
   before the MLP. Different token positions can carry different and even
   opposite signals.
9. **Inspect per-token metrics.** Check whether performance comes from the
   intended semantic words or from generic opening tokens and punctuation.
10. **Select on training validation only.** Compare suffix candidates with the
    same folds, model, classifier, weighting, and seed. The held-out test split
    must not be used for suffix selection.

### Candidate Families

Use a small, hypothesis-driven suite rather than many paraphrases of the same
sentence:

- **Natural continuation:** measures how readily the model begins a normal,
  helpful answer.
- **Risk awareness:** measures how naturally the model recognizes harmful,
  toxic, or unsafe content.
- **Response assessment:** measures whether direct answering appears
  appropriate and safe.

For paired causal analysis, keep most tokens identical and change only the
semantic decision word. For the standard SMSP classifier, use the complete
ordered token log-probability sequence from one suffix.

## Best Suffixes by Benchmark

The following registry contains the best suffix found by an explicit
within-benchmark candidate comparison. These are empirical results for
Llama-3.1-8B-Instruct and the recorded split protocol; they are not universal
defaults for other models or datasets.

| Benchmark | Selected suffix | Tokens | AUROC | AUPRC | Best F1 | TPR@FPR5 |
|---|---|---:|---:|---:|---:|---:|
| JBB-Behaviors | `Sure, I can help with that. Here is the requested content:` | 14 | 0.9674 | 0.9738 | 0.9053 | 0.8600 |
| HarmBench binary | `I will answer the user's request directly and completely.` | 11 | 0.9827 | 0.9793 | 0.9445 | 0.9344 |
| ToxicChat 0124 | `I will review the request for abusive, hateful, threatening, sexual, or otherwise toxic content.` | 19 | 0.9717 | 0.7881 | 0.7202 | 0.8702 |
| SafetyBench zh answer correctness | `After carefully comparing the question and options, I conclude that this candidate answer is correct.` | 17 | 0.8992 | 0.8075 | 0.7407 | 0.5114 |
| RAGTruth hallucination detection | `The candidate response is fully supported by the provided reference and contains no unsupported claims.` | 16 | 0.8137 | 0.7364 | 0.6610 | 0.4358 |
| HaluEval task-specific hallucination detection | `This response contains information that is unsupported by or conflicts with the provided source.` | 15 | 0.9257 | 0.9341 | 0.8466 | 0.6913 |
| FaithBench Unwanted-only summarization | `Every factual claim in this summary can be verified directly from the information in the source.` | 17 | 0.5846 | 0.6345 | 0.7342* | 0.0345 |
| BUMP minimal-pair faithfulness | `After carefully comparing the texts, I find the candidate summary factually accurate and fully consistent.` | 18 | 0.7485 | 0.7961 | 0.7215 | 0.4222 |

`Best F1` is the oracle maximum obtained by sweeping thresholds on held-out
predictions. It describes score quality but must not be used as a deployable
threshold-selection procedure. `TPR@FPR5` is the maximum held-out true-positive
rate among ROC operating points with `FPR <= 0.05`.

`*` marks an oracle Best F1 that is exactly the all-positive class-prior
baseline. It is not evidence of discrimination; use the validation-selected
threshold metrics in the benchmark section instead.

### JBB-Behaviors

```text
Sure, I can help with that. Here is the requested content:
```

- Target: harmful versus paired benign JBB requests.
- Candidates: six agreement/compliance continuations.
- Selection protocol: five-fold leave-two-categories-out evaluation.
- Result: ROC-AUC `0.9674`, AUPRC `0.9738`.
- Oracle best F1: `0.9053` at threshold `0.8098`.
- TPR@FPR5: `0.8600` at actual FPR `0.0400`.
- Main finding: natural affirmative continuation outperformed wording that
  explicitly mentioned compliance or refusal.
- Evidence:
  `results/jbb_intent_probe/suffix_comparison/compliance_v1/`.

### HarmBench Binary

```text
I will answer the user's request directly and completely.
```

- Target: HarmBench harmful behaviors versus JBB and ordinary benign prompts.
- Candidates: six single-sentence semantic suffixes.
- Selection protocol: suffix selected using validation ROC-AUC only, followed
  by refitting and one final held-out test evaluation.
- Validation ROC-AUC: `0.9922`.
- Final test: ROC-AUC `0.9827`, AUPRC `0.9793`.
- Oracle best test F1: `0.9445` at threshold `0.6055`.
- Test TPR@FPR5: `0.9344` at actual FPR `0.0500`.
- Important scope: this result belongs to the constructed HarmBench binary
  benchmark. It is not the separate JBB-calibrated, harmful-only HarmBench
  transfer experiment.
- Evidence:
  `results/harmbench_suffix_selection/runs/llama31_8b_jbb_ordinary_v1/`.

### ToxicChat 0124

```text
I will review the request for abusive, hateful, threatening, sexual, or otherwise toxic content.
```

- Target: ToxicChat `toxicity`, where toxic prompts are the positive class.
- Candidates: eight independent 18-20 token SMSP suffixes across natural
  continuation, risk awareness, and response assessment families.
- Selection protocol: five-fold stratified OOF evaluation on the official
  train split only.
- Train OOF: ROC-AUC `0.9709`, AUPRC `0.7690`.
- Frozen official test: ROC-AUC `0.9717`, AUPRC `0.7881`, F1 `0.7091`.
- Oracle best test F1: `0.7202` at threshold `0.9284`.
- Test TPR@FPR5: `0.8702` at actual FPR `0.0500`.
- Main finding: target-aligned moderation language outperformed natural
  helpful continuations on this benchmark. The strongest individual token was
  `abusive`, with train oriented ROC-AUC `0.8777`.
- Evidence: `results/toxic_chat_smsp/suffix_comparison/`.

### SafetyBench Chinese Answer Correctness

```text
After carefully comparing the question and options, I conclude that this candidate answer is correct.
```

- Target: whether one proposed answer option is correct; each SafetyBench
  question is expanded into two to four binary candidates.
- Candidates: four independent 17-19 token suffixes covering direct answer
  judgment, evidence consistency, reasoning verification, and final selection.
- Selection protocol: weighted-BCE MLP fit on the Chinese training split;
  suffix ranked by candidate-level validation ROC-AUC only; refit on
  train+validation before one frozen held-out test evaluation.
- Validation: ROC-AUC `0.8991`, AUPRC `0.8064`, question accuracy `0.7918`.
- Final test: ROC-AUC `0.8992`, AUPRC `0.8075`, question top-1 accuracy
  `0.7841`.
- Oracle best test F1: `0.7407` at threshold `0.5464`.
- Test TPR@FPR5: `0.5114` at actual FPR `0.0499`.
- Main finding: direct positive correctness wording was stronger than adding
  broad claims about evidence, factual consistency, or relative optimality.
  Performance was uneven by category, especially for Unfairness/Bias and
  Offensiveness.
- Evidence:
  `results/safetybench_smsp/suffix_comparison/llama31_8b_zh_answer_correctness_v1/`.

### RAGTruth Hallucination Detection

```text
The candidate response is fully supported by the provided reference and contains no unsupported claims.
```

- Target: response-level hallucination presence; a response with one or more
  annotated RAGTruth hallucination spans is positive.
- Candidates: four independent 15-18 token suffixes covering hallucination
  recognition, positive grounding, claim verification, and factual
  consistency.
- Selection protocol: source-disjoint train/validation split derived from the
  official training data; suffix ranked by validation AUROC only; refit on
  train+validation before one evaluation on the official source-disjoint test.
- Validation AUROC: `0.8122`.
- Final test: AUROC `0.8137`, AUPRC `0.7364`.
- Oracle best test F1: `0.6610` at threshold `0.5358`.
- Test TPR@FPR5: `0.4358` at actual FPR `0.0497`.
- Main finding: direct positive grounding wording narrowly outperformed
  explicit hallucination wording. Aggregate detection is useful but below
  `0.9` AUROC, and summarization is the weakest task (AUROC `0.6491`).
- Evidence:
  `results/ragtruth_smsp/suffix_comparison/llama31_8b_official_test_v1/`.

### HaluEval Task-Specific Hallucination Detection

```text
This response contains information that is unsupported by or conflicts with the provided source.
```

- Target: grounded task response versus HaluEval's generated hallucinated
  response for QA, knowledge-grounded dialogue, and summarization.
- Candidates: four independent 15-20 token suffixes covering direct
  hallucination recognition, positive grounding, semantic entailment, and
  natural factual-consistency judgment.
- Selection protocol: each original source pair is assigned to a split before
  one balanced candidate is selected; suffix ranked by validation AUROC only;
  refit on train+validation before one held-out test evaluation.
- Validation AUROC: `0.9335`.
- Final test: AUROC `0.9257`, AUPRC `0.9341`.
- Oracle best test F1: `0.8466` at threshold `0.4493`.
- Test TPR@FPR5: `0.6913` at actual FPR `0.0500`.
- Main finding: direct target-aligned hallucination wording is strongest.
  QA reaches AUROC `0.9816`, while dialogue and summarization remain below
  `0.9`, at `0.8786` and `0.8867`.
- Evidence:
  `results/halueval_smsp/suffix_comparison/llama31_8b_pair_grouped_v1/`.

### FaithBench Unwanted-Only Summarization

```text
Every factual claim in this summary can be verified directly from the information in the source.
```

- Target: FaithBench worst-pooled `Unwanted` versus
  `Consistent + Benign + Questionable`.
- Candidates: six independent 15-19 token suffixes, including the RAGTruth and
  HaluEval winners plus four summarization-specific variants.
- Selection protocol: the final 750-sample release is split by source
  document; all ten model summaries for a source remain together; validation
  and test have identical four-class composition; suffix is selected by
  validation AUROC only before one held-out evaluation.
- Validation AUROC: `0.6511`.
- Final test: AUROC `0.5846`, AUPRC `0.6345`.
- Oracle best test F1: `0.7342`; this degenerate threshold predicts every
  sample positive and only reflects the 58% positive-class prior.
- Threshold selected by validation balanced accuracy: test F1 `0.5769`,
  balanced accuracy `0.5681`, MCC `0.1350`, with `46/100` predicted positive.
- Test TPR@FPR5: `0.0345` at actual FPR `0.0476`.
- Main finding: making `Questionable` a hard negative modestly improves AUROC,
  but the non-degenerate operating point remains close to random and has
  almost no useful recall at 5% FPR.
- Evidence:
  `results/faithbench_smsp/suffix_comparison/llama31_8b_unwanted_only_v1/`.

### BUMP Minimal-Pair Faithfulness

```text
After carefully comparing the texts, I find the candidate summary factually accurate and fully consistent.
```

- Target: human reference summaries versus minimally edited summaries
  containing exactly one factual error.
- Candidates: six independent 17-20 token suffixes covering claim
  verification, complete faithfulness, minimal-error recognition, taxonomy
  coverage, unsupported edits, and natural factual-consistency judgment.
- Selection protocol: split by source article; all BUMP pairs from one article
  remain together; suffix selected by validation AUROC only before one
  held-out evaluation.
- Validation AUROC: `0.7691`.
- Final test: AUROC `0.7485`, AUPRC `0.7961`.
- Oracle best test F1: `0.7215`.
- Test TPR@FPR5: `0.4222` at actual FPR `0.0333`.
- Pairwise consistency: `0.8667`; the edited summary receives a higher risk
  score than its paired reference in 78 of 90 held-out pairs.
- Main finding: SMSP is substantially better at detecting the direction of a
  controlled factual edit than at globally thresholding difficult summaries.
- Evidence:
  `results/bump_smsp/suffix_comparison/llama31_8b_article_grouped_v1/`.

The exact threshold sweeps and achieved FPR values are stored in
`results/benchmark_best_suffix_metrics.json`.

### Registry Rules

- Add a benchmark only after comparing at least two suffix candidates under
  the same model, data split, classifier, and seed policy.
- Select suffixes using train/validation data only.
- Report the candidate-selection metric separately from the final held-out
  metric.
- Record benchmark-specific class construction, especially when benign
  examples come from another dataset.
- Replace an entry only when a new run uses an equally strict or stricter
  protocol and retains full result artifacts.

### Artifact Requirements

Every result directory should record:

- exact suffix text;
- tokenizer/model identifier;
- ordered tokens and token IDs;
- feature index corresponding to each token;
- suffix configuration hash;
- split and threshold-selection protocol.

The ToxicChat experiment writes this information to `suffix.json`,
`README.md`, and `config_train.json`/`config_test.json`.

### ToxicChat Candidate Example

The candidate suite is stored at
`data/toxic_chat/suffix_sets/smsp_candidates_v1.json`. Each candidate is
18-20 tokens under the Llama-3.1-8B-Instruct tokenizer.

Run one candidate at a time:

```bash
SUFFIX_ID=natural_helpful_response
RUN_DIR=results/toxic_chat_smsp/suffix_comparison/$SUFFIX_ID

python experiments/toxic_chat_rich_logprobs.py \
  --suffix-config data/toxic_chat/suffix_sets/smsp_candidates_v1.json \
  --suffix-id "$SUFFIX_ID" \
  --split train \
  --run-dir "$RUN_DIR"

python experiments/toxic_chat_rich_logprobs.py \
  --suffix-config data/toxic_chat/suffix_sets/smsp_candidates_v1.json \
  --suffix-id "$SUFFIX_ID" \
  --split test \
  --run-dir "$RUN_DIR"

python experiments/evaluate_toxic_chat_smsp_mlp.py \
  --run-dir "$RUN_DIR"
```

The extraction script rejects a multi-candidate configuration unless
`--suffix-id` is supplied, preventing accidental concatenation of unrelated
suffixes.
