# Prompt/suffix optimization method review

## Decision

Use semantic evolutionary search first, with a nested held-out protocol and a
joint AUROC/TPR@FPR5 objective. Add gradient-guided token search only when the
Llama weights can run under autograd.

## Methods considered

### Manual and evolutionary semantic search

Promptbreeder evolves task prompts and mutation prompts, while OPRO represents
the optimization trajectory in text and asks an LLM to propose better prompts.
Both are compatible with a black-box scorer. For SMSP the evaluator is not task
accuracy but a trained detector over ordered suffix-token log probabilities, so
every proposal must be scored end-to-end rather than by suffix likelihood alone.

- Promptbreeder: https://arxiv.org/abs/2309.16797
- OPRO: https://arxiv.org/abs/2309.03409

The implemented first population crosses four Chinese openers, three subjects,
and four positive judgments, plus manually designed English, Chinese, bilingual,
minimal, entailment, verification, and contrastive-selection seeds.

### AutoPrompt / GCG-style discrete token search

AutoPrompt and GCG use gradients with respect to discrete token choices to rank
coordinate replacements. GCG is attractive because it can search non-semantic
token sequences, but its standard objective maximizes the likelihood of a target
continuation. That objective is not equivalent to separating correct from
incorrect SafetyBench candidates, and an OpenAI-compatible prompt-logprob API
does not expose token-choice gradients.

- AutoPrompt: https://aclanthology.org/2020.emnlp-main.346/
- GCG: https://arxiv.org/abs/2307.15043

For a genuine SMSP-GCG variant, use a balanced minibatch of correct/incorrect
candidates and optimize a pairwise ranking surrogate such as
`softplus(-(score_pos - score_neg))`. Candidate token replacements should then
be accepted only by the nested AUROC/TPR evaluation. Restricting replacements
to natural-language tokens is optional, but comparing semantic and unrestricted
frontiers is necessary to quantify transfer and interpretability.

## Objective and selection

The existing baseline selects by validation AUROC only. Its validation results
already show the conflict:

| Suffix | AUROC | TPR@FPR5 |
|---|---:|---:|
| careful correctness | 0.8991 | 0.4899 |
| confident selection | 0.8981 | 0.5223 |

The baseline audit in `suffix_evolve/results/baseline_audit.json` recomputes
scores without reading test. Under the joint objective, confident selection has
the best point score and bootstrap lower bound. This is evidence that objective
design matters; it is not yet evidence of a held-out test improvement.

The search therefore uses:

`joint = 0.5 * AUROC + 0.5 * TPR@FPR5`

and ranks by the 10th-percentile question-bootstrap lower confidence bound.
Pareto status is retained so a modest global-AUROC gain cannot hide a large loss
at the requested operating point.

## Leakage controls

- Candidate answer rows from the same MCQ always remain in one group.
- Evolution happens inside the original training split.
- Original validation is reserved for confirming finalists.
- Test remains untouched until one suffix and training protocol are frozen.
- Bootstrap resamples whole questions, not correlated candidate rows.

## Generation-1 empirical result

The first nested search evaluated 59 suffixes and retained the original suffix
as a protected control. On the largest inner split (900 fit questions, 450
search questions), `仔细核对后，这个候选答案是正确的。` was the only Pareto
candidate:

| Suffix | AUROC | TPR@FPR5 | Joint LCB |
|---|---:|---:|---:|
| generation-1 winner | 0.8918 | 0.5400 | 0.6882 |
| original English suffix | 0.8878 | 0.4667 | 0.6563 |

Repeated token-feature permutation attributes most of the winner's signal to
`是` and `正确`, with smaller positive contributions distributed across
`仔细核对后`. Generation 2 therefore freezes the correctness core and mutates
only the verification prefix, scope, judgment frame, certainty, and punctuation
coordinates. Its candidates are stored in
`configs/semantic_generation2_v1.json`.

## Extraction reproducibility finding

Independent concurrent extractions of identical prompts showed material
log-probability variation (maximum absolute differences around 0.5 on small
repeats). Inspection of the installed vLLM source confirms that its prompt
logprob mapping is keyed by token ID and inserts the observed token first, so
this variation is not sufficient evidence of selecting a top-k alternative.
It is more consistent with numerical variation under different continuous
batching schedules. The evolution extractor now matches the observed suffix
token ID explicitly. Before any final metric is accepted, the same small cache
must also be reproduced twice with `workers=1`; final validation/test should be
serial or use repeated extraction with a predeclared aggregation rule.

The mechanism-level candidate set in `configs/mechanism_candidates_v1.json`
moves beyond semantic paraphrases. Its primary family renders the candidate's
actual option letter into suffixes such as `最终答案是{candidate_letter}。`,
directly probing the model's native multiple-choice distribution. Other
families use structured verdicts, negative judgments, reasoning scaffolds,
contrastive claims, entailment labels, and calibrated confidence formats.
