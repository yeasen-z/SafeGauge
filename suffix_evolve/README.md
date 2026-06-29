# SafetyBench suffix evolution

This directory isolates suffix-search work from the published SafetyBench
baseline. The baseline is read-only input; all new configs, caches, reports,
and code live here.

The search target is Chinese SafetyBench candidate-answer correctness for
Llama-3.1-8B-Instruct. It jointly rewards global ranking quality (AUROC) and
the high-specificity operating region (TPR at FPR <= 5%). Test data is never
used during suffix search.

## Protocol

1. Split the original training questions into `fit` and `search` groups.
2. Extract the ordered suffix-token log-probability vector for every candidate.
3. Fit the same weighted-BCE MLP used by the baseline on `fit`.
4. Rank suffixes on `search` using a question-bootstrap lower confidence bound
   of `0.5 * AUROC + 0.5 * TPR@FPR5`.
5. Evaluate only the final Pareto candidates on the untouched validation split.
6. Choose one suffix before extracting the held-out test split.

This nested protocol matters: repeatedly optimizing the public validation score
would turn validation into training data and make a small apparent gain weak
evidence.

## Run

The existing vLLM-compatible server must expose prompt log-probabilities:

```bash
python suffix_evolve/src/run_search.py \
  --base-url http://127.0.0.1:24509/v1 \
  --candidate-config suffix_evolve/configs/semantic_candidates_v1.json \
  --run-dir suffix_evolve/results/semantic_v1
```

Use `--dry-run` to validate the split, candidates, and tokenizer without model
requests. Interrupted extraction is resumable from JSONL caches.

## Why not call this GCG?

Original GCG uses gradients with respect to one-hot token choices to shortlist
coordinate replacements. A completions API that returns prompt log-probability
values does not expose those gradients. This implementation is therefore a
black-box evolutionary search over semantic phrases. A genuine GCG backend
requires local model weights and autograd; it should optimize a differentiable
pairwise-ranking surrogate on `fit`, then pass candidates through the same
nested evaluation protocol.



白盒试一下，看看上xian
embedding 分布
