# SuffixProbe core library

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
| `dataset.py` | Tensor dataset wrapper used by training scripts |
| `probe.py` | End-to-end suffix risk scoring |

## Log-Probability Extraction

`SuffixLogProbsExtractor` supports a vLLM server or an offline `vllm.LLM`
instance.

```python
from suffixprobe import SuffixLogProbsExtractor

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
from suffixprobe import SuffixRiskProbe

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

The suffix is the measurement instrument in SuffixProbe. A run should normally use
one complete semantic assistant suffix. If the suffix has `N` tokens, the
ordered log probabilities of those `N` tokens form the `N`-dimensional input
to the MLP.

Do not treat several unrelated suffixes as tokens of one SuffixProbe model. To compare
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
semantic decision word. For the standard SuffixProbe classifier, use the complete
ordered token log-probability sequence from one suffix.

Benchmark protocols and empirical results are documented in the repository
root README. This package README is intentionally limited to the core SuffixProbe
library behavior and API.
