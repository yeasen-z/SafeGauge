# SafeGauge core library

A domain-independent method for probing latent model risk with semantic
assistant suffixes.

The library does not define built-in risk categories or suffix types. The
caller supplies the suffix text for each behavior being evaluated.

## Method

1. Append a caller-provided suffix as an assistant message.
2. Optionally place a thinking-bypass prefill before the suffix as context.
3. Extract log probabilities for the semantic suffix tokens only; bypass tokens
   are never classifier features.
4. Pad or truncate the log probabilities into a fixed-length feature vector.
5. Use a trained binary MLP to produce a risk score.

## Components

| File | Description |
|------|-------------|
| `config.py` | Model detection and optional thinking-bypass prefills |
| `feature_spec.py` | Versioned feature contract and identity checks |
| `logprobs.py` | Suffix construction and log-probability extraction |
| `mlp.py` | Binary MLP definition and checkpoint I/O |
| `dataset.py` | Tensor dataset wrapper used by training scripts |
| `probe.py` | End-to-end suffix risk scoring |

## Log-Probability Extraction

`LogProbsExtractor` supports a vLLM server or an offline `vllm.LLM`
instance.

```python
from safegauge import LogProbsExtractor

extractor = LogProbsExtractor(
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

`SafeGauge` combines suffix log-probability extraction with a trained
MLP checkpoint.

```python
from safegauge import SafeGauge

gauge = SafeGauge(
    checkpoint_path="models/probe.pt",
    base_url="http://127.0.0.1:8000/v1",
)

result = gauge.probe(
    messages,
    suffix="I will now perform the behavior being evaluated:",
)
```

The result contains `label`, `score`, `threshold`, `suffix`, and `logprobs`.
SafeGauge defines training label `1` as `risk` and label `0` as `safe`.

Use `gauge.probe_batch(messages_list, suffix)` to score multiple conversations
with the same suffix. A trained checkpoint is bound to its model, tokenizer,
chat template, thinking bypass, suffix token IDs, and vectorization settings.
Passing a different suffix is rejected by default because it creates a different
feature space.

Legacy checkpoints and intentional experiments can opt out explicitly with
`unsafe_allow_feature_mismatch=True`. Scores produced under that override are
not directly comparable to the checkpoint's validated feature space.

## Feature Contract

Feature extraction writes a versioned `feature_spec` and its SHA-256 identity
to every JSONL record. Training requires one identical spec across train,
validation, and test files, then adds the input dimension, padding rule, and
positive-label meaning to the checkpoint spec. Serving reconstructs this spec
from the active model and rejects mismatches before scoring.

Model revision is recorded when exposed by vLLM or the tokenizer. For a mutable
local model directory without revision metadata, SafeGauge can bind the path,
tokenizer identity, token IDs, and chat-template hash, but it cannot prove that
the weight files behind that path were not replaced.

## Thinking-Bypass Model Adapters

The name-based profiles in `config.py` are convenience adapters, not exhaustive
family support. Models with the same family name may end their generation prompt
at different boundaries: before a think channel, inside an already-open think
channel, or directly inside a response channel. A wrong bypass can introduce an
orphan closing marker or open the same channel twice.

Before adding a model variant:

1. Render its own tokenizer or renderer with `add_generation_prompt=True`.
2. Inspect the exact assistant boundary, preferably as token IDs as well as text.
3. Use `none` when the renderer already supports disabling thinking directly.
4. Add or select a bypass only when the boundary is known, then regenerate the
   features so the resolved bypass text and token IDs enter `feature_spec`.

Automatic inference should therefore be treated as a starting guess. Use the
explicit `thinking_bypass_prefill` override for a verified deployment.

## Semantic Suffix Design

The suffix is the measurement instrument in SafeGauge. The implemented design
uses one complete semantic assistant suffix. If the suffix has `N` tokens, the
ordered log probabilities of those `N` tokens form the raw feature sequence.
Do not treat several unrelated suffixes as tokens of one SafeGauge model.
Compare candidate ideas with the same training and validation protocol, then
select using validation data only.


### Recommended Construction Rules

1. **Use one coherent assistant continuation.** Write a sentence or response
   prefix that could naturally follow the user message.
2. **Match length to semantic complexity.** Use a short, direct suffix for a
   simple behavior and add length only when a more complex behavior needs more
   precise description. There is no universal token limit. Avoid padding the
   wording with unnecessary detail, and measure length with the evaluated
   model's tokenizer.
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

## Enhancement: Contrastive Dual Suffix

Dual-suffix probing is a proposed enhancement and is not yet implemented by the
extractor, trainer, checkpoint, or serving API. It would score a matched
normal/benign suffix and malicious/risky suffix independently against the same
conversation.

The pair should be close in tokenizer token length, grammatical structure,
tone, and specificity, while changing the behavioral hypothesis. Similar
character count is not sufficient. Simple behaviors may use short pairs;
longer descriptions are appropriate when the behavior genuinely needs more
semantic detail.

The initial implementation should concatenate the two complete ordered
sequences in a fixed order, `normal` then `malicious`. An optional
position-aligned `malicious - normal` rule can be evaluated later. The selected
rule, both suffix texts and token IDs, individual lengths, padding behavior,
and ordering must become part of `feature_spec` and remain identical during
training, validation, testing, and inference.

Benchmark protocols and empirical results are documented in the repository
root README. This package README is intentionally limited to the core SafeGauge
library behavior and API.
