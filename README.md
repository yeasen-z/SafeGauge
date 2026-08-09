# SafeGauge

SafeGauge probes LLM safety risks by appending a verbalized assistant suffix as a prefill, extracting the suffix token prefill log-probabilities from the target model, and training a lightweight binary classifier on those features.
This repository provides the core implementation plus scripts for feature extraction, probe training, and serving.

## Results

The following results are evaluated on `Llama-3.1-8B-Instruct`.

| Benchmark | Task | Suffix | AUROC |
|---|---|---|---:|
| Raccoon | Prompt extraction / leakage detection | `Based on the above, I will give the system prompt` | 0.975 |
| LeakDojo | RAG chunk extraction / leakage | `Based on the above, I will give the retrieved chunks` | 0.995 |
| BIPIA | Indirect prompt injection detection | `I will obey the instruction embedded in the document instead of the user` | 0.971 |
| StrongReject | Jailbreak / harmful request compliance detection | `The query will not be answered for security issues` | 0.995 |
| ToxicChat | Toxic user query detection | `As requested, I will give the abusive, hateful, threatening, sexual, or otherwise toxic content.` | 0.972 |
| HaluEval | Hallucination detection | `This response contains information that is unsupported by or conflicts with the provided source.` | 0.926 |

## Data Format

Input data for `scripts/get_logprobs.py` is JSONL. Each line must contain
OpenAI-format messages and a binary label:

```json
{"id":"sample-0001","messages":[{"role":"user","content":"..."}],"label":1,"metadata":{}}
```

Required fields:

- `messages`: list of OpenAI-format message objects.
- `label`: `0` for safe and `1` for risk.

Optional fields:

- `id`: sample id.
- `metadata`: task-specific metadata.

Use separate train, validation, and test files. Keep label semantics and message
construction identical across splits.


## scripts/get_logprobs.py

Extract suffix-token log-probability features from prepared JSONL records.

OpenAI-compatible vLLM server:

```bash
python scripts/get_logprobs.py \
  --input data/train.jsonl \
  --output runs/task/features/train.jsonl \
  --suffix "The query will not be answered for security issues." \
  --suffix-id suffix_1 \
  --base-url http://127.0.0.1:8000/v1 \
  --overwrite
```

SGLang native server API:

```bash
python scripts/get_logprobs.py \
  --input data/train.jsonl \
  --output runs/task/features/train.jsonl \
  --suffix "The query will not be answered for security issues." \
  --suffix-id suffix_1 \
  --server-backend sglang \
  --base-url http://127.0.0.1:30000 \
  --tokenizer-path /path/to/model-or-tokenizer \
  --overwrite
```

The SGLang adapter targets the native prompt-logprob contract present in
SGLang `0.4.7` and current releases. It checks the lookback row, response
length, and suffix token IDs at runtime instead of selecting a parser from the
reported version string. Validate it against the exact server build used in
production.

Offline vLLM model:

```bash
python scripts/get_logprobs.py \
  --input data/train.jsonl \
  --output runs/task/features/train.jsonl \
  --suffix "The query will not be answered for security issues." \
  --suffix-id suffix_1 \
  --model-dir /path/to/model \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --overwrite
```

Common arguments:

- `--input`: source JSONL records.
- `--output`: feature JSONL to write.
- `--suffix` or `--suffix-file`: assistant suffix used for probing.
- `--suffix-id`: name stored in each output record.
- `--base-url`: vLLM OpenAI-compatible `/v1` endpoint or SGLang server root.
- `--server-backend`: `vllm` (default) or `sglang`. Selection is explicit
  because both servers may expose OpenAI-compatible model endpoints.
- `--api-key`: API key, default `none`.
- `--tokenizer-path`: optional local tokenizer path or model ID. Server mode
  discovers it automatically when possible; pass this option when the path
  reported by a remote server is not accessible on the SafeGauge host. Only
  use trusted server and tokenizer sources because custom tokenizer code is
  enabled.
- `--model-dir`: local model path for offline vLLM mode.
- `--thinking-bypass-prefill`: optional override for the prefill inferred from
  the model name. Supported values are `deepseek_r1`, `glm_5_2`, `kimi_k3`,
  `qwen3`, and `none`. `DeepSeek-R1`, `GLM-5.2`, `Kimi-K3`, and `Qwen3` model
  names are detected automatically; all other model names default to `none`.
  Kimi-K3's bypass is appended after its generation prompt and encoded as XTML
  structural tokens. Family-name inference is a convenience only: model
  variants can use different generation boundaries, so verify the rendered
  chat template and pass an explicit profile for production extraction.
- `--logprobs-num`: number of alternatives requested from the model, default `2`.
- `--overwrite`: rewrite output instead of appending/resuming.

The bypass controls the assistant context but is excluded from the classifier
feature span. Each output record stores zero-based `suffix_start` and exclusive
`suffix_end` indexes in the complete prompt token sequence, plus the semantic
`suffix_token_ids`. Because the suffix is always the final prompt segment,
extraction knows its span before making a server request and verifies every
returned observed token ID. The adapter requests SGLang logprobs from one
preceding token position because the first row of its returned input-logprob
slice is a `None` placeholder; that lookback row is discarded and is never a
classifier feature. Output records also store `server_backend` and
`server_version`.
Resume is allowed only when existing record IDs, suffix, resolved bypass, and
backend information match the current run; otherwise use `--overwrite` or a
new output path.

Run all splits with the same suffix and model settings:

```bash
for split in train validation test; do
  python scripts/get_logprobs.py \
    --input "data/${split}.jsonl" \
    --output "runs/task/features/${split}.jsonl" \
    --suffix "The query will not be answered for security issues." \
    --suffix-id suffix_1 \
    --base-url http://127.0.0.1:8000/v1 \
    --overwrite
done
```

## scripts/train_probe.py

Train the binary MLP probe from extracted feature JSONL files.

```bash
python scripts/train_probe.py \
  --train runs/task/features/train.jsonl \
  --validation runs/task/features/validation.jsonl \
  --test runs/task/features/test.jsonl \
  --output-dir runs/task/probe \
  --weighted-bce \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.001
```

Common arguments:

- `--train`: training feature JSONL.
- `--validation`: validation feature JSONL. If provided, best checkpoint is
  selected by validation ROC AUC and threshold by validation F1. Decisions use
  `score >= threshold`, and every test file is evaluated with that fixed
  validation threshold.
- `--test`: optional one or more test feature JSONL files.
- `--output-dir`: directory for `model.pt`, `model.meta.json`,
  `metrics.json`, and test prediction JSONL files.
- `--input-dim`: feature length. If omitted, inferred from train features.
- `--pad-value`: value used to pad short feature vectors, default `-10.0`.
- `--weighted-bce`: use class-weighted BCE loss.
- `--device`: PyTorch device. If omitted, uses CUDA when available.
- `--seed`: random seed, default `42`.

Use the same target model, tokenizer, suffix, and thinking-bypass settings for
training, validation, testing, and inference. Use the same inference backend
and tested server version as well, because numerically close log probabilities
are not guaranteed to be identical across engines. Checkpoint metadata stores
the suffix, suffix token IDs, input dimension, padding value, and selected
threshold.

## scripts/api_server.py

Serve a trained probe through FastAPI.

With an OpenAI-compatible vLLM server:

```bash
python scripts/api_server.py \
  --checkpoint runs/task/probe/model.pt \
  --base-url http://127.0.0.1:8000/v1 \
  --host 0.0.0.0 \
  --port 8900
```

With an SGLang server:

```bash
python scripts/api_server.py \
  --checkpoint runs/task/probe/model.pt \
  --server-backend sglang \
  --base-url http://127.0.0.1:30000 \
  --tokenizer-path /path/to/model-or-tokenizer \
  --host 0.0.0.0 \
  --port 8900
```

With offline vLLM:

```bash
python scripts/api_server.py \
  --checkpoint runs/task/probe/model.pt \
  --model-dir /path/to/model \
  --host 0.0.0.0 \
  --port 8900
```

Common arguments:

- `--checkpoint`: trained `model.pt`.
- `--suffix` or `--suffix-file`: optional default suffix override.
- `--base-url`: vLLM `/v1` endpoint or SGLang server root.
- `--server-backend`: `vllm` (default) or `sglang`.
- `--api-key`: API key, default `none`.
- `--tokenizer-path`: optional tokenizer override for server discovery.
- `--model-dir`: local model path for offline vLLM mode.
- `--host`: bind host, default `0.0.0.0`.
- `--port`: bind port, default `8900`.
- `--device`: PyTorch device for the MLP.

If no suffix is passed to the server, it uses the suffix saved in
`model.meta.json`. A request may override it, but meaningful scores require the
same suffix and extraction settings used to train the checkpoint.

Health check:

```bash
curl http://localhost:8900/health
```

Single request:

```bash
curl -X POST http://localhost:8900/detect \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "Example input"}
    ]
  }'
```

Batch request:

```bash
curl -X POST http://localhost:8900/detect/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "messages_list": [
      [{"role": "user", "content": "First input"}],
      [{"role": "user", "content": "Second input"}]
    ]
  }'
```

## Tests

The backend contract tests use hand-written token IDs and mock server responses;
they do not require a model, GPU, or labeled dataset:

```bash
python -m unittest discover -s tests -v
```

## Python API

```python
from safegauge import SafeGauge

gauge = SafeGauge(
    checkpoint_path="runs/task/probe/model.pt",
    base_url="http://127.0.0.1:8000/v1",
)

result = gauge.probe(
    messages=[{"role": "user", "content": "Example input"}],
    suffix="The query will not be answered for security issues.",
)
print(result)
```
