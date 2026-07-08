# Suffix Registry

This directory stores the versioned suffix candidate registry used by benchmark
runs.

Prepared benchmark JSONL files under `data/` can be regenerated from raw
snapshots in `benchmarks/`. The suffix candidates are experiment configuration,
so they live in the source tree as a small JSON file:

```text
suffix_evolve/suffix_sets.json
```

`scripts/run_existing_data.py` reads suffix sets from this registry using
references such as:

```text
suffix_evolve/suffix_sets.json::safetybench.answer_correctness_v1
```

This keeps GitHub commits small while preserving the exact suffix candidates
used by the benchmark protocols documented in the root README.
