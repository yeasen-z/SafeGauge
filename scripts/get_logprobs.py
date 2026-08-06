#!/usr/bin/env python
"""Extract SafeGauge suffix log-probability features from JSONL records."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safegauge import LogProbsExtractor  # noqa: E402
from safegauge.feature_spec import feature_spec_hash  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def validate_resume(
    existing: list[dict[str, Any]],
    inputs: list[dict[str, Any]],
    expected_feature_spec_hash: str,
) -> None:
    """Ensure an append is a continuation of the same ordered extraction run."""
    if len(existing) > len(inputs):
        raise ValueError(
            "Output contains more records than the input; pass --overwrite or "
            "choose a different output path"
        )
    for index, output_record in enumerate(existing):
        expected_id = str(inputs[index].get("id", str(index)))
        actual_id = str(output_record.get("id"))
        if actual_id != expected_id:
            raise ValueError(
                f"Resume mismatch at record {index}: output id {actual_id!r} "
                f"!= input id {expected_id!r}. Pass --overwrite."
            )
        expected_label = int(inputs[index]["label"])
        try:
            actual_label = int(output_record.get("label"))
        except (TypeError, ValueError):
            actual_label = None
        if (
            output_record.get("messages") != inputs[index].get("messages")
            or actual_label != expected_label
        ):
            raise ValueError(
                f"Resume input mismatch at record {index} ({expected_id!r}); "
                "messages or label changed. Pass --overwrite."
            )
        actual_hash = output_record.get("feature_spec_hash")
        if actual_hash != expected_feature_spec_hash:
            raise ValueError(
                f"Resume feature spec mismatch at record {index}. "
                "The model, tokenizer, bypass, or suffix changed; pass --overwrite."
            )


def load_suffix(args: argparse.Namespace) -> str:
    if args.suffix is not None:
        return args.suffix
    if args.suffix_file is not None:
        return args.suffix_file.read_text(encoding="utf-8").strip()
    raise ValueError("Pass either --suffix or --suffix-file")


def make_extractor(args: argparse.Namespace) -> LogProbsExtractor:
    if args.base_url:
        return LogProbsExtractor(
            thinking_bypass_prefill=args.thinking_bypass_prefill,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    if not args.model_dir:
        raise ValueError("Offline mode requires --model-dir")
    from vllm import LLM

    llm = LLM(
        model=args.model_dir,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    return LogProbsExtractor(
        thinking_bypass_prefill=args.thinking_bypass_prefill,
        llm=llm,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SafeGauge logprob features")
    parser.add_argument("--input", type=Path, required=True, help="SafeGauge records JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Feature JSONL")
    parser.add_argument("--suffix")
    parser.add_argument("--suffix-file", type=Path)
    parser.add_argument("--suffix-id", default="suffix")
    parser.add_argument("--base-url", help="OpenAI-compatible vLLM URL")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--model-dir", help="Local model path for offline vLLM")
    parser.add_argument(
        "--thinking-bypass-prefill",
        choices=("deepseek_r1", "glm_5_2", "kimi_k3", "qwen3", "none"),
        help="Override the thinking-bypass prefill inferred from the model name",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--logprobs-num", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = load_suffix(args)
    records = load_jsonl(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    extractor = make_extractor(args)
    extraction_spec = extractor.feature_spec(suffix, args.suffix_id)
    extraction_spec_hash = feature_spec_hash(extraction_spec)
    existing = []
    if not args.overwrite and args.output.exists():
        existing = load_jsonl(args.output)
        validate_resume(existing, records, extraction_spec_hash)
    done = len(existing)
    mode = "w" if args.overwrite else "a"
    print(f"records={len(records)} done={done} output={args.output}", flush=True)
    with args.output.open(mode, encoding="utf-8") as f:
        for index, record in enumerate(records[done:], start=done):
            messages = record["messages"]
            label = int(record["label"])
            if label not in (0, 1):
                raise ValueError(
                    f"Record {record.get('id', index)!r} has label {label}; "
                    "SafeGauge requires 0=safe and 1=risk"
                )
            suffixed_messages = extractor.apply_suffix(messages, suffix)
            result = extractor.get_logprobs(
                suffixed_messages,
                logprobs_num=args.logprobs_num,
            )
            out = {
                "id": record.get("id", str(index)),
                "label": label,
                "messages": messages,
                "metadata": record.get("metadata", {}),
                "suffix_id": args.suffix_id,
                "suffix": suffix,
                "thinking_bypass_prefill": extractor.thinking_bypass_prefill,
                "feature_spec": extraction_spec,
                "feature_spec_hash": extraction_spec_hash,
                "suffix_logprobs": result["suffix_logprobs"],
                "logprobs": result["all_logprobs"],
                "rank": result["all_rank"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            if (index + 1) % 25 == 0 or index + 1 == len(records):
                print(f"[{index + 1}/{len(records)}] extracted", flush=True)


if __name__ == "__main__":
    main()
