#!/usr/bin/env python
"""Extract SMSP suffix log-probability features from JSONL records."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smsp import SuffixLogProbsExtractor  # noqa: E402


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


def count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1
    return count


def load_suffix(args: argparse.Namespace) -> str:
    if args.suffix is not None:
        return args.suffix
    if args.suffix_file is not None:
        return args.suffix_file.read_text(encoding="utf-8").strip()
    raise ValueError("Pass either --suffix or --suffix-file")


def make_extractor(args: argparse.Namespace) -> SuffixLogProbsExtractor:
    if args.base_url:
        return SuffixLogProbsExtractor(
            reasoning_parser=args.reasoning_parser,
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
    return SuffixLogProbsExtractor(reasoning_parser=args.reasoning_parser, llm=llm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SMSP logprob features")
    parser.add_argument("--input", type=Path, required=True, help="SMSP records JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Feature JSONL")
    parser.add_argument("--suffix")
    parser.add_argument("--suffix-file", type=Path)
    parser.add_argument("--suffix-id", default="suffix")
    parser.add_argument("--base-url", help="OpenAI-compatible vLLM URL")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--model-dir", help="Local model path for offline vLLM")
    parser.add_argument("--reasoning-parser", default="none")
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
    done = 0 if args.overwrite else count_existing(args.output)
    mode = "w" if args.overwrite else "a"
    extractor = make_extractor(args)
    print(f"records={len(records)} done={done} output={args.output}", flush=True)
    with args.output.open(mode, encoding="utf-8") as f:
        for index, record in enumerate(records[done:], start=done):
            messages = record["messages"]
            suffixed_messages = extractor.apply_suffix(messages, suffix)
            result = extractor.get_logprobs(
                suffixed_messages,
                logprobs_num=args.logprobs_num,
            )
            out = {
                "id": record.get("id", str(index)),
                "label": int(record["label"]),
                "messages": messages,
                "metadata": record.get("metadata", {}),
                "suffix_id": args.suffix_id,
                "suffix": suffix,
                "reasoning_parser": args.reasoning_parser,
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
