#!/usr/bin/env python
"""Serve a SafeGauge model over FastAPI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safegauge import SafeGauge


class DetectRequest(BaseModel):
    messages: list[dict] = Field(..., description="OpenAI-format messages")
    suffix: Optional[str] = Field(None, description="Override suffix text")


class BatchDetectRequest(BaseModel):
    messages_list: list[list[dict]] = Field(..., description="Batch of OpenAI-format messages")
    suffix: Optional[str] = Field(None, description="Override suffix text")


app = FastAPI(
    title="SafeGauge API",
    description="SafeGauge service",
    version="0.1.0",
)

probe: SafeGauge | None = None
default_suffix: str | None = None


def suffix_for_request(override: str | None) -> str:
    suffix = override or default_suffix
    if not suffix:
        raise ValueError("No suffix provided. Pass --suffix/--suffix-file or include suffix in the request.")
    return suffix


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/model/info")
async def model_info() -> dict:
    if probe is None:
        raise RuntimeError("SafeGauge is not initialized")
    return probe.get_model_info()


@app.post("/detect")
async def detect(req: DetectRequest) -> dict:
    if probe is None:
        raise RuntimeError("SafeGauge is not initialized")
    return probe.probe(req.messages, suffix_for_request(req.suffix))


@app.post("/detect/batch")
async def detect_batch(req: BatchDetectRequest) -> dict:
    if probe is None:
        raise RuntimeError("SafeGauge is not initialized")
    suffix = suffix_for_request(req.suffix)
    results = probe.probe_batch(req.messages_list, suffix)
    return {
        "results": results,
        "total": len(results),
        "errors": sum(result["label"] == "error" for result in results),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafeGauge API Server")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained MLP checkpoint")
    parser.add_argument("--suffix", help="Default suffix text")
    parser.add_argument("--suffix-file", type=Path, help="File containing default suffix text")
    parser.add_argument("--base-url", help="vLLM /v1 URL or SGLang server root URL")
    parser.add_argument(
        "--server-backend",
        choices=("vllm", "sglang"),
        default="vllm",
        help="Server implementation used by --base-url (default: vllm)",
    )
    parser.add_argument("--api-key", default="none")
    parser.add_argument(
        "--tokenizer-path",
        help="Local tokenizer path or model ID overriding server discovery",
    )
    parser.add_argument("--model-dir", help="Local model path for offline vLLM")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    global probe, default_suffix
    args = parse_args()
    if args.suffix_file:
        default_suffix = args.suffix_file.read_text(encoding="utf-8").strip()
    else:
        default_suffix = args.suffix
    llm = None
    if args.base_url is None:
        if not args.model_dir:
            raise ValueError("Provide either --base-url or --model-dir")
        from vllm import LLM

        llm = LLM(model=args.model_dir)
    probe = SafeGauge(
        checkpoint_path=str(args.checkpoint),
        base_url=args.base_url,
        api_key=args.api_key,
        llm=llm,
        device=args.device,
        server_backend=args.server_backend,
        tokenizer_path=args.tokenizer_path,
    )
    if default_suffix is None:
        default_suffix = probe.meta.get("suffix")
    if default_suffix is not None:
        probe.validate_suffix(default_suffix)
    print(f"Model loaded: {probe.get_model_info()}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
