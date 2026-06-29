from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from experiments.safetybench_smsp import load_cache


def render_suffix(suffix: dict, row: dict) -> str:
    template = suffix.get("template")
    if template is None:
        return suffix["text"]
    allowed = {"candidate_letter": str(row["candidate_letter"])}
    return template.format_map(allowed)


def verify_suffix(extractor, suffix: dict) -> dict:
    if "template" not in suffix:
        texts = {"static": suffix["text"]}
    else:
        texts = {
            letter: suffix["template"].format(candidate_letter=letter)
            for letter in ("A", "B", "C", "D")
        }
    tokenizations = {
        key: extractor.tokenizer.encode(text, add_special_tokens=False)
        for key, text in texts.items()
    }
    lengths = {len(token_ids) for token_ids in tokenizations.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"Dynamic suffix {suffix['id']} has inconsistent feature dimensions: "
            f"{ {key: len(value) for key, value in tokenizations.items()} }"
        )
    canonical_key = next(iter(tokenizations))
    canonical_ids = tokenizations[canonical_key]
    return {
        **suffix,
        "token_ids": canonical_ids,
        "tokens": extractor.tokenizer.convert_ids_to_tokens(canonical_ids),
        "token_count": len(canonical_ids),
        "render_mode": "candidate_letter_template" if "template" in suffix else "static",
        "rendered_token_ids": tokenizations,
    }


def extract_one(row: dict, suffix: dict, extractor, attempts: int = 3) -> dict:
    error = None
    rendered = render_suffix(suffix, row)
    for attempt in range(attempts):
        try:
            result = extractor.get_logprobs(
                extractor.apply_suffix(
                    [{"role": "user", "content": row["prompt"]}], rendered
                )
            )
            values = result["all_logprobs"]
            if len(values) != suffix["token_count"]:
                raise ValueError(
                    f"Expected {suffix['token_count']} values, got {len(values)}"
                )
            if any(value is None for value in values):
                raise ValueError("Server returned missing prompt logprobs")
            return {
                "sample_id": row["sample_id"],
                "question_id": int(row["question_id"]),
                "split": row["split"],
                "language": row["language"],
                "label": int(row["label"]),
                "category": row["category"],
                "candidate_index": int(row["candidate_index"]),
                "candidate_letter": row["candidate_letter"],
                "suffix_id": suffix["id"],
                "rendered_suffix": rendered,
                "logprob_selection": result.get("selection", "unknown"),
                "logprobs": [float(value) for value in values],
            }
        except Exception as caught:
            error = caught
            if attempt + 1 < attempts:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(
        f"Extraction failed for {row['sample_id']} / {suffix['id']}"
    ) from error


def extract_split(frame, suffix, extractor, cache_path: Path, workers: int):
    cache = load_cache(cache_path)
    missing = [
        row._asdict()
        for row in frame.itertuples(index=False)
        if row.sample_id not in cache
    ]
    print(
        f"{suffix['id']} / {frame.iloc[0]['split']}: "
        f"{len(cache)} cached, {len(missing)} remaining",
        flush=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock = Lock()
    completed = len(frame) - len(missing)

    def persist(record):
        nonlocal completed
        with lock:
            with cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            cache[record["sample_id"]] = record
            completed += 1
            if completed % 250 == 0 or completed == len(frame):
                print(
                    f"{suffix['id']}: [{completed}/{len(frame)}]", flush=True
                )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(extract_one, row, suffix, extractor)
            for row in missing
        ]
        try:
            for future in as_completed(futures):
                persist(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return [cache[sample_id] for sample_id in frame["sample_id"]]
