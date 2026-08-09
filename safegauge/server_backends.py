"""Server-specific prompt log-probability adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SERVER_BACKENDS = ("vllm", "sglang")


@dataclass
class ServerModelInfo:
    model: str | None = None
    model_root: str | None = None
    tokenizer_path: str | None = None
    version: str | None = None


@dataclass
class SuffixLogprobs:
    text: str
    positions: list[Any]


class ServerRequestError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _trailing_positions(positions, count: int, backend: str) -> list[Any]:
    if positions is None:
        raise ValueError(f"{backend} did not return prompt logprobs")
    positions = list(positions)
    if len(positions) < count:
        raise ValueError(
            f"{backend} returned fewer prompt-logprob positions than the "
            f"semantic suffix: {len(positions)} < {count}"
        )
    return positions[-count:]


class VLLMServerBackend:
    name = "vllm"

    def __init__(self, base_url: str, api_key: str, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model: str | None = None

    def discover(self) -> ServerModelInfo:
        models = self.client.models.list()
        if not models.data:
            raise ValueError("vLLM server returned an empty model list")

        server_model = models.data[0]
        self.model = server_model.id
        model_root = getattr(server_model, "root", None)

        permissions = getattr(server_model, "permission", [])
        if permissions:
            permission = permissions[0]
            if not isinstance(permission, dict):
                permission = vars(permission)
            if permission.get("allow_logprobs") is False:
                raise PermissionError(
                    f"Server model {self.model!r} does not allow logprobs"
                )

        return ServerModelInfo(
            model=self.model,
            model_root=model_root,
            tokenizer_path=model_root,
        )

    def score(
        self,
        input_ids: list[int],
        suffix_start: int,
        top_k: int,
        temperature: float,
        top_p: float,
    ) -> SuffixLogprobs:
        if not 0 <= suffix_start < len(input_ids):
            raise ValueError("suffix_start must point inside the prompt")
        suffix_length = len(input_ids) - suffix_start
        response = self.client.completions.create(
            model=self.model,
            prompt=input_ids,
            max_tokens=1,
            temperature=temperature,
            top_p=top_p,
            extra_body={
                "prompt_logprobs": top_k,
                "add_special_tokens": False,
            },
        )
        choice = response.choices[0]
        return SuffixLogprobs(
            text=choice.text,
            positions=_trailing_positions(
                getattr(choice, "prompt_logprobs", None),
                suffix_length,
                self.name,
            ),
        )


def _sglang_root_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root.rstrip("/")


def _http_json(
    url: str,
    *,
    method: str,
    payload: dict | None,
    headers: dict[str, str],
    timeout: float,
):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise ServerRequestError(
            f"SGLang request failed with HTTP {exc.code}: {detail}",
            status_code=exc.code,
        ) from exc
    except URLError as exc:
        raise ServerRequestError(f"SGLang request failed: {exc.reason}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ServerRequestError("SGLang returned invalid JSON") from exc


def _parse_sglang_entry(entry, field: str):
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        raise ValueError(f"Invalid SGLang {field} entry: {entry!r}")
    logprob, token_id = entry[:2]
    decoded_token = entry[2] if len(entry) > 2 else None
    try:
        token_id = int(token_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid token ID in SGLang {field}: {token_id!r}") from exc
    return logprob, token_id, decoded_token


def _normalize_sglang_positions(
    meta_info: dict,
    expected_token_ids: list[int],
    lookback_token_id: int,
    top_k: int,
) -> list[dict[int, dict[str, Any]]]:
    observed_rows = meta_info.get("input_token_logprobs")
    if observed_rows is None:
        raise ValueError("SGLang did not return meta_info.input_token_logprobs")
    observed_rows = list(observed_rows)
    expected_length = len(expected_token_ids) + 1
    if len(observed_rows) != expected_length:
        raise ValueError(
            "SGLang input-token logprob length does not match the requested "
            f"lookback plus suffix: {len(observed_rows)} != {expected_length}"
        )
    lookback_logprob, returned_lookback_id, _ = _parse_sglang_entry(
        observed_rows[0],
        "input_token_logprobs lookback",
    )
    if returned_lookback_id != lookback_token_id or lookback_logprob is not None:
        raise ValueError(
            "SGLang input-token lookback alignment mismatch: expected "
            f"[None, {lookback_token_id}, ...], got {observed_rows[0]!r}"
        )
    observed_rows = observed_rows[1:]

    top_rows = meta_info.get("input_top_logprobs")
    if top_k > 0:
        if top_rows is None:
            raise ValueError("SGLang did not return meta_info.input_top_logprobs")
        top_rows = list(top_rows)
        if len(top_rows) != expected_length:
            raise ValueError(
                "SGLang input-top-logprob length does not match the requested "
                f"lookback plus suffix: {len(top_rows)} != {expected_length}"
            )
        if top_rows[0]:
            raise ValueError("SGLang input-top-logprob lookback position must be empty")
        top_rows = top_rows[1:]
    else:
        top_rows = [None] * len(expected_token_ids)

    positions = []
    for offset, (observed_row, top_row, expected_id) in enumerate(
        zip(observed_rows, top_rows, expected_token_ids)
    ):
        observed_logprob, observed_id, observed_text = _parse_sglang_entry(
            observed_row,
            "input_token_logprobs",
        )
        if observed_id != expected_id:
            raise ValueError(
                "SGLang input-token alignment mismatch at suffix offset "
                f"{offset}: returned {observed_id}, expected {expected_id}"
            )
        if observed_logprob is None:
            raise ValueError(
                f"SGLang returned no logprob for suffix token at offset {offset}"
            )

        mapping = {}
        for rank, entry in enumerate(top_row or [], start=1):
            logprob, token_id, decoded_token = _parse_sglang_entry(
                entry,
                "input_top_logprobs",
            )
            mapping[token_id] = {
                "logprob": logprob,
                "rank": rank,
                "decoded_token": decoded_token,
            }

        observed_rank = mapping.get(observed_id, {}).get("rank")
        decoded_token = observed_text
        if decoded_token is None and observed_id in mapping:
            decoded_token = mapping[observed_id]["decoded_token"]
        mapping[observed_id] = {
            "logprob": observed_logprob,
            "rank": observed_rank,
            "decoded_token": decoded_token,
        }
        positions.append(mapping)
    return positions


class SGLangServerBackend:
    name = "sglang"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 600.0,
    ):
        self.base_url = _sglang_root_url(base_url)
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key and api_key != "none":
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _request(self, path: str, method: str = "GET", payload=None):
        return _http_json(
            f"{self.base_url}{path}",
            method=method,
            payload=payload,
            headers=self.headers,
            timeout=self.timeout,
        )

    def _get_first(self, paths: tuple[str, ...], required: bool):
        last_error = None
        for path in paths:
            try:
                return self._request(path)
            except ServerRequestError as exc:
                last_error = exc
                if exc.status_code not in (404, 405):
                    raise
        if required:
            raise last_error or ServerRequestError("SGLang endpoint not found")
        return None

    def discover(self) -> ServerModelInfo:
        model_info = self._get_first(
            ("/model_info", "/get_model_info"),
            required=True,
        )
        if not isinstance(model_info, dict):
            raise ValueError("SGLang model-info response must be an object")

        server_info = self._get_first(
            ("/server_info", "/get_server_info"),
            required=False,
        )
        if not isinstance(server_info, dict):
            server_info = {}

        model_root = model_info.get("model_path") or model_info.get("model")
        tokenizer_path = model_info.get("tokenizer_path") or model_root
        model = model_info.get("served_model_name") or model_root
        version = server_info.get("version")
        return ServerModelInfo(
            model=None if model is None else str(model),
            model_root=None if model_root is None else str(model_root),
            tokenizer_path=(None if tokenizer_path is None else str(tokenizer_path)),
            version=None if version is None else str(version),
        )

    def score(
        self,
        input_ids: list[int],
        suffix_start: int,
        top_k: int,
        temperature: float,
        top_p: float,
    ) -> SuffixLogprobs:
        if not 0 < suffix_start < len(input_ids):
            raise ValueError(
                "SGLang requires a non-empty suffix with one preceding prompt token"
            )
        expected_token_ids = input_ids[suffix_start:]
        lookback_start = suffix_start - 1
        response = self._request(
            "/generate",
            method="POST",
            payload={
                "input_ids": input_ids,
                "sampling_params": {
                    "max_new_tokens": 1,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                "return_logprob": True,
                "logprob_start_len": lookback_start,
                "top_logprobs_num": top_k,
                "return_text_in_logprobs": False,
            },
        )
        if isinstance(response, list):
            if len(response) != 1:
                raise ValueError("SGLang returned an unexpected batch response")
            response = response[0]
        if not isinstance(response, dict):
            raise ValueError("SGLang generate response must be an object")
        meta_info = response.get("meta_info")
        if not isinstance(meta_info, dict):
            raise ValueError("SGLang response does not contain meta_info")

        return SuffixLogprobs(
            text=str(response.get("text") or ""),
            positions=_normalize_sglang_positions(
                meta_info,
                expected_token_ids,
                input_ids[lookback_start],
                top_k,
            ),
        )


def create_server_backend(name: str, base_url: str, api_key: str):
    normalized_name = name.lower()
    if normalized_name == "vllm":
        return VLLMServerBackend(base_url, api_key)
    if normalized_name == "sglang":
        return SGLangServerBackend(base_url, api_key)
    supported = ", ".join(SERVER_BACKENDS)
    raise ValueError(f"Unknown server_backend {name!r}. Supported values: {supported}")
