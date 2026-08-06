THINKING_BYPASS_PREFILL = {
    "deepseek_r1": "<think>\n</think>",
    "glm_5_2": "</think>",
    "kimi_k3": (
        "<|close|>think<|sep|>"
        "<|open|>response<|sep|>"
    ),
    "qwen3": "</think>\n\n",
    "none": "",
}

THINKING_BYPASS_PREFILL_BY_MODEL = (
    ("deepseek-r1", "deepseek_r1"),
    ("glm-5.2", "glm_5_2"),
    ("kimi-k3", "kimi_k3"),
    ("qwen3", "qwen3"),
)


def infer_thinking_bypass_prefill(model_name: str | None) -> str:
    """Infer the thinking-bypass prefill name from a model id or path."""
    normalized_model_name = str(model_name or "").lower().replace("_", "-")
    for model_pattern, prefill_name in THINKING_BYPASS_PREFILL_BY_MODEL:
        if model_pattern in normalized_model_name:
            return prefill_name
    return "none"
