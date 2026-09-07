"""Context-length estimation for the routing "context floor".

The router does not tokenize (no tokenizer dependency, PRD §36: keep it
minimal). Instead it estimates the prompt size from the serialized request
body — messages *and* tool schemas, i.e. everything the backend will see —
with a conservative chars-per-token divisor. The server-side chat template
adds per-message tokens the client never sends, so overestimating is the safe
direction: a false bump costs one slower model; an undercount costs a failed
request on a too-small context.

The estimate is deliberately simple and monotonic in request size; it is a
routing signal, not a billing number.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .config import ContextRoutingConfig


def estimate_prompt_tokens(body: dict[str, Any], chars_per_token: float) -> int:
    """Rough token count of the prompt as the backend will receive it."""
    try:
        size = len(json.dumps(body))
    except (TypeError, ValueError):  # pragma: no cover - body came from JSON parse
        size = 0
    return max(1, int(size / chars_per_token))


def required_context(body: dict[str, Any], cfg: ContextRoutingConfig) -> Optional[int]:
    """Estimated context the request needs (prompt + completion headroom).

    Returns ``None`` when context routing is disabled — callers then skip the
    floor entirely.
    """
    if not cfg.enabled:
        return None
    prompt = estimate_prompt_tokens(body, cfg.chars_per_token)
    # Honor an explicit generation cap; otherwise reserve headroom so the model
    # can actually answer before the context fills up.
    completion = body.get("max_tokens") or body.get("max_completion_tokens") or cfg.completion_reserve
    return prompt + int(completion)
