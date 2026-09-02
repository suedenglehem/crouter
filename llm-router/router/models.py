"""API-level models and response builders."""

from __future__ import annotations

import time
from typing import Any, Optional

from pydantic import BaseModel, Field


class EventIn(BaseModel):
    """Payload for POST /events (PRD §17). Unknown events are accepted upstream."""

    session_id: Optional[str] = None
    task_id: Optional[str] = None
    event: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def error_body(message: str, err_type: str = "invalid_request_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": err_type}}


def x_router_meta(
    *,
    request_id: str,
    session_id: str,
    task_id: str,
    tier: str,
    backend_name: str,
    reason: str,
    escalation: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """The ``x_router`` block attached to every response (PRD §27, §32)."""
    meta: dict[str, Any] = {
        "request_id": request_id,
        "session_id": session_id,
        "task_id": task_id,
        "route": tier,
        "backend": backend_name,
        "reason": reason,
    }
    if escalation is not None:
        meta["escalation"] = escalation
    return meta


def build_escalation_required(
    *,
    request_id: str,
    model_alias: Optional[str],
    x_router: dict[str, Any],
) -> dict[str, Any]:
    """A valid chat-completion body telling the client a stronger tier is needed.

    Returned with HTTP 200 so OpenAI-compatible clients do not crash; the
    ``x_router.escalation.required`` flag carries the signal (PRD §21).
    """
    reason = x_router.get("escalation", {}).get("reason", "unknown")
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_alias or "auto",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        f"[llm-router] escalation required: the frontier tier is currently blocked "
                        f"({reason}). Retry with header 'X-LLM-Escalate: frontier' to force it, "
                        f"or raise the cloud limits in config."
                    ),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "x_router": x_router,
    }
