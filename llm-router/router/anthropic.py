"""Anthropic Messages API <-> OpenAI chat-completions translation (PRD §51).

Claude Code speaks the Anthropic Messages API; our backends speak
OpenAI-compatible chat completions. This module converts in both directions so
the whole Claude Code agent loop can run through the router directly:

* :func:`to_openai_request` — request body, Anthropic -> OpenAI
* :func:`to_anthropic_message` — non-streaming response, OpenAI -> Anthropic
* :func:`anthropic_stream` — streaming response, OpenAI SSE lines -> Anthropic
  SSE events (message_start / content_block_* / message_delta / message_stop)

The translation is deliberately lossy in safe directions: prompt-caching
markers, ``metadata``, ``output_config`` and friends are dropped; the model's
reasoning (llama.cpp ``reasoning_content``) becomes a ``thinking`` block only
when the client asked for thinking. Unknown request fields never fail the
request — Claude Code ships new betas faster than we ship releases.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Optional

#: OpenAI finish_reason -> Anthropic stop_reason (PRD §51).
STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def _sse(event: str, data: dict[str, Any]) -> str:
    """One complete SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Request: Anthropic -> OpenAI
# ---------------------------------------------------------------------------

def wants_thinking(body: dict[str, Any]) -> bool:
    """True when the client sent a ``thinking`` parameter (any shape)."""
    return body.get("thinking") is not None


def _system_text(system: Any) -> Optional[str]:
    """Top-level ``system`` (string or list of text blocks) -> one string."""
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        parts = [b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"]
        text = "\n\n".join(p for p in parts if p)
        return text or None
    return None


def _openai_image_part(block: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Anthropic image block -> OpenAI ``image_url`` content part."""
    src = block.get("source") or {}
    stype = src.get("type")
    if stype == "base64":
        url = f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
    elif stype == "url":
        url = src.get("url", "")
    else:
        return None
    if not url:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def _tool_result_content(block: dict[str, Any]) -> Any:
    """Anthropic tool_result content (string or block list) -> OpenAI payload."""
    c = block.get("content")
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        return ""
    texts: list[str] = []
    images: list[dict[str, Any]] = []
    for b in c:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            texts.append(b.get("text", ""))
        elif b.get("type") == "image":
            part = _openai_image_part(b)
            if part is not None:
                images.append(part)
    if not images:
        return "\n".join(t for t in texts if t)
    parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(texts)}] if any(texts) else []
    parts.extend(images)
    return parts


def _tool_choice(choice: Any) -> Optional[Any]:
    """Anthropic tool_choice -> OpenAI tool_choice (best effort)."""
    if not isinstance(choice, dict):
        return choice  # already OpenAI-shaped or absent
    t = choice.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


def to_openai_request(body: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic Messages request body to OpenAI chat-completions.

    The result is what the backend will actually receive (the router rewrites
    ``model`` on top of it). Everything not understood here is dropped.
    """
    out: dict[str, Any] = {}
    if body.get("model") is not None:
        out["model"] = body["model"]

    messages: list[dict[str, Any]] = []
    sys_text = _system_text(body.get("system"))
    if sys_text:
        messages.append({"role": "system", "content": sys_text})

    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or "user"
        content = m.get("content")
        if isinstance(content, str) or content is None:
            messages.append({"role": role, "content": content or ""})
            continue

        texts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[tuple[str, Any]] = []
        images: list[dict[str, Any]] = []
        for b in content if isinstance(content, list) else []:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                texts.append(b.get("text", ""))
            elif t == "tool_use":
                tool_calls.append(
                    {
                        "id": b.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {"name": b.get("name") or "unknown", "arguments": json.dumps(b.get("input") or {})},
                    }
                )
            elif t == "tool_result":
                tool_results.append((b.get("tool_use_id") or "", _tool_result_content(b)))
            elif t == "image":
                part = _openai_image_part(b)
                if part is not None:
                    images.append(part)
            # thinking / redacted_thinking and unknown blocks are dropped

        if role in ("assistant", "system"):
            msg: dict[str, Any] = {"role": role}
            text = "".join(texts)
            msg["content"] = text if (text or not tool_calls) else None
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
        else:  # user (or anything else carrying blocks)
            parts: list[dict[str, Any]] = []
            if any(texts):
                parts.append({"type": "text", "text": "".join(texts)})
            parts.extend(images)
            if parts:
                messages.append(
                    {"role": role, "content": parts[0]["text"] if len(parts) == 1 and parts[0]["type"] == "text" else parts}
                )
            for tool_call_id, result in tool_results:
                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})

    out["messages"] = messages

    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        out["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description") or "",
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
            for t in tools
            if isinstance(t, dict) and t.get("name")
        ]

    tc = _tool_choice(body.get("tool_choice"))
    if tc is not None:
        out["tool_choice"] = tc

    for src, dst in (("max_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p")):
        if body.get(src) is not None:
            out[dst] = body[src]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]

    if body.get("stream") is not None:
        out["stream"] = bool(body["stream"])
        if out["stream"]:
            # llama.cpp reports real token counts in a final usage chunk only
            # when asked; Claude Code needs them for context tracking.
            out.setdefault("stream_options", {"include_usage": True})

    return out


# ---------------------------------------------------------------------------
# Non-streaming response: OpenAI -> Anthropic
# ---------------------------------------------------------------------------

def _tool_use_block(tc: dict[str, Any]) -> dict[str, Any]:
    fn = tc.get("function") or {}
    raw = fn.get("arguments") or "{}"
    try:
        inp = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(inp, dict):
            inp = {"_raw": inp}
    except ValueError:
        inp = {"_raw": raw}
    return {
        "type": "tool_use",
        "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
        "name": fn.get("name") or "unknown",
        "input": inp,
    }


def to_anthropic_message(data: dict[str, Any], *, model_alias: str, thinking: bool = False) -> dict[str, Any]:
    """Translate a non-streaming OpenAI chat completion to an Anthropic message."""
    choice = (data.get("choices") or [{}])[0] if isinstance(data, dict) else {}
    msg = choice.get("message") or {}
    finish = choice.get("finish_reason") or "stop"

    content: list[dict[str, Any]] = []
    reasoning = msg.get("reasoning_content")
    if thinking and isinstance(reasoning, str) and reasoning:
        content.append({"type": "thinking", "thinking": reasoning})
    text = msg.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            content.append(_tool_use_block(tc))

    usage = data.get("usage") or {} if isinstance(data, dict) else {}
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model_alias,
        "content": content,
        "stop_reason": STOP_REASON_MAP.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens") or 0,
            "output_tokens": usage.get("completion_tokens") or 0,
        },
    }


# ---------------------------------------------------------------------------
# Streaming response: OpenAI SSE -> Anthropic SSE
# ---------------------------------------------------------------------------

async def anthropic_stream(
    lines: AsyncIterator[bytes],
    *,
    model_alias: str,
    thinking: bool = False,
    estimated_input_tokens: int = 0,
    x_router: Optional[dict[str, Any]] = None,
) -> AsyncIterator[str]:
    """Translate an OpenAI SSE stream (one line per item) to Anthropic events.

    ``estimated_input_tokens`` seeds the usage in ``message_start`` — the real
    prompt token count only arrives with the final usage chunk, which is then
    reported in ``message_delta.usage`` (Claude Code re-estimates every turn
    anyway, so a close seed is fine).
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    start_message: dict[str, Any] = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model_alias,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": estimated_input_tokens, "output_tokens": 0},
    }
    if x_router is not None:
        start_message["x_router"] = x_router
    yield _sse("message_start", {"type": "message_start", "message": start_message})

    next_index = 0
    open_blocks: list[int] = []          # at most one block is open at a time (serialized)
    open_kind: dict[int, str] = {}
    tool_state: dict[int, dict[str, Any]] = {}  # OpenAI tool_call index -> state
    finish_reason: Optional[str] = None
    input_tokens = estimated_input_tokens
    output_tokens = 0

    def _close_open(frames: list[str]) -> None:
        while open_blocks:
            idx = open_blocks.pop(0)
            frames.append(_sse("content_block_stop", {"type": "content_block_stop", "index": idx}))

    def _ensure_block(kind: str, frames: list[str], start_payload: Optional[dict] = None) -> int:
        nonlocal next_index
        if open_blocks and open_kind.get(open_blocks[-1]) == kind:
            return open_blocks[-1]
        _close_open(frames)
        idx = next_index
        next_index += 1
        open_blocks.append(idx)
        open_kind[idx] = kind
        payload = start_payload or ({"type": "text", "text": ""} if kind == "text" else {"type": "thinking", "thinking": ""})
        frames.append(_sse("content_block_start", {"type": "content_block_start", "index": idx, "content_block": payload}))
        return idx

    async for raw in lines:
        text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue

            frames: list[str] = []

            u = obj.get("usage")
            if isinstance(u, dict):
                if isinstance(u.get("prompt_tokens"), int):
                    input_tokens = u["prompt_tokens"]
                if isinstance(u.get("completion_tokens"), int):
                    output_tokens = u["completion_tokens"]

            for ch in obj.get("choices") or []:
                if not isinstance(ch, dict):
                    continue
                delta = ch.get("delta") or {}
                fr = ch.get("finish_reason")
                if fr:
                    finish_reason = fr

                rc = delta.get("reasoning_content")
                if thinking and isinstance(rc, str) and rc:
                    idx = _ensure_block("thinking", frames)
                    frames.append(
                        _sse("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "thinking_delta", "thinking": rc}})
                    )

                c = delta.get("content")
                if isinstance(c, str) and c:
                    idx = _ensure_block("text", frames)
                    frames.append(
                        _sse("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": c}})
                    )

                for tc in delta.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    tidx = tc.get("index", 0)
                    fn = tc.get("function") or {}
                    st = tool_state.setdefault(tidx, {"id": None, "name": None, "args": "", "block": None})
                    if tc.get("id"):
                        st["id"] = str(tc["id"])
                    name = fn.get("name")
                    if name:
                        st["name"] = str(name)
                    frag = fn.get("arguments")
                    if isinstance(frag, str):
                        st["args"] += frag
                    # Open the block once identity is known (or args arrive first).
                    if st["block"] is None and (st["id"] or st["name"] or st["args"]):
                        st["block"] = _ensure_block(
                            "tool_use",
                            frames,
                            start_payload={
                                "type": "tool_use",
                                "id": st["id"] or f"toolu_{uuid.uuid4().hex[:24]}",
                                "name": st["name"] or "unknown",
                                "input": {},
                            },
                        )
                    if isinstance(frag, str) and frag:
                        frames.append(
                            _sse("content_block_delta", {"type": "content_block_delta", "index": st["block"], "delta": {"type": "input_json_delta", "partial_json": frag}})
                        )

            for f in frames:
                yield f

    # Close whatever is still open, then finish.
    _close_open_tail = []
    while open_blocks:
        idx = open_blocks.pop(0)
        _close_open_tail.append(_sse("content_block_stop", {"type": "content_block_stop", "index": idx}))
    for f in _close_open_tail:
        yield f

    if finish_reason is None and tool_state:  # stream ended mid-tool-call
        finish_reason = "tool_calls"
    stop_reason = STOP_REASON_MAP.get(finish_reason or "stop", "end_turn")
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Errors and the cloud-gate sentinel (Anthropic-shaped twins of models.py)
# ---------------------------------------------------------------------------

def anthropic_error(message: str, err_type: str = "api_error") -> dict[str, Any]:
    """Anthropic error envelope: ``{"type":"error","error":{...}}``."""
    return {"type": "error", "error": {"type": err_type, "message": message}}


def build_escalation_required_message(
    *,
    request_id: str,
    model_alias: Optional[str],
    x_router: dict[str, Any],
) -> dict[str, Any]:
    """Anthropic-shaped twin of :func:`router.models.build_escalation_required`.

    A valid message telling the client a stronger tier is needed; returned with
    HTTP 200 so Claude Code does not crash (PRD §21).
    """
    reason = x_router.get("escalation", {}).get("reason", "unknown")
    return {
        "id": f"msg-{request_id}",
        "type": "message",
        "role": "assistant",
        "model": model_alias or "auto",
        "content": [
            {
                "type": "text",
                "text": (
                    f"[llm-router] escalation required: the frontier tier is currently blocked "
                    f"({reason}). Retry with header 'X-LLM-Escalate: frontier' to force it, "
                    f"or raise the cloud limits in config."
                ),
            }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "x_router": x_router,
    }
