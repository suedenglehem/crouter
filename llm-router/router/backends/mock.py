"""Mock backend (PRD §37) — mandatory so the whole suite runs without GPUs.

Behavior is selected per-backend in YAML via ``behavior``:

* ``success``   — plain completion with ``response_text``
* ``stream``    — same, but always streamed as SSE chunks
* ``tool_call`` — completion whose message contains a tool call
* ``error``     — HTTP ``error_status`` (default 500) with an error body
* ``timeout``   — sleeps ``delay_seconds`` then raises a timeout BackendError
* ``echo``      — returns the received request JSON as the content, so tests
                  can assert that tools/tool_choice/messages were forwarded
                  verbatim

The mock always reports itself healthy; request-path failures come from the
configured behavior.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator

from ..config import BackendConfig
from .base import Backend, BackendError, BackendResult


class MockBackend(Backend):
    def __init__(self, tier: str, cfg: BackendConfig) -> None:
        super().__init__(tier, cfg.name)
        self._cfg = cfg

    # -- public API ---------------------------------------------------------

    async def chat_completion(self, body: dict[str, Any], *, stream: bool) -> BackendResult:
        if self._cfg.delay_seconds > 0:
            await asyncio.sleep(self._cfg.delay_seconds)

        behavior = self._cfg.behavior
        if behavior == "timeout":
            raise BackendError("timeout", f"mock backend {self.name} timed out")
        if behavior == "error":
            err = {"error": {"message": f"mock backend error ({self.name})", "type": "backend_error"}}
            return BackendResult(self._cfg.error_status, body=json.dumps(err).encode())

        # Rewrite the model field exactly like a real backend would, so "echo"
        # shows what was actually forwarded.
        payload = dict(body)
        if self._cfg.model:
            payload["model"] = self._cfg.model
        completion = self._completion(payload)
        do_stream = stream or behavior == "stream"
        if do_stream:
            async def gen() -> AsyncIterator[bytes]:
                for chunk in self._chunks(completion):
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return BackendResult(200, "text/event-stream", stream=gen())

        return BackendResult(200, body=json.dumps(completion).encode(), usage=completion["usage"])

    async def health_check(self) -> tuple[bool, float]:
        return True, 0.5

    # -- payload builders -----------------------------------------------------

    def _completion(self, body: dict[str, Any]) -> dict[str, Any]:
        cfg = self._cfg
        model = cfg.model or f"mock-{self.tier}"
        if cfg.behavior == "echo":
            content = json.dumps({"received": body})
        else:
            content = cfg.response_text

        message: dict[str, Any] = {"role": "assistant", "content": content}
        finish_reason = "stop"
        if cfg.behavior == "tool_call":
            message["content"] = None
            message["tool_calls"] = [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ]
            finish_reason = "tool_calls"

        return {
            "id": f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
        }

    def _chunks(self, completion: dict[str, Any]) -> list[dict[str, Any]]:
        """Split a completion into OpenAI-style streaming chunks."""
        base = {
            "id": completion["id"],
            "object": "chat.completion.chunk",
            "created": completion["created"],
            "model": completion["model"],
        }
        message = completion["choices"][0]["message"]

        if "tool_calls" in message:
            return [
                {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}]},
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": message["tool_calls"]},
                            "finish_reason": completion["choices"][0]["finish_reason"],
                        }
                    ],
                },
            ]

        text = message.get("content") or ""
        words = text.split(" ")
        parts: list[str] = []
        buf = ""
        for w in words:
            buf = f"{buf} {w}".strip() if buf else w
            parts.append(buf)
            buf = ""
        # Cap the number of chunks so tests stay fast.
        if len(parts) > 4:
            step = -(-len(parts) // 4)  # ceil division
            parts = [text[i : i + step] for i in range(0, len(text), step)] or [""]

        chunks: list[dict[str, Any]] = []
        for i, part in enumerate(parts):
            delta: dict[str, Any] = {"content": part}
            if i == 0:
                delta["role"] = "assistant"
            finish = None if i < len(parts) - 1 else completion["choices"][0]["finish_reason"]
            chunk = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if i == len(parts) - 1:
                chunk["usage"] = completion["usage"]
            chunks.append(chunk)
        return chunks
