"""Anthropic Messages API endpoint (PRD §51).

Covers the pure translation functions directly, plus the /v1/messages
endpoint end-to-end against mock backends (non-streaming, streaming, tool
calls, usage, error paths, session-header fallback).
"""

from __future__ import annotations

import asyncio
import json

from fastapi.testclient import TestClient

from conftest import make_config
from router.api import create_app
from router.anthropic import (
    anthropic_stream,
    to_anthropic_message,
    to_openai_request,
)


def _app_client(**overrides):
    """A TestClient on a fresh app with mock-backend config overrides."""
    return TestClient(create_app(make_config(**overrides)))


# ---------------------------------------------------------------------------
# Request translation: Anthropic -> OpenAI
# ---------------------------------------------------------------------------

def test_system_blocks_become_single_system_message():
    body = {
        "model": "auto",
        "max_tokens": 100,
        "system": [
            {"type": "text", "text": "You are terse."},
            {"type": "text", "text": "Use tools.", "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = to_openai_request(body)
    assert out["messages"][0] == {"role": "system", "content": "You are terse.\n\nUse tools."}
    assert out["messages"][1] == {"role": "user", "content": "hi"}


def test_system_string_passthrough():
    out = to_openai_request({"model": "auto", "system": "plain system", "messages": [{"role": "user", "content": "x"}]})
    assert out["messages"][0]["content"] == "plain system"


def test_assistant_tool_use_becomes_openai_tool_calls():
    body = {
        "model": "auto",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
                ],
            },
        ],
    }
    out = to_openai_request(body)
    asst = out["messages"][1]
    assert asst["role"] == "assistant"
    assert asst["content"] == "Checking."
    assert asst["tool_calls"] == [
        {"id": "toolu_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
    ]


def test_tool_results_become_role_tool_messages():
    body = {
        "model": "auto",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "sunny"}]},
                ],
            },
        ],
    }
    out = to_openai_request(body)
    assert out["messages"] == [{"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"}]


def test_tool_result_string_content_and_mid_conversation_system():
    body = {
        "model": "auto",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t9", "content": "ok"}]},
            {"role": "system", "content": [{"type": "text", "text": "mid-conversation note"}]},
        ],
    }
    out = to_openai_request(body)
    assert out["messages"][0] == {"role": "tool", "tool_call_id": "t9", "content": "ok"}
    assert out["messages"][1] == {"role": "system", "content": "mid-conversation note"}


def test_tools_and_unknown_fields():
    body = {
        "model": "auto",
        "max_tokens": 4096,
        "temperature": 0.2,
        "stop_sequences": ["\n\nHuman:"],
        "stream": True,
        "system": [{"type": "text", "text": "s"}],
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object", "properties": {}}},
        ],
        # fields Claude Code sends that we deliberately ignore:
        "metadata": {"user_id": "..."},
        "output_config": {"effort": "high"},
        "thinking": {"type": "adaptive", "display": "omitted"},
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
    }
    out = to_openai_request(body)
    assert out["tools"] == [
        {
            "type": "function",
            "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {}}},
        }
    ]
    assert out["max_tokens"] == 4096
    assert out["temperature"] == 0.2
    assert out["stop"] == ["\n\nHuman:"]
    # `stream` itself must survive the translation — without it llama-server
    # answers a single non-streamed JSON body and the SSE translator sees nothing.
    assert out["stream"] is True
    assert out["stream_options"] == {"include_usage": True}
    for dropped in ("metadata", "output_config", "thinking", "context_management", "system"):
        assert dropped not in out


def test_no_stream_means_no_stream_options():
    out = to_openai_request({"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert "stream" not in out
    assert "stream_options" not in out


# ---------------------------------------------------------------------------
# Non-streaming response translation: OpenAI -> Anthropic
# ---------------------------------------------------------------------------

def _completion(content="hello", finish="stop", tool_calls=None, reasoning=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-x",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    }


def test_to_anthropic_message_text():
    msg = to_anthropic_message(_completion(), model_alias="auto")
    assert msg["type"] == "message"
    assert msg["role"] == "assistant"
    assert msg["model"] == "auto"
    assert msg["content"] == [{"type": "text", "text": "hello"}]
    assert msg["stop_reason"] == "end_turn"
    assert msg["usage"] == {"input_tokens": 12, "output_tokens": 7}


def test_to_anthropic_message_stop_reasons():
    assert to_anthropic_message(_completion(finish="length"), model_alias="auto")["stop_reason"] == "max_tokens"
    tc = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]
    msg = to_anthropic_message(_completion(content=None, finish="tool_calls", tool_calls=tc), model_alias="auto")
    assert msg["stop_reason"] == "tool_use"
    assert msg["content"][0]["type"] == "tool_use"
    assert msg["content"][0]["input"] == {"a": 1}


def test_to_anthropic_message_thinking_only_when_requested():
    data = _completion(reasoning="hmm")
    without = to_anthropic_message(data, model_alias="auto", thinking=False)
    assert [b["type"] for b in without["content"]] == ["text"]
    with_ = to_anthropic_message(data, model_alias="auto", thinking=True)
    assert [b["type"] for b in with_["content"]] == ["thinking", "text"]
    assert with_["content"][0]["thinking"] == "hmm"


def test_to_anthropic_message_malformed_tool_arguments():
    tc = [{"id": "c1", "function": {"name": "f", "arguments": "{not json"}}]
    msg = to_anthropic_message(_completion(content=None, finish="tool_calls", tool_calls=tc), model_alias="auto")
    assert msg["content"][0]["input"] == {"_raw": "{not json"}


# ---------------------------------------------------------------------------
# Streaming state machine (direct, with synthetic OpenAI SSE)
# ---------------------------------------------------------------------------

def _sse_lines(*payloads: dict | str):
    async def gen():
        for p in payloads:
            yield f"data: {p}\n".encode() if isinstance(p, str) else f"data: {json.dumps(p)}\n".encode()

    return gen()


async def _collect(gen) -> list[dict]:
    events = []
    async for frame in gen:
        assert frame.startswith("event: ") and "\ndata: " in frame
        data = frame.split("\ndata: ", 1)[1].strip()
        events.append(json.loads(data))
    return events


def test_stream_text_and_usage():
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}]},
        {"choices": [{"index": 0, "delta": {"content": "hel"}}]},
        {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 312, "completion_tokens": 5}},
    ]

    async def run():
        return await _collect(anthropic_stream(_sse_lines(*chunks), model_alias="auto", estimated_input_tokens=99))

    events = asyncio.run(run())

    types = [e["type"] for e in events]
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    # message_start carries the seed estimate; message_delta carries real usage.
    assert events[0]["message"]["usage"] == {"input_tokens": 99, "output_tokens": 0}
    text = "".join(e["delta"]["text"] for e in events if e["type"] == "content_block_delta" and e["delta"].get("type") == "text_delta")
    assert text == "hello"
    delta = events[-2]
    assert delta["delta"]["stop_reason"] == "end_turn"
    assert delta["usage"] == {"input_tokens": 312, "output_tokens": 5}


def test_stream_thinking_then_text_blocks():
    chunks = [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}]},
        {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
    ]

    async def run():
        return await _collect(anthropic_stream(_sse_lines(*chunks), model_alias="auto", thinking=True))

    events = asyncio.run(run())
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert [s["content_block"]["type"] for s in starts] == ["thinking", "text"]
    assert starts[0]["index"] == 0 and starts[1]["index"] == 1
    thinking = "".join(
        e["delta"]["thinking"] for e in events if e["type"] == "content_block_delta" and e["delta"].get("type") == "thinking_delta"
    )
    assert thinking == "hmm"


def test_stream_thinking_dropped_when_not_requested():
    chunks = [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}]},
        {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]},
    ]

    async def run():
        return await _collect(anthropic_stream(_sse_lines(*chunks), model_alias="auto", thinking=False))

    events = asyncio.run(run())
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert [s["content_block"]["type"] for s in starts] == ["text"]


def test_stream_incremental_tool_call():
    def tc(frag: str) -> dict:
        return {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": frag}}]}}]}

    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}}]},
                }
            ]
        },
        tc('{"ci'),
        tc('ty": "Paris"}'),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]

    async def run():
        return await _collect(anthropic_stream(_sse_lines(*chunks), model_alias="auto"))

    events = asyncio.run(run())

    starts = [e for e in events if e["type"] == "content_block_start"]
    assert len(starts) == 1
    block = starts[0]["content_block"]
    assert block == {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {}}

    args = "".join(
        e["delta"]["partial_json"] for e in events if e["type"] == "content_block_delta" and e["delta"].get("type") == "input_json_delta"
    )
    assert json.loads(args) == {"city": "Paris"}

    # Every block opened is closed, in order.
    stops = [e for e in events if e["type"] == "content_block_stop"]
    assert [s["index"] for s in stops] == [0]

    delta = events[-2]
    assert delta["delta"]["stop_reason"] == "tool_use"


def test_stream_no_finish_reason_with_tools_defaults_to_tool_use():
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f", "arguments": "{}"}}]},
                }
            ]
        },
    ]

    async def run():
        return await _collect(anthropic_stream(_sse_lines(*chunks), model_alias="auto"))

    events = asyncio.run(run())
    assert events[-2]["delta"]["stop_reason"] == "tool_use"


# ---------------------------------------------------------------------------
# Endpoint: /v1/messages (mock backends)
# ---------------------------------------------------------------------------

def _anthropic_events(body: str) -> list[dict]:
    """Parse Anthropic SSE frames (event: + data:) out of a response body."""
    events = []
    for block in body.split("\n\n"):
        data_lines = [l[len("data: "):] for l in block.splitlines() if l.startswith("data: ")]
        if not data_lines:
            continue
        events.append(json.loads(data_lines[0]))
    return events


def test_messages_non_streaming_shape(client):
    r = client.post(
        "/v1/messages",
        json={
            "model": "auto",
            "max_tokens": 64,
            "system": [{"type": "text", "text": "be brief"}],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "auto"
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "FAST-SAYS-HI"
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 12, "output_tokens": 7}
    assert body["x_router"]["route"] == "fast"


def test_messages_tool_call_response():
    with _app_client(**{"backends": {"fast": {"behavior": "tool_call"}}}) as c:
        r = c.post("/v1/messages", json={"model": "auto", "max_tokens": 64, "messages": [{"role": "user", "content": "weather?"}]})
    assert r.status_code == 200
    body = r.json()
    block = body["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "get_weather"
    assert block["input"] == {"city": "Paris"}
    assert body["stop_reason"] == "tool_use"


def test_messages_streaming_events(client):
    r = client.post(
        "/v1/messages",
        json={"model": "auto", "max_tokens": 64, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-LLM-Session-ID": "sess-msg"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = _anthropic_events(r.text)
    types = [e["type"] for e in events]
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"
    assert "content_block_start" in types and "content_block_delta" in types and "content_block_stop" in types

    # x_router rides along in message_start.
    start = events[0]["message"]
    assert start["x_router"]["route"] == "fast"
    assert start["x_router"]["session_id"] == "sess-msg"
    assert start["usage"]["input_tokens"] > 0  # seeded estimate

    text = "".join(
        e["delta"]["text"] for e in events if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta"
    )
    assert text == "FAST-SAYS-HI"

    delta = [e for e in events if e["type"] == "message_delta"][0]
    assert delta["delta"]["stop_reason"] == "end_turn"
    # Real usage from the mock's final chunk replaces the seed estimate.
    assert delta["usage"] == {"input_tokens": 12, "output_tokens": 7}


def test_messages_streaming_tool_call():
    with _app_client(**{"backends": {"fast": {"behavior": "tool_call"}}}) as c:
        r = c.post(
            "/v1/messages",
            json={"model": "auto", "max_tokens": 64, "stream": True, "messages": [{"role": "user", "content": "weather?"}]},
        )
    events = _anthropic_events(r.text)
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert len(starts) == 1
    block = starts[0]["content_block"]
    assert block["type"] == "tool_use" and block["name"] == "get_weather"

    args = "".join(
        e["delta"]["partial_json"] for e in events if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "input_json_delta"
    )
    assert json.loads(args) == {"city": "Paris"}

    delta = [e for e in events if e["type"] == "message_delta"][0]
    assert delta["delta"]["stop_reason"] == "tool_use"


def test_messages_claude_code_session_header_fallback(client):
    r = client.post(
        "/v1/messages",
        json={"model": "auto", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Claude-Code-Session-ID": "cc-sess-42"},
    )
    assert r.json()["x_router"]["session_id"] == "cc-sess-42"


def test_messages_unknown_model_404(client):
    r = client.post("/v1/messages", json={"model": "nope", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "model_not_found"


def test_messages_backend_error_forwarded():
    with _app_client(**{"backends": {"fast": {"behavior": "error", "error_status": 400}}}) as c:
        r = c.post("/v1/messages", json={"model": "auto", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    body = r.json()
    assert body["type"] == "error"
    assert "mock backend error" in body["error"]["message"]
    assert "x_router" in body


def test_messages_all_backends_down_502():
    with _app_client(**{"backends": {"fast": {"behavior": "timeout", "delay_seconds": 0.01}}}) as c:
        r = c.post("/v1/messages", json={"model": "auto", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "backend_error"


def test_messages_escalation_required_sentinel():
    with _app_client(**{"cloud": {"enabled": False}}) as c:
        r = c.post("/v1/messages", json={"model": "frontier", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message"
    assert body["content"][0]["text"].startswith("[llm-router] escalation required")
    assert body["x_router"]["escalation"]["required"] is True


def test_messages_malformed_body_400(client):
    r = client.post("/v1/messages", json={"model": "auto"})  # no messages
    assert r.status_code == 400
    assert r.json()["type"] == "error"
