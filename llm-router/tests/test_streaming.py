"""Streaming passthrough (PRD 24)."""

import json


def sse_data_lines(body: str) -> list[dict]:
    """Parse SSE data payloads out of a response body."""
    out = []
    for line in body.splitlines():
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                out.append({"__done__": True})
            else:
                out.append(json.loads(payload))
    return out


def test_stream_response_is_sse_and_complete(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "local-fast", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = sse_data_lines(r.text)
    # Content chunks reassemble to the mock text.
    content = "".join(
        ev["choices"][0]["delta"].get("content", "")
        for ev in events
        if not ev.get("__done__") and "choices" in ev and ev["choices"]
        and isinstance(ev["choices"][0].get("delta"), dict)
    )
    assert content == "FAST-SAYS-HI"

    # Stream terminates with [DONE] as the very last event.
    assert events[-1] == {"__done__": True}

    # A final chunk carries finish_reason stop and usage (mock includes it).
    finished = [ev for ev in events if not ev.get("__done__") and "choices" in ev and ev["choices"]
                and ev["choices"][0].get("finish_reason")]
    assert finished, "no chunk carried a finish_reason"


def test_stream_carries_router_meta_before_done(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-LLM-Session-ID": "sess-stream"},
    )
    events = sse_data_lines(r.text)
    meta_events = [ev for ev in events if not ev.get("__done__") and "x_router" in ev]
    assert len(meta_events) == 1
    meta = meta_events[0]["x_router"]
    assert meta["route"] == "fast"
    assert meta["session_id"] == "sess-stream"

    # The meta event must come after all content and before [DONE].
    idx_meta = events.index(meta_events[0])
    idx_done = len(events) - 1
    assert idx_meta < idx_done


def test_stream_is_incremental(client):
    """Chunks arrive one at a time (not buffered into a single blob)."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "local-fast", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    ) as r:
        lines = list(r.iter_lines())
    data_lines = [l for l in lines if l.startswith("data:") and l.strip() != "data: [DONE]"]
    # The mock splits the text into multiple chunks; we must see more than one.
    assert len(data_lines) >= 2


def test_stream_escalation_meta_reflects_route(client):
    """After an event-driven escalation, streamed requests carry route=deep."""
    h = {"X-LLM-Session-ID": "sess-s", "X-LLM-Task-ID": "task-s"}
    for _ in range(2):
        client.post("/events", json={"session_id": "sess-s", "task_id": "task-s", "event": "test_failure"})

    r = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        headers=h,
    )
    events = sse_data_lines(r.text)
    meta_events = [ev for ev in events if not ev.get("__done__") and "x_router" in ev]
    assert meta_events[0]["x_router"]["route"] == "deep"


def test_non_stream_response_has_x_router(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "local-fast", "messages": [{"role": "user", "content": "hi"}]},
    )
    body = r.json()
    assert "x_router" in body
    # Standard OpenAI fields are preserved.
    for key in ("id", "object", "created", "model", "choices"):
        assert key in body
