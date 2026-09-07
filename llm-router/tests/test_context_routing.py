"""Context-length routing: the "context floor" rule.

A request whose estimated context need (prompt + completion headroom) exceeds
the selected tier's ``max_context`` is bumped up the escalation chain until it
fits — regardless of how the tier was chosen. Per-request, no task-state change.
"""

import json
from fastapi.testclient import TestClient

from conftest import content_of, make_config
from router.api import create_app
from router.context import required_context
from router.config import ContextRoutingConfig


def _ctx_client(**overrides) -> TestClient:
    """Mock-backed app with small, predictable context limits.

    fast=200 tokens, deep=10_000, frontier=None (unknown -> assumed to fit).
    completion_reserve=50 keeps the arithmetic easy to reason about.
    """
    overrides.setdefault(
        "backends", {"fast": {"max_context": 200}, "deep": {"max_context": 10_000}}
    )
    overrides.setdefault(
        "routing", {"context": {"enabled": True, "chars_per_token": 3.0, "completion_reserve": 50}}
    )
    app = create_app(make_config(**overrides))
    return TestClient(app)


def _post(client: TestClient, content: str, model="auto", **headers):
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": content}]},
        headers=headers,
    )


# -- basic floor behavior ------------------------------------------------------

def test_small_request_stays_on_fast():
    with _ctx_client() as c:
        r = _post(c, "hello")
        assert r.status_code == 200
        meta = r.json()["x_router"]
        assert meta["route"] == "fast"
        assert meta["reason"] == "auto_policy"
        assert "escalation" not in meta


def test_overflow_bumps_fast_to_deep():
    with _ctx_client() as c:
        # ~1000 chars of content -> est. prompt ~357 tokens + 50 reserve > 200 (fast).
        r = _post(c, "a" * 1000)
        assert r.status_code == 200
        assert content_of(r) == "DEEP-SAYS-HI"
        meta = r.json()["x_router"]
        assert meta["route"] == "deep"
        assert meta["reason"] == "context_overflow"
        assert meta["escalation"]["from"] == "fast"
        assert meta["escalation"]["to"] == "deep"


def test_overflow_reaches_frontier_when_deep_cannot_fit():
    with _ctx_client() as c:
        # ~40k chars -> est. > 10_000 (deep); frontier has no max_context, so it fits.
        r = _post(c, "a" * 40_000)
        assert r.status_code == 200
        assert content_of(r) == "FRONTIER-SAYS-HI"
        meta = r.json()["x_router"]
        assert meta["route"] == "frontier"
        assert meta["reason"] == "context_overflow"
        assert meta["escalation"]["from"] == "fast"
        assert meta["escalation"]["to"] == "frontier"


# -- the floor overrides explicit choices --------------------------------------

def test_floor_overrides_explicit_model():
    with _ctx_client() as c:
        r = _post(c, "a" * 1000, model="local-fast")
        assert content_of(r) == "DEEP-SAYS-HI"
        meta = r.json()["x_router"]
        assert meta["route"] == "deep"
        assert meta["reason"] == "context_overflow"


def test_floor_overrides_escalate_header():
    with _ctx_client() as c:
        # Explicitly asked for deep, but the context does not fit there either.
        r = _post(c, "a" * 40_000, **{"X-LLM-Escalate": "deep"})
        assert content_of(r) == "FRONTIER-SAYS-HI"
        meta = r.json()["x_router"]
        assert meta["route"] == "frontier"
        assert meta["reason"] == "context_overflow"


# -- completion headroom -------------------------------------------------------

def test_client_max_tokens_counts_toward_requirement():
    with _ctx_client() as c:
        # Small prompt, but the client asks for a long generation.
        r = _post(c, "hello")  # baseline fits fast
        assert r.json()["x_router"]["route"] == "fast"

        body = {"model": "auto", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 9000}
        r = c.post("/v1/chat/completions", json=body)
        # ~20 + 9000 > 200 (fast), still <= 10_000 (deep).
        assert content_of(r) == "DEEP-SAYS-HI"
        assert r.json()["x_router"]["reason"] == "context_overflow"

        body["max_tokens"] = 999_999
        r = c.post("/v1/chat/completions", json=body)
        assert content_of(r) == "FRONTIER-SAYS-HI"


# -- config switches -----------------------------------------------------------

def test_disabled_context_routing_never_bumps():
    with _ctx_client(routing={"context": {"enabled": False}}) as c:
        r = _post(c, "a" * 40_000)
        assert content_of(r) == "FAST-SAYS-HI"
        assert r.json()["x_router"]["reason"] == "auto_policy"


def test_allow_frontier_false_stops_at_deep():
    with _ctx_client(
        routing={
            "context": {"enabled": True, "chars_per_token": 3.0, "completion_reserve": 50},
            "escalation": {"allow_frontier": False},
        }
    ) as c:
        r = _post(c, "a" * 40_000)
        # Bumped off fast (too small), but the chain stops at deep.
        assert content_of(r) == "DEEP-SAYS-HI"
        meta = r.json()["x_router"]
        assert meta["route"] == "deep"
        assert meta["reason"] == "context_overflow"


def test_unknown_max_context_is_assumed_to_fit():
    # Only fast has a limit; deep (None) must never be bumped away from.
    with _ctx_client(backends={"fast": {"max_context": 200}}) as c:
        r = _post(c, "a" * 40_000)
        assert content_of(r) == "DEEP-SAYS-HI"
        meta = r.json()["x_router"]
        assert meta["route"] == "deep"
        assert meta["escalation"]["to"] == "deep"


# -- streaming + metrics -------------------------------------------------------

def test_streaming_overflow_meta_reports_deep():
    with _ctx_client() as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "stream": True,
                "messages": [{"role": "user", "content": "a" * 1000}],
            },
        )
        assert r.status_code == 200
        meta_chunks = [
            line[len("data:"):].strip()
            for line in r.text.splitlines()
            if line.startswith("data:") and "x_router" in line
        ]
        assert meta_chunks, "no router-meta chunk in stream"
        meta = json.loads(meta_chunks[-1])["x_router"]
        assert meta["route"] == "deep"
        assert meta["reason"] == "context_overflow"


def test_metric_counts_context_escalation():
    with _ctx_client() as c:
        _post(c, "a" * 1000)
        text = c.get("/metrics").text
        assert 'router_escalations_total{from="fast",reason="context_overflow",to="deep"}' in text


# -- estimator unit checks -----------------------------------------------------

def test_required_context_disabled_returns_none():
    cfg = ContextRoutingConfig(enabled=False)
    assert required_context({"messages": []}, cfg) is None


def test_required_context_honors_generation_caps():
    cfg = ContextRoutingConfig(chars_per_token=4.0, completion_reserve=100)
    base = {"messages": [{"role": "user", "content": "x" * 40}]}
    no_cap = required_context(base, cfg)  # prompt estimate + reserve (100)
    small_cap = required_context({**base, "max_tokens": 5}, cfg)
    assert small_cap < no_cap  # an explicit cap replaces the larger reserve

    # max_completion_tokens is honored too.
    alt_cap = required_context({**base, "max_completion_tokens": 7}, cfg)
    assert alt_cap < no_cap
