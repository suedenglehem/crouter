"""API-level behavior: malformed requests, health, metrics, secrets, fallbacks (PRD 30-33, 38)."""

import logging

from fastapi.testclient import TestClient

from router.api import create_app, mask_headers
from conftest import chat, content_of, event, make_config


# -- malformed requests -----------------------------------------------------------

def test_malformed_json_400(client):
    r = client.post("/v1/chat/completions", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_missing_messages_400(client):
    r = client.post("/v1/chat/completions", json={"model": "auto"})
    assert r.status_code == 400


def test_non_object_body_400(client):
    r = client.post("/v1/chat/completions", json=[1, 2, 3])
    assert r.status_code == 400


def test_events_malformed_payload_400(client):
    r = client.post("/events", json={"no_event_field": True})
    assert r.status_code == 400


# -- health -------------------------------------------------------------------------

def test_health_shape_all_healthy(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    for tier in ("fast", "deep", "frontier"):
        assert body["backends"][tier]["healthy"] is True
        assert isinstance(body["backends"][tier]["latency_ms"], (int, float))


def test_health_backends_endpoint(client):
    r = client.get("/health/backends")
    assert set(r.json().keys()) == {"fast", "deep", "frontier"}


def test_health_degraded_when_backend_down():
    """An OpenAI-compatible backend on a dead port reports unhealthy."""
    cfg = make_config(
        **{
            "backends": {
                "fast": {
                    "type": "openai_compatible",
                    "name": "local-fast",
                    "base_url": "http://127.0.0.1:9/v1",  # discard port: almost certainly closed
                    "model": "m",
                }
            },
            "health": {"check_interval_seconds": 10, "timeout_seconds": 1},
        }
    )
    with TestClient(create_app(cfg)) as c:
        body = c.get("/health").json()
        assert body["status"] == "degraded"
        assert body["backends"]["fast"]["healthy"] is False


# -- metrics --------------------------------------------------------------------------

def test_metrics_endpoint_exposes_counters(client):
    chat(client, model="local-fast")
    event(client, "s", "t", "test_failure")
    r = client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    for name in (
        "router_requests_total",
        "router_request_latency_seconds",
        "router_backend_requests_total",
        "router_escalations_total",
        "router_cloud_requests_total",
        "router_tool_failures_total",
        "router_test_failures_total",
    ):
        assert name in text, f"missing metric {name}"
    assert 'route="fast"' in text


def test_metrics_disabled_returns_204():
    cfg = make_config(**{"metrics": {"enabled": False}})
    with TestClient(create_app(cfg)) as c:
        assert c.get("/metrics").status_code == 204


# -- secrets ----------------------------------------------------------------------------

def test_mask_headers_masks_secrets():
    masked = mask_headers({"Authorization": "Bearer sk-abc", "X-Custom": "v"})
    assert masked["Authorization"] == "***"
    assert masked["X-Custom"] == "v"


def test_api_key_never_in_logs(client, monkeypatch, caplog):
    """The OpenRouter key (read from env) must not leak into any log record."""
    cfg = make_config(
        **{
            "backends": {
                "frontier": {
                    "type": "openrouter",
                    "name": "frontier",
                    "base_url": "http://127.0.0.1:9/v1",
                    "model": "m",
                    "api_key_env": "TEST_OR_KEY",
                }
            }
        }
    )
    monkeypatch.setenv("TEST_OR_KEY", "sk-supersecret123")
    with TestClient(create_app(cfg)) as c:
        with caplog.at_level(logging.DEBUG):
            chat(c, model="local-fast")  # routes to the mock fast backend
    assert "sk-supersecret123" not in caplog.text


# -- backend fallback (PRD 31) -------------------------------------------------------------

def test_backend_failure_falls_back_when_configured():
    """A *backend* failure (timeout) is handled by the fallback policy, not escalation."""
    cfg = make_config(
        **{
            "backends": {"fast": {"type": "mock", "name": "local-fast", "model": "local-fast", "behavior": "timeout"}},
            "routing": {"fallbacks": {"fast": ["deep"]}},
        }
    )
    with TestClient(create_app(cfg)) as c:
        h = {"X-LLM-Session-ID": "sess-fb", "X-LLM-Task-ID": "task-fb"}
        r = chat(c, model="auto", **h)
        assert content_of(r) == "DEEP-SAYS-HI"  # served by the fallback backend
        meta = r.json()["x_router"]
        assert meta["route"] == "deep"
        assert meta["reason"] == "backend_fallback"

        # Fallback is NOT an escalation: the second request also starts at fast
        # (task state unchanged), falls back again, and carries no escalation block.
        r2 = chat(c, model="auto", **h)
        meta2 = r2.json()["x_router"]
        assert meta2["reason"] == "backend_fallback"
        assert "escalation" not in meta2


def test_backend_failure_without_fallback_is_502():
    cfg = make_config(
        **{"backends": {"fast": {"type": "mock", "name": "local-fast", "model": "local-fast", "behavior": "timeout"}}}
    )
    with TestClient(create_app(cfg)) as c:
        r = chat(c, model="auto")
        assert r.status_code == 502
        body = r.json()
        assert body["error"]["type"] == "backend_error"
        assert body["x_router"]["route"] == "fast"


def test_backend_failure_does_not_escalate_by_default():
    """PRD 12: with the backend_error signal off, a crashed llama-server is not 'model too weak'."""
    cfg = make_config(
        **{
            "backends": {"fast": {"type": "mock", "name": "local-fast", "model": "local-fast", "behavior": "timeout"}},
            "routing": {"fallbacks": {"fast": ["deep"]}},
        }
    )
    with TestClient(create_app(cfg)) as c:
        h = {"X-LLM-Session-ID": "sess-be", "X-LLM-Task-ID": "task-be"}
        r = None
        for _ in range(5):
            r = chat(c, model="auto", **h)  # each falls back to deep; no escalation recorded
        assert c.get("/health").status_code == 200
        meta = r.json()["x_router"]
        # Still a plain fallback after five failures: no escalation was recorded.
        assert meta["reason"] == "backend_fallback"
        assert "escalation" not in meta


def test_cloud_fallback_requires_cloud_enabled():
    """A fallback to frontier only happens when cloud.enabled is true."""
    cfg = make_config(
        **{
            "backends": {"fast": {"type": "mock", "name": "local-fast", "model": "local-fast", "behavior": "timeout"}},
            "routing": {"fallbacks": {"fast": ["frontier"]}},
            "cloud": {"enabled": False, "allow_automatic_escalation": True},
        }
    )
    with TestClient(create_app(cfg)) as c:
        r = chat(c, model="auto")
        assert r.status_code == 502  # frontier skipped (cloud disabled), no other fallback


# -- observability ---------------------------------------------------------------------------

def test_request_log_line_format(client, caplog):
    with caplog.at_level(logging.INFO):
        chat(client, model="local-fast", **{"X-LLM-Session-ID": "sess-log"})
    lines = [r.message for r in caplog.records if r.message.startswith("request=")]
    assert any(
        "route=fast" in line and "backend=local-fast" in line and "status=200" in line and "session=sess-log" in line
        for line in lines
    )


def test_escalation_log_line(client, caplog):
    with caplog.at_level(logging.INFO):
        event(client, "sess-l", "task-l", "test_failure")
        event(client, "sess-l", "task-l", "test_failure")
    esc_lines = [r.message for r in caplog.records if r.message.startswith("ESCALATION")]
    assert any(
        "from=fast" in line and "to=deep" in line and "reason=repeated_test_failure" in line for line in esc_lines
    )


def test_prompts_not_logged_by_default(client, caplog):
    with caplog.at_level(logging.DEBUG):
        chat(client, model="local-fast", **{"X-LLM-Session-ID": "sess-p"})
    assert not any("hello" in r.message for r in caplog.records)


def test_prompts_logged_when_enabled():
    cfg = make_config(**{"logging": {"level": "DEBUG", "log_requests": True, "log_responses": False, "log_prompts": True}})
    records: list[str] = []

    class H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("router.api")
    old_level = logger.level
    handler = H()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        with TestClient(create_app(cfg)) as c:
            c.post(
                "/v1/chat/completions",
                json={"model": "local-fast", "messages": [{"role": "user", "content": "secret-prompt-text"}]},
            )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    assert any("secret-prompt-text" in m for m in records)
