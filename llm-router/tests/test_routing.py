"""Explicit routing and priority order (PRD §6, §7)."""

from conftest import chat, content_of


def test_explicit_model_fast(client):
    r = chat(client, model="local-fast")
    assert r.status_code == 200
    body = r.json()
    assert content_of(r) == "FAST-SAYS-HI"
    assert body["x_router"]["route"] == "fast"
    assert body["x_router"]["reason"] == "explicit_model"


def test_explicit_model_deep(client):
    r = chat(client, model="local-deep")
    assert content_of(r) == "DEEP-SAYS-HI"
    assert r.json()["x_router"]["route"] == "deep"


def test_explicit_model_frontier(client):
    r = chat(client, model="frontier")
    assert content_of(r) == "FRONTIER-SAYS-HI"
    assert r.json()["x_router"]["route"] == "frontier"


def test_auto_starts_on_fast(client):
    r = chat(client, model="auto")
    assert content_of(r) == "FAST-SAYS-HI"
    meta = r.json()["x_router"]
    assert meta["route"] == "fast"
    assert meta["reason"] == "auto_policy"


def test_missing_model_uses_default_route(client):
    # defaults.route is "auto" -> starts on fast.
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert content_of(r) == "FAST-SAYS-HI"


def test_unknown_model_404(client):
    r = chat(client, model="gpt-9-turbo")
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "model_not_found"


def test_route_header_wins_over_model(client):
    # Priority: explicit header (1) beats explicit model (2).
    r = chat(client, model="local-fast", **{"X-LLM-Route": "deep"})
    assert content_of(r) == "DEEP-SAYS-HI"
    assert r.json()["x_router"]["reason"] == "explicit_route_header"


def test_escalate_header_wins_over_route_header_and_model(client):
    r = chat(
        client,
        model="local-fast",
        **{"X-LLM-Route": "fast", "X-LLM-Escalate": "deep"},
    )
    assert content_of(r) == "DEEP-SAYS-HI"
    assert r.json()["x_router"]["reason"] == "explicit_escalate"


def test_route_header_auto_falls_through_to_policy(client):
    # A route header of "auto" is not a tier: it falls through to the policy.
    r = chat(client, model="auto", **{"X-LLM-Route": "auto"})
    # auto policy -> task's current route (fast for a fresh task).
    assert content_of(r) == "FAST-SAYS-HI"


def test_escalate_header_records_escalation_in_x_router(client):
    r = chat(client, model="auto", **{"X-LLM-Session-ID": "s1", "X-LLM-Escalate": "deep"})
    meta = r.json()["x_router"]
    assert meta["route"] == "deep"
    assert meta["escalation"]["from"] == "fast"
    assert meta["escalation"]["to"] == "deep"
    assert meta["escalation"]["reason"] == "explicit_request"


def test_models_endpoint_lists_aliases(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert {"auto", "local-fast", "local-deep", "frontier"} <= ids


def test_session_and_task_ids_in_metadata(client):
    r = chat(
        client,
        model="auto",
        **{"X-LLM-Session-ID": "sess-abc", "X-LLM-Task-ID": "task-1"},
    )
    meta = r.json()["x_router"]
    assert meta["session_id"] == "sess-abc"
    assert meta["task_id"] == "task-1"


def test_generated_ids_when_headers_absent(client):
    r = chat(client, model="auto")
    meta = r.json()["x_router"]
    assert meta["session_id"].startswith("sess-")
    assert meta["request_id"]
