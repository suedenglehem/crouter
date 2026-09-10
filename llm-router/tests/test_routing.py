"""Explicit routing and priority order (PRD §6, §7)."""

from fastapi.testclient import TestClient

from conftest import chat, content_of, make_config
from router.api import create_app
from router.routing import extract_prompt_marker, last_user_message_text


# -- prompt markers (@@fast / @@deep) ------------------------------------------

def _body(content):
    return {"messages": [{"role": "user", "content": content}]}


def test_marker_extracted_and_stripped_from_string_content():
    body = _body("create a python code to find oldest files in a folder @@deep")
    assert extract_prompt_marker(body) == "deep"
    assert body["messages"][0]["content"] == "create a python code to find oldest files in a folder"


def test_marker_rightmost_wins_and_all_occurrences_stripped():
    body = _body("first @@fast then @@DEEP")
    assert extract_prompt_marker(body) == "deep"
    assert "@@" not in body["messages"][0]["content"]


def test_marker_no_match_leaves_body_untouched():
    text = "no marker here, even @fast or @@fastest"
    body = _body(text)
    assert extract_prompt_marker(body) is None
    assert body["messages"][0]["content"] == text


def test_marker_in_list_content_parts():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "run this @@fast please"},
        {"type": "image_url", "image_url": {"url": "http://x"}},
    ]}]}
    assert extract_prompt_marker(body) == "fast"
    assert body["messages"][0]["content"][0]["text"] == "run this please"


def test_marker_only_scans_last_user_message():
    body = {"messages": [
        {"role": "user", "content": "earlier @@deep"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "now plain"},
    ]}
    assert extract_prompt_marker(body) is None
    assert body["messages"][0]["content"] == "earlier @@deep"  # untouched


def test_last_user_message_text_shapes():
    assert last_user_message_text(_body("hi")) == "hi"
    parts = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": {"url": "http://x"}},
        {"type": "text", "text": "b"},
    ]}]}
    assert last_user_message_text(parts) == "ab"
    assert last_user_message_text({"messages": [{"role": "assistant", "content": "x"}]}) is None


def test_prompt_marker_routes_deep(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "fix the build @@deep"}]},
    )
    assert content_of(r) == "DEEP-SAYS-HI"
    meta = r.json()["x_router"]
    assert meta["route"] == "deep"
    assert meta["reason"] == "prompt_marker"


def test_prompt_marker_routes_fast(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "rename a variable @@fast"}]},
    )
    assert content_of(r) == "FAST-SAYS-HI"
    meta = r.json()["x_router"]
    assert meta["route"] == "fast"
    assert meta["reason"] == "prompt_marker"


def test_prompt_marker_beats_route_header(client):
    # Priority: prompt marker (1) beats explicit headers (2).
    r = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "task @@fast"}]},
        headers={"X-LLM-Route": "deep"},
    )
    assert content_of(r) == "FAST-SAYS-HI"
    meta = r.json()["x_router"]
    assert meta["route"] == "fast"
    assert meta["reason"] == "prompt_marker"


def test_pin_beats_prompt_marker(client):
    # Priority: the authoritative pin (0) beats everything.
    client.get("/route/all/deep")
    r = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "task @@fast"}]},
    )
    assert content_of(r) == "DEEP-SAYS-HI"
    meta = r.json()["x_router"]
    assert meta["route"] == "deep"
    assert meta["reason"] == "pinned"


# -- complexity classifier (opt-in, PRD §20 exception) --------------------------

def _complexity_client(**overrides) -> TestClient:
    """App with the classifier on; the fast mock doubles as judge and answers DEEP."""
    overrides.setdefault("backends", {"fast": {"response_text": "DEEP"}})
    overrides.setdefault("routing", {"complexity": {"enabled": True, "tier": "fast"}})
    return TestClient(create_app(make_config(**overrides)))


def test_complexity_verdict_routes_auto_request():
    with _complexity_client() as c:
        r = chat(c, model="auto")
        assert content_of(r) == "DEEP-SAYS-HI"
        meta = r.json()["x_router"]
        assert meta["route"] == "deep"
        assert meta["reason"] == "complexity"


def test_complexity_skipped_for_explicit_model():
    with _complexity_client() as c:
        r = chat(c, model="local-fast")
        # Served by the fast mock (whose response_text is now "DEEP"); the judge
        # must not have influenced routing.
        assert content_of(r) == "DEEP"
        assert r.json()["x_router"]["reason"] == "explicit_model"


def test_complexity_skipped_for_claude_family():
    with _complexity_client() as c:
        r = chat(c, model="claude-haiku-4-5")
        # Family mapping -> fast; if the classifier had run it would have said DEEP.
        assert content_of(r) == "DEEP"  # served by the fast mock
        meta = r.json()["x_router"]
        assert meta["route"] == "fast"
        assert meta["reason"] == "claude_family"


def test_prompt_marker_beats_complexity():
    with _complexity_client() as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "task @@fast"}]},
        )
        meta = r.json()["x_router"]
        assert meta["route"] == "fast"
        assert meta["reason"] == "prompt_marker"


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


def test_claude_family_names_map_to_tiers(client):
    # Claude Code background tasks (e.g. the auto-mode Bash safety classifier)
    # send their own model names; they must not 404 or every gated command is
    # blocked with "classifier unavailable".
    r = chat(client, model="claude-sonnet-5")
    assert r.status_code == 200
    assert content_of(r) == "DEEP-SAYS-HI"
    meta = r.json()["x_router"]
    assert meta["route"] == "deep"
    assert meta["reason"] == "claude_family"


def test_claude_haiku_maps_to_fast(client):
    r = chat(client, model="claude-haiku-4-5")
    assert r.status_code == 200
    assert content_of(r) == "FAST-SAYS-HI"
    meta = r.json()["x_router"]
    assert meta["route"] == "fast"
    assert meta["reason"] == "claude_family"


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
