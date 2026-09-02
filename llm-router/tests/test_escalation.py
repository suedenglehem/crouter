"""Deterministic escalation rules A-D (PRD 19, 20)."""

from conftest import chat, content_of, event


S = {"X-LLM-Session-ID": "sess-e", "X-LLM-Task-ID": "task-e"}


def test_rule_b_repeated_test_failure_fast_to_deep(client):
    # One failure: no escalation yet.
    r1 = event(client, "sess-e", "task-e", "test_failure")
    assert r1.json()["escalated"] is False
    assert chat(client, model="auto", **S).json()["x_router"]["route"] == "fast"

    # Second failure: threshold (2) reached -> deep.
    r2 = event(client, "sess-e", "task-e", "test_failure")
    body = r2.json()
    assert body["escalated"] is True
    assert body["route"] == "deep"

    # Next auto request goes to the deep backend.
    r3 = chat(client, model="auto", **S)
    assert content_of(r3) == "DEEP-SAYS-HI"
    meta = r3.json()["x_router"]
    assert meta["route"] == "deep"


def test_rule_b_deep_to_frontier(client):
    for _ in range(2):
        event(client, "sess-e", "task-e", "test_failure")  # fast -> deep
    assert chat(client, model="auto", **S).json()["x_router"]["route"] == "deep"

    r = None
    for _ in range(2):
        r = event(client, "sess-e", "task-e", "test_failure")  # deep -> frontier
    assert r.json()["escalated"] is True
    assert r.json()["route"] == "frontier"

    r3 = chat(client, model="auto", **S)
    assert content_of(r3) == "FRONTIER-SAYS-HI"


def test_rule_c_repeated_tool_failure(client):
    event(client, "sess-e", "task-e", "tool_failure")
    assert chat(client, model="auto", **S).json()["x_router"]["route"] == "fast"
    r = event(client, "sess-e", "task-e", "tool_failure")
    assert r.json()["escalated"] is True
    assert content_of(chat(client, model="auto", **S)) == "DEEP-SAYS-HI"


def test_single_isolated_failure_does_not_escalate(client):
    event(client, "sess-e", "task-e", "test_failure")
    r = chat(client, model="auto", **S)
    assert content_of(r) == "FAST-SAYS-HI"


def test_rule_d_retry_limit_without_events(client):
    """Rule D: repeated retries with at least one observed failure escalate.

    The fast backend errors on every request (backend_error signal enabled),
    so the second auto attempt for the same task escalates to deep.
    """
    from fastapi.testclient import TestClient

    from router.api import create_app
    from conftest import make_config

    cfg = make_config(
        **{
            "backends": {"fast": {"type": "mock", "name": "local-fast", "model": "local-fast", "behavior": "error"}},
            "routing": {
                "escalation": {
                    "signals": {"explicit_request": True, "repeated_tool_failure": True,
                                "repeated_test_failure": True, "timeout": False, "backend_error": True}
                }
            },
        }
    )
    with TestClient(create_app(cfg)) as c:
        h = {"X-LLM-Session-ID": "sess-d", "X-LLM-Task-ID": "task-d"}
        r1 = chat(c, model="auto", **h)
        # First attempt hits the failing fast backend (500 forwarded).
        assert r1.status_code == 500
        assert r1.json()["x_router"]["route"] == "fast"

        r2 = chat(c, model="auto", **h)
        # Second attempt: attempts=2 >= max_fast_attempts and failures>0 -> deep.
        assert content_of(r2) == "DEEP-SAYS-HI"
        meta = r2.json()["x_router"]
        assert meta["route"] == "deep"
        assert meta["escalation"]["reason"] == "retry_limit"


def test_rule_d_requires_a_failure_signal(client):
    """Retries alone (no failure signal) must not escalate a long conversation."""
    for _ in range(5):
        r = chat(client, model="auto", **S)
        assert content_of(r) == "FAST-SAYS-HI"


def test_rule_a_explicit_escalation_event_with_target(client):
    r = event(client, "sess-e", "task-e", "explicit_escalation", target="frontier")
    assert r.json()["escalated"] is True
    assert r.json()["route"] == "frontier"
    assert content_of(chat(client, model="auto", **S)) == "FRONTIER-SAYS-HI"


def test_rule_a_explicit_escalation_event_default_next_tier(client):
    r = event(client, "sess-e", "task-e", "explicit_escalation")
    assert r.json()["route"] == "deep"


def test_signal_disabled_no_auto_escalation():
    from fastapi.testclient import TestClient

    from router.api import create_app
    from conftest import make_config

    cfg = make_config(
        **{
            "routing": {
                "escalation": {
                    "signals": {"explicit_request": True, "repeated_tool_failure": False,
                                "repeated_test_failure": False, "timeout": False, "backend_error": False}
                }
            }
        }
    )
    with TestClient(create_app(cfg)) as c:
        event(c, "sess-x", "task-x", "test_failure")
        event(c, "sess-x", "task-x", "test_failure")
        assert content_of(chat(c, model="auto", **{"X-LLM-Session-ID": "sess-x", "X-LLM-Task-ID": "task-x"})) == "FAST-SAYS-HI"


def test_escalation_disabled_entirely():
    from fastapi.testclient import TestClient

    from router.api import create_app
    from conftest import make_config

    cfg = make_config(**{"routing": {"escalation": {"enabled": False}}})
    with TestClient(create_app(cfg)) as c:
        event(c, "sess-y", "task-y", "test_failure")
        event(c, "sess-y", "task-y", "test_failure")
        assert content_of(chat(c, model="auto", **{"X-LLM-Session-ID": "sess-y", "X-LLM-Task-ID": "task-y"})) == "FAST-SAYS-HI"


def test_allow_frontier_false_stops_at_deep():
    from fastapi.testclient import TestClient

    from router.api import create_app
    from conftest import make_config

    cfg = make_config(**{"routing": {"escalation": {"allow_frontier": False}}})
    with TestClient(create_app(cfg)) as c:
        h = {"X-LLM-Session-ID": "sess-f", "X-LLM-Task-ID": "task-f"}
        for _ in range(2):
            event(c, "sess-f", "task-f", "test_failure")  # fast -> deep
        for _ in range(2):
            r = event(c, "sess-f", "task-f", "test_failure")  # would go to frontier
        assert r.json()["route"] == "deep"  # frontier disallowed: stays on deep
        assert content_of(chat(c, model="auto", **h)) == "DEEP-SAYS-HI"


def test_task_complete_resets_counters(client):
    event(client, "sess-e", "task-e", "test_failure")
    r = event(client, "sess-e", "task-e", "task_complete")
    assert r.json()["route"] == "fast"
    # Fresh failure count: one more failure is not enough to escalate.
    event(client, "sess-e", "task-e", "test_failure")
    assert content_of(chat(client, model="auto", **S)) == "FAST-SAYS-HI"


def test_unknown_event_accepted_not_crashing(client):
    r = event(client, "sess-e", "task-e", "quantum_flux_capacitor_malfunction", detail="weird")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_escalation_metadata_in_response_after_event_driven_change(client):
    for _ in range(2):
        event(client, "sess-e", "task-e", "test_failure")
    meta = chat(client, model="auto", **S).json()["x_router"]
    assert meta["route"] == "deep"


def test_explicit_model_not_counted_for_rule_d(client):
    """Explicitly pinned requests do not consume the auto retry budget."""
    for _ in range(5):
        chat(client, model="local-fast", **S)
    # A single failure + one more auto attempt must NOT escalate: attempts on
    # fast via auto are still 0 (explicit pins don't count).
    event(client, "sess-e", "task-e", "test_failure")
    r = chat(client, model="auto", **S)
    assert content_of(r) == "FAST-SAYS-HI"
