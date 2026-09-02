"""Cloud cost protection (PRD 21, 22)."""

from fastapi.testclient import TestClient

from router.api import create_app
from conftest import chat, content_of, event, make_config


H = {"X-LLM-Session-ID": "sess-cloud", "X-LLM-Task-ID": "task-cloud"}


def escalate_to_frontier(c: TestClient):
    for _ in range(2):
        event(c, "sess-cloud", "task-cloud", "test_failure")  # fast -> deep
    for _ in range(2):
        event(c, "sess-cloud", "task-cloud", "test_failure")  # deep -> frontier


def test_auto_escalation_blocked_when_not_opted_in():
    """PRD 21: with allow_automatic_escalation=false, a deep failure must not
    silently call OpenRouter — an escalation-required response is returned."""
    cfg = make_config(**{"cloud": {"allow_automatic_escalation": False}})
    with TestClient(create_app(cfg)) as c:
        escalate_to_frontier(c)
        r = chat(c, model="auto", **H)
        assert r.status_code == 200  # valid completion shape, so clients don't crash
        body = r.json()
        esc = body["x_router"]["escalation"]
        assert esc["required"] is True
        assert esc["reason"] == "cloud_auto_disabled"
        assert body["choices"][0]["message"]["content"].startswith("[llm-router] escalation required")


def test_manual_frontier_allowed_when_auto_blocked():
    cfg = make_config(**{"cloud": {"allow_automatic_escalation": False}})
    with TestClient(create_app(cfg)) as c:
        r = chat(c, model="auto", **{**H, "X-LLM-Escalate": "frontier"})
        assert content_of(r) == "FRONTIER-SAYS-HI"


def test_rate_limit_blocks_third_request():
    cfg = make_config(**{"cloud": {"allow_automatic_escalation": True, "max_requests_per_hour": 2}})
    with TestClient(create_app(cfg)) as c:
        escalate_to_frontier(c)
        assert content_of(chat(c, model="auto", **H)) == "FRONTIER-SAYS-HI"
        assert content_of(chat(c, model="auto", **H)) == "FRONTIER-SAYS-HI"
        r3 = chat(c, model="auto", **H)
        esc = r3.json()["x_router"]["escalation"]
        assert esc["required"] is True
        assert esc["reason"] == "cloud_rate_limit"


def test_manual_allowed_at_rate_limit_by_default():
    cfg = make_config(**{"cloud": {"allow_automatic_escalation": True, "max_requests_per_hour": 1}})
    with TestClient(create_app(cfg)) as c:
        escalate_to_frontier(c)
        assert content_of(chat(c, model="auto", **H)) == "FRONTIER-SAYS-HI"  # uses the one slot
        r = chat(c, model="auto", **{**H, "X-LLM-Escalate": "frontier"})
        assert content_of(r) == "FRONTIER-SAYS-HI"  # manual bypasses the limit


def test_manual_blocked_at_rate_limit_when_configured():
    cfg = make_config(
        **{"cloud": {"allow_automatic_escalation": True, "max_requests_per_hour": 1, "allow_manual_when_limited": False}}
    )
    with TestClient(create_app(cfg)) as c:
        escalate_to_frontier(c)
        assert content_of(chat(c, model="auto", **H)) == "FRONTIER-SAYS-HI"
        r = chat(c, model="auto", **{**H, "X-LLM-Escalate": "frontier"})
        assert r.json()["x_router"]["escalation"]["reason"] == "cloud_rate_limit"


def test_cost_limit_blocks_after_budget_spent():
    # Mock usage is 12 prompt + 7 completion tokens. At $10/MTok that is
    # $0.00019 per request; a $0.00035 daily budget allows exactly two.
    cfg = make_config(
        **{
            "cloud": {
                "allow_automatic_escalation": True,
                "max_estimated_cost_usd_per_day": 0.00035,
                "pricing": {"input_per_mtok": 10.0, "output_per_mtok": 10.0},
            }
        }
    )
    with TestClient(create_app(cfg)) as c:
        escalate_to_frontier(c)
        assert content_of(chat(c, model="auto", **H)) == "FRONTIER-SAYS-HI"
        assert content_of(chat(c, model="auto", **H)) == "FRONTIER-SAYS-HI"
        r3 = chat(c, model="auto", **H)
        esc = r3.json()["x_router"]["escalation"]
        assert esc["required"] is True
        assert esc["reason"] == "cloud_cost_limit"


def test_cloud_disabled_blocks_even_manual():
    cfg = make_config(**{"cloud": {"enabled": False}})
    with TestClient(create_app(cfg)) as c:
        r = chat(c, model="frontier")
        assert r.json()["x_router"]["escalation"]["reason"] == "cloud_disabled"


def test_local_requests_do_not_consume_cloud_budget():
    cfg = make_config(**{"cloud": {"allow_automatic_escalation": True, "max_requests_per_hour": 2}})
    with TestClient(create_app(cfg)) as c:
        for _ in range(5):
            assert content_of(chat(c, model="local-fast")) == "FAST-SAYS-HI"
        # Budget untouched: frontier still available.
        escalate_to_frontier(c)
        assert content_of(chat(c, model="auto", **H)) == "FRONTIER-SAYS-HI"
