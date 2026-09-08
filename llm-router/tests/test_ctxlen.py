"""Runtime context-bound steering: ``GET /ctxlen`` and ``GET /ctxlen/<tier>=<tokens|reset>``.

The context floor reads each tier's max_context live per request, so a bound
changed here applies to the very next routed request. ``reset`` restores the
startup value — the config value, or the queried size when query_context_size
overrode it at startup (PRD §52). Changes are in-memory only.
"""

from fastapi.testclient import TestClient

from conftest import content_of, make_config
from router.api import create_app
from router.backends.mock import MockBackend


def _ctx_client(**overrides) -> TestClient:
    """Mock-backed app with small, predictable bounds: fast=200, deep=10_000."""
    overrides.setdefault(
        "backends", {"fast": {"max_context": 200}, "deep": {"max_context": 10_000}}
    )
    overrides.setdefault(
        "routing", {"context": {"enabled": True, "chars_per_token": 3.0, "completion_reserve": 50}}
    )
    return TestClient(create_app(make_config(**overrides)))


def _post(c: TestClient, content: str):
    return c.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": content}]},
    )


#: ~357 estimated tokens + 50 reserve > 200 (fast) -> overflows to deep.
BIG = "a" * 1000


# -- read ------------------------------------------------------------------------

def test_get_ctxlen_reports_current_and_initial():
    with _ctx_client() as c:
        r = c.get("/ctxlen")
        assert r.status_code == 200
        body = r.json()
        assert body["fast"] == {"max_context": 200, "initial_max_context": 200}
        assert body["deep"]["max_context"] == 10_000
        assert body["frontier"]["max_context"] is None


# -- set / reset -------------------------------------------------------------------

def test_set_bound_applies_to_next_request():
    with _ctx_client() as c:
        r = _post(c, BIG)
        assert content_of(r) == "DEEP-SAYS-HI"  # overflowed fast's 200

        s = c.get("/ctxlen/fast=5000")
        assert s.status_code == 200
        assert s.json() == {"tier": "fast", "max_context": 5000, "previous_max_context": 200}

        r = _post(c, BIG)
        assert content_of(r) == "FAST-SAYS-HI"  # now fits fast


def test_reset_restores_startup_value():
    with _ctx_client() as c:
        c.get("/ctxlen/fast=5000")
        s = c.get("/ctxlen/fast=reset")
        assert s.status_code == 200
        assert s.json()["max_context"] == 200

        r = _post(c, BIG)
        assert content_of(r) == "DEEP-SAYS-HI"  # floor active again


def test_reset_restores_queried_startup_size(monkeypatch):
    """When query_context_size overrode the manual value at startup, reset goes
    back to the QUERIED size (the effective baseline), not the stale config one."""

    async def fake_query(self):
        return 50_000

    monkeypatch.setattr(MockBackend, "query_context_size", fake_query)
    cfg = make_config(
        backends={"fast": {"max_context": 132_768, "query_context_size": True}},
        routing={"context": {"enabled": True, "chars_per_token": 3.0, "completion_reserve": 50}},
    )
    with TestClient(create_app(cfg)) as c:
        assert c.get("/ctxlen").json()["fast"]["initial_max_context"] == 50_000
        s = c.get("/ctxlen/fast=reset")
        assert s.json() == {"tier": "fast", "max_context": 50_000, "previous_max_context": 50_000}


def test_reset_with_unset_initial_gives_none():
    with TestClient(create_app(make_config())) as c:  # no max_context anywhere
        s = c.get("/ctxlen/fast=reset")
        assert s.status_code == 200
        assert s.json()["max_context"] is None


def test_set_is_in_memory_only():
    with _ctx_client() as c:
        c.get("/ctxlen/deep=123")
    with _ctx_client() as c:  # fresh app -> startup value again
        assert c.get("/ctxlen").json()["deep"]["max_context"] == 10_000


# -- validation ----------------------------------------------------------------------

def test_unknown_tier_404():
    with _ctx_client() as c:
        r = c.get("/ctxlen/frontier-x=100")
        assert r.status_code == 404
        assert "unknown tier" in r.json()["error"]


def test_malformed_spec_400():
    with _ctx_client() as c:
        r = c.get("/ctxlen/fast")  # missing '=value'
        assert r.status_code == 400
        assert "expected <tier>=<tokens|reset>" in r.json()["error"]


def test_bad_value_400():
    with _ctx_client() as c:
        for spec in ("fast=abc", "fast=0", "fast=-5"):
            r = c.get(f"/ctxlen/{spec}")
            assert r.status_code == 400, spec
