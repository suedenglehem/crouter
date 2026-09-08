"""Authoritative routing pin: ``GET /route/all/<tier>``, ``/route/reset``, ``/route/last``.

While pinned, every request goes to the pinned tier regardless of model,
headers or policy (reason ``pinned``), and the context floor is skipped —
the pin runs the tier at its maximum known window. ``reset`` returns to the
default configuration; ``last`` restores the bounds that were in effect
right before the pin was set. In-memory only.
"""

from fastapi.testclient import TestClient

from conftest import content_of, make_config
from router.api import create_app


def _ctx_client(**overrides) -> TestClient:
    """Mock-backed app with small, predictable bounds: fast=200, deep=10_000."""
    overrides.setdefault(
        "backends", {"fast": {"max_context": 200}, "deep": {"max_context": 10_000}}
    )
    overrides.setdefault(
        "routing", {"context": {"enabled": True, "chars_per_token": 3.0, "completion_reserve": 50}}
    )
    return TestClient(create_app(make_config(**overrides)))


def _post(c: TestClient, content: str, model="auto", **headers):
    return c.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": content}]},
        headers=headers,
    )


#: ~357 estimated tokens + 50 reserve > 200 (fast) -> overflows to deep when unpinned.
BIG = "a" * 1000


# -- pin behavior ------------------------------------------------------------------

def test_pin_forces_tier_and_skips_floor():
    with _ctx_client() as c:
        r = _post(c, BIG)
        assert content_of(r) == "DEEP-SAYS-HI"  # unpinned: overflowed fast's 200

        s = c.get("/route/all/fast")
        assert s.status_code == 200
        body = s.json()
        assert body["pin"] == "fast"
        # The pinned tier runs at its maximum known window (startup value).
        assert body["bounds"]["fast"] == 200

        r = _post(c, BIG)
        meta = r.json()["x_router"]
        assert content_of(r) == "FAST-SAYS-HI"  # stays on fast despite the overflow
        assert meta["reason"] == "pinned"


def test_pin_overrides_explicit_model_and_headers():
    with _ctx_client() as c:
        c.get("/route/all/fast")
        r = _post(c, "hello", model="local-deep")
        assert content_of(r) == "FAST-SAYS-HI"
        assert r.json()["x_router"]["reason"] == "pinned"

        r = _post(c, "hello", **{"X-LLM-Escalate": "deep"})
        assert content_of(r) == "FAST-SAYS-HI"
        assert r.json()["x_router"]["reason"] == "pinned"


def test_pin_restores_max_window_when_bound_was_lowered():
    with _ctx_client() as c:
        c.get("/ctxlen/fast=100")  # custom lower bound
        s = c.get("/route/all/fast")
        assert s.json()["bounds"]["fast"] == 200  # back to the startup maximum


def test_pin_unknown_tier_404():
    with _ctx_client() as c:
        r = c.get("/route/all/nope")
        assert r.status_code == 404
        assert "unknown tier" in r.json()["error"]


# -- reset ---------------------------------------------------------------------------

def test_reset_returns_to_defaults():
    with _ctx_client() as c:
        c.get("/ctxlen/fast=100")
        c.get("/route/all/deep")
        s = c.get("/route/reset")
        assert s.status_code == 200
        body = s.json()
        assert body["pin"] is None
        assert body["bounds"]["fast"] == 200   # startup value, not the custom 100
        assert body["bounds"]["deep"] == 10_000

        r = _post(c, BIG)
        assert content_of(r) == "DEEP-SAYS-HI"  # floor active again (fast=200)


def test_reset_is_idempotent_without_pin():
    with _ctx_client() as c:
        s = c.get("/route/reset")
        assert s.status_code == 200
        assert s.json()["pin"] is None


# -- last --------------------------------------------------------------------------------

def test_last_restores_pre_pin_bounds_and_unpins():
    with _ctx_client() as c:
        c.get("/ctxlen/fast=100")   # custom state before the pin
        c.get("/route/all/deep")
        s = c.get("/route/last")
        assert s.status_code == 200
        body = s.json()
        assert body["pin"] is None
        assert body["bounds"]["fast"] == 100    # the pre-pin custom value, not startup's 200

        r = _post(c, BIG)
        assert content_of(r) == "DEEP-SAYS-HI"  # floor active with fast=100


def test_last_without_prior_state_400():
    with _ctx_client() as c:
        r = c.get("/route/last")
        assert r.status_code == 400
        assert "no previous state" in r.json()["error"]


def test_repin_keeps_original_snapshot():
    with _ctx_client() as c:
        c.get("/ctxlen/fast=100")
        c.get("/route/all/deep")
        c.get("/route/all/frontier")  # re-pin while pinned
        s = c.get("/route/last")
        assert s.json()["bounds"]["fast"] == 100  # snapshot from before the FIRST pin


def test_pin_is_in_memory_only():
    with _ctx_client() as c:
        c.get("/route/all/deep")
    with _ctx_client() as c:  # fresh app -> no pin, startup bounds
        body = c.get("/route").json()
        assert body["pin"] is None
        assert body["bounds"]["fast"] == 200
