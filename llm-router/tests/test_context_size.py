"""``query_context_size``: backends report their real context window at startup
and override the manual ``max_context``, which may be stale (PRD §52).

llama-server exposes it via GET /props -> default_generation_settings.n_ctx —
the effective --ctx-size actually in use. Non-llama servers usually lack
/props; a failed query keeps the manual value.
"""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import make_config
from router.api import create_app, sync_context_sizes
from router.backends.mock import MockBackend
from router.backends.openai_compatible import OpenAICompatibleBackend
from router.config import BackendConfig


def _props_client(n_ctx=None, status=200, raw: bytes | None = None, props_path="/props") -> httpx.AsyncClient:
    """httpx client serving the given payload at ``props_path`` only."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == props_path:
            body = raw if raw is not None else json.dumps(
                {"default_generation_settings": {"n_ctx": n_ctx}}
            ).encode()
            return httpx.Response(status, content=body)
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test/v1")


def _backend(client: httpx.AsyncClient | None = None) -> OpenAICompatibleBackend:
    cfg = BackendConfig(type="openai_compatible", name="t", base_url="http://test/v1")
    return OpenAICompatibleBackend("fast", cfg, client=client or _props_client())


# -- backend-level query ---------------------------------------------------------

def test_query_returns_n_ctx_from_server_root():
    # llama-server serves /props at the root, outside the /v1 base_url prefix.
    be = _backend(_props_client(n_ctx=132864, props_path="/props"))
    assert asyncio.run(be.query_context_size()) == 132864


def test_query_falls_back_to_v1_props():
    # Hypothetical server exposing it under the /v1 prefix instead.
    be = _backend(_props_client(n_ctx=90_000, props_path="/v1/props"))
    assert asyncio.run(be.query_context_size()) == 90_000


def test_query_404_returns_none():
    # Non-llama OpenAI-compatible servers usually have no /props.
    be = _backend(_props_client(status=404))
    assert asyncio.run(be.query_context_size()) is None


def test_query_bad_json_returns_none():
    be = _backend(_props_client(raw=b"not json"))
    assert asyncio.run(be.query_context_size()) is None


@pytest.mark.parametrize("n_ctx", [None, -5, 0, "big"])
def test_query_invalid_n_ctx_returns_none(n_ctx):
    be = _backend(_props_client(n_ctx=n_ctx))
    assert asyncio.run(be.query_context_size()) is None


# -- startup override (sync_context_sizes) ----------------------------------------

class _Stub:
    def __init__(self, n):
        self._n = n

    async def query_context_size(self):
        return self._n


def test_override_applied_when_flag_set():
    cfg = make_config(backends={"fast": {"max_context": 100, "query_context_size": True}})
    asyncio.run(sync_context_sizes(cfg, {"fast": _Stub(999), "deep": _Stub(None), "frontier": _Stub(None)}))
    assert cfg.backends["fast"].max_context == 999


def test_manual_value_kept_when_query_fails():
    cfg = make_config(backends={"fast": {"max_context": 100, "query_context_size": True}})
    asyncio.run(sync_context_sizes(cfg, {"fast": _Stub(None), "deep": _Stub(5), "frontier": _Stub(6)}))
    assert cfg.backends["fast"].max_context == 100


def test_no_override_when_flag_unset():
    cfg = make_config(backends={"fast": {"max_context": 100}})
    asyncio.run(sync_context_sizes(cfg, {"fast": _Stub(999), "deep": _Stub(None), "frontier": _Stub(None)}))
    assert cfg.backends["fast"].max_context == 100


# -- end to end: queried size drives the context floor -----------------------------

def _post(client: TestClient, content: str):
    return client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": content}]},
    )


def test_queried_size_overrides_manual_and_drives_floor(monkeypatch):
    async def fake_query(self):
        return 50_000

    monkeypatch.setattr(MockBackend, "query_context_size", fake_query)
    cfg = make_config(
        backends={"fast": {"max_context": 132_768, "query_context_size": True}},
        routing={"context": {"enabled": True, "chars_per_token": 3.0, "completion_reserve": 50}},
    )
    with TestClient(create_app(cfg)) as c:
        # ~160k chars -> est. ~53.4k tokens + 50 reserve: over the QUERIED 50_000,
        # under the manual 132_768 — so only a working override bumps to deep.
        r = _post(c, "a" * 160_000)
    assert r.status_code == 200
    meta = r.json()["x_router"]
    assert meta["route"] == "deep"
    assert meta["reason"] == "context_overflow"


def test_manual_size_stands_without_flag(monkeypatch):
    async def fake_query(self):
        return 50_000

    monkeypatch.setattr(MockBackend, "query_context_size", fake_query)
    cfg = make_config(
        backends={"fast": {"max_context": 132_768}},  # flag unset -> manual value stands
        routing={"context": {"enabled": True, "chars_per_token": 3.0, "completion_reserve": 50}},
    )
    with TestClient(create_app(cfg)) as c:
        r = _post(c, "a" * 160_000)
    assert r.status_code == 200
    meta = r.json()["x_router"]
    assert meta["route"] == "fast"
    assert meta["reason"] == "auto_policy"
