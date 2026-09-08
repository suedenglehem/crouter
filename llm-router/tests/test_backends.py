"""Backend abstraction unit tests (PRD 12, 29) — no network via httpx.MockTransport."""

import asyncio
import json

import httpx
import pytest

from router.backends.base import BackendError
from router.backends.openai_compatible import OpenAICompatibleBackend
from router.backends.openrouter import OpenRouterBackend
from router.config import BackendConfig, HealthConfig


def make_cfg(**kw) -> BackendConfig:
    base = {"type": "openai_compatible", "name": "test-backend", "base_url": "http://unit.test/v1", "model": "real-model"}
    base.update(kw)
    return BackendConfig.model_validate(base)


def make_backend(handler, cfg=None, timeout=5.0):
    cfg = cfg or make_cfg()
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://unit.test/v1", timeout=httpx.Timeout(timeout))
    return OpenAICompatibleBackend("fast", cfg, health_cfg=HealthConfig(), client=client)


def run(coro):
    return asyncio.run(coro)


# -- request forwarding ---------------------------------------------------------

def test_model_name_rewritten_and_body_forwarded():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "x", "choices": []})

    b = make_backend(handler)
    body = {"model": "local-fast", "messages": [{"role": "user", "content": "hi"}], "tools": [1, 2]}
    result = run(b.chat_completion(body, stream=False))
    assert result.status_code == 200
    assert seen["body"]["model"] == "real-model"
    assert seen["body"]["messages"] == body["messages"]
    assert seen["body"]["tools"] == [1, 2]


def test_5xx_raises_backend_error():
    def handler(request):
        return httpx.Response(503, json={"error": "unavailable"})

    b = make_backend(handler)
    with pytest.raises(BackendError) as ei:
        run(b.chat_completion({"messages": []}, stream=False))
    assert ei.value.kind == "http_5xx"
    assert ei.value.status_code == 503


def test_4xx_forwarded_not_raised():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    b = make_backend(handler)
    result = run(b.chat_completion({"messages": []}, stream=False))
    assert result.status_code == 400
    assert b"bad request" in (result.body or b"")


def test_connection_error_kind():
    def handler(request):
        raise httpx.ConnectError("refused")

    b = make_backend(handler)
    with pytest.raises(BackendError) as ei:
        run(b.chat_completion({"messages": []}, stream=False))
    assert ei.value.kind == "connection"


def test_timeout_kind():
    # httpx 0.28+ enforces timeouts inside the real transport (via request
    # extensions) which MockTransport ignores, so raise the exception directly:
    # what we are testing is the mapping to BackendError(kind="timeout").
    def handler(request):
        raise httpx.ReadTimeout("read timed out")

    b = make_backend(handler)
    with pytest.raises(BackendError) as ei:
        run(b.chat_completion({"messages": []}, stream=False))
    assert ei.value.kind == "timeout"


def test_usage_parsed_from_response():
    def handler(request):
        return httpx.Response(200, json={"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}})

    b = make_backend(handler)
    result = run(b.chat_completion({"messages": []}, stream=False))
    assert result.usage == {"prompt_tokens": 5, "completion_tokens": 3}


# -- streaming -------------------------------------------------------------------

def test_streaming_passthrough_lines():
    sse = b'data: {"a":1}\n\ndata: {"b":2}\n\ndata: [DONE]\n\n'

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

    b = make_backend(handler)

    async def consume():
        result = await b.chat_completion({"messages": []}, stream=True)
        assert result.status_code == 200
        return [line async for line in result.stream]

    lines = run(consume())
    assert lines[0].startswith(b"data: ")
    assert any(line.strip() == b"data: [DONE]" for line in lines)


def test_stream_5xx_raises():
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    b = make_backend(handler)
    with pytest.raises(BackendError) as ei:
        run(b.chat_completion({"messages": []}, stream=True))
    assert ei.value.kind == "http_5xx"


# -- health ------------------------------------------------------------------------

def test_health_ok():
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    b = make_backend(handler)
    healthy, latency = run(b.health_check())
    assert healthy is True
    assert latency >= 0


def test_health_unhealthy_on_error():
    def handler(request):
        return httpx.Response(500)

    b = make_backend(handler)
    healthy, _ = run(b.health_check())
    assert healthy is False


# -- openrouter ----------------------------------------------------------------------

def test_openrouter_auth_header_from_env(monkeypatch):
    monkeypatch.setenv("TEST_OR_KEY", "sk-supersecret123")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={})

    cfg = make_cfg(type="openrouter", api_key_env="TEST_OR_KEY")
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://unit.test/v1")
    b = OpenRouterBackend("frontier", cfg, health_cfg=HealthConfig(), client=client)
    run(b.chat_completion({"messages": []}, stream=False))

    assert seen["headers"]["authorization"] == "Bearer sk-supersecret123"
    # Attribution headers present.
    assert "x-title" in seen["headers"]


def test_openrouter_missing_key_no_auth_header(monkeypatch):
    monkeypatch.delenv("TEST_OR_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    cfg = make_cfg(type="openrouter", api_key_env="TEST_OR_KEY")
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://unit.test/v1")
    b = OpenRouterBackend("frontier", cfg, health_cfg=HealthConfig(), client=client)

    headers = b._request_headers()  # sync method; no event loop needed
    assert "authorization" not in headers


# -- api key (literal in YAML, env fallback) ---------------------------------------

def test_api_key_literal_sends_auth_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={})

    b = make_backend(handler, cfg=make_cfg(api_key="sk-lm-test"))
    run(b.chat_completion({"messages": []}, stream=False))
    assert seen["headers"]["authorization"] == "Bearer sk-lm-test"


def test_api_key_literal_applies_to_health_check():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"data": []})

    b = make_backend(handler, cfg=make_cfg(api_key="sk-lm-test"))
    healthy, _ = run(b.health_check())
    assert healthy is True
    assert seen["headers"]["authorization"] == "Bearer sk-lm-test"


def test_api_key_env_fallback_for_openai_compatible(monkeypatch):
    monkeypatch.setenv("TEST_FAST_KEY", "sk-from-env")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    b = make_backend(handler, cfg=make_cfg(api_key_env="TEST_FAST_KEY"))
    assert b._request_headers()["Authorization"] == "Bearer sk-from-env"


def test_api_key_literal_wins_over_env(monkeypatch):
    monkeypatch.setenv("TEST_FAST_KEY", "sk-from-env")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    b = make_backend(handler, cfg=make_cfg(api_key="sk-literal", api_key_env="TEST_FAST_KEY"))
    assert b._request_headers()["Authorization"] == "Bearer sk-literal"


def test_no_key_no_auth_header(monkeypatch):
    monkeypatch.delenv("TEST_FAST_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    b = make_backend(handler)  # no api_key, no api_key_env
    assert "authorization" not in b._request_headers()


def test_extra_headers_authorization_wins_over_api_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    b = make_backend(handler, cfg=make_cfg(api_key="sk-lm-test", extra_headers={"Authorization": "Bearer sk-explicit"}))
    assert b._request_headers()["Authorization"] == "Bearer sk-explicit"


def test_openrouter_literal_api_key_wins(monkeypatch):
    monkeypatch.setenv("TEST_OR_KEY", "sk-from-env")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    cfg = make_cfg(type="openrouter", api_key="sk-literal", api_key_env="TEST_OR_KEY")
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://unit.test/v1")
    b = OpenRouterBackend("frontier", cfg, health_cfg=HealthConfig(), client=client)

    assert b._request_headers()["Authorization"] == "Bearer sk-literal"
