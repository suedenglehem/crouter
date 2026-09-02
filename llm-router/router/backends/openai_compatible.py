"""Generic OpenAI-compatible HTTP backend.

``llama-server`` is just one instance of this: an OpenAI-compatible HTTP
server. No llama.cpp-specific logic lives here (PRD §29).

The request body is forwarded verbatim except that ``model`` is rewritten to
the backend's configured model name. Streaming responses are proxied line by
line without buffering the whole response (PRD §24).
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import httpx

from ..config import BackendConfig, HealthConfig
from .base import Backend, BackendError, BackendResult, cached_health


class OpenAICompatibleBackend(Backend):
    def __init__(
        self,
        tier: str,
        cfg: BackendConfig,
        health_cfg: Optional[HealthConfig] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        super().__init__(tier, cfg.name)
        self._cfg = cfg
        self._health_cfg = health_cfg or HealthConfig()
        # ``client`` is an injection point for tests (httpx.MockTransport).
        self._client = client or httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            timeout=httpx.Timeout(cfg.timeout_seconds),
        )
        self._owns_client = client is None
        self._health_cache: Optional[tuple[float, bool, float]] = None

    # -- headers ---------------------------------------------------------

    def _request_headers(self) -> dict[str, str]:
        """Headers sent with every request. Subclasses may add auth here."""
        return dict(self._cfg.extra_headers or {})

    # -- chat completion ---------------------------------------------------

    def _payload(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = dict(body)
        if self._cfg.model:
            payload["model"] = self._cfg.model
        return payload

    async def chat_completion(self, body: dict[str, Any], *, stream: bool) -> BackendResult:
        payload = self._payload(body)
        headers = self._request_headers() or None

        try:
            if stream:
                req = self._client.build_request(
                    "POST", "/chat/completions", json=payload, headers=headers
                )
                resp = await self._client.send(req, stream=True)
            else:
                resp = await self._client.post(
                    "/chat/completions", json=payload, headers=headers
                )
        except httpx.TimeoutException as e:
            raise BackendError("timeout", f"backend {self.name} timed out") from e
        except httpx.HTTPError as e:
            raise BackendError("connection", f"backend {self.name} unreachable: {e}") from e

        if stream:
            return await self._stream_result(resp)
        return self._full_result(resp)

    async def _stream_result(self, resp: httpx.Response) -> BackendResult:
        """Wrap a streamed response; 5xx becomes a BackendError, other non-200 is forwarded."""
        if resp.status_code >= 500:
            await resp.aread()
            await resp.aclose()
            raise BackendError(
                "http_5xx",
                f"backend {self.name} returned HTTP {resp.status_code}",
                status_code=resp.status_code,
            )
        if resp.status_code != 200:
            data = await resp.aread()
            await resp.aclose()
            return BackendResult(resp.status_code, self._ct(resp), body=data)

        async def gen():
            try:
                async for line in resp.aiter_lines():
                    yield (line + "\n").encode("utf-8")
            finally:
                await resp.aclose()

        return BackendResult(200, "text/event-stream", stream=gen())

    def _full_result(self, resp: httpx.Response) -> BackendResult:
        if resp.status_code >= 500:
            raise BackendError(
                "http_5xx",
                f"backend {self.name} returned HTTP {resp.status_code}",
                status_code=resp.status_code,
            )
        usage = None
        try:
            parsed = json.loads(resp.content)
            if isinstance(parsed, dict):
                u = parsed.get("usage")
                if isinstance(u, dict):
                    usage = u
        except (ValueError, UnicodeDecodeError):
            pass  # non-JSON body is forwarded as-is
        return BackendResult(resp.status_code, self._ct(resp), body=resp.content, usage=usage)

    @staticmethod
    def _ct(resp: httpx.Response) -> str:
        ct = resp.headers.get("content-type", "application/json")
        return ct.split(";")[0].strip() or "application/json"

    # -- health ------------------------------------------------------------

    async def health_check(self) -> tuple[bool, float]:
        hit = cached_health(self._health_cache, self._health_cfg.check_interval_seconds)
        if hit is not None:
            return hit

        t0 = time.perf_counter()
        try:
            resp = await self._client.get(
                "/models", timeout=self._health_cfg.timeout_seconds, headers=self._request_headers() or None
            )
            healthy = resp.status_code == 200
        except httpx.HTTPError:
            healthy = False
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._health_cache = (time.monotonic(), healthy, latency_ms)
        return healthy, latency_ms

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
