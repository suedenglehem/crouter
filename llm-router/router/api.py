"""FastAPI application (PRD §5, §30–33).

Endpoints:

    POST /v1/chat/completions   OpenAI-compatible chat completions (streaming + non-streaming)
    GET  /v1/models             logical model aliases
    GET  /health                overall status + per-backend health
    GET  /health/backends       per-backend health only
    GET  /metrics               Prometheus text format
    POST /events                lifecycle events from Claude Code hooks

The request body is proxied to the selected backend verbatim (except the
``model`` field, which is rewritten there) — no summarizing, truncating or
rewriting in v1 (PRD §26). Every response carries an ``x_router`` block with
request/session/task/route metadata and escalation context (PRD §27).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from . import __version__
from .backends import build_backends
from .backends.base import Backend, BackendError, BackendResult
from .config import AppConfig
from .escalation import EscalationController
from .events import handle_event
from .metrics import MetricsRegistry, register_router_metrics
from .models import EventIn, build_escalation_required, error_body, x_router_meta
from .routing import ALIAS_TO_TIER, CloudBudget, Router, UnknownModel
from .sessions import SessionState, SessionStore, TaskState

log = logging.getLogger("router.api")

SENSITIVE_HEADERS = {"authorization", "x-api-key"}


def mask_headers(headers: dict[str, str]) -> dict[str, str]:
    """Copy of a header map with secret values masked (PRD §9, §32)."""
    return {k: ("***" if k.lower() in SENSITIVE_HEADERS else v) for k, v in headers.items()}


def create_app(cfg: AppConfig) -> FastAPI:
    backends = build_backends(cfg)
    store = SessionStore(default_route=cfg.routing.default)
    controller = EscalationController(cfg.routing.escalation, cfg.cloud)
    router = Router(cfg, backends, controller)
    registry = MetricsRegistry()
    M = register_router_metrics(registry)
    cloud = CloudBudget(cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        for b in backends.values():
            await b.close()

    app = FastAPI(title="llm-router", version=__version__, lifespan=lifespan)

    # ------------------------------------------------------------------ /v1/models

    @app.get("/v1/models")
    async def list_models() -> Any:
        data = [
            {"id": alias, "object": "model", "created": 0, "owned_by": tier}
            for alias, tier in (
                ("auto", cfg.routing.default),
                ("local-fast", "fast"),
                ("local-deep", "deep"),
                ("frontier", "frontier"),
            )
        ]
        return {"object": "list", "data": data}

    # ------------------------------------------------------------------ health

    async def _backend_health() -> dict[str, Any]:
        results: dict[str, Any] = {}
        for tier in sorted(backends):
            healthy, latency_ms = await backends[tier].health_check()
            results[tier] = {"healthy": healthy, "latency_ms": round(latency_ms, 1)}
        return results

    @app.get("/health")
    async def health() -> Any:
        backends_status = await _backend_health()
        status = "ok" if all(b["healthy"] for b in backends_status.values()) else "degraded"
        return {"status": status, "backends": backends_status}

    @app.get("/health/backends")
    async def health_backends() -> Any:
        return await _backend_health()

    # ------------------------------------------------------------------ metrics

    @app.get("/metrics")
    async def metrics() -> Response:
        if not cfg.metrics.enabled:
            return Response(status_code=204)
        return PlainTextResponse(registry.render(), media_type="text/plain; version=0.0.4")

    # ------------------------------------------------------------------ events

    @app.post("/events")
    async def post_events(request: Request) -> Any:
        try:
            payload = EventIn.model_validate(await request.json())
        except Exception as e:  # pydantic ValidationError or bad JSON
            return JSONResponse(status_code=400, content=error_body(f"invalid event payload: {e}"))
        return handle_event(store, controller, registry, payload)

    # ------------------------------------------------------------------ chat completions

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        request_id = uuid.uuid4().hex[:8]
        lc_headers = {k.lower(): v for k, v in request.headers.items()}

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content=error_body("request body must be valid JSON"))
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return JSONResponse(
                status_code=400,
                content=error_body("'messages' (a non-empty list) is required"),
            )

        session = store.get_or_create_session(lc_headers.get("x-llm-session-id"))
        task_id_hdr = lc_headers.get("x-llm-task-id")
        if task_id_hdr:
            task_id = task_id_hdr
        else:
            session.request_seq += 1
            task_id = f"{session.session_id}-r{session.request_seq}"
        task = store.get_or_create_task(session, task_id)

        t0 = time.perf_counter()
        stream = bool(body.get("stream", False))

        if cfg.logging.log_requests:
            log.info(
                "request=%s session=%s task=%s model=%s stream=%s headers=%s",
                request_id, session.session_id, task_id, body.get("model"), stream, mask_headers(lc_headers),
            )
        if cfg.logging.log_prompts:
            log.debug("prompt messages=%d first=%r", len(body["messages"]), str(body["messages"][0])[:500])

        # -- route resolution (priority order, PRD §7) -------------------------
        try:
            decision = router.resolve(
                model=body.get("model"), headers=lc_headers, session=session, task=task
            )
        except UnknownModel as e:
            return JSONResponse(
                status_code=404,
                content=error_body(
                    f"unknown model {e.args[0]!r}; expected one of auto|local-fast|local-deep|frontier",
                    "model_not_found",
                ),
            )

        decision = router.apply_attempt(decision, session, task)
        esc_dict = decision.escalation.as_dict() if decision.escalation else None

        def _meta(tier: str, backend_name: str, reason: str, escalation: Optional[dict]) -> dict[str, Any]:
            return x_router_meta(
                request_id=request_id,
                session_id=session.session_id,
                task_id=task_id,
                tier=tier,
                backend_name=backend_name,
                reason=reason,
                escalation=escalation,
            )

        # -- cloud gate for frontier (PRD §21/§22) ------------------------------
        if decision.tier == "frontier":
            allowed, why = cloud.allows(manual=decision.manual)
            if not allowed:
                M["cloud_blocked_total"].inc(reason=why or "unknown")
                x_router = _meta(
                    "frontier", backends["frontier"].name, decision.reason,
                    {"required": True, "to": "frontier", "reason": why},
                )
                resp_body = build_escalation_required(
                    request_id=request_id, model_alias=body.get("model"), x_router=x_router
                )
                _finish(request_id, session.session_id, task_id, "frontier", backends["frontier"].name, t0, 200)
                return JSONResponse(status_code=200, content=resp_body)

        # -- dispatch with backend fallback (PRD §31; NOT an escalation) --------
        result: Optional[BackendResult] = None
        serving: Optional[Backend] = None
        last_error: Optional[BackendError] = None
        for candidate in _candidates(decision.tier):
            M["backend_requests_total"].inc(backend=candidate.name)
            if candidate.tier == "frontier":
                M["cloud_requests_total"].inc()
            try:
                result = await candidate.chat_completion(body, stream=stream)
                serving = candidate
                break
            except BackendError as e:
                last_error = e
                M["backend_errors_total"].inc(backend=candidate.name, kind=e.kind)
                log.warning("backend %s failed (kind=%s): %s", candidate.name, e.kind, e)

        if result is None or serving is None:
            assert last_error is not None
            store.mark_error(session, str(last_error))
            _note_request_failure(session, task, serving_tier=decision.tier, manual=decision.manual, kind=last_error.kind)
            x_router = _meta(decision.tier, backends[decision.tier].name, "backend_unavailable", esc_dict)
            M["requests_total"].inc(route=decision.tier, status="error")
            M["request_latency_seconds"].observe(time.perf_counter() - t0, route=decision.tier)
            return JSONResponse(
                status_code=502,
                content={**error_body(f"backend '{decision.tier}' unavailable: {last_error}", "backend_error"), "x_router": x_router},
            )

        # A non-2xx from the backend (e.g. 400) is forwarded to the client; if
        # the backend_error signal is enabled it also counts toward rule D.
        if result.status_code >= 400:
            _note_request_failure(session, task, serving_tier=serving.tier, manual=decision.manual, kind="backend_error")

        store.record_request(session, task, serving.tier, serving.name)

        # -- response -----------------------------------------------------------
        served_reason = decision.reason if serving.tier == decision.tier else "backend_fallback"
        x_router = _meta(serving.tier, serving.name, served_reason, esc_dict)
        latency = time.perf_counter() - t0

        # A backend may answer a streaming request with a plain (non-2xx) JSON
        # body; in that case result.stream is None and we fall through to the
        # non-streaming forwarding path below.
        if stream and result.stream is not None:
            meta_chunk = {
                "id": f"router-meta-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": serving.name,
                "choices": [],
                "x_router": x_router,
            }
            is_frontier = serving.tier == "frontier"

            async def gen() -> AsyncIterator[bytes]:
                usage: Optional[dict] = None
                done_seen = False
                async for line in result.stream:  # type: ignore[union-attr]
                    if is_frontier:
                        u = _usage_from_sse_line(line)
                        if u is not None:
                            usage = u
                    text = line.decode("utf-8", "replace") if isinstance(line, (bytes, bytearray)) else str(line)
                    if text.strip() == "data: [DONE]":
                        yield f"data: {json.dumps(meta_chunk)}\n\n".encode()
                        done_seen = True
                    yield line
                if not done_seen:  # backend did not terminate the stream properly
                    yield f"data: {json.dumps(meta_chunk)}\n\n".encode()
                if is_frontier:
                    cloud.record(usage)

            M["requests_total"].inc(route=serving.tier, status="ok")
            M["request_latency_seconds"].observe(latency, route=serving.tier)
            _finish(request_id, session.session_id, task_id, serving.tier, serving.name, t0, 200)
            return StreamingResponse(
                gen(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-streaming: forward the JSON body and attach x_router.
        if serving.tier == "frontier":
            cloud.record(result.usage)  # count rate/cost for this request now
        status_code = result.status_code
        try:
            data = json.loads(result.body or b"{}")
            if isinstance(data, dict):
                data["x_router"] = x_router
                out_bytes = json.dumps(data).encode()
            else:  # pragma: no cover - defensive
                out_bytes = result.body or b"null"
        except (ValueError, UnicodeDecodeError):
            out_bytes = result.body or b"null"

        if cfg.logging.log_responses and status_code == 200:
            log.debug("response body=%s", out_bytes[:2000].decode("utf-8", "replace"))

        M["requests_total"].inc(route=serving.tier, status="ok" if status_code < 400 else "error")
        M["request_latency_seconds"].observe(latency, route=serving.tier)
        _finish(request_id, session.session_id, task_id, serving.tier, serving.name, t0, status_code)

        return Response(content=out_bytes, status_code=status_code, media_type="application/json")

    # -- helpers -----------------------------------------------------------------

    def _candidates(tier: str):
        """Primary backend plus the configured fallback chain (one level)."""
        yield backends[tier]
        for fb in cfg.routing.fallbacks.get(tier, []):
            if fb == "frontier" and not cfg.cloud.enabled:
                continue  # cloud fallback must be explicitly enabled
            yield backends[fb]

    def _note_request_failure(
        session: SessionState, task: TaskState, *, serving_tier: str, manual: bool, kind: str
    ) -> None:
        """Count a request-path failure toward rule D when the signal is enabled."""
        if manual or serving_tier == "frontier":
            return
        sig = cfg.routing.escalation.signals
        if (kind == "timeout" and sig.timeout) or (kind != "timeout" and sig.backend_error):
            controller.note_failure(session, task, kind)

    def _finish(request_id: str, session_id: str, task_id: str, tier: str, backend_name: str, t0: float, status: int) -> None:
        if cfg.logging.log_requests:
            log.info(
                "request=%s session=%s task=%s route=%s backend=%s latency=%.2fs status=%d",
                request_id, session_id, task_id, tier, backend_name, time.perf_counter() - t0, status,
            )

    return app


def _usage_from_sse_line(line: bytes | str) -> Optional[dict]:
    """Best-effort extraction of a ``usage`` object from one SSE line."""
    text = line.decode("utf-8", "replace") if isinstance(line, (bytes, bytearray)) else str(line)
    text = text.strip()
    if not text.startswith("data:"):
        return None
    try:
        obj = json.loads(text[len("data:"):].strip())
    except ValueError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("usage"), dict):
        return obj["usage"]
    return None
