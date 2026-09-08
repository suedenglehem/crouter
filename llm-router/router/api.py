"""FastAPI application (PRD §5, §30–33).

Endpoints:

    POST /v1/chat/completions   OpenAI-compatible chat completions (streaming + non-streaming)
    POST /v1/messages           Anthropic Messages API for Claude Code (PRD §51, streaming + non-streaming)
    GET  /v1/models             logical model aliases
    GET  /health                overall status + per-backend health
    GET  /health/backends       per-backend health only
    GET  /metrics               Prometheus text format
    POST /events                lifecycle events from Claude Code hooks
    GET  /ctxlen                current context-length bounds per tier (admin)
    GET  /ctxlen/<tier>=<n|reset>  set/reset a tier's max_context at runtime (admin)
    GET  /route                 current routing mode (pin) + bounds (admin)
    GET  /route/all/<tier>      pin ALL traffic to one tier at its max window (admin)
    GET  /route/reset           back to defaults: no pin, startup bounds (admin)
    GET  /route/last            restore the pre-pin context bounds (admin)

The request body is proxied to the selected backend verbatim (except the
``model`` field, which is rewritten there) — no summarizing, truncating or
rewriting in v1 (PRD §26). ``/v1/messages`` is the exception: it translates
Anthropic <-> OpenAI so Claude Code can run its whole agent loop through the
router. Every response carries an ``x_router`` block with request/session/task/
route metadata and escalation context (PRD §27).
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
from .anthropic import (
    anthropic_error,
    anthropic_stream,
    build_escalation_required_message,
    to_anthropic_message,
    to_openai_request,
    wants_thinking,
)
from .backends import build_backends
from .backends.base import Backend, BackendError, BackendResult
from .config import AppConfig
from .context import estimate_prompt_tokens, required_context
from .escalation import EscalationController
from .events import handle_event
from .metrics import MetricsRegistry, register_router_metrics
from .models import EventIn, build_escalation_required, error_body, x_router_meta
from .routing import ALIAS_TO_TIER, CloudBudget, REASON_CONTEXT_OVERFLOW, Router, UnknownModel
from .sessions import SessionState, SessionStore, TaskState

log = logging.getLogger("router.api")

SENSITIVE_HEADERS = {"authorization", "x-api-key"}


def mask_headers(headers: dict[str, str]) -> dict[str, str]:
    """Copy of a header map with secret values masked (PRD §9, §32)."""
    return {k: ("***" if k.lower() in SENSITIVE_HEADERS else v) for k, v in headers.items()}


async def sync_context_sizes(cfg: AppConfig, backends: dict[str, Backend]) -> None:
    """Apply ``query_context_size`` overrides at startup (PRD §52).

    Backends flagged with ``query_context_size: true`` report their real
    context window; that value replaces the manual ``max_context`` in config,
    which may be stale. A failed query keeps the manual value.
    """
    for tier, bcfg in cfg.backends.items():
        if not bcfg.query_context_size:
            continue
        n_ctx = await backends[tier].query_context_size()
        if n_ctx is None:
            log.warning(
                "backend %s: context-size query failed; keeping configured max_context=%s",
                tier, bcfg.max_context,
            )
            continue
        if n_ctx != bcfg.max_context:
            log.info(
                "backend %s: using queried context size %d (config had %s)",
                tier, n_ctx, bcfg.max_context,
            )
        bcfg.max_context = n_ctx


def create_app(cfg: AppConfig) -> FastAPI:
    backends = build_backends(cfg)
    store = SessionStore(default_route=cfg.routing.default)
    controller = EscalationController(cfg.routing.escalation, cfg.cloud)
    router = Router(cfg, backends, controller)
    registry = MetricsRegistry()
    M = register_router_metrics(registry)
    cloud = CloudBudget(cfg)

    # Effective max_context per tier as of startup — what ``/ctxlen/<tier>=reset``
    # restores. Seeded from the config and re-snapshotted after the
    # query_context_size sync, so "initial" means the value the router actually
    # started serving with (queried size wins over a stale manual one).
    initial_ctx: dict[str, Optional[int]] = {t: b.max_context for t, b in cfg.backends.items()}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await sync_context_sizes(cfg, backends)
        initial_ctx.update({t: b.max_context for t, b in cfg.backends.items()})
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

    # ------------------------------------------------------------------ ctxlen (admin)

    @app.get("/ctxlen")
    async def get_ctxlen() -> Any:
        """Current context-length bounds per tier, plus the startup baseline."""
        return {
            tier: {"max_context": bcfg.max_context, "initial_max_context": initial_ctx.get(tier)}
            for tier, bcfg in cfg.backends.items()
        }

    @app.get("/ctxlen/{spec}")
    async def set_ctxlen(spec: str) -> Any:
        """Set or reset a tier's context-length bound at runtime.

        ``GET /ctxlen/fast=32000`` sets fast's max_context to 32000;
        ``GET /ctxlen/deep=reset`` restores deep's startup value (the config
        value, or the queried size when query_context_size is on). The context
        floor reads these live per request, so a change applies from the very
        next routed request. In-memory only — restarting the router reverts to
        the configuration file.
        """
        tier, sep, value = spec.partition("=")
        if not sep:
            return JSONResponse(
                status_code=400, content={"error": f"expected <tier>=<tokens|reset>, got {spec!r}"}
            )
        bcfg = cfg.backends.get(tier)
        if bcfg is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"unknown tier {tier!r}; expected one of {sorted(cfg.backends)}"},
            )
        old = bcfg.max_context
        if value == "reset":
            new = initial_ctx.get(tier)  # None when the config set no bound
        else:
            try:
                new = int(value)
            except ValueError:
                return JSONResponse(
                    status_code=400, content={"error": f"invalid token count {value!r} for tier {tier!r}"}
                )
            if new <= 0:
                return JSONResponse(status_code=400, content={"error": f"token count must be > 0, got {new}"})
        bcfg.max_context = new
        log.info("ctxlen %s: max_context %s -> %s", tier, old, new)
        return {"tier": tier, "max_context": new, "previous_max_context": old}

    # ------------------------------------------------------------------ route (admin)

    #: Authoritative pin state (GET /route/all/<tier>): every request goes to
    #: one tier at its maximum known window until reset/last. In-memory only —
    #: restarting the router clears the pin and reverts bounds to the config.
    route_state: dict[str, Any] = {"pin": None, "saved_bounds": None}

    def _route_view() -> dict[str, Any]:
        return {
            "pin": route_state["pin"],
            "bounds": {t: b.max_context for t, b in cfg.backends.items()},
            "initial_bounds": initial_ctx,
        }

    @app.get("/route")
    async def get_route() -> Any:
        """Current routing mode (pin) and context bounds per tier."""
        return _route_view()

    @app.get("/route/all/{tier}")
    async def route_all(tier: str) -> Any:
        """Authoritatively pin ALL traffic to one tier at its maximum window.

        Saves the current context bounds first (restored by ``/route/last``),
        sets the pinned tier's bound to its startup value — the config number,
        or the queried size when query_context_size overrode it: the largest
        window the router knows about — and forces every request onto that
        tier until ``/route/reset`` or ``/route/last``. While pinned, the
        context floor is skipped (reason ``pinned``), so even a prompt bigger
        than the window gets the backend's own precise error instead of a
        silent bump away from the pinned tier. Re-pinning to another tier
        keeps the original pre-pin snapshot.
        """
        if tier not in cfg.backends:
            return JSONResponse(
                status_code=404,
                content={"error": f"unknown tier {tier!r}; expected one of {sorted(cfg.backends)}"},
            )
        if route_state["pin"] is None:
            # Snapshot the pre-pin bounds once; re-pinning keeps the original.
            route_state["saved_bounds"] = {t: b.max_context for t, b in cfg.backends.items()}
        old_pin = route_state["pin"]
        route_state["pin"] = tier
        cfg.backends[tier].max_context = initial_ctx.get(tier)
        log.info("route pin: %s -> all/%s", old_pin or "none", tier)
        return _route_view()

    @app.get("/route/reset")
    async def route_reset() -> Any:
        """Back to the default configuration: no pin, bounds at startup values."""
        for t, b in cfg.backends.items():
            b.max_context = initial_ctx.get(t)
        route_state["pin"] = None
        route_state["saved_bounds"] = None
        log.info("route reset: pin cleared, bounds restored to startup defaults")
        return _route_view()

    @app.get("/route/last")
    async def route_last() -> Any:
        """Restore the context bounds in effect right before /route/all/<tier>."""
        if route_state["saved_bounds"] is None:
            return JSONResponse(
                status_code=400,
                content={"error": "no previous state to restore (no /route/all/... was issued)"},
            )
        for t, v in route_state["saved_bounds"].items():
            if t in cfg.backends:
                cfg.backends[t].max_context = v
        route_state["pin"] = None
        route_state["saved_bounds"] = None
        log.info("route last: restored pre-pin bounds")
        return _route_view()

    # ------------------------------------------------------------------ shared dispatch

    async def _dispatch(
        *,
        request_id: str,
        lc_headers: dict[str, str],
        openai_body: dict[str, Any],
        stream: bool,
    ) -> dict[str, Any]:
        """Route resolution + backend dispatch, shared by both API endpoints.

        ``openai_body`` is what the backend will receive (the OpenAI endpoint
        passes the client body through; /v1/messages passes its translation).
        Returns a dict whose ``kind`` is one of:

          * ``"unknown_model"`` — model not in the alias table -> 404
          * ``"escalation"``    — frontier cloud gate blocked -> 200 sentinel body
          * ``"error"``         — every backend for the tier failed -> 502
          * ``"ok"``            — ``result``/``serving``/``x_router`` ready to render

        Session identity: ``X-LLM-Session-ID``, falling back to Claude Code's
        own ``X-Claude-Code-Session-ID`` so escalation state works out of the
        box when ANTHROPIC_BASE_URL points here (PRD §17, §51).
        """
        session_key = lc_headers.get("x-llm-session-id") or lc_headers.get("x-claude-code-session-id")
        session = store.get_or_create_session(session_key)
        task_id_hdr = lc_headers.get("x-llm-task-id")
        if task_id_hdr:
            task_id = task_id_hdr
        else:
            session.request_seq += 1
            task_id = f"{session.session_id}-r{session.request_seq}"
        task = store.get_or_create_task(session, task_id)

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

        if cfg.logging.log_requests:
            log.info(
                "request=%s session=%s task=%s model=%s stream=%s headers=%s",
                request_id, session.session_id, task_id, openai_body.get("model"), stream, mask_headers(lc_headers),
            )
        if cfg.logging.log_prompts:
            log.debug("prompt messages=%d first=%r", len(openai_body["messages"]), str(openai_body["messages"][0])[:500])

        # -- route resolution (priority order, PRD §7 + context floor) ---------
        required_ctx = required_context(openai_body, cfg.routing.context)
        try:
            decision = router.resolve(
                model=openai_body.get("model"), headers=lc_headers, session=session, task=task,
                required_context=required_ctx,
                pinned_tier=route_state["pin"],
            )
        except UnknownModel as e:
            return {"kind": "unknown_model", "model": e.args[0], "session_id": session.session_id, "task_id": task_id}

        decision = router.apply_attempt(decision, session, task)
        esc_dict = decision.escalation.as_dict() if decision.escalation else None

        # Context-floor bumps are route escalations too (PRD §33).
        if decision.reason == REASON_CONTEXT_OVERFLOW and decision.escalation is not None:
            M["escalations_total"].inc(
                **{
                    "from": decision.escalation.from_tier or "none",
                    "to": decision.tier,
                    "reason": REASON_CONTEXT_OVERFLOW,
                }
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
                return {
                    "kind": "escalation", "x_router": x_router,
                    "session_id": session.session_id, "task_id": task_id,
                }

        # -- dispatch with backend fallback (PRD §31; NOT an escalation) --------
        result: Optional[BackendResult] = None
        serving: Optional[Backend] = None
        last_error: Optional[BackendError] = None
        for candidate in _candidates(decision.tier):
            M["backend_requests_total"].inc(backend=candidate.name)
            if candidate.tier == "frontier":
                M["cloud_requests_total"].inc()
            try:
                result = await candidate.chat_completion(openai_body, stream=stream)
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
            return {
                "kind": "error", "tier": decision.tier,
                "message": f"backend '{decision.tier}' unavailable: {last_error}",
                "x_router": x_router,
                "session_id": session.session_id, "task_id": task_id,
            }

        # A non-2xx from the backend (e.g. 400) is forwarded to the client; if
        # the backend_error signal is enabled it also counts toward rule D.
        if result.status_code >= 400:
            _note_request_failure(session, task, serving_tier=serving.tier, manual=decision.manual, kind="backend_error")

        store.record_request(session, task, serving.tier, serving.name)

        # -- response metadata ---------------------------------------------------
        served_reason = decision.reason if serving.tier == decision.tier else "backend_fallback"
        x_router = _meta(serving.tier, serving.name, served_reason, esc_dict)
        return {
            "kind": "ok", "result": result, "serving": serving, "x_router": x_router,
            "session_id": session.session_id, "task_id": task_id,
        }

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

        t0 = time.perf_counter()
        stream = bool(body.get("stream", False))

        outcome = await _dispatch(request_id=request_id, lc_headers=lc_headers, openai_body=body, stream=stream)

        if outcome["kind"] == "unknown_model":
            return JSONResponse(
                status_code=404,
                content=error_body(
                    f"unknown model {outcome['model']!r}; expected one of auto|local-fast|local-deep|frontier",
                    "model_not_found",
                ),
            )

        if outcome["kind"] == "escalation":
            x_router = outcome["x_router"]
            resp_body = build_escalation_required(
                request_id=request_id, model_alias=body.get("model"), x_router=x_router
            )
            _finish(request_id, outcome["session_id"], outcome["task_id"], "frontier", backends["frontier"].name, t0, 200)
            return JSONResponse(status_code=200, content=resp_body)

        if outcome["kind"] == "error":
            M["requests_total"].inc(route=outcome["tier"], status="error")
            M["request_latency_seconds"].observe(time.perf_counter() - t0, route=outcome["tier"])
            return JSONResponse(
                status_code=502,
                content={**error_body(outcome["message"], "backend_error"), "x_router": outcome["x_router"]},
            )

        result: BackendResult = outcome["result"]
        serving: Backend = outcome["serving"]
        x_router = outcome["x_router"]

        # -- response -----------------------------------------------------------
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
            _finish(request_id, outcome["session_id"], outcome["task_id"], serving.tier, serving.name, t0, 200)
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
        _finish(request_id, outcome["session_id"], outcome["task_id"], serving.tier, serving.name, t0, status_code)

        return Response(content=out_bytes, status_code=status_code, media_type="application/json")

    # ------------------------------------------------------------------ messages (Anthropic)

    @app.post("/v1/messages")
    async def create_message(request: Request) -> Response:
        """Anthropic Messages API for Claude Code (PRD §51).

        Translates the request to OpenAI chat-completions, runs it through the
        same routing/escalation machinery as /v1/chat/completions, and
        translates the response back — including the streaming event protocol.
        """
        request_id = uuid.uuid4().hex[:8]
        lc_headers = {k.lower(): v for k, v in request.headers.items()}

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content=anthropic_error("request body must be valid JSON"))
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return JSONResponse(
                status_code=400,
                content=anthropic_error("'messages' (a non-empty list) is required"),
            )

        t0 = time.perf_counter()
        stream = bool(body.get("stream", False))
        openai_body = to_openai_request(body)
        thinking = wants_thinking(body)
        model_alias = body.get("model") or "auto"

        outcome = await _dispatch(request_id=request_id, lc_headers=lc_headers, openai_body=openai_body, stream=stream)

        if outcome["kind"] == "unknown_model":
            return JSONResponse(
                status_code=404,
                content=anthropic_error(
                    f"unknown model {outcome['model']!r}; expected one of auto|local-fast|local-deep|frontier",
                    "model_not_found",
                ),
            )

        if outcome["kind"] == "escalation":
            x_router = outcome["x_router"]
            resp_body = build_escalation_required_message(
                request_id=request_id, model_alias=body.get("model"), x_router=x_router
            )
            _finish(request_id, outcome["session_id"], outcome["task_id"], "frontier", backends["frontier"].name, t0, 200)
            return JSONResponse(status_code=200, content=resp_body)

        if outcome["kind"] == "error":
            M["requests_total"].inc(route=outcome["tier"], status="error")
            M["request_latency_seconds"].observe(time.perf_counter() - t0, route=outcome["tier"])
            return JSONResponse(
                status_code=502,
                content={**anthropic_error(outcome["message"], "backend_error"), "x_router": outcome["x_router"]},
            )

        result: BackendResult = outcome["result"]
        serving: Backend = outcome["serving"]
        x_router = outcome["x_router"]
        latency = time.perf_counter() - t0

        # A backend may answer a streaming request with a plain (non-2xx) JSON
        # body; in that case result.stream is None and we fall through to the
        # non-streaming rendering below.
        if stream and result.stream is not None:
            est_input = estimate_prompt_tokens(openai_body, cfg.routing.context.chars_per_token)

            async def gen() -> AsyncIterator[bytes]:
                async for frame in anthropic_stream(
                    result.stream,  # type: ignore[arg-type]
                    model_alias=model_alias,
                    thinking=thinking,
                    estimated_input_tokens=est_input,
                    x_router=x_router,
                ):
                    yield frame.encode("utf-8")

            M["requests_total"].inc(route=serving.tier, status="ok")
            M["request_latency_seconds"].observe(latency, route=serving.tier)
            _finish(request_id, outcome["session_id"], outcome["task_id"], serving.tier, serving.name, t0, 200)
            return StreamingResponse(
                gen(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-streaming rendering.
        if serving.tier == "frontier":
            cloud.record(result.usage)  # count rate/cost for this request now
        status_code = result.status_code
        try:
            data = json.loads(result.body or b"{}")
        except (ValueError, UnicodeDecodeError):
            data = None

        if isinstance(data, dict) and isinstance(data.get("choices"), list):
            out = to_anthropic_message(data, model_alias=model_alias, thinking=thinking)
            out["x_router"] = x_router
            out_bytes = json.dumps(out).encode()
        else:  # backend error body (or non-JSON) -> Anthropic error envelope
            err = data.get("error") if isinstance(data, dict) else None
            message = (
                err.get("message")
                if isinstance(err, dict) and isinstance(err.get("message"), str)
                else f"backend error (HTTP {status_code})"
            )
            out_bytes = json.dumps({**anthropic_error(message), "x_router": x_router}).encode()

        if cfg.logging.log_responses and status_code == 200:
            log.debug("response body=%s", out_bytes[:2000].decode("utf-8", "replace"))

        M["requests_total"].inc(route=serving.tier, status="ok" if status_code < 400 else "error")
        M["request_latency_seconds"].observe(latency, route=serving.tier)
        _finish(request_id, outcome["session_id"], outcome["task_id"], serving.tier, serving.name, t0, status_code)

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
