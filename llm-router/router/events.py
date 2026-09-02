"""Lifecycle event handling (PRD §17).

Claude Code hooks (or any client) POST events to ``/events``:

    {"session_id": "abc", "task_id": "task-123", "event": "test_failure",
     "metadata": {"command": "pytest", "exit_code": 1}}

The EscalationController consumes them; unknown events are accepted and logged
at debug level rather than crashing the hook.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .escalation import EscalationController
from .metrics import MetricsRegistry
from .models import EventIn
from .sessions import SessionStore

log = logging.getLogger("router.events")

KNOWN_EVENTS = {
    "task_start",
    "task_complete",
    "task_failure",
    "tool_failure",
    "test_failure",
    "explicit_escalation",
}


def handle_event(
    store: SessionStore,
    controller: EscalationController,
    registry: MetricsRegistry,
    payload: EventIn,
) -> dict[str, Any]:
    session = store.get_or_create_session(payload.session_id)
    task_id = payload.task_id or f"{session.session_id}-default"
    task = store.get_or_create_task(session, task_id)

    if payload.event not in KNOWN_EVENTS:
        log.debug("unknown event %r accepted (session=%s task=%s)", payload.event, session.session_id, task_id)

    esc = controller.record_event(session, task, payload.event, payload.metadata)

    # Metrics (low-cardinality labels only).
    if payload.event == "tool_failure":
        registry.counter("router_tool_failures_total", "Tool failure events received via /events.").inc()
    elif payload.event == "test_failure":
        registry.counter("router_test_failures_total", "Test failure events received via /events.").inc()
    if esc is not None:
        registry.counter(
            "router_escalations_total", "Route escalations, by from/to tier and reason."
        ).inc(**{"from": esc.from_tier or "none", "to": esc.to_tier, "reason": esc.reason})

    return {
        "status": "ok",
        "event": payload.event,
        "session_id": session.session_id,
        "task_id": task.task_id,
        "route": task.current_route,
        "escalated": esc is not None,
    }
