"""EscalationController (PRD §18).

Keeps task state and decides, with deterministic signals only:

    stay | escalate to deep | escalate to frontier

It never performs inference. The rules (PRD §19):

* **A — explicit request**: a client header or ``explicit_escalation`` event
  selects the requested tier immediately.
* **B — repeated test failure**: N test failures within one task on the
  current tier escalate one step up the chain.
* **C — repeated tool failure**: same, for tool failures (default N = 2).
* **D — retry limit**: if a task has made ``max_*_attempts`` requests to its
  current tier *and* at least one failure signal was recorded, escalate.

Model self-assessment is deliberately NOT used (PRD §20): the architecture
works without the model knowing it is struggling.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from .config import CloudConfig, EscalationConfig
from .sessions import SessionState, TaskState

log = logging.getLogger("router.escalation")

REASON_EXPLICIT = "explicit_request"
REASON_TEST_FAILURE = "repeated_test_failure"
REASON_TOOL_FAILURE = "repeated_tool_failure"
REASON_RETRY_LIMIT = "retry_limit"
REASON_BACKEND_ERROR = "backend_error"
REASON_TIMEOUT = "timeout"

#: Events that count as a failure signal for rule D's "without success" check.
FAILURE_EVENTS = {
    "tool_failure": "tool",
    "test_failure": "test",
    "task_failure": "task",
}


@dataclass(frozen=True)
class Escalation:
    from_tier: str | None
    to_tier: str
    reason: str
    count: int  # failure/attempt count that triggered it

    def as_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_tier,
            "to": self.to_tier,
            "reason": self.reason,
            "attempts": self.count,
        }


class EscalationController:
    def __init__(self, esc_cfg: EscalationConfig, cloud_cfg: CloudConfig) -> None:
        self._esc = esc_cfg
        self._cloud = cloud_cfg

    # -- chain helpers --------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._esc.enabled

    @property
    def explicit_requests_enabled(self) -> bool:
        """Rule A is active only when the controller and its signal are both on."""
        return self.enabled and self._esc.signals.explicit_request

    def next_tier(self, tier: str) -> Optional[str]:
        """Next tier up the configured chain; ``None`` at the top or when frontier is disallowed."""
        chain = self._esc.chain
        if tier not in chain:
            return None
        nxt = chain[chain.index(tier) + 1] if chain.index(tier) + 1 < len(chain) else None
        if nxt == "frontier" and not self._esc.allow_frontier:
            return None
        return nxt

    def _max_attempts(self, tier: str) -> Optional[int]:
        if tier == "fast":
            return self._esc.max_fast_attempts
        if tier == "deep":
            return self._esc.max_deep_attempts
        return None  # frontier is the top of the chain by default

    # -- events -----------------------------------------------------------------

    def record_event(
        self, session: SessionState, task: TaskState, event: str, metadata: Optional[dict[str, Any]] = None
    ) -> Optional[Escalation]:
        """Apply one lifecycle event (from POST /events). Returns an Escalation if the route changed."""
        metadata = metadata or {}

        if event in ("task_start", "task_complete"):
            task.failures.clear()
            task.attempts.clear()
            task.done = event == "task_complete"
            task.touch()
            return None

        if event == "explicit_escalation":
            if not self.explicit_requests_enabled:
                return None  # rule A switched off via signals.explicit_request
            target = metadata.get("target") or self.next_tier(task.current_route)
            if target and target != task.current_route:
                return self.escalate(session, task, target, REASON_EXPLICIT)
            return None

        kind = FAILURE_EVENTS.get(event)
        if kind is not None:
            task.failures[kind] = task.failures.get(kind, 0) + 1
            task.touch()
            log.debug(
                "failure event session=%s task=%s kind=%s count=%d",
                session.session_id, task.task_id, kind, task.failures[kind],
            )
        else:
            # Unknown events are accepted and logged at debug level (PRD §17).
            log.debug("unknown event %r accepted session=%s task=%s", event, session.session_id, task.task_id)

        return self._check_rules(session, task)

    def note_failure(self, session: SessionState, task: TaskState, kind: str) -> None:
        """Record a failure observed from the request path itself (backend error/timeout)."""
        if not self.enabled:
            return
        task.failures[kind] = task.failures.get(kind, 0) + 1
        task.touch()

    # -- per-request accounting ---------------------------------------------------

    def note_attempt(self, session: SessionState, task: TaskState, tier: str) -> Optional[Escalation]:
        """Count one auto-routed request to ``tier``; may trigger rule D."""
        if not self.enabled:
            return None
        task.attempts[tier] = task.attempts.get(tier, 0) + 1
        task.touch()
        return self._check_rules(session, task)

    # -- rules ----------------------------------------------------------------------

    def _check_rules(self, session: SessionState, task: TaskState) -> Optional[Escalation]:
        if not self.enabled or task.done:
            return None
        tier = task.current_route
        target = self.next_tier(tier)
        if target is None:
            return None

        sig = self._esc.signals
        threshold = self._esc.failure_threshold

        # Rule B — repeated test failure.
        if sig.repeated_test_failure and task.failures.get("test", 0) >= threshold:
            return self.escalate(session, task, target, REASON_TEST_FAILURE)

        # Rule C — repeated tool failure.
        if sig.repeated_tool_failure and task.failures.get("tool", 0) >= threshold:
            return self.escalate(session, task, target, REASON_TOOL_FAILURE)

        # Rule D — retry limit: max attempts on this tier AND at least one
        # recorded failure signal ("retries without success").
        max_att = self._max_attempts(tier)
        total_failures = sum(task.failures.values())
        if (
            max_att is not None
            and task.attempts.get(tier, 0) >= max_att
            and total_failures > 0
        ):
            return self.escalate(session, task, target, REASON_RETRY_LIMIT)

        return None

    # -- escalation -------------------------------------------------------------------

    def escalate(
        self, session: SessionState, task: TaskState, to_tier: str, reason: str
    ) -> Escalation:
        count = sum(task.failures.values()) or task.attempts.get(task.current_route, 0)
        esc = Escalation(task.current_route, to_tier, reason, count)

        task.escalation_history.append(
            {**esc.as_dict(), "at": time.time()}
        )
        from_tier = task.current_route
        task.current_route = to_tier
        # Failure counters are per-tier: a fresh start on the stronger model.
        task.failures.clear()
        task.touch()

        session.current_route = to_tier
        session.escalation_count += 1
        session.touch()

        log.info(
            "ESCALATION session=%s task=%s from=%s to=%s reason=%s count=%d",
            session.session_id, task.task_id, from_tier, to_tier, reason, count,
        )
        return esc
