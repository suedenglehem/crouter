"""Lightweight in-memory session and task state (PRD §13, §14).

A *session* groups related requests (one Claude Code conversation); a *task*
is one unit of work inside a session. Escalation decisions are made per task,
so an escalated Task A does not drag the whole session up to a stronger model:

    Task A -> escalates to deep
    Task B -> remains on fast

No conversation content is stored — only counters and timestamps (PRD §13).
State is in-memory by design; it is lost when the router restarts, which is
acceptable for v1. Idle sessions are swept lazily after ``ttl_seconds``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


def _now() -> float:
    return time.time()


@dataclass
class TaskState:
    task_id: str
    session_id: str
    #: Tier the task is currently routed to (moves up as it escalates).
    current_route: str
    #: tier -> number of requests dispatched to that tier for this task.
    attempts: dict[str, int] = field(default_factory=dict)
    #: failure kind (test | tool | task | backend_error | timeout) -> count on the current tier.
    failures: dict[str, int] = field(default_factory=dict)
    escalation_history: list[dict] = field(default_factory=list)
    done: bool = False
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    def touch(self) -> None:
        self.updated_at = _now()


@dataclass
class SessionState:
    session_id: str
    current_route: str
    fast_attempts: int = 0
    deep_attempts: int = 0
    last_backend: str | None = None
    last_error: str | None = None
    escalation_count: int = 0
    request_seq: int = 0
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    def touch(self) -> None:
        self.updated_at = _now()


class SessionStore:
    SWEEP_THRESHOLD = 1024  # sweep idle entries once the store grows past this

    def __init__(self, default_route: str = "fast", ttl_seconds: float = 3600.0) -> None:
        self._default_route = default_route
        self._ttl = ttl_seconds
        self._sessions: dict[str, SessionState] = {}
        self._tasks: dict[str, TaskState] = {}          # by task_id
        self._session_tasks: dict[str, set[str]] = {}   # session_id -> task_ids

    # -- sessions -----------------------------------------------------------

    def get_or_create_session(self, session_id: str | None = None) -> SessionState:
        sid = session_id or f"sess-{uuid.uuid4().hex[:12]}"
        s = self._sessions.get(sid)
        if s is None:
            s = SessionState(session_id=sid, current_route=self._default_route)
            self._sessions[sid] = s
            self._session_tasks.setdefault(sid, set())
        s.touch()
        return s

    def get_session(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    # -- tasks ---------------------------------------------------------------

    def get_or_create_task(self, session: SessionState, task_id: str) -> TaskState:
        t = self._tasks.get(task_id)
        if t is None:
            t = TaskState(
                task_id=task_id,
                session_id=session.session_id,
                current_route=self._default_route,
            )
            self._tasks[task_id] = t
            self._session_tasks.setdefault(session.session_id, set()).add(task_id)
        t.touch()
        return t

    def get_task(self, task_id: str) -> TaskState | None:
        return self._tasks.get(task_id)

    # -- updates ---------------------------------------------------------------

    def record_request(self, session: SessionState, task: TaskState, tier: str, backend_name: str) -> None:
        """Bookkeeping after a request was dispatched to ``tier``.

        Per-task attempt counting is owned by
        :meth:`router.escalation.EscalationController.note_attempt` (auto-routed
        requests only); this method keeps session-level counters in sync.
        """
        if tier == "fast":
            session.fast_attempts += 1
        elif tier == "deep":
            session.deep_attempts += 1
        session.last_backend = backend_name
        session.touch()
        task.touch()
        self._maybe_sweep()

    def mark_error(self, session: SessionState, error: str) -> None:
        session.last_error = error
        session.touch()

    # -- lifecycle ---------------------------------------------------------------

    def _maybe_sweep(self) -> None:
        if len(self._sessions) < self.SWEEP_THRESHOLD:
            return
        now = time.time()
        stale = [sid for sid, s in self._sessions.items() if now - s.updated_at > self._ttl]
        for sid in stale:
            del self._sessions[sid]
            for tid in self._session_tasks.pop(sid, ()):  # type: ignore[arg-type]
                self._tasks.pop(tid, None)

    def __len__(self) -> int:
        return len(self._sessions)
