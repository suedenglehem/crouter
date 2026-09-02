"""Router: "where should this request go?" (PRD §3, §7).

Routing priority (highest first):

    1. explicit escalation/control header   (X-LLM-Escalate / X-LLM-Route)
    2. explicit model                       (local-fast | local-deep | frontier)
    3. session route                        (task's current tier after escalations)
    4. automatic policy                     (deterministic escalation rules)
    5. configured default                   (routing.default / defaults.route)

``auto`` always starts on ``routing.default`` (fast in the example config);
it only moves up when the EscalationController says so (PRD §10). No LLM is
invoked to judge difficulty.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import date
from typing import Optional

from .backends.base import Backend
from .config import AppConfig
from .escalation import Escalation, REASON_EXPLICIT, EscalationController
from .sessions import SessionState, TaskState

log = logging.getLogger("router.routing")


class UnknownModel(ValueError):
    """Raised when the request names a model that is not one of our aliases."""


#: Client-facing alias -> tier key.
ALIAS_TO_TIER = {
    "local-fast": "fast",
    "fast": "fast",
    "local-deep": "deep",
    "deep": "deep",
    "frontier": "frontier",
}

REASON_EXPLICIT_ESCALATE = "explicit_escalate"
REASON_EXPLICIT_ROUTE_HEADER = "explicit_route_header"
REASON_EXPLICIT_MODEL = "explicit_model"
REASON_AUTO_POLICY = "auto_policy"
REASON_DEFAULT = "default"


@dataclass(frozen=True)
class RouteDecision:
    tier: str
    reason: str
    #: True when the client explicitly chose this tier (affects cloud policy).
    manual: bool = False
    escalation: Optional[Escalation] = None


class Router:
    def __init__(
        self,
        cfg: AppConfig,
        backends: dict[str, Backend],
        controller: EscalationController,
    ) -> None:
        self._cfg = cfg
        self._backends = backends
        self._controller = controller

    # -- resolution -----------------------------------------------------------

    def resolve(
        self,
        *,
        model: Optional[str],
        headers: dict[str, str],
        session: SessionState,
        task: TaskState,
    ) -> RouteDecision:
        """Apply the routing priority. ``headers`` keys must be lower-cased."""
        # 1a. explicit escalation header (rule A) — subject to cloud limits later.
        h_esc = headers.get("x-llm-escalate", "").strip().lower()
        if h_esc in ALIAS_TO_TIER:
            tier = ALIAS_TO_TIER[h_esc]
            esc = None
            # The header still routes (it is an explicit request); it only
            # records state changes when rule A's signal is enabled.
            if tier != task.current_route and self._controller.explicit_requests_enabled:
                esc = self._controller.escalate(session, task, tier, REASON_EXPLICIT)
            return RouteDecision(tier=tier, reason=REASON_EXPLICIT_ESCALATE, manual=True, escalation=esc)

        # 1b. explicit route header (direct routing for this request only).
        h_route = headers.get("x-llm-route", "").strip().lower()
        if h_route in ALIAS_TO_TIER:
            return RouteDecision(tier=ALIAS_TO_TIER[h_route], reason=REASON_EXPLICIT_ROUTE_HEADER, manual=True)

        # 2. explicit model name (or configured default when absent).
        m = (model or self._cfg.defaults.route).strip().lower()
        if m in ALIAS_TO_TIER:
            return RouteDecision(
                tier=ALIAS_TO_TIER[m],
                reason=REASON_EXPLICIT_MODEL if model else REASON_DEFAULT,
                manual=bool(model),
            )

        # 3 + 4. auto: the task's current route (already reflects any
        # event-driven escalation) — this is where session state matters.
        if m == "auto" or h_route == "auto":
            return RouteDecision(tier=task.current_route, reason=REASON_AUTO_POLICY)

        raise UnknownModel(m)

    def apply_attempt(self, decision: RouteDecision, session: SessionState, task: TaskState) -> RouteDecision:
        """Count an auto-routed attempt; rule D may bump the tier for this request."""
        if decision.manual or not self._controller.enabled:
            return decision
        esc = self._controller.note_attempt(session, task, decision.tier)
        if esc is None:
            return decision
        return replace(
            decision,
            tier=esc.to_tier,
            reason=REASON_AUTO_POLICY,
            escalation=decision.escalation or esc,
        )

    # -- backend fallback (PRD §31) ---------------------------------------------

    def next_fallback(self, tier: str) -> Optional[str]:
        """Next configured fallback tier for a *backend* failure (not an escalation)."""
        for candidate in self._cfg.routing.fallbacks.get(tier, []):
            if candidate == "frontier" and not self._cfg.cloud.enabled:
                continue  # cloud fallback must be explicitly enabled
            return candidate
        return None


class CloudBudget:
    """Hard limits on cloud usage (PRD §21, §22).

    Tracks a rolling one-hour request count and an estimated daily cost.
    Limits block *automatic* cloud calls; explicit manual frontier requests
    are governed by ``allow_manual_when_limited``.
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg.cloud
        self._hour: deque[float] = deque()
        self._day_key: str | None = None
        self._day_cost: float = 0.0

    def allows(self, manual: bool) -> tuple[bool, Optional[str]]:
        c = self._cfg
        if not c.enabled:
            return False, "cloud_disabled"
        # Automatic cloud escalation is opt-in (PRD §21): when disabled, a deep
        # failure must NOT silently call OpenRouter — an "escalation required"
        # response is returned instead. Explicit manual requests are unaffected.
        if not manual and not c.allow_automatic_escalation:
            return False, "cloud_auto_disabled"

        now = time.time()
        while self._hour and now - self._hour[0] > 3600.0:
            self._hour.popleft()

        rate_ok = len(self._hour) < c.max_requests_per_hour
        cost_ok = self._day_cost < c.max_estimated_cost_usd_per_day
        if rate_ok and cost_ok:
            return True, None
        if manual and c.allow_manual_when_limited:
            return True, None

        reason = "cloud_rate_limit" if not rate_ok else "cloud_cost_limit"
        log.info("cloud blocked (manual=%s) reason=%s", manual, reason)
        return False, reason

    def record(self, usage: Optional[dict]) -> None:
        self._hour.append(time.time())
        cost = self.estimate_cost(usage or {})
        if cost > 0:
            today = date.today().isoformat()
            if self._day_key != today:
                self._day_key, self._day_cost = today, 0.0
            self._day_cost += cost

    def estimate_cost(self, usage: dict) -> float:
        p = self._cfg.pricing
        prompt = float(usage.get("prompt_tokens", 0) or 0)
        completion = float(usage.get("completion_tokens", 0) or 0)
        return (prompt / 1e6) * p.input_per_mtok + (completion / 1e6) * p.output_per_mtok

    @property
    def day_cost(self) -> float:
        return self._day_cost
