"""Router: "where should this request go?" (PRD §3, §7).

Routing priority (highest first):

    0. authoritative pin                    (GET /route/all/<tier>)
    1. prompt marker                        (@@fast / @@deep in the last user message)
    2. explicit escalation/control header   (X-LLM-Escalate / X-LLM-Route)
    3. explicit model                       (local-fast | local-deep | frontier)
    4. automatic policy                     (complexity verdict when enabled, else the
                                             task's current tier after escalations)
    5. configured default                   (routing.default / defaults.route)

``auto`` always starts on ``routing.default`` (fast in the example config);
it only moves up when the EscalationController says so (PRD §10). No LLM is
invoked to judge difficulty — except via the opt-in complexity classifier
(``routing.complexity``, PRD §20 exception), whose verdict feeds step 4.

After the priority above, a **context floor** applies: if the request's
estimated context need exceeds the chosen tier's ``max_context``, the tier is
bumped up the escalation chain until it fits (reason ``context_overflow``).
This is per-request and does not change task state — long conversations must
use the bigger window even when the task itself is easy.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Optional

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
#: A Claude Code background task (safety classifier, context collapse, ...)
#: sent its own model name; mapped to a tier by family.
REASON_CLAUDE_FAMILY = "claude_family"
REASON_AUTO_POLICY = "auto_policy"
REASON_DEFAULT = "default"
#: The request's estimated context exceeds the selected tier's max_context.
REASON_CONTEXT_OVERFLOW = "context_overflow"
#: An authoritative pin (GET /route/all/<tier>) forces every request to one tier.
REASON_PINNED = "pinned"
#: A @@fast / @@deep marker in the last user message steered this request.
REASON_PROMPT_MARKER = "prompt_marker"
#: The opt-in complexity classifier judged the task (routing.complexity).
REASON_COMPLEXITY = "complexity"


# -- prompt markers -----------------------------------------------------------

#: Magic words that steer a single user message to a tier: ``@@fast``,
#: ``@@deep`` or ``@@frontier`` anywhere in the last user message. The leading
#: whitespace is consumed on strip so "do X @@fast" becomes "do X".
PROMPT_MARKER_RE = re.compile(r"\s*@@(fast|deep|frontier)\b", re.IGNORECASE)


def _rightmost_marker(text: str) -> Optional[str]:
    matches = list(PROMPT_MARKER_RE.finditer(text))
    return matches[-1].group(1).lower() if matches else None


def last_user_message_text(openai_body: dict[str, Any]) -> Optional[str]:
    """Text of the last user message ("" when it carries none), or None if absent.

    Handles both content shapes the router sees — a plain string and a list of
    parts (text blocks joined in order; images/tool results ignored). Never
    raises: a malformed message must not 500 the request.
    """
    try:
        messages = openai_body.get("messages")
        if not isinstance(messages, list):
            return None
        for m in reversed(messages):
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
                ]
                return "".join(parts)
            return ""
    except Exception:  # pragma: no cover - defensive; body came from JSON parse
        log.debug("last-user-message extraction failed", exc_info=True)
        return None


def extract_prompt_marker(openai_body: dict[str, Any]) -> Optional[str]:
    """Find a routing marker in the last user message and strip it in place.

    Returns the tier key (``fast`` | ``deep`` | ``frontier``) when present —
    the rightmost occurrence wins — else None. All occurrences are removed so
    the backend never sees them and the context estimate reflects what is
    actually sent. Never raises.
    """
    try:
        messages = openai_body.get("messages")
        if not isinstance(messages, list):
            return None
        for m in reversed(messages):
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            content = m.get("content")
            tier: Optional[str] = None
            if isinstance(content, str):
                tier = _rightmost_marker(content)
                if tier is not None:
                    m["content"] = PROMPT_MARKER_RE.sub("", content)
            elif isinstance(content, list):
                for b in content:  # parts in order -> last match wins overall
                    if (
                        isinstance(b, dict)
                        and b.get("type") == "text"
                        and isinstance(b.get("text"), str)
                    ):
                        t = _rightmost_marker(b["text"])
                        if t is not None:
                            tier = t
                            b["text"] = PROMPT_MARKER_RE.sub("", b["text"])
            return tier
    except Exception:  # pragma: no cover - defensive; body came from JSON parse
        log.debug("prompt-marker extraction failed", exc_info=True)
        return None


def _header_tiers(headers: dict[str, str]) -> tuple[Optional[str], Optional[str]]:
    """Explicit control headers as (escalate tier or None, raw route value).

    Shared by :meth:`Router._resolve_base` and :meth:`Router.is_plain_auto` so
    the two can't drift. The route value may be ``"auto"`` — not an alias; it
    falls through to the auto policy in ``_resolve_base``.
    """
    h_esc = headers.get("x-llm-escalate", "").strip().lower()
    h_route = headers.get("x-llm-route", "").strip().lower()
    return (ALIAS_TO_TIER.get(h_esc) if h_esc else None, h_route or None)


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
        required_context: Optional[int] = None,
        pinned_tier: Optional[str] = None,
        marker_tier: Optional[str] = None,
        complexity_tier: Optional[str] = None,
    ) -> RouteDecision:
        """Apply the routing priority, then the context floor.

        ``headers`` keys must be lower-cased. ``required_context`` is the
        request's estimated context need (prompt + completion headroom); when
        given and larger than the chosen tier's ``max_context``, the decision
        is bumped up the escalation chain until it fits — even over an
        explicitly requested tier, since that backend would reject the prompt.

        ``pinned_tier`` (set via GET /route/all/<tier>) is authoritative: every
        request goes to that tier regardless of model/headers/policy, and the
        context floor is skipped — the pin already runs the tier at its maximum
        known window, so a too-big prompt should get the backend's own precise
        error rather than a silent bump away from the pinned tier.

        ``marker_tier`` comes from a @@fast / @@deep marker in the last user
        message (see :func:`extract_prompt_marker`): it outranks headers and
        model names — the most recent human intent — but not the pin, and the
        context floor still applies afterwards. Per-request only: no session or
        task state is touched, so the next message reverts to automatic routing.

        ``complexity_tier`` is the opt-in classifier's verdict for a plain auto
        request; it replaces the task's current route at step 4 and keeps
        ``manual=False`` so rule D accounting continues as usual.
        """
        if pinned_tier is not None and pinned_tier in self._cfg.backends:
            return RouteDecision(tier=pinned_tier, reason=REASON_PINNED, manual=True)
        decision = self._resolve_base(
            model=model, headers=headers, session=session, task=task,
            marker_tier=marker_tier, complexity_tier=complexity_tier,
        )
        return self._apply_context_floor(decision, required_context, session, task)

    def _resolve_base(
        self,
        *,
        model: Optional[str],
        headers: dict[str, str],
        session: SessionState,
        task: TaskState,
        marker_tier: Optional[str] = None,
        complexity_tier: Optional[str] = None,
    ) -> RouteDecision:
        """Routing priority only (no context floor)."""
        # 1a'. prompt marker (@@fast / @@deep in the last user message) — the
        # most recent human intent; outranks headers and model names.
        if marker_tier is not None and marker_tier in self._cfg.backends:
            return RouteDecision(tier=marker_tier, reason=REASON_PROMPT_MARKER, manual=True)

        h_esc_tier, h_route = _header_tiers(headers)

        # 1a. explicit escalation header (rule A) — subject to cloud limits later.
        if h_esc_tier is not None:
            tier = h_esc_tier
            esc = None
            # The header still routes (it is an explicit request); it only
            # records state changes when rule A's signal is enabled.
            if tier != task.current_route and self._controller.explicit_requests_enabled:
                esc = self._controller.escalate(session, task, tier, REASON_EXPLICIT)
            return RouteDecision(tier=tier, reason=REASON_EXPLICIT_ESCALATE, manual=True, escalation=esc)

        # 1b. explicit route header (direct routing for this request only).
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

        # 3 + 4. auto: the complexity verdict when one was produced for this
        # request (opt-in classifier), else the task's current route (already
        # reflects any event-driven escalation) — where session state matters.
        if m == "auto" or h_route == "auto":
            if complexity_tier is not None and complexity_tier in self._cfg.backends:
                return RouteDecision(tier=complexity_tier, reason=REASON_COMPLEXITY)
            return RouteDecision(tier=task.current_route, reason=REASON_AUTO_POLICY)

        # 5. Claude Code background tasks (Bash safety classifier, context
        # collapse, small-fast helpers) send their own model names — e.g. the
        # auto-mode classifier sends claude-sonnet-5 regardless of what the
        # user runs CC with. Without a mapping they 404 and every gated Bash
        # command is blocked ("classifier unavailable"). Map by family:
        # haiku -> fast, everything else -> deep (falling back to fast when
        # the config has no deep tier).
        if m.startswith("claude"):
            candidates = ("fast",) if "haiku" in m else ("deep", "fast")
            for tier in candidates:
                if tier in self._cfg.backends:
                    return RouteDecision(tier=tier, reason=REASON_CLAUDE_FAMILY, manual=True)

        raise UnknownModel(m)

    def is_plain_auto(self, model: Optional[str], headers: dict[str, str]) -> bool:
        """True iff :meth:`resolve` would reach step 4 (the auto policy).

        Mirrors steps 1a'/1a/1b/2 of ``_resolve_base`` (via the shared
        :func:`_header_tiers`) so callers can decide whether an expensive
        pre-step — e.g. the complexity classifier — would ever be used. The
        pin is checked by the caller, since it lives outside this class.
        """
        h_esc_tier, h_route = _header_tiers(headers)
        if h_esc_tier is not None or h_route in ALIAS_TO_TIER:
            return False
        m = (model or self._cfg.defaults.route).strip().lower()
        return m == "auto" or h_route == "auto"

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

    # -- context floor ---------------------------------------------------------

    def _apply_context_floor(
        self,
        decision: RouteDecision,
        required: Optional[int],
        session: SessionState,
        task: TaskState,
    ) -> RouteDecision:
        """Bump the tier up the chain until its max_context fits ``required``.

        Per-request only (like backend fallback): it does not mutate task or
        session state, so a short subagent conversation still starts on fast.
        Tiers with no configured ``max_context`` are assumed to fit. The walk
        stops at the top of the chain — if even that cannot fit, the request
        goes there anyway and the backend's own error is more precise than ours.
        """
        if required is None:
            return decision
        tier = decision.tier
        while True:
            bcfg = self._cfg.backends.get(tier)
            limit = bcfg.max_context if bcfg is not None else None
            if limit is None or required <= limit:
                break
            nxt = self._controller.next_tier(tier)
            if nxt is None:
                break
            tier = nxt
        if tier == decision.tier:
            return decision

        # Reuse Escalation for the x_router block; its ``count`` field carries
        # the estimated token requirement (not an attempt count).
        esc = Escalation(decision.tier, tier, REASON_CONTEXT_OVERFLOW, required)
        log.info(
            "CONTEXT-OVERFLOW session=%s task=%s from=%s to=%s required_tokens~%d",
            session.session_id,
            task.task_id,
            decision.tier,
            tier,
            required,
        )
        return replace(decision, tier=tier, reason=REASON_CONTEXT_OVERFLOW, escalation=esc)

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
