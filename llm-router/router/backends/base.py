"""Common backend interface.

A backend executes one chat-completion request against a concrete model
server and returns either a full response body or a raw SSE stream. The
routing layer never inspects the payload — it is proxied through as-is,
except for the ``model`` field which is rewritten to the backend's real
model name.

Two failure classes are distinguished (PRD §12):

* :class:`BackendError` — transport/5xx problems (crash, timeout, 503...).
  These do NOT mean "the model was too weak"; they trigger the configured
  backend-fallback policy, not an escalation.
* Any other non-2xx status is returned in :class:`BackendResult` and
  forwarded to the client unchanged (e.g. a 400 bad request).
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional


@dataclass
class BackendResult:
    status_code: int
    content_type: str = "application/json"
    #: Full response body (non-streaming path).
    body: Optional[bytes] = None
    #: Raw SSE lines for the streaming path; each item ends with ``\n``.
    stream: Optional[AsyncIterator[bytes]] = field(default=None, repr=False)
    #: Parsed ``usage`` object when available (used for cloud cost estimates).
    usage: Optional[dict[str, Any]] = None


class BackendError(Exception):
    """Transport-level or 5xx failure of a backend."""

    def __init__(self, kind: str, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        #: One of: connection | timeout | http_5xx | bad_response | other
        self.kind = kind
        self.status_code = status_code


class Backend(abc.ABC):
    """Abstract base for all backends."""

    def __init__(self, tier: str, name: str) -> None:
        self.tier = tier
        self.name = name or tier

    @abc.abstractmethod
    async def chat_completion(self, body: dict[str, Any], *, stream: bool) -> BackendResult:
        """Execute one chat completion. ``body`` is the client's request JSON."""

    @abc.abstractmethod
    async def health_check(self) -> tuple[bool, float]:
        """Return ``(healthy, latency_ms)``. Must be cheap (cached)."""

    async def close(self) -> None:  # pragma: no cover - trivial default
        pass

    async def query_context_size(self) -> Optional[int]:
        """The backend's real context window (PRD §52), or ``None`` when it
        doesn't report one. Used to override a possibly-stale manual
        ``max_context`` at startup."""
        return None


def cached_health(
    last: Optional[tuple[float, bool, float]], interval_seconds: float
) -> Optional[tuple[bool, float]]:
    """Return the cached health result if it is still fresh, else ``None``."""
    if last is not None and (time.monotonic() - last[0]) < interval_seconds:
        return last[1], last[2]
    return None
