"""Optional difficulty classification by an LLM before routing (PRD §20 opt-in).

When ``routing.complexity.enabled`` is true, plain ``auto`` requests are judged
once per user message: the configured backend answers exactly one word —
FAST or DEEP — and the verdict feeds step 4 of the routing priority. The judge
is advisory by design: any failure (timeout, backend error, unparseable
answer) yields ``None`` and the request falls back to the normal auto policy.

Verdicts are cached per (session, message hash). Only successful verdicts are
cached — a timeout while the judging tier is busy must not pin "no verdict"
for an entire Claude Code tool loop that reuses the same last user message.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import OrderedDict
from typing import Any, Optional

from .backends.base import Backend, BackendError
from .config import AppConfig

log = logging.getLogger("router.complexity")

_SYSTEM_PROMPT = (
    "You classify coding tasks by difficulty for model routing. "
    "Reply with exactly one word, nothing else: FAST or DEEP."
)

_RUBRIC = (
    "FAST: single-file edits, renames, simple lookups/reads, running code, "
    "small well-scoped changes.\n"
    "DEEP: multi-file refactors, architecture/design decisions, subtle debugging, "
    "performance work, ambiguous or open-ended tasks."
)

_NON_ALPHA_RE = re.compile(r"[^A-Za-z]")


class ComplexityClassifier:
    """Judge task difficulty with a configured backend; advisory only."""

    def __init__(self, cfg: AppConfig, backends: dict[str, Backend]) -> None:
        cx = cfg.routing.complexity
        self._tier = cx.tier
        self._timeout = cx.timeout_seconds
        self._max_chars = cx.max_input_chars
        self._disable_thinking = cx.disable_thinking
        self._backend = backends.get(cx.tier)
        self._cache: "OrderedDict[tuple[str, str], str]" = OrderedDict()
        self._cache_size = cx.cache_size

    async def classify(self, session_id: str, user_text: str) -> Optional[str]:
        """Return ``"fast"``/``"deep"`` for this user message, or None.

        ``None`` means "no verdict" — the caller falls back to auto policy.
        Never raises.
        """
        if self._backend is None or not user_text.strip():
            return None
        key = (session_id, hashlib.sha1(user_text.encode("utf-8", "replace")).hexdigest())
        cached = self._cache.get(key)
        if cached is not None:
            log.debug("complexity cache hit session=%s -> %s", session_id, cached)
            return cached

        body: dict[str, Any] = {
            # Placeholder — the backend rewrites "model" to its configured name.
            "model": "complexity-judge",
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"{_RUBRIC}\n\nTask:\n{user_text[: self._max_chars]}"},
            ],
            # Thinking models burn reasoning tokens before the answer; 512 can
            # be fully consumed by reasoning and leave content empty.
            "max_tokens": 1024,
            "temperature": 0,
        }
        if self._disable_thinking:
            # Qwen3 llama.cpp servers honor this (skips reasoning entirely);
            # servers that ignore unknown fields just think anyway; a strict
            # 400 degrades safely to "no verdict" below.
            body["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            result = await asyncio.wait_for(
                self._backend.chat_completion(body, stream=False), timeout=self._timeout
            )
        except Exception as e:  # noqa: BLE001 - advisory step must never raise (incl. BackendError/timeout)
            log.debug("complexity judge failed session=%s: %r", session_id, e)
            return None

        verdict = self._parse(result.body if result.status_code == 200 else None)
        if verdict is not None:
            self._cache[key] = verdict
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            log.info("COMPLEXITY session=%s tier=%s", session_id, verdict)
        return verdict

    @staticmethod
    def _parse(body: Optional[bytes]) -> Optional[str]:
        """Extract FAST/DEEP from the judge's answer; None when unparseable.

        Only ``message.content`` is read — never ``reasoning_content``, whose
        prose may mention both words.
        """
        if not body:
            return None
        try:
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
            return None
        normalized = _NON_ALPHA_RE.sub("", str(content)).upper()
        if normalized in ("FAST", "DEEP"):
            return normalized.lower()
        return None
