"""ComplexityClassifier unit tests against a stub judging backend (no network)."""

from __future__ import annotations

import asyncio
import json

from conftest import make_config
from router.backends.base import Backend, BackendError, BackendResult
from router.complexity import ComplexityClassifier


class StubBackend(Backend):
    """Returns a fixed completion; counts calls for cache assertions."""

    def __init__(
        self,
        content: str | None = "FAST",
        status_code: int = 200,
        delay_seconds: float = 0.0,
        error: BackendError | None = None,
    ) -> None:
        super().__init__("fast", "stub-fast")
        self.content = content
        self.status_code = status_code
        self.delay_seconds = delay_seconds
        self.error = error
        self.calls = 0

    async def chat_completion(self, body: dict, *, stream: bool) -> BackendResult:
        self.calls += 1
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.error is not None:
            raise self.error
        message = {"role": "assistant", "content": self.content or ""}
        return BackendResult(
            self.status_code, body=json.dumps({"choices": [{"message": message}]}).encode()
        )

    async def health_check(self) -> tuple[bool, float]:
        return True, 0.1


def _classifier(backend: StubBackend, **cx_overrides) -> ComplexityClassifier:
    cfg = make_config(routing={"complexity": {"enabled": True, "tier": "fast", **cx_overrides}})
    return ComplexityClassifier(cfg, {"fast": backend})


# -- verdict parsing -----------------------------------------------------------

def test_verdict_fast():
    c = _classifier(StubBackend(content="FAST"))
    assert asyncio.run(c.classify("s1", "rename a variable")) == "fast"


def test_verdict_deep_with_punctuation():
    c = _classifier(StubBackend(content="deep."))
    assert asyncio.run(c.classify("s1", "refactor the auth module")) == "deep"


def test_empty_content_is_no_verdict():
    # A thinking model can burn its whole budget on reasoning -> empty content.
    c = _classifier(StubBackend(content=""))
    assert asyncio.run(c.classify("s1", "task")) is None


def test_unparseable_answer_is_no_verdict():
    c = _classifier(StubBackend(content="I would say DEEP, probably."))
    assert asyncio.run(c.classify("s1", "task")) is None


# -- failure handling (advisory: never raises) ---------------------------------

def test_backend_error_is_no_verdict():
    c = _classifier(StubBackend(error=BackendError("timeout", "stub timed out")))
    assert asyncio.run(c.classify("s1", "task")) is None


def test_non_200_is_no_verdict():
    c = _classifier(StubBackend(content="FAST", status_code=400))
    assert asyncio.run(c.classify("s1", "task")) is None


def test_timeout_is_no_verdict():
    c = _classifier(StubBackend(content="FAST", delay_seconds=0.2), timeout_seconds=0.05)
    assert asyncio.run(c.classify("s1", "task")) is None


# -- caching -------------------------------------------------------------------

def test_cache_hit_avoids_second_call():
    b = StubBackend(content="DEEP")
    c = _classifier(b)
    assert asyncio.run(c.classify("s1", "same task")) == "deep"
    assert asyncio.run(c.classify("s1", "same task")) == "deep"
    assert b.calls == 1


def test_cache_is_per_session_and_text():
    b = StubBackend(content="DEEP")
    c = _classifier(b)
    asyncio.run(c.classify("s1", "task A"))
    asyncio.run(c.classify("s2", "task A"))  # different session -> re-judge
    asyncio.run(c.classify("s1", "task B"))  # different text -> re-judge
    assert b.calls == 3


def test_negative_results_not_cached():
    # First call fails (busy judge); the retry must be judged, not cached as None.
    class Flaky(StubBackend):
        def __init__(self) -> None:
            super().__init__(content="DEEP")
            self.flake_once = True

        async def chat_completion(self, body: dict, *, stream: bool) -> BackendResult:
            if self.flake_once:
                self.flake_once = False
                self.calls += 1  # super() is never reached on the flaky path
                raise BackendError("timeout", "busy")
            return await super().chat_completion(body, stream=stream)

    b = Flaky()
    c = _classifier(b)
    assert asyncio.run(c.classify("s1", "task")) is None
    assert asyncio.run(c.classify("s1", "task")) == "deep"
    assert b.calls == 2


def test_lru_eviction_at_cache_size():
    b = StubBackend(content="FAST")
    c = _classifier(b, cache_size=2)
    asyncio.run(c.classify("s1", "one"))
    asyncio.run(c.classify("s1", "two"))
    asyncio.run(c.classify("s1", "three"))  # evicts "one"
    assert b.calls == 3
    asyncio.run(c.classify("s1", "one"))  # re-judged after eviction
    assert b.calls == 4


# -- judge body hygiene ----------------------------------------------------------

def test_input_truncated_to_max_chars():
    captured: dict[str, str] = {}

    class Capturing(StubBackend):
        async def chat_completion(self, body: dict, *, stream: bool) -> BackendResult:
            captured["user"] = body["messages"][1]["content"]
            return await super().chat_completion(body, stream=stream)

    c = _classifier(Capturing(content="FAST"), max_input_chars=64)
    asyncio.run(c.classify("s1", "x" * 500))
    assert captured["user"].endswith("x" * 64)
    assert "x" * 65 not in captured["user"]


def test_judge_body_is_minimal():
    captured: dict = {}

    class Capturing(StubBackend):
        async def chat_completion(self, body: dict, *, stream: bool) -> BackendResult:
            captured.update(body)
            return await super().chat_completion(body, stream=stream)

    c = _classifier(Capturing(content="FAST"))
    asyncio.run(c.classify("s1", "task"))
    assert "tools" not in captured
    assert captured["temperature"] == 0
    # disable_thinking defaults to true -> chat_template_kwargs present.
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}


def test_disable_thinking_off_omits_chat_template_kwargs():
    captured: dict = {}

    class Capturing(StubBackend):
        async def chat_completion(self, body: dict, *, stream: bool) -> BackendResult:
            captured.update(body)
            return await super().chat_completion(body, stream=stream)

    c = _classifier(Capturing(content="FAST"), disable_thinking=False)
    asyncio.run(c.classify("s1", "task"))
    assert "chat_template_kwargs" not in captured
