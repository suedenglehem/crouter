"""Shared fixtures: a fully mock-backed router app (no GPU, no network)."""

from __future__ import annotations

import copy
import pytest
from fastapi.testclient import TestClient

from router.api import create_app
from router.config import AppConfig


def base_config_dict() -> dict:
    """A complete config where every backend is a mock."""
    return {
        "server": {"host": "127.0.0.1", "port": 8000},
        "defaults": {"route": "auto"},
        "backends": {
            "fast": {
                "type": "mock",
                "name": "local-fast",
                "model": "local-fast",
                "behavior": "success",
                "response_text": "FAST-SAYS-HI",
            },
            "deep": {
                "type": "mock",
                "name": "local-deep",
                "model": "local-deep",
                "behavior": "success",
                "response_text": "DEEP-SAYS-HI",
            },
            "frontier": {
                "type": "mock",
                "name": "frontier",
                "model": "frontier-model",
                "behavior": "success",
                "response_text": "FRONTIER-SAYS-HI",
            },
        },
        "routing": {
            "default": "fast",
            "fallbacks": {},
            "escalation": {
                "enabled": True,
                "max_fast_attempts": 2,
                "max_deep_attempts": 2,
                "failure_threshold": 2,
                "allow_frontier": True,
                "signals": {
                    "explicit_request": True,
                    "repeated_tool_failure": True,
                    "repeated_test_failure": True,
                    "timeout": False,
                    "backend_error": False,
                },
                "chain": ["fast", "deep", "frontier"],
            },
        },
        # Cloud is ON and automatic escalation allowed by default in tests so the
        # full fast->deep->frontier chain can be exercised; individual tests
        # override this.
        "cloud": {
            "enabled": True,
            "allow_automatic_escalation": True,
            "max_requests_per_hour": 100,
            "max_estimated_cost_usd_per_day": 10.0,
            "allow_manual_when_limited": True,
            "pricing": {"input_per_mtok": 0.0, "output_per_mtok": 0.0},
        },
        "logging": {
            "level": "DEBUG",
            "log_requests": True,
            "log_responses": False,
            "log_prompts": False,
        },
        "metrics": {"enabled": True},
        "health": {"check_interval_seconds": 10.0, "timeout_seconds": 2.0},
    }


def make_config(**overrides) -> AppConfig:
    """Build an AppConfig from the mock base config with deep-merged overrides."""
    data = copy.deepcopy(base_config_dict())

    def merge(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = copy.deepcopy(v)

    merge(data, overrides)
    return AppConfig.model_validate(data)


@pytest.fixture()
def cfg() -> AppConfig:
    return make_config()


@pytest.fixture()
def client(cfg) -> TestClient:
    app = create_app(cfg)
    with TestClient(app) as c:
        yield c


# -- convenience helpers -------------------------------------------------------

def chat(client: TestClient, model="auto", **headers):
    """POST a minimal non-streaming chat completion."""
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hello"}]},
        headers=headers,
    )


def event(client: TestClient, session_id: str, task_id: str, ev: str, **metadata):
    return client.post(
        "/events",
        json={"session_id": session_id, "task_id": task_id, "event": ev, "metadata": metadata},
    )


def content_of(resp) -> str:
    return resp.json()["choices"][0]["message"]["content"]
