"""Backend construction. Maps config tier keys to Backend instances."""

from __future__ import annotations

import httpx

from ..config import AppConfig, ConfigError
from .base import Backend, BackendError, BackendResult  # noqa: F401 (re-export)
from .mock import MockBackend
from .openai_compatible import OpenAICompatibleBackend
from .openrouter import OpenRouterBackend


def build_backends(cfg: AppConfig) -> dict[str, Backend]:
    """Instantiate one backend per configured tier key."""
    backends: dict[str, Backend] = {}
    for tier, bcfg in cfg.backends.items():
        if bcfg.type == "mock":
            backends[tier] = MockBackend(tier, bcfg)
        elif bcfg.type == "openrouter":
            backends[tier] = OpenRouterBackend(tier, bcfg, health_cfg=cfg.health)
        else:  # openai_compatible
            backends[tier] = OpenAICompatibleBackend(tier, bcfg, health_cfg=cfg.health)
    return backends


def build_backends_with_clients(
    cfg: AppConfig, clients: dict[str, httpx.AsyncClient] | None = None
) -> dict[str, Backend]:
    """Like :func:`build_backends` but allows injecting httpx clients (tests)."""
    clients = clients or {}
    backends: dict[str, Backend] = {}
    for tier, bcfg in cfg.backends.items():
        if bcfg.type == "mock":
            backends[tier] = MockBackend(tier, bcfg)
        elif bcfg.type == "openrouter":
            backends[tier] = OpenRouterBackend(
                tier, bcfg, health_cfg=cfg.health, client=clients.get(tier)
            )
        else:
            backends[tier] = OpenAICompatibleBackend(
                tier, bcfg, health_cfg=cfg.health, client=clients.get(tier)
            )
    return backends


__all__ = [
    "Backend",
    "BackendError",
    "BackendResult",
    "MockBackend",
    "OpenAICompatibleBackend",
    "OpenRouterBackend",
    "build_backends",
    "build_backends_with_clients",
]
