"""OpenRouter backend (PRD §23).

OpenRouter is OpenAI-compatible, so this only adds:

* an ``Authorization`` header — from the literal ``api_key`` in config if set,
  else lazily from the environment variable named by ``api_key_env`` (default
  ``OPENROUTER_API_KEY``);
* the optional attribution headers OpenRouter recommends.

Nothing else about the request is modified.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from ..config import BackendConfig, HealthConfig
from .openai_compatible import OpenAICompatibleBackend


class OpenRouterBackend(OpenAICompatibleBackend):
    def __init__(
        self,
        tier: str,
        cfg: BackendConfig,
        health_cfg: Optional[HealthConfig] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        super().__init__(tier, cfg, health_cfg=health_cfg, client=client)
        self._api_key_env = cfg.api_key_env or "OPENROUTER_API_KEY"

    def _request_headers(self) -> dict[str, str]:
        headers = super()._request_headers()
        # Attribution headers (OpenRouter docs); overridable via extra_headers.
        headers.setdefault("HTTP-Referer", "http://localhost:8000")
        headers.setdefault("X-Title", "llm-router")
        # A literal api_key was already applied by the base class; otherwise
        # fall back to this backend's default env var name.
        if not any(k.lower() == "authorization" for k in headers):
            key = os.environ.get(self._api_key_env, "")
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers
