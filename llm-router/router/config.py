"""Configuration loading and validation.

The configuration is a single YAML file (see ``config.example.yaml``).
API keys are preferably kept out of the YAML: each backend can name an
environment variable via ``api_key_env`` and the key is read lazily at
request time. A literal ``api_key`` in the YAML also works and takes
precedence — handy for local llama-servers started with ``--api-key``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

VALID_ROUTES = {"fast", "deep", "frontier"}
VALID_MODELS = VALID_ROUTES | {"auto"}


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class DefaultsConfig(BaseModel):
    """Fallback model used when a request omits ``model`` (or names an unknown one)."""

    route: str = "auto"

    @field_validator("route")
    @classmethod
    def _valid_route(cls, v: str) -> str:
        if v not in VALID_MODELS:
            raise ValueError(f"defaults.route must be one of {sorted(VALID_MODELS)}, got {v!r}")
        return v


class BackendConfig(BaseModel):
    """One backend entry. ``type`` selects the implementation."""

    type: Literal["openai_compatible", "openrouter", "mock"] = "openai_compatible"
    name: str
    base_url: Optional[str] = None
    model: Optional[str] = None
    #: Environment variable holding the API key (read lazily at request time).
    api_key_env: Optional[str] = None
    #: Literal API key in the YAML. Takes precedence over ``api_key_env`` —
    #: for local servers where keeping the key out of the file is not worth it
    #: (e.g. llama-server started with --api-key). Sent as ``Authorization: Bearer <key>``.
    api_key: Optional[str] = None
    timeout_seconds: float = 300.0
    extra_headers: dict[str, str] = Field(default_factory=dict)
    #: Hard context window of this backend's model (prompt + completion).
    #: ``None`` means "unknown / assume it fits" — the context floor never
    #: bumps away from such a tier on size grounds.
    max_context: Optional[int] = None
    #: When true, query the backend at startup for its real context size and
    #: use that instead of ``max_context`` (PRD §52). llama-server reports it
    #: via GET /props (the effective --ctx-size actually in use); a failed
    #: query keeps the manual value.
    query_context_size: bool = False

    # Mock-backend knobs (ignored by the other types).
    behavior: Literal["success", "stream", "tool_call", "error", "timeout", "echo"] = "success"
    response_text: str = "mock response"
    error_status: int = 500
    delay_seconds: float = 0.0

    def resolved_api_key(self) -> Optional[str]:
        """Effective API key, or ``None`` when the backend needs no auth.

        The literal ``api_key`` wins; otherwise the variable named by
        ``api_key_env`` is read from the environment at call time (so a
        rotated key needs no restart).
        """
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            value = os.environ.get(self.api_key_env, "")
            if value:
                return value
        return None


class EscalationSignals(BaseModel):
    """Which deterministic signals are allowed to trigger escalation."""

    explicit_request: bool = True
    repeated_tool_failure: bool = True
    repeated_test_failure: bool = True
    timeout: bool = False
    backend_error: bool = False


class EscalationConfig(BaseModel):
    enabled: bool = True
    max_fast_attempts: int = 2
    max_deep_attempts: int = 2
    #: Number of repeated test/tool failures (within one task, on the current tier)
    #: that trigger an escalation.
    failure_threshold: int = 2
    allow_frontier: bool = True
    signals: EscalationSignals = Field(default_factory=EscalationSignals)
    chain: list[str] = Field(default_factory=lambda: ["fast", "deep", "frontier"])


class ContextRoutingConfig(BaseModel):
    """Context-length routing ("context floor").

    A request whose estimated size (prompt + completion headroom) exceeds the
    selected tier's ``max_context`` is bumped up the escalation chain until it
    fits — regardless of how the tier was chosen. This complements the
    complexity-based escalation rules: context only grows within a session, so
    long conversations must move to the bigger window even when the task is easy.
    """

    enabled: bool = True
    #: Conservative chars-per-token divisor applied to the serialized request
    #: body (messages + tools + JSON overhead). Smaller = more tokens estimated
    #: = earlier bump to a bigger tier. 3 suits code-heavy agent traffic; the
    #: server-side chat template adds per-message tokens the client never sees,
    #: so erring high is deliberate.
    chars_per_token: float = Field(default=3.0, gt=0)
    #: Tokens reserved for the completion when the client gives no max_tokens
    #: (llama-server would otherwise generate until EOS or a full context).
    completion_reserve: int = Field(default=8192, ge=0)


class RoutingConfig(BaseModel):
    #: Tier that ``auto`` starts on (and the tier used when no model is given).
    default: str = "fast"
    #: Backend-fallback policy, keyed by tier. A backend failure (crash, 5xx,
    #: timeout) falls back to the listed tiers in order. This is NOT an
    #: escalation and does not change task state. Cloud fallback only happens
    #: when ``cloud.enabled`` is true.
    fallbacks: dict[str, list[str]] = Field(default_factory=dict)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    context: ContextRoutingConfig = Field(default_factory=ContextRoutingConfig)

    @field_validator("default")
    @classmethod
    def _valid_default(cls, v: str) -> str:
        if v not in VALID_ROUTES:
            raise ValueError(f"routing.default must be one of {sorted(VALID_ROUTES)}, got {v!r}")
        return v


class CloudPricing(BaseModel):
    """Optional per-1M-token prices used to estimate daily cloud cost."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0


class CloudConfig(BaseModel):
    enabled: bool = True
    #: When false, automatic escalation never calls the cloud; an
    #: "escalation required" response is returned instead.
    allow_automatic_escalation: bool = False
    max_requests_per_hour: int = 100
    max_estimated_cost_usd_per_day: float = 10.0
    #: Explicit (manual) frontier requests are still allowed when a limit is hit.
    allow_manual_when_limited: bool = True
    pricing: CloudPricing = Field(default_factory=CloudPricing)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_requests: bool = True
    log_responses: bool = False
    log_prompts: bool = False


class MetricsConfig(BaseModel):
    enabled: bool = True


class HealthConfig(BaseModel):
    #: How long a backend health result is cached (seconds).
    check_interval_seconds: float = 10.0
    timeout_seconds: float = 2.0


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    backends: dict[str, BackendConfig]
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate the YAML configuration file."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        data: Any = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {p}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"top level of {p} must be a mapping")

    try:
        cfg = AppConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigError(str(e)) from e

    _cross_validate(cfg)
    return cfg


def _cross_validate(cfg: AppConfig) -> None:
    """Validate relationships between sections that pydantic can't see."""
    esc = cfg.routing.escalation
    for tier in esc.chain:
        if tier not in cfg.backends:
            raise ConfigError(f"escalation chain references unknown backend {tier!r}")
    if len(esc.chain) < 2:
        raise ConfigError("escalation chain must have at least two tiers")

    for src, targets in cfg.routing.fallbacks.items():
        if src not in cfg.backends:
            raise ConfigError(f"fallback source {src!r} is not a configured backend")
        for t in targets:
            if t not in cfg.backends:
                raise ConfigError(f"fallback target {t!r} (of {src}) is not a configured backend")

    for tier, bcfg in cfg.backends.items():
        if bcfg.type == "mock":
            continue
        if not bcfg.base_url:
            raise ConfigError(f"backend {tier!r}: base_url is required for type={bcfg.type}")
        if not bcfg.model:
            raise ConfigError(f"backend {tier!r}: model is required for type={bcfg.type}")
