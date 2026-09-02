"""Server bootstrap (uvicorn)."""

from __future__ import annotations

import logging

from .api import create_app
from .config import AppConfig


def run(cfg: AppConfig, host: str | None = None, port: int | None = None) -> None:
    """Create the app and serve it with uvicorn (blocking)."""
    import uvicorn

    app = create_app(cfg)
    log_level = cfg.logging.level.lower()
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # we log requests ourselves
    uvicorn.run(
        app,
        host=host or cfg.server.host,
        port=port or cfg.server.port,
        log_level=log_level,
    )
