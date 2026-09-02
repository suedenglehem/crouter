"""CLI entry point: ``python -m router`` or the ``llm-router`` script (PRD §34)."""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm-router",
        description="Local-first LLM routing & escalation service (OpenAI-compatible).",
    )
    parser.add_argument("--config", default="config.yaml", help="path to the YAML config file")
    parser.add_argument("--host", default=None, help="bind host (overrides config)")
    parser.add_argument("--port", type=int, default=None, help="bind port (overrides config)")
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="log level (overrides config)")
    args = parser.parse_args(argv)

    from .config import ConfigError, load_config

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"llm-router: config error: {e}", file=sys.stderr)
        return 2

    level = (args.log_level or cfg.logging.level).upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    from .server import run

    try:
        run(cfg, host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
