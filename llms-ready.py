#!/dd2/andrei/crouter/llm-router/.venv/bin/python
"""Check that the router's backends actually answer a real chat request.

Probes fast and deep with a tiny /chat/completions call sent straight to each
backend's base_url — bypassing llm-router, so it works even when the router
itself is down. Frontier (OpenRouter) is probed only when cloud.enabled is
true in the config; otherwise it is reported as SKIP.

Exit codes: 0 = every enabled backend answered, 1 = at least one failed,
2 = config/usage error.

Usage:
    ./llms-ready.py [--config PATH] [--timeout SECONDS] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "llm-router"))

import httpx  # noqa: E402

from router.config import ConfigError, load_config  # noqa: E402

TIERS = ("fast", "deep", "frontier")
PROMPT = "Reply with exactly one word: ok"


def probe(bcfg, timeout: float) -> dict:
    """Send a minimal chat completion to the backend; return a result dict."""
    url = bcfg.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = bcfg.resolved_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "model": bcfg.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    t0 = time.monotonic()
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    except httpx.TimeoutException:
        return {"ok": False, "detail": f"timeout after {timeout:.0f}s", "ms": None}
    except httpx.HTTPError as e:
        return {"ok": False, "detail": f"{type(e).__name__}: {e}", "ms": None}
    ms = int((time.monotonic() - t0) * 1000)
    if resp.status_code != 200:
        body = " ".join(resp.text[:160].split())
        return {"ok": False, "detail": f"HTTP {resp.status_code}: {body}", "ms": ms}
    try:
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {})
    except ValueError:
        return {"ok": False, "detail": "non-JSON response body", "ms": ms}
    # Thinking models may spend the whole tiny budget on reasoning — any
    # generated token (content or reasoning_content) proves the server serves.
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    if not content and not reasoning:
        fr = choice.get("finish_reason", "?")
        return {"ok": False, "detail": f"empty completion (finish_reason={fr})", "ms": ms}
    snippet = f'"{content[:40]}"' if content else "thinking only"
    return {"ok": True, "detail": snippet, "ms": ms}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check fast/deep/frontier backends answer a real chat request.",
    )
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "llm-router" / "config.yaml"),
        help="router config YAML (default: llm-router/config.yaml)",
    )
    ap.add_argument(
        "--timeout", type=float, default=30.0,
        help="per-request timeout in seconds (default 30)",
    )
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    results: dict[str, dict] = {}
    for tier in TIERS:
        bcfg = cfg.backends.get(tier)
        if bcfg is None:
            results[tier] = {"status": "skip", "detail": "not configured"}
            continue
        if tier == "frontier" and not cfg.cloud.enabled:
            results[tier] = {
                "status": "skip", "detail": "cloud disabled (cloud.enabled=false)"
            }
            continue
        if bcfg.type == "mock" or not bcfg.base_url or not bcfg.model:
            results[tier] = {"status": "skip", "detail": f"type={bcfg.type}, no live endpoint"}
            continue
        r = probe(bcfg, args.timeout)
        results[tier] = {
            "status": "ok" if r["ok"] else "fail",
            "url": bcfg.base_url,
            **r,
        }

    if args.json:
        print(json.dumps(
            {"config": str(args.config), "results": results}, indent=2))
    else:
        print(f"llms-ready — config {args.config}")
        for tier in TIERS:
            r = results[tier]
            line = f"  {tier:<9}{r['status'].upper():<5}"
            if "url" in r:
                line += f"{r['url']}"
                if r.get("ms") is not None:
                    line += f"  {r['ms']} ms"
            line += f"  {r['detail']}"
            print(line)

    failed = [t for t in TIERS if results[t]["status"] == "fail"]
    checked = [t for t in TIERS if results[t]["status"] != "skip"]
    if not args.json:
        if failed:
            print(f"\nNOT READY — failing: {', '.join(failed)}")
        elif checked:
            print(f"\nREADY — all enabled backends respond ({', '.join(checked)})")
        else:
            print("\nREADY — no enabled backends to check")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
