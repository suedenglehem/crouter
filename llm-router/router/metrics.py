"""Minimal Prometheus-compatible metrics registry.

Hand-rolled on purpose (PRD §36: keep dependencies minimal). Supports the two
metric types this service needs — counters and histograms — and renders the
standard Prometheus text exposition format at ``/metrics``.

Labels must be low-cardinality (PRD §33): never put raw session/task IDs in a
label; use bounded values like tier names, backend names, or reason codes.
"""

from __future__ import annotations

import threading
from typing import Any


def _fmt_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + inner + "}"


class Counter:
    def __init__(self, name: str, help_text: str) -> None:
        self.name = name
        self.help = help_text
        self._values: dict[tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        for key in sorted(self._values):
            labels = dict(key)
            lines.append(f"{self.name}{_fmt_labels(labels)} {self._values[key]:g}")
        return lines


class Histogram:
    DEFAULT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)

    def __init__(self, name: str, help_text: str, buckets: tuple[float, ...] = DEFAULT_BUCKETS) -> None:
        self.name = name
        self.help = help_text
        self._buckets = sorted(buckets)
        # label-key -> [bucket_counts..., sum, count]
        self._values: dict[tuple[str, ...], list[float]] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            row = self._values.get(key)
            if row is None:
                row = [0.0] * (len(self._buckets) + 2)
                self._values[key] = row
            for i, b in enumerate(self._buckets):
                if value <= b:
                    row[i] += 1
            row[-2] += value  # sum
            row[-1] += 1      # count

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for key in sorted(self._values):
            labels = dict(key)
            row = self._values[key]
            cumulative = 0.0
            for i, b in enumerate(self._buckets):
                cumulative += row[i]
                lbl = dict(labels)
                lbl["le"] = f"{b:g}"
                lines.append(f"{self.name}_bucket{_fmt_labels(lbl)} {cumulative:g}")
            inf = dict(labels)
            inf["le"] = "+Inf"
            lines.append(f"{self.name}_bucket{_fmt_labels(inf)} {row[-1]:g}")
            lines.append(f"{self.name}_sum{_fmt_labels(labels)} {row[-2]:g}")
            lines.append(f"{self.name}_count{_fmt_labels(labels)} {row[-1]:g}")
        return lines


class MetricsRegistry:
    """Holds all metrics and renders the exposition format."""

    def __init__(self) -> None:
        self._metrics: dict[str, Counter | Histogram] = {}

    def counter(self, name: str, help_text: str) -> Counter:
        m = self._metrics.get(name)
        if isinstance(m, Counter):
            return m
        c = Counter(name, help_text)
        self._metrics[name] = c
        return c

    def histogram(self, name: str, help_text: str, buckets: tuple[float, ...] | None = None) -> Histogram:
        m = self._metrics.get(name)
        if isinstance(m, Histogram):
            return m
        h = Histogram(name, help_text, buckets or Histogram.DEFAULT_BUCKETS)
        self._metrics[name] = h
        return h

    def render(self) -> str:
        out: list[str] = []
        for name in sorted(self._metrics):
            out.extend(self._metrics[name].render())  # type: ignore[union-attr]
        return "\n".join(out) + ("\n" if out else "")


def register_router_metrics(registry: MetricsRegistry) -> dict[str, Any]:
    """Create the standard router metrics (PRD §33). Returns them by name."""
    return {
        "requests_total": registry.counter(
            "router_requests_total", "Total requests handled by the router."
        ),
        "request_latency_seconds": registry.histogram(
            "router_request_latency_seconds", "Router request latency in seconds."
        ),
        "backend_requests_total": registry.counter(
            "router_backend_requests_total", "Requests sent to backends."
        ),
        "backend_errors_total": registry.counter(
            "router_backend_errors_total", "Backend failures (transport/5xx), by kind."
        ),
        "escalations_total": registry.counter(
            "router_escalations_total", "Route escalations, by from/to tier and reason."
        ),
        "cloud_requests_total": registry.counter(
            "router_cloud_requests_total", "Requests sent to the cloud (frontier) backend."
        ),
        "cloud_blocked_total": registry.counter(
            "router_cloud_blocked_total", "Cloud requests blocked by cost/rate limits."
        ),
        "tool_failures_total": registry.counter(
            "router_tool_failures_total", "Tool failure events received via /events."
        ),
        "test_failures_total": registry.counter(
            "router_test_failures_total", "Test failure events received via /events."
        ),
    }
