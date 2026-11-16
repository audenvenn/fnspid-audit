"""Findings and report formatting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SEVERITIES = ("info", "warn", "error")


@dataclass
class Finding:
    check: str                     # e.g. "prices.splice_dates"
    severity: str                  # info | warn | error
    message: str                   # one-line human summary
    metrics: dict[str, Any] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.severity not in SEVERITIES:
            raise ValueError(f"bad severity {self.severity!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.check, "severity": self.severity, "message": self.message,
                "metrics": self.metrics, "examples": self.examples}


def _fmt_val(v: Any) -> str:
    if isinstance(v, float):
        if v != v:
            return "nan"
        if abs(v) >= 1000 or v == int(v):
            return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:g}"
        return f"{v:.4g}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def format_report(findings: list[Finding], title: str = "finlint report", max_examples: int = 3) -> str:
    order = {"error": 0, "warn": 1, "info": 2}
    lines = [title, "=" * len(title)]
    counts = {s: sum(f.severity == s for f in findings) for s in SEVERITIES}
    lines.append(f"{counts['error']} error(s), {counts['warn']} warning(s), {counts['info']} info")
    for f in sorted(findings, key=lambda f: (order[f.severity], f.check)):
        lines.append(f"[{f.severity.upper():5}] {f.check}: {f.message}")
        if f.metrics:
            kv = ", ".join(f"{k}={_fmt_val(v)}" for k, v in f.metrics.items() if not isinstance(v, (dict, list)))
            if kv:
                lines.append(f"        {kv}")
        for ex in f.examples[:max_examples]:
            lines.append("        e.g. " + ", ".join(f"{k}={_fmt_val(v)}" for k, v in ex.items()))
    return "\n".join(lines)
