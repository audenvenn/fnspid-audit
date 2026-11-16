"""finlint: a small linter for free financial news and price datasets.

Modules
  finlint.prices     split/splice detection, adjustment-vintage resets, coverage breaks, stale runs,
                     duplicates, OHLC and calendar sanity
  finlint.news       date-only share, timezone sanity from session-anchored headlines, hour profile, duplicates
  finlint.lookahead  leakage-free news->session alignment and a look-ahead report for a given join
  finlint.cli        `finlint prices|news|join ...`
"""
from .report import Finding, format_report

__all__ = ["Finding", "format_report"]
__version__ = "0.1.0"
