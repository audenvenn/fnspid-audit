#!/usr/bin/env python3
"""Fetch an independent reference price history (Yahoo Finance via yfinance) for a sample of tickers.

Politeness: one request per ticker, sequential, >= 1.5 s between requests, results cached on disk
(data/raw/yahoo/<ticker>.csv); re-running only fetches what is missing.

Ticker groups (written to data/derived/results/reference_sample.csv):
  jump       every ticker whose close moves >25% across a detected splice boundary
             (2015-01-02, 2020-04-02, 2020-07-02, 2020-07-06)  -> repair candidates
  control    120 random tickers with data on both sides of 2020-07-06 and no large boundary move
  truncated  150 random tickers whose series ends 2020-04-01..2020-07-31, plus GOOGL, NFLX, FB
  absent     NVDA, META (no FNSPID price file at all)
Second pass (--retry-short): tickers for which period="max" returned <100 rows are re-requested with an
explicit start date.
Stooq (the first choice) now serves a JavaScript proof-of-work page to scripts, so it was not used.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = common.RAW / "yahoo"
SEED = 20261116


def yahoo_symbol(t: str) -> str:
    return t.replace(".", "-")


def build_sample() -> pd.DataFrame:
    b = pd.read_csv(common.RESULTS / "price_boundary_jumps.csv")
    b["alr"] = np.abs(np.log(b["r"]))
    jump = sorted(b.loc[b["alr"] > np.log(1.25), "ticker"].unique())
    sp = pd.read_csv(common.RESULTS / "price_series_span.csv", parse_dates=["first", "last"])
    both = set(b.loc[b["date"] == "2020-07-06", "ticker"])
    rng = np.random.default_rng(SEED)
    ctrl_pool = sorted(both - set(jump))
    control = sorted(rng.choice(ctrl_pool, size=120, replace=False))
    tr_pool = sorted(sp.loc[(sp["last"] >= "2020-04-01") & (sp["last"] <= "2020-07-31"), "ticker"])
    trunc = sorted(set(rng.choice(tr_pool, size=150, replace=False)) | {"GOOGL", "NFLX", "FB"})
    rows = ([(t, "jump") for t in jump] + [(t, "control") for t in control] +
            [(t, "truncated") for t in trunc] + [(t, "absent") for t in ["NVDA", "META"]])
    s = pd.DataFrame(rows, columns=["ticker", "group"]).drop_duplicates("ticker")
    s.to_csv(common.RESULTS / "reference_sample.csv", index=False)
    return s


def main():
    import yfinance as yf
    OUT.mkdir(parents=True, exist_ok=True)
    retry_short = "--retry-short" in sys.argv
    s = pd.read_csv(common.RESULTS / "reference_sample.csv") if retry_short else build_sample()
    print(s.group.value_counts().to_dict(), flush=True)
    log = []
    for i, t in enumerate(s["ticker"]):
        f = OUT / f"{t}.csv"
        if retry_short:
            # Yahoo sometimes answers period="max" with only the last few days; ask for an explicit range
            if not f.exists() or len(pd.read_csv(f)) >= 100 or len(pd.read_csv(f)) == 0:
                continue
        elif f.exists():
            continue
        ys = yahoo_symbol(t)
        t0 = time.time()
        try:
            if retry_short:
                h = yf.Ticker(ys).history(start="1962-01-01", end="2026-11-14", auto_adjust=False, actions=True,
                                          raise_errors=False)
            else:
                h = yf.Ticker(ys).history(period="max", auto_adjust=False, actions=True, raise_errors=False)
            status = "ok" if len(h) else "empty"
        except Exception as e:  # network or symbol errors are recorded, not fatal
            h, status = pd.DataFrame(), f"error:{type(e).__name__}"
        if len(h):
            h = h.reset_index()
            h["Date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None).dt.date
        if retry_short and len(h) <= len(pd.read_csv(f)):
            status = "retry_not_longer"
        else:
            h.to_csv(f, index=False)
        log.append((t, ys, status, len(h), time.strftime("%Y-%m-%dT%H:%M:%S")))
        print(i, t, status, len(h), flush=True)
        time.sleep(max(0.0, 1.5 - (time.time() - t0)) + 0.5)
    lp = common.RESULTS / "reference_fetch_log.csv"
    old = pd.read_csv(lp) if lp.exists() else pd.DataFrame()
    new = pd.DataFrame(log, columns=["ticker", "yahoo_symbol", "status", "rows", "fetched_at"])
    pd.concat([old, new]).drop_duplicates("ticker", keep="last").to_csv(lp, index=False)


if __name__ == "__main__":
    main()
