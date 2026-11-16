"""D10: astronomically large historical prices with zero volume from cumulative reverse-split adjustment.

Counts price rows above a level threshold, their zero-volume share, the cross-sectional open statistics, and
for the most extreme tickers compares the inflation of the first price with the cumulative product of Yahoo
split ratios between FNSPID's first and last dates (yfinance; cached). Writes results/prices_extremes.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I does not add the script dir
import common  # noqa: E402

THRESH = 10_000.0
TOP = 10


def split_product(ticker: str, start, end, cache: Path) -> dict:
    f = cache / f"{ticker}.splits.json"
    if f.exists():
        ev = json.loads(f.read_text())
    else:
        import yfinance as yf
        sp = yf.Ticker(ticker).splits
        ev = [[str(ix.date()), float(v)] for ix, v in sp.items()]
        f.write_text(json.dumps(ev))
    inside = [v for d, v in ev if str(start) < d <= str(end)]
    return {"n_splits": len(inside), "product": float(np.prod(inside)) if inside else 1.0}


def main():
    p = pd.read_parquet(common.DERIVED / "prices.parquet", columns=["ticker", "date", "open", "close", "volume"])
    hi = p[(p["open"] > THRESH) | (p["close"] > THRESH)]
    per = (hi.groupby("ticker").agg(rows=("close", "size"), max_close=("close", "max"),
                                    zero_volume=("volume", lambda v: int((v == 0).sum())))
           .sort_values("max_close", ascending=False))
    cache = common.RAW / "yahoo_splits"
    cache.mkdir(parents=True, exist_ok=True)
    checks = []
    for t in per.index[:TOP]:
        s = p[p["ticker"] == t].sort_values("date")
        first, last = s.iloc[0], s.iloc[-1]
        try:
            sp = split_product(t, first["date"], last["date"], cache)
        except Exception as e:  # network or symbol problems: record and continue
            checks.append({"ticker": t, "error": str(e)[:200]})
            continue
        implied_first = float(first["close"]) * sp["product"]
        checks.append({"ticker": t, "first_date": str(first["date"]), "first_close": float(first["close"]),
                       "first_volume": float(first["volume"]), "n_splits_in_span": sp["n_splits"],
                       "split_product_in_span": sp["product"], "first_close_times_split_product": implied_first})
    out = {"threshold": THRESH, "rows_above": int(len(hi)), "tickers_above": int(hi["ticker"].nunique()),
           "rows_above_zero_volume": int((hi["volume"] == 0).sum()),
           "open_mean": float(p["open"].mean()), "open_std": float(p["open"].std()), "open_max": float(p["open"].max()),
           "open_median": float(p["open"].median()),
           "top_tickers": per.head(TOP).reset_index().to_dict(orient="records"), "split_checks": checks}
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    (common.RESULTS / "prices_extremes.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("top_tickers", "split_checks")}, indent=1))
    for c in checks:
        print(c)


if __name__ == "__main__":
    main()
