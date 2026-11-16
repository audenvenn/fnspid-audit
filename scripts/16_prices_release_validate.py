#!/usr/bin/env python3
"""End-to-end check of the repaired price columns against the independent Yahoo download.

For every sampled ticker with Yahoo data, the ratio close_rc / Yahoo Close (and adj_close_rc / Yahoo Adj
Close) should be constant across the whole 2013-2023 window if the repair put all segments on one
adjustment basis (the constant is the split/dividend adjustment after 2023-12-28). We report the share of
series whose log-ratio IQR is below 1% before (original columns) and after repair, by repair status.
Output: data/derived/results/prices_rc_validation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

YDIR = common.RAW / "yahoo"


def spread(x):
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 50:
        return np.nan
    lx = np.log(x)
    return float(np.quantile(lx, 0.95) - np.quantile(lx, 0.05))


def main():
    samp = pd.read_csv(common.RESULTS / "reference_sample.csv")
    R = pd.read_parquet(common.DERIVED / "release" / "prices_rc.parquet",
                        columns=["ticker", "date", "close", "adj_close", "close_factor", "adj_factor", "rc_status"],
                        filters=[("ticker", "in", list(samp["ticker"]))])
    R["ticker"] = R["ticker"].astype(str)
    R["close_rc"] = R["close"] * R["close_factor"]
    R["adj_close_rc"] = R["adj_close"] * R["adj_factor"]
    R["date"] = pd.to_datetime(R["date"])
    R = R[(R["date"] >= "2013-01-01") & (R["date"] <= "2023-12-28")]
    flags = pd.read_parquet(common.DERIVED / "release" / "prices_series_flags.parquet").set_index("ticker")
    rows = []
    for t, f in R.groupby("ticker"):
        p = YDIR / f"{t}.csv"
        if not p.exists():
            continue
        y = pd.read_csv(p)
        if len(y) < 100:
            continue
        y["Date"] = pd.to_datetime(y["Date"])
        m = f.merge(y[["Date", "Close", "Adj Close"]], left_on="date", right_on="Date", how="inner")
        if len(m) < 100:
            continue
        rows.append({"ticker": t, "group": samp.set_index("ticker").at[t, "group"],
                     "series_status": flags.at[t, "series_status"] if t in flags.index else None,
                     "n": len(m), "close_spread_before": spread(m["close"] / m["Close"]),
                     "close_spread_after": spread(m["close_rc"] / m["Close"]),
                     "adj_spread_before": spread(m["adj_close"] / m["Adj Close"]),
                     "adj_spread_after": spread(m["adj_close_rc"] / m["Adj Close"]),
                     "share_rows_repaired_or_latest": float(m["close_rc"].notna().mean())})
    V = pd.DataFrame(rows)
    V.to_csv(common.RESULTS / "prices_rc_validation.csv", index=False)
    out = {}
    for st, g in V.groupby("series_status"):
        out[str(st)] = {"n": int(len(g)),
                        "close_consistent_before_(p5-p95<2%)": int((g["close_spread_before"] < 0.02).sum()),
                        "close_consistent_after": int((g["close_spread_after"] < 0.02).sum()),
                        "adj_consistent_before": int((g["adj_spread_before"] < 0.02).sum()),
                        "adj_consistent_after": int((g["adj_spread_after"] < 0.02).sum()),
                        "median_close_spread_before": float(g["close_spread_before"].median()),
                        "median_close_spread_after": float(g["close_spread_after"].median()),
                        "median_adj_spread_before": float(g["adj_spread_before"].median()),
                        "median_adj_spread_after": float(g["adj_spread_after"].median())}
    (common.RESULTS / "prices_rc_validation.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
