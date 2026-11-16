#!/usr/bin/env python3
"""Detect dataset-wide splice dates in the FNSPID price panel and list per-series boundary jumps.

Signals per (ticker, date), computed on consecutive rows of each series:
  r   = close_t / close_{t-1}                         (split-adjusted close)
  dq  = (adj_t/close_t) / (adj_{t-1}/close_{t-1})     (dividend-adjustment factor change)
In a single consistent download, adj/close can only rise over time (it is the product of dividend
factors for ex-dates after t), so dq < 1 marks a change of adjustment vintage, i.e. a splice.

Outputs (data/derived/results/):
  price_date_events.csv     per date: n series, n extreme |r|, n dq decrease, robust z-scores
  price_boundary_jumps.csv  per (ticker, boundary date): r, dq
  price_boundaries.json     summary numbers quoted in AUDIT.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BOUNDARIES = ["2015-01-02", "2020-04-02", "2020-07-02", "2020-07-06"]


def load_sorted(columns=("close", "adj_close")):
    """Load the panel sorted by (ticker, date) as numpy arrays (memory-lean)."""
    import pyarrow.parquet as pq
    t = pq.read_table(common.DERIVED / "prices.parquet", columns=["ticker", "date", *columns])
    tk = pd.Categorical(t.column("ticker").to_pandas())
    cats = np.asarray(tk.categories.astype(str))
    code = tk.codes.astype(np.int32)
    del tk
    day = t.column("date").to_numpy().astype("datetime64[D]").astype(np.int32)
    order = np.lexsort((day, code))
    out = {"code": code[order], "day": day[order], "cats": cats}
    del code, day
    for c in columns:
        out[c] = t.column(c).to_numpy()[order]
    del t
    same = np.r_[False, out["code"][1:] == out["code"][:-1]]
    out["same"] = same
    return out


def main():
    common.ensure_dirs()
    a = load_sorted()
    code, day, close, adj, same = a["code"], a["day"], a["close"], a["adj_close"], a["same"]
    cats = a["cats"]
    prev_close = np.r_[np.nan, close[:-1]]
    r = np.where(same, close / prev_close, np.nan)
    q = (adj / close).astype(np.float32)
    dq = np.where(same, q / np.r_[np.float32(np.nan), q[:-1]], np.float32(np.nan)).astype(np.float32)
    del q
    prev_day = np.where(same, np.r_[0, day[:-1]], -1).astype(np.int32)
    with np.errstate(invalid="ignore", divide="ignore"):
        extreme = (r > 2) | (r < 0.5)
        dq_dec = dq < 0.995
        big = np.abs(np.log(r)) > 0.2
    d0, d1 = day.min(), day.max()
    idx = day - d0
    nd = d1 - d0 + 1
    n = np.bincount(idx, minlength=nd)
    n_pair = np.bincount(idx, weights=same, minlength=nd)
    n_ext = np.bincount(idx, weights=extreme, minlength=nd)
    n_dq = np.bincount(idx, weights=dq_dec, minlength=nd)
    n_big = np.bincount(idx, weights=big, minlength=nd)
    keep = n > 0
    dates = (np.arange(nd) + d0).astype("datetime64[D]")
    ev = pd.DataFrame({"date": dates[keep], "n": n[keep], "n_pairs": n_pair[keep].astype(int),
                       "n_extreme": n_ext[keep].astype(int), "n_dq_dec": n_dq[keep].astype(int),
                       "n_abs_lr_gt_0p2": n_big[keep].astype(int)})
    for c in ["n_extreme", "n_dq_dec"]:
        frac = ev[c] / ev["n_pairs"].clip(lower=1)
        med = frac.rolling(251, center=True, min_periods=50).median()
        mad = (frac - med).abs().rolling(251, center=True, min_periods=50).median()
        ev["z_" + c] = ((frac - med) / (1.4826 * mad + 1e-4)).round(1)
    ev.to_csv(common.RESULTS / "price_date_events.csv", index=False)
    top_dq = ev.sort_values("n_dq_dec", ascending=False).head(8)
    top_ext = ev.sort_values("n_extreme", ascending=False).head(8)
    print(top_dq.to_string()); print(top_ext.to_string())

    bdays = np.array([np.datetime64(b, "D").astype(np.int32) for b in BOUNDARIES])
    m = np.isin(day, bdays) & same
    bj = pd.DataFrame({"ticker": cats[code[m]], "date": day[m].astype("datetime64[D]"),
                       "prev_date": prev_day[m].astype("datetime64[D]"), "prev_close": prev_close[m],
                       "close": close[m], "r": r[m], "dq": dq[m]})
    bj.to_csv(common.RESULTS / "price_boundary_jumps.csv", index=False)

    # per-series first/last
    last_idx = np.r_[np.nonzero(code[1:] != code[:-1])[0], len(code) - 1]
    first_idx = np.r_[0, last_idx[:-1] + 1]
    last = pd.Series(day[last_idx].astype("datetime64[D]"), index=cats[code[last_idx]])
    first = pd.Series(day[first_idx].astype("datetime64[D]"), index=cats[code[first_idx]])
    pd.DataFrame({"first": first, "last": last}).to_csv(common.RESULTS / "price_series_span.csv", index_label="ticker")

    def D(s):
        return np.datetime64(s, "D")

    # within one download, adj_close == close on the last days before the download (no later dividends yet)
    eq = np.isclose(adj, close, rtol=1e-9)
    adj_eq_close_share = {}
    for dd in ["2019-06-03", "2020-03-31", "2020-04-01", "2020-06-01", "2020-07-02", "2020-07-06", "2021-06-01", "2023-06-01", "2023-12-28"]:
        m_ = day == np.datetime64(dd, "D").astype(np.int32)
        adj_eq_close_share[dd] = {"series": int(m_.sum()), "share": round(float(eq[m_].mean()), 4) if m_.any() else None}
    x = bj[bj["date"] == D("2020-07-06")]
    covid = ev[(ev["date"] >= D("2020-02-15")) & (ev["date"] <= D("2020-05-01"))].sort_values("n_extreme").iloc[-1]
    summ = {
        "n_series": int(len(cats)), "n_rows": int(len(code)),
        "series_with_row_2020_07_06": int(len(x)),
        "adj_close_equals_close_share": adj_eq_close_share,
        "extreme_2020_07_06": int(((x.r > 2) | (x.r < .5)).sum()),
        "extreme_2020_07_06_up": int((x.r > 2).sum()), "extreme_2020_07_06_down": int((x.r < .5).sum()),
        "abs_logr_gt_0p2_2020_07_06": int((np.abs(np.log(x.r)) > 0.2).sum()),
        "move_gt_25pct_2020_07_06": int((np.abs(np.log(x.r)) > np.log(1.25)).sum()),
        "dq_dec_2020_07_06": int((x.dq < 0.995).sum()),
        "r_quantiles_2020_07_06": {str(k): round(float(v), 4) for k, v in x.r.quantile([.01, .05, .5, .95, .99]).items()},
        "worst_covid_day": str(covid["date"].date()), "worst_covid_day_extreme": int(covid["n_extreme"]),
        "worst_covid_day_pairs": int(covid["n_pairs"]),
        "events_top_dq": top_dq[["date", "n_pairs", "n_extreme", "n_dq_dec"]].astype(str).to_dict("records"),
        "events_top_extreme": top_ext[["date", "n_pairs", "n_extreme", "n_dq_dec"]].astype(str).to_dict("records"),
        "boundary_stats": {b: {"pairs": int((bj.date == D(b)).sum()),
                               "extreme": int(((bj.date == D(b)) & ((bj.r > 2) | (bj.r < .5))).sum()),
                               "dq_dec": int(((bj.date == D(b)) & (bj.dq < .995)).sum())} for b in BOUNDARIES},
        "series_end_counts": {
            "2020_04_to_2020_07": int(((last >= D("2020-04-01")) & (last <= D("2020-07-31"))).sum()),
            "on_2020_07_02": int((last == D("2020-07-02")).sum()),
            "on_2020_06_19": int((last == D("2020-06-19")).sum()),
            "on_2020_04_01": int((last == D("2020-04-01")).sum()),
            "on_2023_12_28": int((last == D("2023-12-28")).sum()),
            "before_2020": int((last < D("2020-01-01")).sum()),
            "2020_08_to_2023_11": int(((last >= D("2020-08-01")) & (last <= D("2023-11-30"))).sum()),
        },
        "series_start_2015_01_02": int((first == D("2015-01-02")).sum()),
        "first_date_min": str(first.min()), "last_date_max": str(last.max()),
        "examples_2020_07_06": x[x.ticker.isin(["AAPL", "TSLA", "AMZN", "GOOG", "MSFT"])][
            ["ticker", "prev_close", "close", "r"]].round(4).to_dict("records"),
        "missing_or_truncated_examples": {t: (str(first.get(t)), str(last.get(t))) if t in first.index else None
                                          for t in ["NVDA", "META", "FB", "GOOGL", "GOOG", "NFLX", "AAPL", "MSFT"]},
    }
    (common.RESULTS / "price_boundaries.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: v for k, v in summ.items() if not k.startswith("events")}, indent=1, default=str))


if __name__ == "__main__":
    main()
