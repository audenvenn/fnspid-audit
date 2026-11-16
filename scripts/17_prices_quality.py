#!/usr/bin/env python3
"""Further price-file quality facts quoted in AUDIT.md (beyond the finlint summary).

  * long constant-price, zero-volume runs (>= 250 rows): placeholder rows, e.g. before a listing date
  * rows with NaN close, weekend rows, all-zero rows
  * header layouts and lower-case ticker file names
Output: data/derived/results/prices_quality.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def chunk_stats(p: pd.DataFrame):
    p = p.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    code = p["ticker"].cat.codes.to_numpy()
    c = p["close"].to_numpy()
    v = p["volume"].to_numpy()
    same = np.r_[False, (code[1:] == code[:-1]) & (c[1:] == c[:-1])]
    rid = np.cumsum(~same)
    L = np.bincount(rid)
    zero = np.bincount(rid, weights=(v == 0))
    long_ = np.nonzero((L >= 250) & (zero >= 0.99 * L))[0]
    first = np.searchsorted(rid, long_)
    series_first = p.groupby(code)["date"].transform("min").to_numpy()
    last = first + L[long_] - 1
    nxt = np.minimum(last + 1, len(p) - 1)
    ok_next = code[nxt] == code[last]
    ex = pd.DataFrame({"ticker": p["ticker"].astype(str).to_numpy()[first], "start": p["date"].to_numpy()[first],
                       "end": p["date"].to_numpy()[last], "rows": L[long_], "close": c[first],
                       "series_start": series_first[first],
                       "next_date": np.where(ok_next, p["date"].to_numpy()[nxt], np.datetime64("NaT")),
                       "next_close": np.where(ok_next, c[nxt], np.nan), "next_volume": np.where(ok_next, v[nxt], np.nan)})
    wk = p["date"].dt.day_of_week >= 5
    st = {"nan_close_rows": int(np.isnan(c).sum()), "weekend_rows": int(wk.sum()),
          "weekend_rows_nan_close": int((wk & np.isnan(c)).sum()),
          "all_zero_ohlc_rows": int(((p["open"] == 0) & (p["high"] == 0) & (p["low"] == 0) & (p["close"] == 0)).sum()),
          "zero_open_rows": int((p["open"] == 0).sum()),
          "zero_open_rows_before_1993": int(((p["open"] == 0) & (p["date"] < "1993-01-01")).sum())}
    return ex, st


def main():
    src = common.DERIVED / "prices.parquet"
    tick = sorted(pq.read_table(src, columns=["ticker"]).column(0).unique().to_pylist()) if False else \
        sorted(set(pd.read_csv(common.RESULTS / "prices_files.csv")["ticker"]))
    exs, tot = [], {}
    for i in range(0, len(tick), 1000):
        t = pq.read_table(src, columns=["ticker", "date", "open", "high", "low", "close", "volume"],
                          filters=[("ticker", "in", tick[i:i + 1000])]).to_pandas(date_as_object=False)
        t["ticker"] = t["ticker"].astype(str).astype("category")
        ex, st = chunk_stats(t)
        exs.append(ex)
        for k, v in st.items():
            tot[k] = tot.get(k, 0) + v
        del t
    ex = pd.concat(exs, ignore_index=True).sort_values("rows", ascending=False)
    files = pd.read_csv(common.RESULTS / "prices_files.csv")
    low = files["ticker"].str.contains("[a-z]")
    out = {
        "placeholder_runs_ge250_zero_volume": {"runs": int(len(ex)), "rows": int(ex["rows"].sum()),
                                               "series": int(ex["ticker"].nunique()),
                                               "longest": ex.head(8).astype(str).to_dict("records")},
        **tot,
        "header_layouts": files["layout"].value_counts().rename({0: "date,open,high,low,close,adj close,volume",
                                                                 1: "date,volume,open,high,low,close,adj close"}).to_dict(),
        "lowercase_ticker_files": int(low.sum()),
        "lowercase_with_uppercase_twin": int(files.loc[low, "ticker"].str.upper().isin(set(files["ticker"])).sum()),
        "files": int(len(files)),
    }
    (common.RESULTS / "prices_quality.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
