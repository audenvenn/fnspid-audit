#!/usr/bin/env python3
"""Run finlint on FNSPID (as published) and on reference data, saving the concise reports.

  1. prices: the full FNSPID panel (29.7M rows; columns ticker, date, close, adj_close, volume)
  2. prices: the Yahoo reference sample (independent download; should show no splice dates)
  3. news:   FNSPID Benzinga rows (timed + date-only), as labelled "UTC"
  4. news:   the upstream Kaggle analyst-ratings file (explicit -04:00 offsets) -> timezone check passes
  5. news:   FNSPID nasdaq-file rows from nasdaq.com
  6. join:   FNSPID Benzinga news -> FNSPID prices, same-date merge as in price_news_integrate.py
Outputs: data/derived/results/finlint_*.txt and finlint_*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import newslib as nl  # noqa: E402

import pandas as pd  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
from finlint import lookahead, news, prices  # noqa: E402
from finlint.report import format_report  # noqa: E402


def save(name, findings, title):
    txt = format_report(findings, title=title)
    (common.RESULTS / f"finlint_{name}.txt").write_text(txt + "\n")
    (common.RESULTS / f"finlint_{name}.json").write_text(json.dumps([f.to_dict() for f in findings], indent=1, default=str))
    print(txt, "\n", flush=True)


def main():
    common.ensure_dirs()
    # 1. FNSPID prices: chunked (400 whole series at a time) to stay within ~3 GB
    import contextlib
    import io
    from finlint import cli
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli.main(["prices", str(common.DERIVED / "prices.parquet"), "--chunk-tickers", "400",
                  "--json", str(common.RESULTS / "finlint_fnspid_prices.json")])
    txt = buf.getvalue().replace("finlint prices\n==============", "finlint prices: FNSPID Stock_price/full_history.zip\n" + "=" * 52)
    (common.RESULTS / "finlint_fnspid_prices.txt").write_text(txt)
    print(txt, flush=True)
    # 2. Yahoo reference sample
    ys = []
    for f in sorted((common.RAW / "yahoo").glob("*.csv")):
        y = pd.read_csv(f)
        if len(y) >= 100:
            y["ticker"] = f.stem
            ys.append(y.rename(columns={"Date": "date", "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
                                        "Open": "open", "High": "high", "Low": "low"}))
    Y = pd.concat(ys, ignore_index=True)
    Y = Y[(Y["date"] >= "2010-01-01") & (Y["date"] <= "2023-12-28")]
    save("yahoo_prices", prices.run_all(Y[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]),
         "finlint prices: independent Yahoo download of the sampled tickers (2010-2023)")
    del Y, ys
    # 3. FNSPID Benzinga news
    b = nl.load(["date_raw", "title", "ticker", "url"], file_ids=(0,), filter=pc.match_substring(ds.field("url"), "benzinga.com"))
    b = b.astype(str).rename(columns={"date_raw": "date"})
    save("fnspid_benzinga_news", news.run_all(b), "finlint news: FNSPID All_external.csv, Benzinga rows")
    # 6. join risk for the same rows
    P = pd.read_parquet(common.DERIVED / "prices.parquet", columns=["ticker", "date"])
    P["ticker"] = P["ticker"].astype(str)
    P["date"] = pd.to_datetime(P["date"])
    sessions = P["date"].value_counts()
    sessions = sessions[sessions >= 50].index
    tz = news.check_timezone(b)
    shift = float(tz[0].metrics["best_shift_h"]) if tz[0].severity == "error" else 0.0
    d = news.to_naive(b["date"]).dt.normalize()
    key = pd.MultiIndex.from_arrays([P["ticker"], P["date"]])
    hit = pd.MultiIndex.from_arrays([b["ticker"], d]).isin(key)
    assigned = d.where(hit)
    save("fnspid_benzinga_join", tz + lookahead.lookahead_report(b["date"], assigned, sessions, shift_h=shift),
         "finlint join: FNSPID Benzinga news -> FNSPID prices (same-date merge)")
    del b, P
    # 4. upstream Kaggle file
    k = pd.read_parquet(common.DERIVED / "kaggle_analyst_ratings.parquet").rename(
        columns={"date_orig": "date", "headline": "title", "stock": "ticker"})
    save("kaggle_upstream_news", news.run_all(k[["date", "title", "ticker", "url"]]),
         "finlint news: upstream Kaggle raw_analyst_ratings (explicit -04:00 offsets)")
    del k
    # 5. nasdaq.com rows of the nasdaq file (the rest of that file is a copy of All_external.csv)
    q = nl.load(["date_raw", "title", "ticker", "url"], file_ids=(1,),
                filter=pc.match_substring(ds.field("url"), "nasdaq.com")).astype(str).rename(columns={"date_raw": "date"})
    save("fnspid_nasdaq_news", news.run_all(q), "finlint news: FNSPID nasdaq_exteral_data.csv, nasdaq.com rows")


if __name__ == "__main__":
    main()
