"""Command line interface.

  finlint prices PRICES.(csv|parquet) [--ticker-col ticker --date-col date --close-col close --adj-col adj_close]
                                      [--chunk-tickers 1000]   (parquet: bounded memory)
  finlint news   NEWS.(csv|parquet)   [--ts-col date --text-col title --ticker-col ticker --url-col url --label-tz UTC]
  finlint join   NEWS PRICES          [--shift-h auto|<hours>]   look-ahead risk of a same-date news->price merge

Every sub-command prints a concise report; --json PATH also writes the findings as JSON.
Exit status: 0 = no errors, 1 = at least one error-level finding, 2 = usage problem.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import lookahead, news, prices
from .report import format_report


def read_table(path: str, columns=None, nrows=None) -> pd.DataFrame:
    if path.endswith(".parquet") or path.endswith(".pq"):
        df = pd.read_parquet(path, columns=columns)
        return df.head(nrows) if nrows else df
    return pd.read_csv(path, usecols=columns, nrows=nrows, dtype=str if columns is None else None, keep_default_na=False)


def _prices(a) -> list:
    cols = {"ticker": a.ticker_col, "date": a.date_col, "close": a.close_col, "adj_close": a.adj_col,
            "open": a.open_col, "high": a.high_col, "low": a.low_col, "volume": a.volume_col}

    def numeric(df):
        for c in ("close", "adj_close", "open", "high", "low", "volume"):
            if cols[c] in df:
                df[cols[c]] = pd.to_numeric(df[cols[c]], errors="coerce")
        return df

    if a.chunk_tickers and a.path.endswith((".parquet", ".pq")):
        # bounded memory: read whole series a few hundred tickers at a time
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
        dset = ds.dataset(a.path, format="parquet")
        col = dset.to_table(columns=[a.ticker_col]).column(0)
        uniq = pc.unique(col.cast("string") if not str(col.type).startswith("dictionary") else col)
        uniq = uniq.dictionary_decode() if hasattr(uniq, "dictionary_decode") else uniq
        tick = sorted(set(str(x) for x in uniq.to_pylist()))
        del col, uniq
        names = [c for c in cols.values() if c in dset.schema.names]

        def frames():
            for i in range(0, len(tick), a.chunk_tickers):
                part = dset.to_table(columns=names, filter=ds.field(a.ticker_col).isin(tick[i:i + a.chunk_tickers]))
                yield numeric(part.to_pandas())
        return prices.run_chunked(frames(), cols)
    return prices.run_all(numeric(read_table(a.path, nrows=a.nrows)), cols)


def _news(a) -> list:
    df = read_table(a.path, nrows=a.nrows)
    return news.run_all(df, ts_col=a.ts_col, text_col=a.text_col, ticker_col=a.ticker_col,
                        url_col=a.url_col, label_tz=a.label_tz, group_col=a.group_col)


def _join(a) -> list:
    nw = read_table(a.news, nrows=a.nrows)
    pr = read_table(a.prices)
    pr_dates = pd.to_datetime(pr[a.date_col]).dt.normalize()
    sessions = pr_dates.unique()
    findings = []
    if a.shift_h == "auto":
        tz = news.check_timezone(nw, a.ts_col, a.text_col, a.label_tz)
        findings += tz
        shift = float(tz[0].metrics.get("best_shift_h", 0)) if tz[0].severity == "error" else 0.0
    else:
        shift = float(a.shift_h)
    # FNSPID-style merge: calendar date of the stamp, joined only where that ticker has a price row that day
    d = news.to_naive(nw[a.ts_col]).dt.normalize()
    key = pd.MultiIndex.from_arrays([pr[a.ticker_col].astype(str), pr_dates])
    hit = pd.MultiIndex.from_arrays([nw[a.ticker_col].astype(str), d]).isin(key)
    assigned = d.where(hit)
    findings += lookahead.lookahead_report(nw[a.ts_col], assigned, sessions, shift_h=shift, label_tz=a.label_tz)
    return findings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="finlint", description="Lint free financial news/price datasets.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prices", help="lint a long-format daily price panel")
    p.add_argument("path")
    for k, v in [("ticker", "ticker"), ("date", "date"), ("close", "close"), ("adj", "adj_close"), ("open", "open"),
                 ("high", "high"), ("low", "low"), ("volume", "volume")]:
        p.add_argument(f"--{k}-col", default=v)
    p.add_argument("--chunk-tickers", type=int, default=0,
                   help="parquet input only: lint N tickers at a time (bounded memory for large panels)")
    n = sub.add_parser("news", help="lint a news table")
    n.add_argument("path")
    j = sub.add_parser("join", help="look-ahead risk of a same-date news->price join")
    j.add_argument("news")
    j.add_argument("prices")
    j.add_argument("--date-col", default="date", help="date column of the price table")
    j.add_argument("--shift-h", default="auto", help="'auto' (infer from session-anchored headlines) or hours")
    for q in (n, j):
        q.add_argument("--ts-col", default="date")
        q.add_argument("--text-col", default="title")
        q.add_argument("--ticker-col", default="ticker")
        q.add_argument("--label-tz", default="UTC")
    n.add_argument("--url-col", default="url")
    n.add_argument("--group-col", default=None)
    for q in (p, n, j):
        q.add_argument("--nrows", type=int, default=None, help="read only the first N rows")
        q.add_argument("--json", default=None, help="also write findings to this JSON file")
    a = ap.parse_args(argv)
    try:
        findings = {"prices": _prices, "news": _news, "join": _join}[a.cmd](a)
    except (KeyError, ValueError, FileNotFoundError) as e:
        print(f"finlint: {e}", file=sys.stderr)
        return 2
    print(format_report(findings, title=f"finlint {a.cmd}"))
    if a.json:
        with open(a.json, "w") as f:
            json.dump([x.to_dict() for x in findings], f, indent=1, default=str)
    return 1 if any(x.severity == "error" for x in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
