#!/usr/bin/env python3
"""Materialize corrected news records (full original text + patch columns) for selected tickers.

The release ships only the sidecar patch (release/news_patch.parquet). This script regenerates full
records for any subset: it looks up the selected rows' byte offsets in the patch, fetches only those byte
ranges of the original CSV from Hugging Face (pinned revision, HTTP range requests), parses them, checks
the record count and the Date field against the patch, and writes one parquet.

Usage: python -I scripts/24_news_materialize.py --file 1 --tickers AAPL,MSFT [--out PATH] [--drop-summaries]
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import requests  # noqa: E402

csv.field_size_limit(sys.maxsize)
HEADERS = {
    0: ["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Author", "Article", "Lsa_summary",
        "Luhn_summary", "Textrank_summary", "Lexrank_summary"],
    1: ["Unnamed: 0", "Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Author", "Article",
        "Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary"],
}


def fetch(url, a, b):
    """One HTTP range request; honours 429 Retry-After (Hugging Face rate-limits per IP)."""
    for k in range(12):
        try:
            r = requests.get(url, headers={"Range": f"bytes={a}-{b - 1}", "User-Agent": "fnspid-audit/0.1"}, timeout=(30, 300))
            if r.status_code == 206 and len(r.content) == b - a:
                return r.content
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 30)) + 2
                print(f"rate limited; waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
        except requests.RequestException:
            pass
        time.sleep(5 * (k + 1))
    raise IOError(f"range {a}-{b} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=int, required=True, choices=[0, 1])
    ap.add_argument("--tickers", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--drop-summaries", action="store_true")
    a = ap.parse_args()
    tickers = a.tickers.split(",")
    path = common.DERIVED / "release" / "news_patch.parquet"
    idx = pq.read_table(path, columns=["row", "off", "ticker"], filters=[("file", "=", a.file)]).to_pandas()
    idx = idx.sort_values("row").reset_index(drop=True)
    idx["off_end"] = np.r_[idx["off"].to_numpy()[1:], common.NEWS_SIZES[a.file]]
    keep = idx[idx["ticker"].isin(tickers)][["row", "off_end"]]
    del idx
    if keep.empty:
        sys.exit("no rows for these tickers")
    full = pq.read_table(path, filters=[("file", "=", a.file), ("ticker", "in", tickers)]).to_pandas()
    sel = full.merge(keep, on="row").sort_values("row").reset_index(drop=True)
    runs = (sel["row"].diff() != 1).cumsum()
    url = common.hf_url(common.NEWS_FILES[a.file])
    recs = []
    for _, g in sel.groupby(runs):
        raw = fetch(url, int(g["off"].iloc[0]), int(g["off_end"].iloc[-1]))
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8", "replace"), newline="")))
        if len(rows) != len(g):
            raise ValueError(f"expected {len(g)} records, parsed {len(rows)}")
        recs.extend(rows)
        print(f"rows {g['row'].iloc[0]}-{g['row'].iloc[-1]}: {len(raw):,} bytes", flush=True)
    df = pd.DataFrame([r + [""] * (len(HEADERS[a.file]) - len(r)) for r in recs], columns=HEADERS[a.file])
    assert (df["Date"].to_numpy() == sel["date_raw"].to_numpy()).all(), "Date mismatch between CSV and patch"
    if a.drop_summaries:
        df = df.drop(columns=["Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary"])
    out = pd.concat([sel.drop(columns=["off_end"]).reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    path = Path(a.out) if a.out else common.DERIVED / "release" / f"news_rc_f{a.file}_{'_'.join(tickers)}.parquet"
    out.to_parquet(path, index=False, compression="zstd")
    print("wrote", path, len(out))


if __name__ == "__main__":
    main()
