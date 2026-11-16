#!/usr/bin/env python3
"""Corrected news release candidate: a sidecar patch with one row per FNSPID news record.

Join key: (file, row) where file 0 = Stock_news/All_external.csv, 1 = Stock_news/nasdaq_exteral_data.csv
and row = 0-based data-record index in file order (``off`` = byte offset of the record, for range reads).

Columns
  source            benzinga | reuters | lenta | bloomberg | nasdaq | partner:<domain> | (no url)
  ticker            Stock_symbol as published (empty for most Reuters/Bloomberg/Lenta rows)
  date_raw          FNSPID Date string as published
  date_only         True if the stamp has no time of day (00:00:00)
  ts_utc            corrected UTC publication time (null when not recoverable)
  ts_status         date_only | fixed_benzinga_+8h | ok_reuters_utc | fixed_reuters_et_wallclock |
                    ambiguous_reuters_switch | ambiguous_dst_hour | uncertain_nasdaq_relative | unparseable
  date_ny           New York calendar date: from ts_utc for timed rows, the stored date for date-only rows
  url_date          publication date embedded in the URL (nasdaq.com, reuters.com, lenta.ru, bloomberg.com), if any
  stamp_before_url_date / stamp_after_url_date   date-only stamp disagrees with the URL date
  session_no_leak   first trading session whose 16:00 New York close is after the news is known to be public
                    (date-only rows: first session strictly after the later of date_ny and url_date);
                    null if not determinable
  has_ticker, ticker_in_prices, cyrillic_title, epoch_date (1969-12-31/1970-01-01 artefact), pre1990
  dup_first_row     row of the first identical record (same ticker, date, url, title, body) in the same file, else -1
  in_other_file     the same (ticker, url) also appears in the other file
  enc_title, enc_article   encoding-problem bitmasks (1 U+FFFD, 2 mojibake, 4 HTML entity, 8 control char, 32 letter-spaced)
Output: data/derived/release/news_patch.parquet  and  release/news_patch_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import newslib as nl  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from finlint import lookahead  # noqa: E402

REL = common.DERIVED / "release"
DICT = pa.dictionary(pa.int32(), pa.string())
SCHEMA = pa.schema([
    ("file", pa.int8()), ("row", pa.int64()), ("off", pa.int64()), ("source", DICT), ("ticker", DICT),
    ("date_raw", pa.string()), ("date_only", pa.bool_()), ("ts_utc", pa.timestamp("ns")), ("ts_status", DICT),
    ("date_ny", pa.date32()), ("url_date", pa.date32()), ("stamp_before_url_date", pa.bool_()),
    ("stamp_after_url_date", pa.bool_()), ("session_no_leak", pa.date32()), ("has_ticker", pa.bool_()),
    ("ticker_in_prices", pa.bool_()), ("cyrillic_title", pa.bool_()), ("epoch_date", pa.bool_()), ("pre1990", pa.bool_()),
    ("dup_first_row", pa.int64()), ("in_other_file", pa.bool_()), ("enc_title", pa.int16()), ("enc_article", pa.int16())])
COLS = ["file", "row", "off", "date_raw", "title", "ticker", "url", "art_hash", "enc_t", "enc_a"]


def h64(s: pd.Series) -> np.ndarray:
    return pd.util.hash_array(s.astype(str).to_numpy(dtype=object)).view(np.int64)


def sessions_from_prices():
    # trading sessions = dates carrying >= 50 price rows (per-date counts from 11_prices_boundaries.py)
    ev = pd.read_csv(common.RESULTS / "price_date_events.csv", parse_dates=["date"])
    vc = ev.set_index("date")["n"]
    return pd.to_datetime(pd.Series(vc[vc >= 50].index)).sort_values().to_numpy("datetime64[D]")


def main():
    REL.mkdir(parents=True, exist_ok=True)
    S = sessions_from_prices()
    price_tickers = set(pd.read_csv(common.RESULTS / "price_series_span.csv")["ticker"])
    # pass 1: keys for duplicate detection (sort-based, one file at a time, to bound memory)
    empty_url = h64(pd.Series([""]))[0]
    dup_first, tkurl_sets, tkurl_rows = {}, {}, {}
    for fid in (0, 1):
        rows, krec, ktu, hasurl = [], [], [], []
        for path in nl.parts(fid):
            d = pq.read_table(path, columns=["row", "date_raw", "title", "ticker", "url", "art_hash"]).to_pandas()
            rows.append(d["row"].to_numpy(np.int64))
            krec.append(h64(d["ticker"] + "\x1f" + d["date_raw"] + "\x1f" + d["url"] + "\x1f" + d["title"])
                        ^ d["art_hash"].to_numpy(np.int64))
            ktu.append(h64(d["ticker"] + "\x1f" + d["url"]))
            hasurl.append((d["url"].str.len() > 0).to_numpy())
        rows, krec, ktu, hasurl = (np.concatenate(x) for x in (rows, krec, ktu, hasurl))
        order = np.argsort(krec, kind="stable")
        xs = krec[order]
        start = np.r_[True, xs[1:] != xs[:-1]]
        grp_first_row = rows[order][start][np.cumsum(start) - 1]     # first row of each record group, per sorted position
        df = np.full(len(rows), -1, np.int64)
        dup_pos = ~start
        df[order[dup_pos]] = grp_first_row[dup_pos]
        dup_first[fid] = df
        tkurl_sets[fid] = np.unique(ktu[hasurl])
        tkurl_rows[fid] = np.where(hasurl, ktu, empty_url)
        del order, xs, start, grp_first_row, krec, rows
    both = np.intersect1d(tkurl_sets[0], tkurl_sets[1], assume_unique=True)
    in_other = {fid: np.isin(tkurl_rows[fid], both) & (tkurl_rows[fid] != empty_url) for fid in (0, 1)}
    del tkurl_sets, tkurl_rows, both

    writer = None
    summ = {"rows": 0, "status": {}, "date_only": 0, "session_no_leak_null": 0, "dups": 0, "in_other_file": 0,
            "stamp_before_url_date": 0, "stamp_after_url_date": 0, "with_url_date": 0}
    for fid in (0, 1):
        for path in nl.parts(fid):
            d = pq.read_table(path, columns=COLS).to_pandas()
            src = nl.source_of(d["url"])
            ts = nl.stamp(d["date_raw"])
            do = nl.is_date_only(ts)
            utc, status = nl.corrected_utc(src, ts, do)
            ny = utc.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
            date_ny = ny.where(~do, ts.dt.normalize())
            ud = nl.url_date(d["url"])
            cons = nl.conservative_date(ts, do, ud)
            true = utc.where(~do, cons)
            sess = lookahead.leakage_free_session(true, do.to_numpy(), S)
            r0, r1 = int(d["row"].iloc[0]), int(d["row"].iloc[-1]) + 1      # rows are 0..n-1 in part order
            kk = {"dup_first_row": dup_first[fid][r0:r1], "in_other_file": in_other[fid][r0:r1]}
            out = pd.DataFrame({
                "file": d["file"].astype(np.int8), "row": d["row"].astype(np.int64), "off": d["off"].astype(np.int64),
                "source": src.astype(str), "ticker": d["ticker"], "date_raw": d["date_raw"], "date_only": do.to_numpy(),
                "ts_utc": utc, "ts_status": status.astype(str), "date_ny": date_ny.dt.date,
                "url_date": ud.dt.date, "stamp_before_url_date": (do & ud.notna() & (ts.dt.normalize() < ud)).to_numpy(),
                "stamp_after_url_date": (do & ud.notna() & (ts.dt.normalize() > ud)).to_numpy(),
                "session_no_leak": pd.to_datetime(sess).dt.date,
                "has_ticker": (d["ticker"].str.len() > 0).to_numpy(), "ticker_in_prices": d["ticker"].isin(price_tickers).to_numpy(),
                "cyrillic_title": d["title"].str.contains("[А-Яа-яЁё]", regex=True).to_numpy(),
                "epoch_date": d["date_raw"].str.startswith(("1969-12-31", "1970-01-01")).to_numpy(),
                "pre1990": (ts < "1990-01-01").to_numpy(),
                "dup_first_row": kk["dup_first_row"], "in_other_file": kk["in_other_file"],
                "enc_title": d["enc_t"].astype(np.int16), "enc_article": d["enc_a"].astype(np.int16)})
            tbl = pa.Table.from_pandas(out, preserve_index=False).select(SCHEMA.names).cast(SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(REL / "news_patch.parquet", SCHEMA, compression="zstd", compression_level=9)
            writer.write_table(tbl)
            summ["rows"] += len(out)
            for k, v in out["ts_status"].value_counts().items():
                summ["status"][k] = summ["status"].get(k, 0) + int(v)
            summ["date_only"] += int(out["date_only"].sum())
            summ["session_no_leak_null"] += int(out["session_no_leak"].isna().sum())
            summ["dups"] += int((out["dup_first_row"] >= 0).sum())
            summ["in_other_file"] += int(out["in_other_file"].sum())
            summ["stamp_before_url_date"] += int(out["stamp_before_url_date"].sum())
            summ["stamp_after_url_date"] += int(out["stamp_after_url_date"].sum())
            summ["with_url_date"] += int(out["url_date"].notna().sum())
            print(path, len(out), flush=True)
    writer.close()
    (REL / "news_patch_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
