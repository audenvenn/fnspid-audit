#!/usr/bin/env python3
"""Convert FNSPID Stock_price/full_history.zip (7,693 per-ticker CSVs) into one normalized parquet.

Reads members directly from the zip (no extraction), keeps every row exactly as published (no dedup,
no sorting fixes), and records per-file metadata (header layout, row counts, ordering).

Output: data/derived/prices.parquet   columns: ticker, date, open, high, low, close, adj_close, volume, layout
        data/derived/results/prices_files.csv   one row per member file
"""
from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

LAYOUTS = {
    "date,open,high,low,close,adj close,volume": 0,
    "date,volume,open,high,low,close,adj close": 1,
}
SCHEMA = pa.schema([("ticker", pa.dictionary(pa.int32(), pa.string())), ("date", pa.date32()),
                    ("open", pa.float64()), ("high", pa.float64()), ("low", pa.float64()),
                    ("close", pa.float64()), ("adj_close", pa.float64()), ("volume", pa.float64()),
                    ("layout", pa.int8())])


def main():
    common.ensure_dirs()
    zp = common.PRICE_ZIP
    h = hashlib.sha256()
    with open(zp, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    assert h.hexdigest() == common.PRICE_ZIP_SHA256, "price zip checksum mismatch"
    zf = zipfile.ZipFile(zp)
    members = sorted((i for i in zf.infolist()
                      if i.filename.startswith("full_history/") and i.filename.endswith(".csv")), key=lambda i: i.filename)
    out = common.DERIVED / "prices.parquet"
    writer = pq.ParquetWriter(out, SCHEMA, compression="zstd", compression_level=6)
    meta, buf, nbuf = [], [], 0
    for k, info in enumerate(members):
        tk = info.filename[len("full_history/"):-4]
        raw = zf.read(info)
        text = raw.decode("utf-8")
        header = text.split("\n", 1)[0].strip()
        layout = LAYOUTS[header]
        df = pd.read_csv(io.StringIO(text), dtype={"date": str})
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        d = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        dd = d.values.astype("datetime64[D]")
        meta.append(dict(ticker=tk, layout=layout, n=len(df), n_bad_date=int(d.isna().sum()),
                         first=str(d.min().date()) if len(df) else "", last=str(d.max().date()) if len(df) else "",
                         sorted_desc=bool((np.diff(dd.astype("int64")) < 0).all()) if len(df) > 1 else True,
                         n_dup_dates=int(pd.Series(dd).duplicated().sum()),
                         mtime="%04d-%02d-%02d %02d:%02d" % info.date_time[:5]))
        part = pd.DataFrame({"ticker": tk, "date": dd, "open": df["open"].astype(float), "high": df["high"].astype(float),
                             "low": df["low"].astype(float), "close": df["close"].astype(float),
                             "adj_close": df["adj_close"].astype(float), "volume": df["volume"].astype(float),
                             "layout": np.int8(layout)})
        buf.append(part)
        nbuf += len(part)
        if nbuf > 2_000_000 or k == len(members) - 1:
            t = pd.concat(buf, ignore_index=True)
            t["ticker"] = t["ticker"].astype("category")
            writer.write_table(pa.Table.from_pandas(t, schema=SCHEMA, preserve_index=False))
            buf, nbuf = [], 0
            print(f"{k + 1}/{len(members)} files", flush=True)
    writer.close()
    m = pd.DataFrame(meta)
    m.to_csv(common.RESULTS / "prices_files.csv", index=False)
    print(m.describe(include="all").T.to_string())
    print("rows", int(m.n.sum()), "files", len(m))


if __name__ == "__main__":
    main()
