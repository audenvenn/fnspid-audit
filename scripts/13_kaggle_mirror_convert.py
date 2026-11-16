#!/usr/bin/env python3
"""Convert the third-party HF mirror of the Kaggle Benzinga file raw_analyst_ratings into a compact parquet.

The Kaggle dataset ("Massive Stock News Analysis DB for NLP/backtests", miguelaenlle) is FNSPID's
upstream for its timed Benzinga rows (FNSPID's README links it). Its timestamps carry an explicit
US-Eastern UTC offset, e.g. "2020-06-05 10:30:54-04:00", which gives row-level ground truth.

Input : data/raw/kaggle_mirror/raw_analyst_ratings.jsonl  (sha256 checked)
Output: data/derived/kaggle_analyst_ratings.parquet  (kid, headline, url, publisher, date_orig, stock)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import pandas as pd  # noqa: E402

SRC = common.RAW / "kaggle_mirror" / "raw_analyst_ratings.jsonl"


def main():
    out = common.DERIVED / "kaggle_analyst_ratings.parquet"
    if not SRC.exists() and out.exists():
        print("already converted:", out)
        return
    h = hashlib.sha256()
    with open(SRC, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    assert h.hexdigest() == common.KAGGLE_MIRROR_SHA256, "mirror checksum mismatch"
    rows = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append((d.get("Unnamed: 0"), d.get("headline"), d.get("url"), d.get("publisher"),
                         d.get("date"), d.get("stock")))
    k = pd.DataFrame(rows, columns=["kid", "headline", "url", "publisher", "date_orig", "stock"])
    k.to_parquet(common.DERIVED / "kaggle_analyst_ratings.parquet", index=False, compression="zstd")
    tz = k["date_orig"].str.extract(r"([+-]\d\d:\d\d)$")[0].fillna("none")
    print(len(k), tz.value_counts().to_dict())


if __name__ == "__main__":
    main()
