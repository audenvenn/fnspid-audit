"""Helpers shared by the news audit, patch and experiment scripts."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402

FILE_NAMES = {0: "All_external.csv", 1: "nasdaq_exteral_data.csv"}

# Reuters rows switch timestamp convention between these two dates (see 22_news_timestamps.py):
REUTERS_UTC_UNTIL = pd.Timestamp("2009-11-14")    # stamps < this are true UTC
REUTERS_ET_FROM = pd.Timestamp("2009-11-16")      # stamps >= this are New York wall-clock labelled "UTC"
BENZINGA_SHIFT_H = 8                              # true UTC = FNSPID stamp + 8 h (row-level check vs upstream)


def parts(file_id: int) -> list[str]:
    p = sorted(glob.glob(str(common.DERIVED / "news_rows" / f"f{file_id}" / "part-*.parquet")))
    if not p:
        raise FileNotFoundError(f"no parsed parts for file {file_id}; run 20_news_stream.py first")
    return p


def dataset(file_ids=(0, 1)):
    files = []
    for f in file_ids:
        files += parts(f)
    return ds.dataset(files, format="parquet")


def load(columns, file_ids=(0, 1), filter=None) -> pd.DataFrame:
    t = dataset(file_ids).to_table(columns=list(columns), filter=filter)
    return t.to_pandas(types_mapper=pd.ArrowDtype)


def domain(url: pd.Series) -> pd.Series:
    return url.astype("string[pyarrow]").str.extract(r"^https?://([^/]+)/", expand=False).fillna("")


def source_of(url: pd.Series) -> pd.Series:
    d = domain(url).astype(str)
    s = np.where(d.str.contains("benzinga.com"), "benzinga",
        np.where(d.str.contains("reuters.com"), "reuters",
        np.where(d.str.contains("lenta.ru"), "lenta",
        np.where(d.str.contains("bloomberg.com"), "bloomberg",
        np.where(d.str.contains("nasdaq.com"), "nasdaq",
        np.where(d == "", "(no url)", "partner:" + d.str.replace("www.", "", regex=False)))))))
    return pd.Series(s, index=url.index)


def stamp(date_raw: pd.Series) -> pd.Series:
    """FNSPID 'Date' strings are 'YYYY-MM-DD HH:MM:SS UTC' (verified: all rows, length 23)."""
    return pd.to_datetime(date_raw.astype(str).str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce")


def is_date_only(ts: pd.Series) -> pd.Series:
    return (ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0)


def ny_to_utc(naive_local: pd.Series) -> pd.Series:
    loc = naive_local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    return loc.dt.tz_convert("UTC").dt.tz_localize(None)


def corrected_utc(src: pd.Series, ts: pd.Series, date_only: pd.Series):
    """Return (utc, status) where status is one of
    date_only | fixed_benzinga_+8h | ok_reuters_utc | fixed_reuters_et_wallclock | ambiguous_reuters_switch |
    uncertain_nasdaq_relative | uncertain_other | unparseable."""
    utc = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    status = pd.Series("uncertain_other", index=ts.index, dtype=object)
    # (converted to a categorical at the end to keep memory low on 8M+ rows)
    s = src.astype(str)
    timed = ~date_only & ts.notna()
    m = timed & (s == "benzinga")
    utc[m] = ts[m] + pd.Timedelta(hours=BENZINGA_SHIFT_H)
    status[m] = "fixed_benzinga_+8h"
    r = timed & (s == "reuters")
    m1 = r & (ts < REUTERS_UTC_UNTIL)
    utc[m1] = ts[m1]
    status[m1] = "ok_reuters_utc"
    m2 = r & (ts >= REUTERS_ET_FROM)
    utc[m2] = ny_to_utc(ts[m2])
    status[m2] = np.where(utc[m2].isna(), "ambiguous_dst_hour", "fixed_reuters_et_wallclock")
    m3 = r & ~m1 & ~m2
    status[m3] = "ambiguous_reuters_switch"
    m4 = timed & (s == "nasdaq")
    status[m4] = "uncertain_nasdaq_relative"
    status[date_only & ts.notna()] = "date_only"
    status[ts.isna()] = "unparseable"
    return utc, status.astype("category")


def url_date(url: pd.Series) -> pd.Series:
    """Publication date embedded in the URL, where the site puts one:
    nasdaq.com '...-2015-11-23', reuters.com '/article/2011/09/27/', lenta.ru '/news/2009/03/17/',
    bloomberg.com '/news/2011-08-26/'."""
    u = url.astype(str)
    a = u.str.extract(r"nasdaq\.com/.*-(\d{4}-\d{2}-\d{2})$", expand=False)
    b = u.str.extract(r"/(?:article|news)/(\d{4})/(\d{2})/(\d{2})/", expand=True)
    b = (b[0] + "-" + b[1] + "-" + b[2]).where(b[0].notna())
    c = u.str.extract(r"bloomberg\.com/news/(\d{4}-\d{2}-\d{2})/", expand=False)
    s = a.fillna(b).fillna(c)
    return pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")


def conservative_date(ts: pd.Series, date_only: pd.Series, url_d: pd.Series) -> pd.Series:
    """For date-only rows: the later of the stamp date and the URL date (the stamp is sometimes a day
    early, e.g. call transcripts stamped with the call date); NaT for timed rows."""
    d = ts.dt.normalize()
    later = d.where(url_d.isna() | (d >= url_d), url_d)
    return later.where(date_only)
