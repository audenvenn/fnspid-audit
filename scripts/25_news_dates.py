#!/usr/bin/env python3
"""What calendar does a date-only FNSPID stamp refer to? Checks on the nasdaq file.

1. Many nasdaq.com URLs end in the publication date ("...-2015-11-23"): compare with the stamp date.
2. Articles that say when they were written ("as of 11:34 a.m. ET on Wednesday", "At 12:16 p.m. ET ...
   Wednesday"): compare the weekday named right after the time with the stamp's weekday.
3. Benzinga URLs carry /YY/MM/: share of date-only rows whose stamp month equals the URL month.
Output: data/derived/results/news_dates.json
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
import pyarrow.compute as pc  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def main():
    out = {}
    q = nl.load(["date_raw", "url", "title", "txt_time_kind", "txt_time_ctx"], file_ids=(1,),
                filter=pc.match_substring(ds.field("url"), "nasdaq.com"))
    q["ts"] = nl.stamp(q["date_raw"])
    q["date_only"] = nl.is_date_only(q["ts"])
    ud = pd.to_datetime(q["url"].astype(str).str.extract(r"-(\d{4}-\d{2}-\d{2})$", expand=False), errors="coerce")
    diff = (q["ts"].dt.normalize() - ud).dt.days
    m = ud.notna()
    vc = diff[m].value_counts()
    out["url_date"] = {"rows_with_url_date": int(m.sum()), "share_of_nasdaq_rows": float(m.mean()), "nasdaq_rows": int(len(q)),
                       "stamp_minus_url_days": {str(int(k)): int(v) for k, v in vc.head(8).items()},
                       "equal_share": float((diff[m] == 0).mean())}
    mm = m & (diff == -1)
    out["url_date"]["minus1_by_year"] = q.loc[mm, "ts"].dt.year.value_counts().sort_index().to_dict()
    out["url_date"]["minus1_date_only_share"] = float(q.loc[mm, "date_only"].mean()) if mm.any() else None
    tr = q["title"].astype(str).str.contains(r"Earnings Call Transcript", case=False, regex=True)
    out["url_date"]["minus1_earnings_call_transcripts"] = int((mm & tr).sum())
    out["url_date"]["transcripts_with_url_date"] = int((m & tr).sum())
    out["url_date"]["transcripts_stamped_minus1_share"] = float((mm & tr).sum() / max((m & tr).sum(), 1))
    # weekday named after a time-of-writing phrase
    k = q[(q["txt_time_kind"] == 2) & q["date_only"]].copy()
    ctx = k["txt_time_ctx"].astype(str)
    after = ctx.str.extract(r"(?:ET|EDT|EST|Eastern(?: Time)?)\b[,.]?\s*(?:on\s+|this\s+)?(" + "|".join(DAYS) + r")\b", expand=False)
    k = k[after.notna()]
    named = after[after.notna()].map({d: i for i, d in enumerate(DAYS)})
    wd = k["ts"].dt.day_of_week
    delta = ((named - wd) % 7)
    out["weekday_check"] = {"rows": int(len(k)), "same_weekday_share": float((delta == 0).mean()),
                            "named_is_next_day_share": float((delta == 1).mean()),
                            "named_is_previous_day_share": float((delta == 6).mean())}
    # Benzinga URL month vs stamp month for date-only rows
    b = nl.load(["date_raw", "url"], file_ids=(0,), filter=pc.match_substring(ds.field("url"), "benzinga.com"))
    b["ts"] = nl.stamp(b["date_raw"])
    ym = b["url"].astype(str).str.extract(r"/(\d\d)/(\d\d)/\d{4,9}/", expand=True)
    ok = ym[0].notna()
    sym = b["ts"].dt.strftime("%y%m")
    eq = (sym[ok] == (ym.loc[ok, 0] + ym.loc[ok, 1]))
    do = nl.is_date_only(b["ts"])[ok]
    out["benzinga_url_month"] = {"rows": int(ok.sum()), "date_only_equal_share": float(eq[do].mean()),
                                 "timed_equal_share": float(eq[~do].mean())}
    (common.RESULTS / "news_dates.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
