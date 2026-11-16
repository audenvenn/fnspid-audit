#!/usr/bin/env python3
"""Timestamp defects in FNSPID news: verification of the conversion error and per-source corrections.

1. Benzinga (All_external): row-level comparison with the upstream Kaggle file, whose stamps carry an
   explicit offset ("2020-06-05 10:30:54-04:00"). Expect FNSPID = upstream local clock - 4 h, i.e. the
   offset was added with the wrong sign; true UTC = FNSPID + 8 h.
2. Session-anchored headlines (independent of the upstream file): in-window share in New York time for
   whole-hour shifts of the FNSPID stamps (finlint.news.session_fit).
3. Reuters (All_external): the same test by regime; the convention switches in Nov 2009.
4. nasdaq file: timed rows are whole hours (relative "N hours ago" stamps).
Output: data/derived/results/timestamps.json (+ session_fit_*.csv)
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
from finlint import news as fn  # noqa: E402

# "Wall St opens lower" is filed minutes after 09:30 New York time; "Wall St ends higher" after 16:00.
# Bare "open" is excluded because it mostly appears in pre-open stories ("set to open", "futures point to").
REUTERS_RULES = [
    ("us_stocks_opens", r"^US STOCKS.*\b(?:opens|opened)\b", (9 * 60 + 30, 10 * 60 + 30)),
    ("us_stocks_ends", r"^US STOCKS.*\b(?:ends|ended|closes|closed)\b", (16 * 60, 18 * 60)),
]


def fit_table(ts, text, rules=fn.SESSION_RULES, shifts=range(-12, 13)):
    f = fn.session_fit(ts.reset_index(drop=True), text.reset_index(drop=True), rules=rules, shifts=shifts, min_rows=10)
    return f


def ny(ts_utc: pd.Series) -> pd.Series:
    return ts_utc.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)


def main():
    common.ensure_dirs()
    out = {}
    # ---------------- Benzinga ----------------
    b = nl.load(["row", "date_raw", "title", "url", "ticker"], file_ids=(0,),
                filter=pc.match_substring(ds.field("url"), "benzinga.com"))
    b["ts"] = nl.stamp(b["date_raw"])
    b["date_only"] = nl.is_date_only(b["ts"])
    # join on a 64-bit hash of (url, ticker) instead of the strings (memory)
    def key(u, t):
        return pd.util.hash_array((u.astype(str) + "\x1f" + t.astype(str)).to_numpy(dtype=object)).view(np.int64)
    b["key"] = key(b["url"], b["ticker"])
    b["title_h"] = pd.util.hash_array(b["title"].astype(str).to_numpy(dtype=object)).view(np.int64)
    b = b.drop(columns=["url", "date_raw"])
    k = pd.read_parquet(common.DERIVED / "kaggle_analyst_ratings.parquet", columns=["url", "stock", "date_orig", "headline"])
    k["key"] = key(k["url"], k["stock"])
    k["headline_h"] = pd.util.hash_array(k["headline"].astype(str).to_numpy(dtype=object)).view(np.int64)
    k = k.drop(columns=["url", "headline", "stock"]).drop_duplicates("key", keep=False)
    bb = b.drop_duplicates("key", keep=False)
    m = bb.drop(columns=["title"]).merge(k, on="key", how="inner")
    m["k_timed"] = m["date_orig"].str.match(r".*[+-]\d\d:\d\d$")
    kt = m[m["k_timed"]].copy()
    kt["k_local"] = pd.to_datetime(kt["date_orig"].str.slice(0, 19))
    kt["k_off_h"] = kt["date_orig"].str.slice(19, 22).astype(int)
    kt["k_utc"] = kt["k_local"] - pd.to_timedelta(kt["k_off_h"], unit="h")
    d_local = (kt["ts"] - kt["k_local"]).dt.total_seconds() / 3600
    d_utc = (kt["ts"] - kt["k_utc"]).dt.total_seconds() / 3600
    kd = m[~m["k_timed"]]
    out["benzinga"] = {
        "fnspid_rows": int(len(b)), "fnspid_timed": int((~b["date_only"]).sum()),
        "fnspid_date_only": int(b["date_only"].sum()), "date_only_share": float(b["date_only"].mean()),
        "upstream_rows": int(len(k)), "matched_unique_url_ticker": int(len(m)),
        "matched_timed": int(len(kt)), "upstream_offsets": kt["k_off_h"].value_counts().to_dict(),
        "fnspid_minus_upstream_local_h": d_local.round(3).value_counts().head(3).to_dict(),
        "fnspid_minus_true_utc_h": d_utc.round(3).value_counts().head(3).to_dict(),
        "share_exactly_minus8h": float((d_utc.round(6) == -8).mean()),
        "matched_date_only": int(len(kd)),
        "date_only_same_date_share": float((kd["ts"].dt.normalize() == pd.to_datetime(kd["date_orig"].str.slice(0, 10))).mean()),
        "headline_identical_share": float((m["title_h"] == m["headline_h"]).mean()),
    }
    t = b[~b["date_only"]].copy()
    h = t["ts"].dt.hour
    out["benzinga"]["timed_label_hour_le_08_share"] = float((h <= 8).mean())
    out["benzinga"]["timed_label_hour_ge_15_share"] = float((h >= 15).mean())
    t["true_utc"] = t["ts"] + pd.Timedelta(hours=nl.BENZINGA_SHIFT_H)
    loc = ny(t["true_utc"])
    out["benzinga"]["true_ny_time_profile"] = {
        "share_before_0930": float((loc.dt.hour * 60 + loc.dt.minute < 570).mean()),
        "share_0930_1600": float(((loc.dt.hour * 60 + loc.dt.minute >= 570) & (loc.dt.hour < 16)).mean()),
        "share_after_1600": float((loc.dt.hour >= 16).mean()),
        "share_weekend": float((loc.dt.day_of_week >= 5).mean()),
    }
    out["benzinga"]["fnspid_date_differs_from_ny_date"] = int((t["ts"].dt.normalize() != loc.dt.normalize()).sum())
    fit = fit_table(t["ts"], t["title"].astype(str))
    fit.to_csv(common.RESULTS / "session_fit_benzinga.csv", index=False)
    sm = fn.session_matches(t["title"].astype(str).reset_index(drop=True))
    out["benzinga"]["session_anchored_rows"] = int(len(sm))
    out["benzinga"]["session_counts"] = sm["session"].value_counts().to_dict()
    out["benzinga"]["in_window_as_labelled_utc"] = float(fit.loc[fit.shift_h == 0, "in_window"].iloc[0])
    out["benzinga"]["in_window_shift_+8h"] = float(fit.loc[fit.shift_h == 8, "in_window"].iloc[0])
    out["benzinga"]["best_shift"] = fit.sort_values("in_window").iloc[-1][["shift_h", "in_window"]].to_dict()
    # per-session in-window shares for the corrected times
    t2 = t.reset_index(drop=True)
    mm = sm.join(t2[["true_utc"]])
    l2 = ny(mm["true_utc"])
    mins = l2.dt.hour * 60 + l2.dt.minute
    mm["inside"] = (mins >= mm["lo"]) & (mins < mm["hi"])
    lab = ny(t2.loc[mm.index, "ts"])
    mins0 = lab.dt.hour * 60 + lab.dt.minute
    mm["inside_lab"] = (mins0 >= mm["lo"]) & (mins0 < mm["hi"])
    # by season: the upstream clock (= FNSPID + 4 h, written with a fixed -04:00 offset) runs one hour ahead
    # of New York wall-clock time in winter, while the corrected New York time is the same in both seasons
    utc_ = t2.loc[mm.index, "true_utc"]
    winter = ((ny(utc_) - utc_).dt.total_seconds() / 3600 == -5).to_numpy()
    up_local = t2.loc[mm.index, "ts"] + pd.Timedelta(hours=4)
    hh = lambda x: x.dt.hour + x.dt.minute / 60  # noqa: E731
    sea = pd.DataFrame({"session": mm["session"].to_numpy(), "winter": winter,
                        "upstream_local_h": hh(up_local).to_numpy(), "ny_corrected_h": hh(l2).to_numpy()})
    out["benzinga"]["season_medians"] = {f"{s_}|{'winter' if w_ else 'summer'}": {k: round(float(v), 2) for k, v in g_[["upstream_local_h", "ny_corrected_h"]].median().items()} | {"n": int(len(g_))}
                                         for (s_, w_), g_ in sea.groupby(["session", "winter"])}
    out["benzinga"]["per_session"] = mm.groupby("session").agg(n=("inside", "size"), in_window_corrected=("inside", "mean"),
                                                               in_window_as_labelled=("inside_lab", "mean")).round(4).to_dict("index")
    del t

    del b, bb, k, m, kt, kd
    import gc
    gc.collect()
    # ---------------- Reuters ----------------
    flt = pc.match_substring(ds.field("url"), "reuters.com")
    proj = {"date_raw": ds.field("date_raw"),
            "us_stocks": pc.starts_with(ds.field("title"), "US STOCKS"),
            "prn": pc.match_substring_regex(ds.field("url"), r"\+PRN\d{8}")}
    r = nl.dataset((0,)).to_table(columns=proj, filter=flt).to_pandas(types_mapper=pd.ArrowDtype)
    tt = nl.dataset((0,)).to_table(columns=["title"], filter=flt & pc.starts_with(ds.field("title"), "US STOCKS")).to_pandas()
    r["title"] = ""
    r.loc[r["us_stocks"].astype(bool).to_numpy(), "title"] = tt["title"].to_numpy()
    r["prn"] = r["prn"].astype(bool)
    r["ts"] = nl.stamp(r["date_raw"])
    r["date_only"] = nl.is_date_only(r["ts"])
    out["reuters"] = {"rows": int(len(r)), "date_only": int(r["date_only"].sum()),
                      "first": str(r["ts"].min()), "last": str(r["ts"].max()),
                      "rows_before_switch": int((r["ts"] < nl.REUTERS_UTC_UNTIL).sum()),
                      "rows_after_switch": int((r["ts"] >= nl.REUTERS_ET_FROM).sum()),
                      "rows_in_switch_gap": int(((r["ts"] >= nl.REUTERS_UTC_UNTIL) & (r["ts"] < nl.REUTERS_ET_FROM)).sum())}
    for name, sel in [("pre_switch", r["ts"] < nl.REUTERS_UTC_UNTIL), ("post_switch", r["ts"] >= nl.REUTERS_ET_FROM)]:
        out["reuters"][name] = {"timed_rows": int((sel & ~r["date_only"]).sum())}
        x = r[sel & ~r["date_only"] & r["us_stocks"].astype(bool)]      # only "US STOCKS..." titles can be anchored
        f = fit_table(x["ts"], x["title"], rules=REUTERS_RULES)
        f.to_csv(common.RESULTS / f"session_fit_reuters_{name}.csv", index=False)
        best = f.sort_values("in_window").iloc[-1]
        # DST split for the post-switch regime: wall-clock labels need +4 h in summer, +5 h in winter
        off = (x["ts"] - nl.ny_to_utc(x["ts"])).dt.total_seconds() / 3600   # -4 in EDT, -5 in EST
        dst = pd.Series(np.where(off == -4, True, np.where(off == -5, False, None)), index=x.index)
        per = {}
        for flag in (True, False):
            y = x[dst == flag]
            g = fit_table(y["ts"], y["title"], rules=REUTERS_RULES)
            if len(g):
                bb_ = g.sort_values("in_window").iloc[-1]
                per["summer" if flag else "winter"] = {"n": int(bb_["n"]), "best_shift_h": int(bb_["shift_h"]),
                                                       "in_window_best": float(bb_["in_window"]),
                                                       "in_window_shift0": float(g.loc[g.shift_h == 0, "in_window"].iloc[0])}
        out["reuters"][name].update({"anchored_rows": int(best["n"]),
                                "best_shift_h": int(best["shift_h"]), "in_window_best": float(best["in_window"]),
                                "in_window_shift0": float(f.loc[f.shift_h == 0, "in_window"].iloc[0]), "by_dst": per})
    # where does the convention switch? PR Newswire releases cluster on 06:00-09:59 New York time and are
    # very rare between 01:00 and 05:00. Label hours 6-9 are therefore common if stamps are New York wall
    # clock, and rare if stamps are UTC (they would be 01:00-05:59 New York time).
    w = r[r["prn"] & ~r["date_only"] & (r["ts"] >= "2009-10-15") & (r["ts"] < "2009-12-16")]
    g = pd.DataFrame({"day": w["ts"].dt.normalize(), "h6_9": w["ts"].dt.hour.between(6, 9)})
    g = g.groupby("day").agg(n=("h6_9", "size"), h6_9=("h6_9", "mean"))
    g = g[g["n"] >= 300]                                   # weekdays only
    wall = (g["h6_9"] > 0.15).to_numpy()
    first_wall = g.index[wall].min()
    out["reuters"]["switch_evidence"] = {
        "share_label_h6_9_utc_regime_max": round(float(g.loc[g.index < first_wall, "h6_9"].max()), 3),
        "share_label_h6_9_wall_regime_min": round(float(g.loc[g.index >= first_wall, "h6_9"].min()), 3),
        "last_weekday_utc_regime": str(g.index[g.index < first_wall].max().date()),
        "first_weekday_wallclock_regime": str(first_wall.date()),
        "weekdays_after_switch_classified_utc": int((~wall[g.index >= first_wall]).sum()),
        "daily_n_and_share_h6_9": {str(k.date()): [int(v["n"]), round(float(v["h6_9"]), 3)]
                                   for k, v in g.loc["2009-11-09":"2009-11-20"].iterrows()}}
    # corrected profile for PR Newswire releases (they cluster on round New York times before the open)
    prn = r["prn"] & ~r["date_only"]
    src = pd.Series("reuters", index=r.index)
    utc, status = nl.corrected_utc(src, r["ts"], r["date_only"])
    out["reuters"]["status_counts"] = status.value_counts().to_dict()
    loc = ny(utc[prn].dropna())
    out["reuters"]["prn_rows"] = int(prn.sum())
    out["reuters"]["prn_top_ny_times_corrected"] = loc.dt.strftime("%H:%M").value_counts().head(6).to_dict()
    lab_post = r.loc[prn & (r["ts"] >= nl.REUTERS_ET_FROM), "ts"]
    out["reuters"]["prn_top_label_times_post_switch"] = lab_post.dt.strftime("%H:%M").value_counts().head(6).to_dict()
    u = utc.dropna()
    out["reuters"]["fnspid_date_differs_from_ny_date"] = int((r.loc[u.index, "ts"].dt.normalize() != ny(u).dt.normalize()).sum())
    del r

    # ---------------- nasdaq.com rows of the nasdaq file ----------------
    q = nl.load(["date_raw"], file_ids=(1,), filter=pc.match_substring(ds.field("url"), "nasdaq.com"))
    q["ts"] = nl.stamp(q["date_raw"])
    q["date_only"] = nl.is_date_only(q["ts"])
    timed = q[~q["date_only"] & q["ts"].notna()]
    out["nasdaq_file"] = {"rows": int(len(q)), "date_only": int(q["date_only"].sum()),
                          "date_only_share": float(q["date_only"].mean()), "timed": int(len(timed)),
                          "timed_whole_hour_share": float(((timed["ts"].dt.minute == 0) & (timed["ts"].dt.second == 0)).mean()),
                          "timed_first": str(timed["ts"].min()), "timed_last": str(timed["ts"].max()),
                          "timed_by_year": timed["ts"].dt.year.value_counts().sort_index().to_dict(),
                          "unparseable": int(q["ts"].isna().sum())}
    (common.RESULTS / "timestamps.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
