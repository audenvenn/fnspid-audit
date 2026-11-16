"""Checks for news tables: date-only share, timezone sanity from session-anchored text, duplicates.

Timestamps are given as a column of naive datetimes (or strings) together with the time zone they are
*labelled* in (``label_tz``, default UTC). The timezone test asks: if the labels were right, would
headlines that name a trading session ("pre-market", "mid-day", "after-hours") fall inside that session
in New York time? It also scans whole-hour shifts and reports the shift that best explains the text.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .report import Finding

MARKET_TZ = "America/New_York"

# (name, regex on headline/text, [start, end) window in New York local minutes after midnight)
SESSION_RULES = [
    ("pre_market", r"\bpre-?market\b|\bbefore the (?:opening )?bell\b", (4 * 60, 9 * 60 + 30)),
    ("mid_day", r"\bmid-?day\b", (11 * 60, 14 * 60 + 30)),
    ("mid_afternoon", r"\bmid-?afternoon\b", (13 * 60, 16 * 60)),
    ("after_hours", r"\bafter-?hours\b|\bpost-?market\b|\bafter the (?:closing )?bell\b", (16 * 60, 20 * 60)),
]


def to_naive(ts) -> pd.Series:
    """Parse to naive datetimes. Strings like '2020-06-05 06:30:54 UTC' are accepted (suffix dropped);
    strings with an explicit offset ('2020-06-05 10:30:54-04:00') are converted to UTC."""
    s = pd.Series(ts)
    if pd.api.types.is_datetime64_any_dtype(s):
        d = s
    else:
        s2 = s.astype(str).str.replace(r"\s*(?:UTC|GMT|Z)$", "", regex=True)
        d = pd.to_datetime(s2, errors="coerce", format="ISO8601", utc=True)
        miss = d.isna() & s2.str.len().gt(0) & s.notna()
        if miss.any():
            d.loc[miss] = pd.to_datetime(s2[miss], errors="coerce", format="mixed", utc=True)
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_convert("UTC").dt.tz_localize(None)
    return d


def is_date_only(ts: pd.Series) -> pd.Series:
    d = to_naive(ts)
    return (d.dt.hour == 0) & (d.dt.minute == 0) & (d.dt.second == 0)


def check_date_only(df: pd.DataFrame, ts_col: str = "date", group_col: str | None = None,
                    disguise_share: float = 0.01) -> list[Finding]:
    d = to_naive(df[ts_col])
    bad = d.isna()
    dn = d[~bad]
    tod = dn.dt.strftime("%H:%M:%S")
    only = tod == "00:00:00"
    share = float(only.mean()) if len(dn) else float("nan")
    sev = "error" if share > 0.5 else ("warn" if share > 0.05 else "info")
    out = [Finding("news.date_only_share", sev,
                   f"{share:.1%} of {len(dn):,} timestamps carry no time of day (00:00:00); "
                   "a same-date news->price merge cannot tell pre- from post-close news",
                   {"rows": int(len(dn)), "date_only": int(only.sum()), "share": share, "unparseable": int(bad.sum())})]
    if group_col and group_col in df:
        g = pd.DataFrame({"g": df.loc[~bad, group_col].astype(str).to_numpy(), "o": only.to_numpy()})
        by = g.groupby("g")["o"].agg(["size", "mean"]).sort_values("size", ascending=False).head(8)
        out[0].examples = [{"group": k, "rows": int(r["size"]), "date_only_share": round(float(r["mean"]), 4)} for k, r in by.iterrows()]
    # exact non-midnight times that hold a large share: date-only rows shifted by a timezone conversion
    vc = tod[~only].value_counts()
    sus = vc[vc > disguise_share * len(dn)]
    if len(sus):
        out.append(Finding("news.constant_time_of_day", "warn",
                           f"{int(sus.sum()):,} rows share {len(sus)} exact time(s) of day "
                           f"({', '.join(sus.index[:4])}): probably date-only rows shifted by a timezone conversion",
                           {"rows": int(sus.sum())}, [{"time": k, "rows": int(v)} for k, v in sus.head(4).items()]))
    return out


def session_matches(text: pd.Series, rules=SESSION_RULES) -> pd.DataFrame:
    """Return a frame with the first matching session rule per row (rows without a match are dropped)."""
    t = text.fillna("").astype(str)
    res = pd.Series(pd.NA, index=t.index, dtype="object")
    lo = pd.Series(np.nan, index=t.index)
    hi = pd.Series(np.nan, index=t.index)
    for name, rx, (a, b) in rules:
        m = res.isna() & t.str.contains(rx, flags=re.IGNORECASE, regex=True)
        res[m] = name
        lo[m] = a
        hi[m] = b
    keep = res.notna()
    return pd.DataFrame({"session": res[keep], "lo": lo[keep], "hi": hi[keep]})


def session_fit(ts: pd.Series, text: pd.Series, label_tz: str = "UTC", shifts=range(-12, 13),
                rules=SESSION_RULES, min_rows: int = 30) -> pd.DataFrame:
    """In-window share of session-anchored rows for each whole-hour shift s (true time = label + s hours)."""
    d = to_naive(ts)
    m = session_matches(text, rules)
    m = m[d.loc[m.index].notna() & ~is_date_only(d.loc[m.index])]
    if len(m) < min_rows:
        return pd.DataFrame(columns=["shift_h", "n", "in_window"])
    base = d.loc[m.index].dt.tz_localize(label_tz, ambiguous="NaT", nonexistent="NaT")
    rows = []
    for s in shifts:
        loc = (base + pd.Timedelta(hours=s)).dt.tz_convert(MARKET_TZ)
        mins = loc.dt.hour * 60 + loc.dt.minute
        inside = (mins >= m["lo"]) & (mins < m["hi"])
        rows.append({"shift_h": s, "n": int(loc.notna().sum()), "in_window": float(inside[loc.notna()].mean())})
    return pd.DataFrame(rows)


def check_timezone(df: pd.DataFrame, ts_col: str = "date", text_col: str = "title", label_tz: str = "UTC",
                   min_rows: int = 30) -> list[Finding]:
    fit = session_fit(df[ts_col], df[text_col], label_tz=label_tz, min_rows=min_rows)
    if fit.empty:
        return [Finding("news.timezone_sessions", "info", f"fewer than {min_rows} timed session-anchored rows; test skipped")]
    at0 = float(fit.loc[fit.shift_h == 0, "in_window"].iloc[0])
    best = fit.sort_values("in_window", ascending=False).iloc[0]
    s, share, n = int(best.shift_h), float(best.in_window), int(best.n)
    if s != 0 and share - at0 > 0.2:
        sev, msg = "error", (f"session-anchored headlines fit their session only {at0:.0%} of the time as labelled ({label_tz}) "
                             f"but {share:.0%} if stamps are shifted by {s:+d} h: timestamps look mis-converted")
    else:
        sev, msg = "info", f"{at0:.0%} of {n:,} session-anchored headlines fall in their session as labelled ({label_tz})"
    return [Finding("news.timezone_sessions", sev, msg,
                    {"rows": n, "in_window_as_labelled": at0, "best_shift_h": s, "in_window_best": share},
                    fit.sort_values("in_window", ascending=False).head(3).to_dict("records"))]


def check_hour_profile(df: pd.DataFrame, ts_col: str = "date", label_tz: str = "UTC") -> list[Finding]:
    d = to_naive(df[ts_col])
    timed = d[d.notna() & ~is_date_only(d)]
    if len(timed) == 0:
        return [Finding("news.hour_profile", "info", "no timed rows")]
    loc = timed.dt.tz_localize(label_tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert(MARKET_TZ)
    h = loc.dt.hour
    night = float(((h >= 0) & (h < 4)).mean())
    rth = float(((h * 60 + loc.dt.minute >= 570) & (h < 16)).mean())
    post = float(((h >= 16) & (h < 24)).mean())
    sev = "warn" if night > 0.15 else "info"
    return [Finding("news.hour_profile", sev,
                    f"timed rows in New York time: {night:.1%} at 00-04h, {rth:.1%} during 09:30-16:00, {post:.1%} after 16:00",
                    {"timed_rows": int(len(timed)), "share_00_04": night, "share_rth": rth, "share_after_close": post,
                     "weekend_share": float((loc.dt.day_of_week >= 5).mean())})]


def check_duplicates(df: pd.DataFrame, ticker_col: str = "ticker", url_col: str | None = "url",
                     title_col: str = "title", ts_col: str = "date") -> list[Finding]:
    out = []
    n = len(df)
    exact = int(df.duplicated().sum())
    out.append(Finding("news.exact_duplicate_rows", "warn" if exact else "info",
                       f"{exact:,} exact duplicate rows ({exact / max(n, 1):.2%})", {"rows": exact, "share": exact / max(n, 1)}))
    if url_col and url_col in df:
        k = [c for c in (ticker_col, url_col) if c in df]
        u = df[df[url_col].astype(str).str.len() > 0]
        du = int(u.duplicated(k).sum())
        out.append(Finding("news.duplicate_ticker_url", "warn" if du else "info",
                           f"{du:,} repeated ({'/'.join(k)}) pairs ({du / max(len(u), 1):.2%})", {"rows": du}))
    k = [c for c in (ticker_col, title_col, ts_col) if c in df]
    dt = int(df.duplicated(k).sum())
    out.append(Finding("news.duplicate_ticker_title_time", "warn" if dt else "info",
                       f"{dt:,} repeated ({'/'.join(k)}) keys ({dt / max(n, 1):.2%})", {"rows": dt}))
    return out


def run_all(df: pd.DataFrame, ts_col="date", text_col="title", ticker_col="ticker", url_col="url",
            label_tz="UTC", group_col=None) -> list[Finding]:
    f = []
    f += check_date_only(df, ts_col, group_col)
    f += check_hour_profile(df, ts_col, label_tz)
    f += check_timezone(df, ts_col, text_col, label_tz)
    f += check_duplicates(df, ticker_col, url_col if url_col in df else None, text_col, ts_col)
    return f
