"""Look-ahead risk of a news -> daily-price join, and a leakage-free alignment.

Convention: a feature attached to trading session d is used at the close of d to predict returns after
that close. A news item is therefore usable at session d only if it was public before the close of d.

* Timed news (true publication time known): usable from the first session whose close is strictly after
  the publication time.
* Date-only news (only a calendar date D in market time): the publication time is somewhere in D, so it is
  only guaranteed to be public before the close of the first session strictly after D.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .news import MARKET_TZ, is_date_only, to_naive
from .report import Finding


def _sessions(sessions) -> np.ndarray:
    s = pd.to_datetime(pd.Series(sessions)).dt.normalize().drop_duplicates().sort_values()
    return s.to_numpy("datetime64[D]")


def _next_session(S: np.ndarray, d: np.ndarray, inclusive: np.ndarray) -> np.ndarray:
    """First session >= d where inclusive, else first session > d (NaT if none)."""
    d = d.astype("datetime64[D]")
    i_right = np.searchsorted(S, d, side="right")
    i_left = np.searchsorted(S, d, side="left")
    i = np.where(inclusive, i_left, i_right)
    out = np.full(len(d), np.datetime64("NaT", "D"), dtype="datetime64[D]")
    ok = (i < len(S)) & ~np.isnat(d)
    out[ok] = S[i[ok]]
    return out


def leakage_free_session(ts_utc, date_only, sessions, close: str = "16:00", tz: str = MARKET_TZ,
                         date_only_policy: str = "next") -> pd.Series:
    """Session at whose close each news item is first known to be public.

    ts_utc: naive UTC datetimes (true publication time for timed rows; for date-only rows the calendar date
    in market time stored at 00:00). date_only: boolean mask. date_only_policy: 'next' or 'drop'."""
    S = _sessions(sessions)
    ts = to_naive(ts_utc)
    do = np.asarray(date_only, dtype=bool)
    hh, mm = (int(x) for x in close.split(":"))
    loc = ts.dt.tz_localize("UTC").dt.tz_convert(tz)
    local_date = loc.dt.tz_localize(None).dt.normalize().to_numpy("datetime64[D]")
    before_close = ((loc.dt.hour * 60 + loc.dt.minute) < hh * 60 + mm).to_numpy()
    timed_sess = _next_session(S, local_date, before_close)
    d_only = ts.dt.normalize().to_numpy("datetime64[D]")          # date-only: the stored calendar date
    do_sess = _next_session(S, d_only, np.zeros(len(d_only), dtype=bool))
    if date_only_policy == "drop":
        do_sess[:] = np.datetime64("NaT", "D")
    out = np.where(do, do_sess, timed_sess)
    return pd.Series(out.astype("datetime64[ns]"), index=ts.index)


def same_date_join(ts_label, sessions) -> pd.Series:
    """FNSPID-style join: the calendar date of the stamp as labelled; rows on non-session dates are not joined."""
    S = _sessions(sessions)
    d = to_naive(ts_label).dt.normalize().to_numpy("datetime64[D]")
    hit = np.isin(d, S)
    out = np.where(hit, d, np.datetime64("NaT", "D"))
    return pd.Series(out.astype("datetime64[ns]"))


def lookahead_report(ts_label, assigned, sessions, date_only=None, shift_h: float = 0.0, label_tz: str = "UTC",
                     close: str = "16:00", tz: str = MARKET_TZ, after_close_prior: float | None = None) -> list[Finding]:
    """Audit a join that attached each news row to session `assigned` (NaT = not joined).

    shift_h: correction to the labelled stamps (true time = label + shift_h hours), e.g. from
    news.check_timezone. after_close_prior: share of news published after the close, used to estimate
    how many same-day date-only rows leak; if None it is estimated from the timed rows."""
    lab = to_naive(ts_label)
    if label_tz != "UTC":
        lab = lab.dt.tz_localize(label_tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC").dt.tz_localize(None)
    do = is_date_only(lab).to_numpy() if date_only is None else np.asarray(date_only, dtype=bool)
    true = lab + pd.to_timedelta(np.where(do, 0.0, shift_h), unit="h")
    lf = leakage_free_session(true, do, sessions, close=close, tz=tz)
    a = pd.to_datetime(pd.Series(np.asarray(assigned))).reset_index(drop=True)
    lf = lf.reset_index(drop=True)
    joined = a.notna().to_numpy()
    leak = joined & lf.notna().to_numpy() & (a.to_numpy() < lf.to_numpy())
    late = joined & lf.notna().to_numpy() & (a.to_numpy() > lf.to_numpy())
    timed = ~do
    nj = int(joined.sum())
    nt = int((joined & timed).sum())
    nd = int((joined & do).sum())
    # date-only rows joined on/before their own date are unverifiable: they leak if published after the close
    d_cal = lab.dt.normalize().to_numpy("datetime64[ns]")
    same_day = joined & do & (a.to_numpy() <= d_cal)
    if after_close_prior is None:
        loc = true[timed].dt.tz_localize("UTC").dt.tz_convert(tz)
        hh, mm = (int(x) for x in close.split(":"))
        after_close_prior = float(((loc.dt.hour * 60 + loc.dt.minute) >= hh * 60 + mm).mean()) if timed.any() else float("nan")
    t_leak = int((leak & timed).sum())
    d_early = joined & do & (a.to_numpy() < d_cal)              # joined before the news' own date
    certain = (leak & timed) | d_early
    findings = [
        Finding("lookahead.timed_after_cutoff", "error" if nt and t_leak / nt > 0.01 else "info",
                f"{t_leak:,} of {nt:,} joined timed rows ({t_leak / max(nt, 1):.1%}) were published after the close of the session they feed",
                {"joined_timed": nt, "leaking": t_leak, "share": t_leak / max(nt, 1)}),
        Finding("lookahead.date_only_same_day", "error" if nd and same_day.sum() / nd > 0.05 else "info",
                f"{int(same_day.sum()):,} of {nd:,} joined date-only rows ({same_day.sum() / max(nd, 1):.1%}) feed their own calendar day (or earlier); "
                f"with an after-close share of {after_close_prior:.0%} that is ~{after_close_prior * same_day.sum():,.0f} leaking rows",
                {"joined_date_only": nd, "same_day": int(same_day.sum()), "before_own_date": int(d_early.sum()),
                 "after_close_prior": after_close_prior, "expected_leaking": float(after_close_prior * same_day.sum())}),
        Finding("lookahead.summary", "info",
                f"{nj:,} joined rows: {int(certain.sum()):,} certainly look-ahead, {int((same_day & ~d_early).sum()):,} unverifiable "
                f"(date-only, same day), {int(late.sum()):,} later than necessary",
                {"joined": nj, "certain_lookahead": int(certain.sum()), "unverifiable": int((same_day & ~d_early).sum()),
                 "late": int(late.sum())}),
    ]
    return findings
