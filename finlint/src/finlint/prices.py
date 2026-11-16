"""Checks for daily price panels in long format (one row per ticker and date).

Expected columns (names configurable through ``cols``): ticker, date, close; optional adj_close, open,
high, low, volume. All checks are vectorised over a (ticker, date)-sorted copy of the panel.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd

from .report import Finding

DEFAULT_COLS = {"ticker": "ticker", "date": "date", "open": "open", "high": "high", "low": "low",
                "close": "close", "adj_close": "adj_close", "volume": "volume"}

# split ratios seen in practice (forward and reverse); a jump r is split-like if r*k ~ 1 for some k
_SPLIT_BASE = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 100, 200,
               Fraction(3, 2), Fraction(4, 3), Fraction(5, 4), Fraction(5, 2), Fraction(5, 3)]
SPLIT_RATIOS = sorted({float(x) for x in _SPLIT_BASE} | {float(1 / Fraction(x)) for x in _SPLIT_BASE})


def _c(cols):
    c = dict(DEFAULT_COLS)
    if cols:
        c.update(cols)
    return c


def prepare(df: pd.DataFrame, cols: dict | None = None) -> pd.DataFrame:
    """Return a (ticker, date)-sorted copy with standard column names and per-series lags.

    Memory-lean: tickers stay categorical, sorting is a single lexsort over integer codes, and only
    one copy of each column is made (a 30M-row panel needs ~2.5 GB)."""
    c = _c(cols)
    keep = {v: k for k, v in c.items() if v in df.columns}
    tk = df[c["ticker"]]
    tk = tk if isinstance(tk.dtype, pd.CategoricalDtype) else tk.astype(str).astype("category")
    if not all(isinstance(x, str) for x in tk.cat.categories):
        tk = tk.cat.rename_categories([str(x) for x in tk.cat.categories])
    d = pd.to_datetime(df[c["date"]])
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)        # daily bars: keep the local trading date
    day = d.dt.normalize().to_numpy("datetime64[ns]")
    order = np.lexsort((day, tk.cat.codes.to_numpy()))
    p = pd.DataFrame({"ticker": pd.Categorical.from_codes(tk.cat.codes.to_numpy()[order], tk.cat.categories),
                      "date": day[order]})
    for name, src in keep.items():
        if src not in ("ticker", "date"):
            p[src] = df[name].to_numpy()[order]
    codes = p["ticker"].cat.codes.to_numpy()
    same = np.r_[False, codes[1:] == codes[:-1]]
    p["_same"] = same
    close = p["close"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.empty(len(p), dtype=np.float64)       # float64: exact 2x moves must sit on the threshold
        r[0] = np.nan
        np.divide(close[1:], close[:-1], out=r[1:])
        r[~same] = np.nan
        p["r"] = r
        if "adj_close" in p:
            q = (p["adj_close"].to_numpy() / close).astype(np.float32)
            dq = np.empty(len(p), dtype=np.float32)
            dq[0] = np.nan
            np.divide(q[1:], q[:-1], out=dq[1:])
            dq[~same] = np.nan
            p["dq"] = dq
            del q
    dd = p["date"].to_numpy()
    prev = np.empty(len(p), dtype="datetime64[ns]")
    prev[0] = np.datetime64("NaT", "ns")
    prev[1:] = dd[:-1]
    prev[~same] = np.datetime64("NaT", "ns")
    p["prev_date"] = prev
    return p


def nearest_split(r: np.ndarray, tol: float = 0.15):
    """For each ratio r return (k, residual) where k is the split ratio minimising |log(r*k)|."""
    r = np.asarray(r, dtype=float)
    ks = np.asarray(SPLIT_RATIOS)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.abs(np.log(r[:, None] * ks[None, :]))
    j = np.nanargmin(np.where(np.isfinite(res), res, np.inf), axis=1)
    best = res[np.arange(len(r)), j]
    k = np.where(best < tol, ks[j], np.nan)
    return k, best


def detect_jumps(p: pd.DataFrame, thresh: float = np.log(2), revert_days: int = 3, split_tol: float = 0.15) -> pd.DataFrame:
    """Per-series jumps with |log r| > thresh, classified as split-like / reverting spike / other."""
    lr = np.log(p["r"].astype(np.float64).where(p["r"] > 0))   # float64: exact 2x moves sit on the threshold
    idx = np.nonzero((np.abs(lr) > thresh).to_numpy())[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=["ticker", "date", "prev_date", "r", "split_k", "kind"])
    j = p.iloc[idx][["ticker", "date", "prev_date", "r"]].copy()
    k, _ = nearest_split(j["r"].to_numpy(), split_tol)
    j["split_k"] = k
    # reverting spike: cumulative log move over the next `revert_days` rows undoes most of the jump
    lrv = lr.to_numpy()
    tick = p["ticker"].cat.codes.to_numpy()
    rev = {}
    for i in idx:
        if rev.get(i):
            continue
        s = 0.0
        for h in range(1, revert_days + 1):
            if i + h >= len(p) or tick[i + h] != tick[i] or not np.isfinite(lrv[i + h]):
                break
            s += lrv[i + h]
            if abs(lrv[i] + s) < 0.5 * abs(lrv[i]):
                rev[i] = True
                rev[i + h] = True          # the move back is part of the same bad tick
                break
    rev = [bool(rev.get(i)) for i in idx]
    j["reverts"] = rev
    j["kind"] = np.where(j["reverts"], "spike", np.where(np.isfinite(j["split_k"]), "split_like", "other"))
    return j.reset_index(drop=True)


def detect_splice_dates(p: pd.DataFrame, thresh: float = np.log(2), min_count: int = 10, z_thresh: float = 8.0,
                        dq_tol: float = 0.005) -> pd.DataFrame:
    """Dates on which unusually many series jump at once, or on which the dividend-adjustment factor
    (adj_close/close) falls for many series. Within one consistent download that factor can only rise
    over time, so a cross-sectional cluster of falls marks a change of adjustment vintage (a splice)."""
    lr = np.log(p["r"].astype(np.float64).where(p["r"] > 0))   # float64: exact 2x moves sit on the threshold
    t = pd.DataFrame({"date": p["date"], "pair": p["_same"], "ext": (np.abs(lr) > thresh).to_numpy()})
    if "dq" in p:
        t["dqdec"] = (p["dq"] < 1 - dq_tol).to_numpy()
    g = t.groupby("date").sum()
    g["pair"] = g["pair"].clip(lower=1)
    res = []
    for c in [c for c in ("ext", "dqdec") if c in g]:
        frac = g[c] / g["pair"]
        med = frac.rolling(251, center=True, min_periods=21).median()
        mad = (frac - med).abs().rolling(251, center=True, min_periods=21).median()
        z = (frac - med) / (1.4826 * mad + 1e-4)
        hit = (g[c] >= min_count) & (z >= z_thresh)
        for d in g.index[hit]:
            res.append({"date": d, "signal": "jumps" if c == "ext" else "adj_factor_reset",
                        "count": int(g.at[d, c]), "pairs": int(g.at[d, "pair"]), "z": float(round(z.at[d], 1))})
    out = pd.DataFrame(res, columns=["date", "signal", "count", "pairs", "z"])
    return out.sort_values(["count"], ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# Partial aggregates. Every check is computed per chunk of whole series and merged, so arbitrarily
# large panels can be linted in bounded memory (run_chunked); run_all is the one-chunk case.
# ----------------------------------------------------------------------------------------------

def partials(p: pd.DataFrame, gap_days: int = 30, min_run: int = 5, thresh: float = np.log(2), dq_tol: float = 0.005,
             blowup_factor: float = 1000.0) -> dict:
    out = {"n": len(p)}
    dup = p.duplicated(["ticker", "date"], keep=False)
    out["dup_rows"] = int(p.duplicated(["ticker", "date"]).sum())
    out["dup_tickers"] = int(p.loc[dup, "ticker"].nunique())
    out["dup_ex"] = p.loc[dup, ["ticker", "date"]].head(3).astype(str).to_dict("records")
    out["nonpos"] = int((p["close"] <= 0).sum())
    out["nan_close"] = int(p["close"].isna().sum())
    out["has_ohlc"] = {"open", "high", "low"} <= set(p.columns)
    if out["has_ohlc"]:
        o, h, l_, c = (p[k].to_numpy() for k in ("open", "high", "low", "close"))
        zero = (o == 0) | (h == 0) | (l_ == 0) | (c == 0)          # missing values stored as 0 (common in Yahoo history)
        out["zero_ohlc"] = int(zero.sum())
        out["zero_open"] = int((o == 0).sum())
        tol = 1e-6 * np.abs(c)
        bad = ~zero & ((h < np.maximum(o, c) - tol) | (l_ > np.minimum(o, c) + tol) | (h < l_ - tol))
        out["ohlc_bad"] = int(bad.sum())
        out["ohlc_ex"] = p.loc[bad, ["ticker", "date", "open", "high", "low", "close"]].head(3).astype(str).to_dict("records")
    out["neg_vol"] = int((p["volume"] < 0).sum()) if "volume" in p else 0
    out["zero_vol"] = int((p["volume"] == 0).sum()) if "volume" in p else 0
    wk = p["date"].dt.day_of_week >= 5
    out["weekend_rows"] = int(wk.sum())
    out["weekend_dates"] = set(p.loc[wk, "date"])
    out["weekend_ex"] = p.loc[wk, ["ticker", "date"]].head(3).astype(str).to_dict("records")
    # per-date counts for splice detection and sparse dates
    with np.errstate(invalid="ignore", divide="ignore"):
        ext = np.abs(np.log(p["r"].to_numpy(np.float64))) > thresh
        t = pd.DataFrame({"date": p["date"].to_numpy(), "rows": 1, "pair": p["_same"].to_numpy(), "ext": ext,
                          "dqdec": (p["dq"].to_numpy() < 1 - dq_tol) if "dq" in p else False})
    out["per_date"] = t.groupby("date").sum()
    del t
    # stale runs
    close = p["close"].to_numpy()
    same_val = p["_same"].to_numpy() & np.r_[False, close[1:] == close[:-1]]
    run_id = np.cumsum(~same_val)
    lengths = np.bincount(run_id)
    long_runs = np.nonzero(lengths >= min_run)[0]
    first_idx = np.searchsorted(run_id, long_runs)
    top = np.argsort(-lengths[long_runs])[:3]
    out["stale"] = {"runs": int(len(long_runs)), "rows": int(lengths[long_runs].sum()) if len(long_runs) else 0,
                    "series": int(p["ticker"].iloc[first_idx].nunique()) if len(long_runs) else 0,
                    "ex": [(int(lengths[long_runs[k]]), str(p["ticker"].iat[first_idx[k]]), str(p["date"].iat[first_idx[k]].date()))
                           for k in top]}
    out["jumps"] = detect_jumps(p, thresh=thresh)
    # level blowups: rows far above the series' recent level (cumulative reverse-split adjustment inflates old
    # prices and deflates old volumes towards 0; returns stay consistent, levels and volumes do not)
    recent = p.groupby("ticker", observed=True).tail(252).groupby("ticker", observed=True)["close"].median()
    ref = p["ticker"].map(recent).to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = p["close"].to_numpy(np.float64) / ref
    blow = np.isfinite(ratio) & (ref > 0) & (ratio > blowup_factor)
    out["blowup_rows"] = int(blow.sum())
    out["blowup_zero_vol"] = int((blow & (p["volume"].to_numpy() == 0)).sum()) if "volume" in p else 0
    out["blowup_series"] = int(p.loc[blow, "ticker"].nunique())
    if blow.any():
        mx = pd.DataFrame({"ticker": p.loc[blow, "ticker"].astype(str).to_numpy(), "ratio": ratio[blow],
                           "close": p.loc[blow, "close"].to_numpy()}).groupby("ticker").agg(max_ratio=("ratio", "max"),
                                                                                               max_close=("close", "max"))
        out["blowup_ex"] = mx.sort_values("max_ratio", ascending=False).head(3).reset_index().to_dict("records")
    else:
        out["blowup_ex"] = []
    out["span"] = p.groupby("ticker", observed=True)["date"].agg(["min", "max", "size"])
    gaps = (p["date"] - p["prev_date"]).dt.days
    out["gaps"] = p.loc[gaps > gap_days, ["ticker", "prev_date", "date"]].astype({"ticker": str})
    return out


def merge(parts: list[dict]) -> dict:
    a = {"n": sum(x["n"] for x in parts)}
    for k in ("dup_rows", "dup_tickers", "nonpos", "nan_close", "neg_vol", "zero_vol", "weekend_rows",
              "blowup_rows", "blowup_zero_vol", "blowup_series"):
        a[k] = sum(x[k] for x in parts)
    a["has_ohlc"] = any(x["has_ohlc"] for x in parts)
    a["ohlc_bad"] = sum(x.get("ohlc_bad", 0) for x in parts)
    a["zero_ohlc"] = sum(x.get("zero_ohlc", 0) for x in parts)
    a["zero_open"] = sum(x.get("zero_open", 0) for x in parts)
    for k in ("dup_ex", "ohlc_ex", "weekend_ex"):
        a[k] = [e for x in parts for e in x.get(k, [])][:3]
    a["weekend_dates"] = set().union(*[x["weekend_dates"] for x in parts])
    a["per_date"] = pd.concat([x["per_date"] for x in parts]).groupby(level=0).sum().sort_index()
    a["stale"] = {"runs": sum(x["stale"]["runs"] for x in parts), "rows": sum(x["stale"]["rows"] for x in parts),
                  "series": sum(x["stale"]["series"] for x in parts),
                  "ex": sorted([e for x in parts for e in x["stale"]["ex"]], reverse=True)[:3]}
    a["blowup_ex"] = sorted([e for x in parts for e in x.get("blowup_ex", [])], key=lambda e: -e["max_ratio"])[:3]
    a["jumps"] = pd.concat([x["jumps"] for x in parts], ignore_index=True)
    a["span"] = pd.concat([x["span"] for x in parts])
    a["gaps"] = pd.concat([x["gaps"] for x in parts], ignore_index=True)
    return a


def splice_dates_from_counts(g: pd.DataFrame, min_count: int = 10, z_thresh: float = 8.0) -> pd.DataFrame:
    g = g.copy()
    g["pair"] = g["pair"].clip(lower=1)
    res = []
    for c in [c for c in ("ext", "dqdec") if c in g and g[c].sum() > 0]:
        frac = g[c] / g["pair"]
        med = frac.rolling(251, center=True, min_periods=21).median()
        mad = (frac - med).abs().rolling(251, center=True, min_periods=21).median()
        z = (frac - med) / (1.4826 * mad + 1e-4)
        hit = (g[c] >= min_count) & (z >= z_thresh)
        for d in g.index[hit]:
            res.append({"date": pd.Timestamp(d), "signal": "jumps" if c == "ext" else "adj_factor_reset",
                        "count": int(g.at[d, c]), "pairs": int(g.at[d, "pair"]), "z": float(round(z.at[d], 1))})
    out = pd.DataFrame(res, columns=["date", "signal", "count", "pairs", "z"])
    return out.sort_values(["count"], ascending=False).reset_index(drop=True)


def detect_splice_dates(p: pd.DataFrame, thresh: float = np.log(2), min_count: int = 10, z_thresh: float = 8.0,
                        dq_tol: float = 0.005) -> pd.DataFrame:
    """Dates on which unusually many series jump at once, or on which the dividend-adjustment factor
    (adj_close/close) falls for many series. Within one consistent download that factor can only rise
    over time, so a cross-sectional cluster of falls marks a change of adjustment vintage (a splice)."""
    return splice_dates_from_counts(partials(p, thresh=thresh, dq_tol=dq_tol)["per_date"], min_count, z_thresh)


def findings(a: dict, sparse_frac: float = 0.05, cluster_min: int = 20, cluster_frac: float = 0.01,
             gap_days: int = 30, min_run: int = 5) -> list[Finding]:
    f = []
    n = a["n"]
    if a["dup_rows"]:
        f.append(Finding("prices.duplicates", "error", f"{a['dup_rows']:,} duplicate (ticker, date) rows",
                         {"dup_rows": a["dup_rows"], "tickers": a["dup_tickers"]}, a["dup_ex"]))
    else:
        f.append(Finding("prices.duplicates", "info", "no duplicate (ticker, date) rows", {"dup_rows": 0}))
    if a["nonpos"]:
        f.append(Finding("prices.nonpositive", "error", f"{a['nonpos']:,} rows with close <= 0", {"rows": a["nonpos"]}))
    if a["nan_close"]:
        f.append(Finding("prices.missing_close", "warn", f"{a['nan_close']:,} rows with a missing (NaN) close", {"rows": a["nan_close"]}))
    if a["has_ohlc"] and a["zero_ohlc"]:
        f.append(Finding("prices.zero_as_missing", "warn",
                         f"{a['zero_ohlc']:,} rows ({a['zero_ohlc'] / max(n, 1):.2%}) have an open/high/low/close of exactly 0 "
                         f"({a['zero_open']:,} zero opens): missing values stored as zero",
                         {"rows": a["zero_ohlc"], "zero_open": a["zero_open"], "share": a["zero_ohlc"] / max(n, 1)}))
    if a["has_ohlc"]:
        nb = a["ohlc_bad"]
        f.append(Finding("prices.ohlc_consistency", "warn" if nb else "info", f"{nb:,} rows (non-zero prices) violate low <= open,close <= high",
                         {"rows": nb, "share": nb / max(n, 1)}, a["ohlc_ex"]))
    if a["neg_vol"]:
        f.append(Finding("prices.negative_volume", "error", f"{a['neg_vol']:,} rows with negative volume", {"rows": a["neg_vol"]}))
    f.append(Finding("prices.weekend_rows", "warn" if a["weekend_rows"] else "info", f"{a['weekend_rows']:,} rows dated on Saturday/Sunday",
                     {"rows": a["weekend_rows"], "dates": len(a["weekend_dates"])}, a["weekend_ex"]))
    per = a["per_date"]["rows"]
    ref = per.rolling(61, center=True, min_periods=10).median()
    sparse = per[(per < sparse_frac * ref) & (ref > 50)]
    if len(sparse):
        f.append(Finding("prices.sparse_dates", "warn",
                         f"{len(sparse)} dates carry <{sparse_frac:.0%} of the usual cross-section (holiday or stray rows?)",
                         {"dates": len(sparse), "rows": int(sparse.sum())},
                         [{"date": str(pd.Timestamp(d).date()), "rows": int(v)} for d, v in sparse.head(3).items()]))
    st = a["stale"]
    f.append(Finding("prices.stale_runs", "warn" if st["rows"] > 0.001 * n else "info",
                     f"{st['runs']:,} runs of >= {min_run} identical closes ({st['rows']:,} rows); {a['zero_vol']:,} zero-volume rows",
                     {"runs": st["runs"], "rows_in_runs": st["rows"], "share_rows": st["rows"] / max(n, 1),
                      "series_with_runs": st["series"], "zero_volume_rows": a["zero_vol"]},
                     [{"ticker": t, "start": d, "len": L} for L, t, d in st["ex"]]))
    if a.get("blowup_rows", 0):
        f.append(Finding("prices.level_blowup", "warn",
                         f"{a['blowup_rows']:,} rows in {a['blowup_series']:,} series sit >1000x above the series' recent level "
                         f"({a['blowup_zero_vol']:,} with zero volume): cumulative reverse-split adjustment suspected; "
                         "price levels and volumes are unusable as features (returns unaffected)",
                         {"rows": a["blowup_rows"], "series": a["blowup_series"], "zero_volume_rows": a["blowup_zero_vol"]},
                         [{"ticker": e["ticker"], "max_ratio": float(e["max_ratio"]), "max_close": float(e["max_close"])}
                          for e in a.get("blowup_ex", [])]))
    j = a["jumps"]
    if len(j):
        kinds = j["kind"].value_counts().to_dict()
        sl = j[j["kind"] == "split_like"]
        f.append(Finding("prices.split_like_jumps", "error" if len(sl) else "warn",
                         f"{len(j):,} moves beyond +100%/-50% in {j['ticker'].nunique():,} series; "
                         f"{kinds.get('split_like', 0):,} look like unadjusted splits, {kinds.get('spike', 0):,} revert (bad ticks)",
                         {"jumps": int(len(j)), "series": int(j["ticker"].nunique()), **{f"kind_{k}": int(v) for k, v in kinds.items()}},
                         sl.assign(date=sl["date"].astype(str), r=sl["r"].astype(float).round(4))[["ticker", "date", "r", "split_k"]].head(3).astype({"ticker": str}).to_dict("records")))
    else:
        f.append(Finding("prices.split_like_jumps", "info", "no moves beyond +100%/-50%", {"jumps": 0}))
    s = splice_dates_from_counts(a["per_date"])
    if len(s):
        top = s.head(5)
        f.append(Finding("prices.splice_dates", "error",
                         f"{s['date'].nunique()} date(s) where many series break at once: " +
                         ", ".join(f"{d.date()} ({sig}: {c})" for d, sig, c in zip(top["date"], top["signal"], top["count"])),
                         {"dates": int(s["date"].nunique())}, top.assign(date=top["date"].astype(str)).to_dict("records")))
    else:
        f.append(Finding("prices.splice_dates", "info", "no cross-sectional break dates", {"dates": 0}))
    f += coverage_findings(a["span"], a["gaps"], cluster_min, cluster_frac, gap_days)
    return f


def coverage_findings(span: pd.DataFrame, gaps: pd.DataFrame, cluster_min: int = 20, cluster_frac: float = 0.01,
                      gap_days: int = 30, margin_days: int = 15) -> list[Finding]:
    end = span["max"].max()
    n = len(span)
    early = span[span["max"] < end]
    ends = early.groupby("max").size().sort_values(ascending=False)
    thr = max(cluster_min, int(cluster_frac * n))
    clusters = ends[ends >= thr]
    out = []
    if len(clusters):
        out.append(Finding("prices.coverage_end_clusters", "error",
                           f"{int(clusters.sum()):,} of {n:,} series stop on {len(clusters)} shared date(s) before the panel end ({end.date()})",
                           {"series_in_clusters": int(clusters.sum()), "series": n, "panel_end": str(end.date())},
                           [{"date": str(d.date()), "series_ending": int(v)} for d, v in clusters.head(5).items()]))
    years = (span["max"].max() - span["min"].min()).days / 365.25
    # survivorship hint: a panel that kept delisted names has many series ending at scattered dates;
    # ends near a shared cut-off date or near the panel end do not count as scattered
    near = np.zeros(len(early), bool)
    for d in list(clusters.index) + [end]:
        near |= (early["max"] - d).abs().dt.days.to_numpy() <= (margin_days if d != end else 31)
    scattered = early[~near]
    share = len(scattered) / max(n, 1)
    if years > 5 and share < 0.02:
        out.append(Finding("prices.survivorship_hint", "warn",
                           f"only {len(scattered):,} of {n:,} series ({share:.2%}) end at scattered dates over a {years:.0f}-year panel: "
                           "delisted securities are probably missing (survivor-only universe)",
                           {"scattered_end_series": int(len(scattered)), "series": n, "years": round(years, 1)}))
    out.append(Finding("prices.internal_gaps", "warn" if len(gaps) else "info",
                       f"{len(gaps):,} within-series gaps longer than {gap_days} calendar days in {gaps['ticker'].nunique():,} series",
                       {"gaps": int(len(gaps)), "series": int(gaps["ticker"].nunique())},
                       gaps.head(3).astype(str).to_dict("records")))
    return out


def check_coverage(p: pd.DataFrame, cluster_min: int = 20, cluster_frac: float = 0.01, gap_days: int = 30) -> list[Finding]:
    a = partials(p, gap_days=gap_days)
    return coverage_findings(a["span"], a["gaps"], cluster_min, cluster_frac, gap_days)


def run_all(df: pd.DataFrame, cols: dict | None = None) -> list[Finding]:
    return findings(partials(prepare(df, cols)))


def run_chunked(frames, cols: dict | None = None) -> list[Finding]:
    """Lint a panel delivered as an iterable of DataFrames, each holding complete series (all rows of its
    tickers). Memory is bounded by the largest chunk."""
    return findings(merge([partials(prepare(f, cols)) for f in frames]))
