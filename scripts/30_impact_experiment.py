#!/usr/bin/env python3
"""Downstream impact: Loughran-McDonald news tone -> next-session return, under FNSPID's alignment and
under leakage-free alignments. CPU only, no learned model, nothing tuned.

Signal. Per news row: tone = (pos - neg) / (pos + neg) from Loughran-McDonald word counts in the title
(for the nasdaq file the article body is also tested). Rows without LM hits carry no signal. Per
(ticker, session): mean tone; sessions with mean tone 0 are dropped. A long-short portfolio goes long
tickers with tone > 0 and short tone < 0, equal weight within each side, for one session.

Alignments (at which session's close the news is assumed known):
  A  fnspid        calendar date of the FNSPID stamp, used only if the ticker trades that day
                   (data_processor/price_news_integrate.py: Date.normalize(), then merge on Date)
  C  tz_fixed      as A, but timed rows use the New York date of the corrected timestamp
  D  trust_labels  what a careful user does with FNSPID's labels: treat stamps as true UTC and apply the
                   16:00 New York close cut-off; date-only rows go to the next session
  B  no_leak       corrected timestamps with the 16:00 cut-off; date-only rows go to the first session after
                   their date (the later of stamp date and URL date) -- finlint.lookahead
Target: close-to-close return (adj_close) from the assigned session to the ticker's next session, split
into overnight (close->open) and intraday (open->close) legs. Returns touching a splice boundary
(2015-01-02, 2020-04-02, 2020-07-02, 2020-07-06) or beyond +100%/-50% are dropped.
Metrics: directional accuracy (pooled), mean daily rank IC (days with >= 10 names), annualised Sharpe of
the daily long-short return. 95% CIs: moving-block bootstrap over sessions (block 21, NBOOT draws),
paired across alignments, so differences have their own CIs.
Outputs: data/derived/results/impact.json, impact_table.csv
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import newslib as nl  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from finlint import lookahead  # noqa: E402

SEED = 7
NBOOT = int(os.environ.get("NBOOT", "2000"))
BLOCK = 21
BOUNDARIES = pd.to_datetime(["2015-01-02", "2020-04-02", "2020-07-02", "2020-07-06"]).to_numpy("datetime64[D]")
SRC_CODES = {"benzinga": 1, "nasdaq": 2}
COLS = ["file", "date_raw", "ticker", "url", "lm_t_pos", "lm_t_neg", "lm_a_pos", "lm_a_neg"]


def sessions_from_prices():
    # trading sessions = dates carrying >= 50 price rows (per-date counts from 11_prices_boundaries.py)
    ev = pd.read_csv(common.RESULTS / "price_date_events.csv", parse_dates=["date"])
    vc = ev.set_index("date")["n"]
    return pd.to_datetime(pd.Series(vc[vc >= 50].index)).sort_values().to_numpy("datetime64[D]")


def day_int(x) -> np.ndarray:
    a = pd.to_datetime(pd.Series(x)).to_numpy("datetime64[D]")
    out = a.astype("int64")
    out[np.isnat(a)] = -1
    return out.astype(np.int32)


def tone(p, n):
    p = p.astype(np.float32)
    n = n.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(p + n > 0, (p - n) / (p + n), np.nan).astype(np.float32)


def load_news(S, tk_code):
    """Per part file: derive compact columns (ticker code, source, alignments, tone)."""
    frames = []
    # nasdaq_exteral_data.csv contains every record of All_external.csv (21_news_audit.py), so file 1 alone
    # is the full dataset; reading both would double-count 13M rows
    for fid in (1,):
        for path in nl.parts(fid):
            d = pq.read_table(path, columns=COLS).to_pandas()
            d = d[d["ticker"].str.len() > 0]
            if d.empty:
                continue
            src = nl.source_of(d["url"])
            ts = nl.stamp(d["date_raw"])
            do = nl.is_date_only(ts)
            utc, status = nl.corrected_utc(src, ts, do)
            ud = nl.url_date(d["url"])
            cons = nl.conservative_date(ts, do, ud)
            stamp_day = ts.dt.normalize()
            ny = utc.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
            C = stamp_day.where(do, ny)
            B = lookahead.leakage_free_session(utc.where(~do, cons), do.to_numpy(), S)
            D = lookahead.leakage_free_session(ts.where(~do, cons), do.to_numpy(), S)
            codes = d["ticker"].map(tk_code)
            frames.append(pd.DataFrame({
                "file": d["file"].to_numpy(np.int8), "tk": codes.fillna(-1).to_numpy(np.int32),
                "src": src.map(SRC_CODES).fillna(0).to_numpy(np.int8), "date_only": do.to_numpy(),
                "A": day_int(stamp_day), "C": day_int(C), "D": day_int(D), "B": day_int(B),
                "tone_t": tone(d["lm_t_pos"].to_numpy(), d["lm_t_neg"].to_numpy()),
                "tone_a": tone(d["lm_a_pos"].to_numpy(), d["lm_a_neg"].to_numpy())}))
    return pd.concat(frames, ignore_index=True)


def load_returns(tk_code, dmin, dmax, chunk=1000):
    """Next-session returns per (ticker, session), built 1,000 tickers at a time to bound memory."""
    tickers = sorted(tk_code)
    out = []
    bset = BOUNDARIES.astype("int64").astype(np.int32)
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        p = pq.read_table(common.DERIVED / "prices.parquet", columns=["ticker", "date", "open", "close", "adj_close"],
                          filters=[("ticker", "in", part), ("date", ">=", dmin), ("date", "<=", dmax)]).to_pandas(date_as_object=False)
        if p.empty:
            continue
        cat = p["ticker"].astype("category")
        mapping = np.array([tk_code.get(str(c), -1) for c in cat.cat.categories], dtype=np.int32)
        tk = mapping[cat.cat.codes.to_numpy()]
        day = p["date"].to_numpy("datetime64[D]").astype(np.int64).astype(np.int32)
        order = np.lexsort((day, tk))
        tk, day = tk[order], day[order]
        adj, cl, op = (p[c].to_numpy(np.float64)[order] for c in ("adj_close", "close", "open"))
        del p, cat
        nxt_same = np.r_[tk[1:] == tk[:-1], False]

        def lead(a):
            return np.r_[a[1:], np.nan]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(nxt_same, lead(adj) / adj - 1, np.nan)
            r_on = np.where(nxt_same, lead(op) / cl - 1, np.nan)
            r_id = np.where(nxt_same, lead(cl) / lead(op) - 1, np.nan)
            bad_on = ~(r_on < 1) | ~(r_on > -0.5) | ~(op > 0) | ~(lead(op) > 0)
        nday = np.r_[day[1:], -1]
        bad = np.isin(day, bset) | np.isin(nday, bset) | ~(r < 1) | ~(r > -0.5)
        bad_on |= bad
        keep = ~bad
        out.append(pd.DataFrame({"key": (tk.astype(np.int64) * 100000 + day)[keep], "day": day[keep],
                                 "r": r[keep].astype(np.float32),
                                 "r_on": np.where(bad_on, np.nan, r_on)[keep].astype(np.float32),
                                 "r_id": np.where(bad_on, np.nan, r_id)[keep].astype(np.float32)}))
    return pd.concat(out, ignore_index=True).set_index("key")


def daily_panel(sub: pd.DataFrame, col: str, tone_col: str, ret: pd.DataFrame) -> pd.DataFrame:
    x = sub[(sub[col] >= 0) & sub[tone_col].notna() & (sub["tk"] >= 0)]
    key = x["tk"].to_numpy(np.int64) * 100000 + x[col].to_numpy(np.int64)
    g = pd.DataFrame({"key": key, "tone": x[tone_col].to_numpy(np.float64)}).groupby("key")["tone"].mean()
    g = g[g != 0]
    return ret.join(g.rename("sig"), how="inner")          # requires a price row for that ticker and session


def per_day_stats(j: pd.DataFrame, leg: str, min_names: int = 10) -> pd.DataFrame:
    x = j[["day", "sig", leg]].dropna()
    x = x[x[leg] != 0]
    s = np.sign(x["sig"].to_numpy())
    df = pd.DataFrame({"day": x["day"].to_numpy(), "hit": (s == np.sign(x[leg].to_numpy())).astype(np.float64),
                       "s": s, "ret": x[leg].to_numpy(np.float64), "sig": x["sig"].to_numpy()})
    by = df.groupby("day")
    hits = by["hit"].sum()
    n = by["hit"].size()
    longs = df[df["s"] > 0].groupby("day")["ret"].mean()
    shorts = df[df["s"] < 0].groupby("day")["ret"].mean()
    ls = (longs - shorts).dropna()
    df["rs"] = by["sig"].rank()
    df["rr"] = by["ret"].rank()
    z = df[by["sig"].transform("size") >= min_names]
    if len(z):
        # Spearman = Pearson on within-day ranks
        zz = z.assign(rs2=z["rs"] ** 2, rr2=z["rr"] ** 2, rsr=z["rs"] * z["rr"]).groupby("day")
        m = zz[["rs", "rr", "rs2", "rr2", "rsr"]].mean()
        ic = (m["rsr"] - m["rs"] * m["rr"]) / np.sqrt((m["rs2"] - m["rs"] ** 2) * (m["rr2"] - m["rr"] ** 2))
    else:
        ic = pd.Series(dtype=float)
    return pd.DataFrame({"hits": hits, "n": n, "ls": ls, "ic": ic})


def metrics(day: pd.DataFrame):
    acc = day["hits"].sum() / max(day["n"].sum(), 1)
    ls = day["ls"].dropna()
    sharpe = ls.mean() / ls.std() * np.sqrt(252) if len(ls) > 2 and ls.std() > 0 else np.nan
    return {"accuracy": float(acc), "ic": float(day["ic"].mean()), "sharpe": float(sharpe),
            "ls_mean_bp": float(ls.mean() * 1e4), "n_obs": int(day["n"].sum()), "n_days": int(len(day)),
            "ic_days": int(day["ic"].notna().sum())}


def block_bootstrap(tables: dict, rng):
    alld = np.array(sorted(set().union(*[set(t.index) for t in tables.values()])))
    T = len(alld)
    nb = int(np.ceil(T / BLOCK))
    arrs = {k: {c: t.reindex(alld)[c].to_numpy(float) for c in ("hits", "n", "ls", "ic")} for k, t in tables.items()}
    res = {k: {"accuracy": np.empty(NBOOT), "ic": np.empty(NBOOT), "sharpe": np.empty(NBOOT)} for k in tables}
    for b in range(NBOOT):
        starts = rng.integers(0, T - BLOCK + 1, nb)
        idx = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel()[:T]
        for k, a in arrs.items():
            res[k]["accuracy"][b] = np.nansum(a["hits"][idx]) / max(np.nansum(a["n"][idx]), 1)
            res[k]["ic"][b] = np.nanmean(a["ic"][idx])
            ls = a["ls"][idx]
            ls = ls[~np.isnan(ls)]
            res[k]["sharpe"][b] = ls.mean() / ls.std() * np.sqrt(252) if len(ls) > 2 and ls.std() > 0 else np.nan
    return res


def ci(a):
    a = a[np.isfinite(a)]
    return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))] if len(a) else [np.nan, np.nan]


def main():
    common.ensure_dirs()
    rng = np.random.default_rng(SEED)
    S = sessions_from_prices()
    price_tk = sorted(pd.read_csv(common.RESULTS / "price_series_span.csv")["ticker"].astype(str))
    tk_code = {t: i for i, t in enumerate(price_tk)}
    d = load_news(S, tk_code)
    dmin = pd.Timestamp(np.datetime64(int(d.loc[d["A"] >= 0, "A"].min()), "D")) - pd.Timedelta(days=10)
    dmax = pd.Timestamp(np.datetime64(int(d["A"].max()), "D")) + pd.Timedelta(days=10)
    ret = load_returns(tk_code, dmin.date(), dmax.date())
    nas = (d["src"] == 2).to_numpy()
    subsets = {
        "all_ticker_news": (np.ones(len(d), bool), "tone_t", ["A", "C", "D", "B"]),
        "nasdaq_com_rows": (nas, "tone_t", ["A", "B"]),
        "nasdaq_com_rows_article_tone": (nas, "tone_a", ["A", "B"]),
        "benzinga_and_partner_rows": (~nas, "tone_t", ["A", "C", "D", "B"]),
        "benzinga_timed": (((d["src"] == 1) & ~d["date_only"]).to_numpy(), "tone_t", ["A", "C", "D", "B"]),
        "date_only_rows": (d["date_only"].to_numpy(), "tone_t", ["A", "B"]),
    }
    names = {"A": "A_fnspid", "C": "C_tz_fixed", "D": "D_trust_labels", "B": "B_no_leak"}
    out = {"n_news_rows_with_ticker": int(len(d)), "n_news_rows_ticker_in_prices": int((d["tk"] >= 0).sum()),
           "sessions": int(len(S)), "returns_rows": int(len(ret)), "window": [str(dmin.date()), str(dmax.date())],
           "signal_coverage_title": float(d["tone_t"].notna().mean()),
           "signal_coverage_article_nasdaq": float(d.loc[d["src"] == 2, "tone_a"].notna().mean()),
           "rows_by_subset": {},
           "nboot": NBOOT, "block": BLOCK}
    table = []
    for name, (mask, tone_col, als) in subsets.items():
        sub = d[mask]
        out["rows_by_subset"][name] = {"rows": int(len(sub)), "with_signal": int(sub[tone_col].notna().sum()),
                                       "date_only_share": float(sub["date_only"].mean())}
        days, point = {}, {}
        for al in als:
            j = daily_panel(sub, al, tone_col, ret)
            for leg in ("r", "r_on", "r_id"):
                key = f"{names[al]}|{leg}"
                days[key] = per_day_stats(j, leg)
                point[key] = metrics(days[key])
        boot = block_bootstrap(days, rng)
        for key, m in point.items():
            al, leg = key.split("|")
            row = {"subset": name, "tone": tone_col, "alignment": al, "target": leg, **m}
            base = f"B_no_leak|{leg}"
            for met in ("accuracy", "ic", "sharpe"):
                row[f"{met}_ci"] = ci(boot[key][met])
                if al != "B_no_leak":
                    row[f"{met}_minus_noleak"] = float(m[met] - point[base][met])
                    row[f"{met}_minus_noleak_ci"] = ci(boot[key][met] - boot[base][met])
            table.append(row)
            print(name, al, leg, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)
    pd.DataFrame(table).to_csv(common.RESULTS / "impact_table.csv", index=False)
    out["table"] = table
    (common.RESULTS / "impact.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
