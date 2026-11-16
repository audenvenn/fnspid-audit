#!/usr/bin/env python3
"""Root cause of the FNSPID price splices, checked against an independent Yahoo download (2026-11).

Model. A split-adjusted close downloaded at time V ("vintage") equals raw(t) / C(t, V), where C(t, V) is
the product of split ratios with ex-dates in (t, V]. Hence for t before both vintages
    FNSPID_close(t) / Yahoo_close(t) = C(V_fnspid, V_yahoo) = product of splits after V_fnspid.
Within a consistently adjusted segment this ratio is constant; across a splice it changes by the product
of the splits that happened between the two vintages. The same holds for adj_close with dividend
factors added.

For each sampled ticker:
  * segments = FNSPID rows between the detected boundary dates (2015-01-02, 2020-04-02, 2020-07-02, 2020-07-06)
  * per segment: median ratio to Yahoo, its dispersion, and the vintage interval implied by Yahoo's split
    history (the V whose split product matches the ratio within 3%)
  * jump test at 2020-07-06: observed discontinuity (FNSPID move / Yahoo move) vs the product of splits
    between 2020-07-03 and 2023-12-28
Outputs: data/derived/results/splice_segments.csv, splice_jumps.csv, splice_truncation.csv, splice.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

YDIR = common.RAW / "yahoo"
BOUNDS = [pd.Timestamp(x) for x in ("2015-01-02", "2020-04-02", "2020-07-02", "2020-07-06")]
V_PRE = pd.Timestamp("2020-07-03")     # day after the last old-vintage row
V_POST = pd.Timestamp("2023-12-28")    # last FNSPID date (files written 2023-12-30)
TOL = 0.03


def load_yahoo(t: str) -> pd.DataFrame | None:
    f = YDIR / f"{t}.csv"
    if not f.exists():
        return None
    y = pd.read_csv(f)
    if len(y) < 50 or "Close" not in y:
        return None
    y["Date"] = pd.to_datetime(y["Date"])
    return y.sort_values("Date").reset_index(drop=True)


def split_product_after(splits: pd.DataFrame, v: pd.Timestamp, until: pd.Timestamp | None = None) -> float:
    s = splits[(splits["Date"] > v) & ((splits["Date"] <= until) if until is not None else True)]
    return float(np.prod(s["ratio"])) if len(s) else 1.0


def vintage_interval(splits: pd.DataFrame, ratio: float, lo: pd.Timestamp, hi: pd.Timestamp):
    """Intervals [a, b) of vintage dates V in [lo, hi] for which the split product after V matches ratio."""
    cuts = [lo] + [d for d in splits["Date"] if lo < d <= hi] + [hi]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if abs(np.log(split_product_after(splits, a) / ratio)) < TOL:
            out.append((a, b))
    return out


def main():
    common.ensure_dirs()
    samp = pd.read_csv(common.RESULTS / "reference_sample.csv")
    P = pd.read_parquet(common.DERIVED / "prices.parquet", columns=["ticker", "date", "close", "adj_close", "volume"],
                        filters=[("ticker", "in", list(samp["ticker"]))])
    P["ticker"] = P["ticker"].astype(str)
    P["date"] = pd.to_datetime(P["date"])
    seg_rows, jump_rows, trunc_rows = [], [], []
    for t, grp in samp.groupby("ticker"):
        group = grp["group"].iloc[0]
        y = load_yahoo(t)
        f = P[P["ticker"] == t].sort_values("date")
        if y is None or f.empty:
            if group in ("truncated", "absent"):
                trunc_rows.append({"ticker": t, "group": group, "yahoo_rows": 0 if y is None else len(y),
                                   "fnspid_last": f["date"].max().date() if len(f) else None})
            jump_rows.append({"ticker": t, "group": group, "status": "no_yahoo_data" if y is None else "no_fnspid"})
            continue
        splits = y.loc[y["Stock Splits"] > 0, ["Date", "Stock Splits"]].rename(columns={"Stock Splits": "ratio"})
        m = f.merge(y[["Date", "Close", "Adj Close"]], left_on="date", right_on="Date", how="inner")
        m["rc"] = m["close"] / m["Close"]
        m["ra"] = m["adj_close"] / m["Adj Close"]
        # segments
        b = [d for d in BOUNDS if (f["date"] >= d).any() and (f["date"] < d).any()]
        edges = [f["date"].min()] + [f.loc[f["date"] >= d, "date"].min() for d in b] + [f["date"].max() + pd.Timedelta(days=1)]
        for i, (a, e) in enumerate(zip(edges[:-1], edges[1:])):
            s = m[(m["date"] >= a) & (m["date"] < e) & (m["date"] >= m["date"].max() - pd.Timedelta(days=99999))]
            s = s[(s["date"] >= max(a, e - pd.Timedelta(days=730)))]          # last <=2 years of each segment
            if len(s) < 3:
                continue
            lr = np.log(s["rc"].where(s["rc"] > 0)).dropna()
            ratio = float(np.exp(lr.median()))
            iqr = float(lr.quantile(0.75) - lr.quantile(0.25))
            vin = vintage_interval(splits, ratio, pd.Timestamp("2015-01-01"), y["Date"].max())
            seg_rows.append({"ticker": t, "group": group, "segment": i, "start": a.date(), "end": (e - pd.Timedelta(days=1)).date(),
                             "n_overlap": len(s), "ratio_close_to_yahoo": ratio, "log_iqr": iqr,
                             "vintage_from": vin[0][0].date() if vin else None, "vintage_to": vin[-1][1].date() if vin else None,
                             "n_vintage_intervals": len(vin)})
        # jump test at 2020-07-06
        def px(df, col, d):
            v = df.loc[df["date"] == d, col] if "date" in df else df.loc[df["Date"] == d, col]
            return float(v.iloc[0]) if len(v) else np.nan
        d0, d1 = pd.Timestamp("2020-07-02"), pd.Timestamp("2020-07-06")
        fj = px(f, "close", d1) / px(f, "close", d0)
        yj = px(y.rename(columns={"Date": "date"}), "Close", d1) / px(y.rename(columns={"Date": "date"}), "Close", d0)
        pred = 1.0 / split_product_after(splits, V_PRE, V_POST)
        obs = fj / yj if np.isfinite(fj) and np.isfinite(yj) else np.nan
        faj = px(f, "adj_close", d1) / px(f, "adj_close", d0)
        yaj = px(y.rename(columns={"Date": "date"}), "Adj Close", d1) / px(y.rename(columns={"Date": "date"}), "Adj Close", d0)
        q_pre = px(f, "adj_close", d0) / px(f, "close", d0)
        q_post = px(f, "adj_close", d1) / px(f, "close", d1)
        yq = y.assign(q=y["Adj Close"] / y["Close"]).set_index("Date")["q"]
        def yq_at(d):
            s = yq[yq.index <= d]
            return float(s.iloc[-1]) if len(s) else np.nan
        div_factor_yahoo = yq_at(V_PRE) / yq_at(V_POST) if len(yq) else np.nan
        jump_rows.append({"ticker": t, "group": group, "status": "ok" if np.isfinite(obs) else "missing_dates",
                          "fnspid_move": fj, "yahoo_move": yj, "discontinuity": obs, "pred_split_factor": pred,
                          "n_splits_between": int(((splits["Date"] > V_PRE) & (splits["Date"] <= V_POST)).sum()),
                          "match": bool(np.isfinite(obs) and abs(np.log(obs / pred)) < 0.02),
                          "adj_discontinuity": (faj / yaj) if np.isfinite(faj) and np.isfinite(yaj) else np.nan,
                          "pred_adj_factor": pred * div_factor_yahoo if np.isfinite(div_factor_yahoo) else np.nan,
                          "internal_div_factor": q_post / q_pre if np.isfinite(q_pre) and np.isfinite(q_post) else np.nan,
                          "yahoo_div_factor": div_factor_yahoo})
        if group in ("truncated", "absent"):
            last = f["date"].max()
            pre = m[m["date"] > last - pd.Timedelta(days=365)]
            lr = np.log(pre["rc"].where(pre["rc"] > 0)).dropna()
            ratio = float(np.exp(lr.median())) if len(lr) else np.nan
            trunc_rows.append({"ticker": t, "group": group, "fnspid_last": last.date(), "yahoo_rows": len(y),
                               "yahoo_last": y["Date"].max().date(),
                               "yahoo_rows_after_fnspid_last": int((y["Date"] > last).sum()),
                               "yahoo_rows_2020_07_to_2023_12": int(((y["Date"] > last) & (y["Date"] <= V_POST)).sum()),
                               "overlap_last_year": len(lr), "ratio": ratio,
                               "ratio_log_iqr": float(lr.quantile(.75) - lr.quantile(.25)) if len(lr) else np.nan,
                               "ratio_is_split_product": bool(len(lr) and any(abs(np.log(split_product_after(splits, v) / ratio)) < TOL
                                                                           for v in [last] + list(splits["Date"][splits["Date"] > last])))})
    S = pd.DataFrame(seg_rows)
    J = pd.DataFrame(jump_rows)
    T = pd.DataFrame([r for r in trunc_rows if r]).reindex(columns=[
        "ticker", "group", "fnspid_last", "yahoo_rows", "yahoo_last", "yahoo_rows_after_fnspid_last",
        "yahoo_rows_2020_07_to_2023_12", "overlap_last_year", "ratio", "ratio_log_iqr", "ratio_is_split_product"])
    T["yahoo_rows"] = T["yahoo_rows"].fillna(0)
    for c in ("yahoo_rows_after_fnspid_last", "yahoo_rows_2020_07_to_2023_12"):
        T[c] = T[c].fillna(0)
    T["ratio_is_split_product"] = T["ratio_is_split_product"].fillna(False).astype(bool)
    S.to_csv(common.RESULTS / "splice_segments.csv", index=False)
    J.to_csv(common.RESULTS / "splice_jumps.csv", index=False)
    T.to_csv(common.RESULTS / "splice_truncation.csv", index=False)
    ok = J[J["status"] == "ok"]
    summ = {"sample": samp["group"].value_counts().to_dict(),
            "with_yahoo_and_both_dates": ok["group"].value_counts().to_dict()}
    for g in ("jump", "control"):
        x = ok[ok["group"] == g].copy()
        x["match"] = x["match"].astype(bool)
        no_break = ~x["match"] & (np.abs(np.log(x["discontinuity"])) < 1e-6)     # vintage change happens at another boundary
        unexplained = ~x["match"] & ~no_break
        has_vintage_change = x["internal_div_factor"] < 0.995                    # adj/close reset visible at 2020-07-06
        e_div = np.abs(np.log(x["internal_div_factor"] / x["yahoo_div_factor"]))
        summ[f"{g}_2020_07_06"] = {
            "n": int(len(x)),
            "discontinuity_matches_split_product": int(x["match"].sum()),
            "with_split_between_vintages": int((x["n_splits_between"] > 0).sum()),
            "matched_and_split": int((x["match"] & (x["n_splits_between"] > 0)).sum()),
            "no_break_at_this_boundary": int(no_break.sum()),
            "unexplained": int(unexplained.sum()), "unexplained_tickers": sorted(x.loc[unexplained, "ticker"]),
            "median_abs_log_err": float(np.nanmedian(np.abs(np.log(x["discontinuity"] / x["pred_split_factor"])))),
            "adj_discontinuity_matches_pred_2pct": int((np.abs(np.log(x["adj_discontinuity"] / x["pred_adj_factor"])) < 0.02).sum()),
            "with_visible_dividend_vintage_change": int(has_vintage_change.sum()),
            "dividend_factor_internal_vs_yahoo_within_0p5pct": int((has_vintage_change & (e_div < 0.005)).sum()),
            "dividend_factor_internal_vs_yahoo_median_abs_log_err": float(np.nanmedian(e_div[has_vintage_change])),
        }
    if len(S):
        last_seg = S.sort_values("segment").groupby("ticker").tail(1)
        pre = S[(S["end"].astype(str) >= "2020-06-01") & (S["end"].astype(str) <= "2020-07-02")]
        summ["segment_vintages"] = {
            "last_segment_vintage_from_counts": last_seg["vintage_from"].astype(str).value_counts().head(8).to_dict(),
            "segments_ending_2020_07_02_vintage_to_counts": pre["vintage_to"].astype(str).value_counts().head(8).to_dict(),
            "segments_with_consistent_ratio_iqr_lt_1pct": int((S["log_iqr"] < 0.01).sum()), "segments": int(len(S))}
    if len(T):
        tr = T[T["group"] == "truncated"]
        summ["truncated"] = {"n": int(len(tr)), "with_yahoo": int((tr["yahoo_rows"] > 0).sum()),
                             "yahoo_has_rows_after_fnspid_end": int((tr["yahoo_rows_after_fnspid_last"] > 20).sum()),
                             "same_security_and_continues": int(((tr["yahoo_rows_after_fnspid_last"] > 20) &
                                                                 tr["ratio_is_split_product"] & (tr["ratio_log_iqr"] < 0.02)).sum()),
                             "continues_through_2023_12": int((tr["yahoo_rows_2020_07_to_2023_12"] > 800).sum())}
        summ["absent"] = T[T["group"] == "absent"].to_dict("records")
    ex = {}
    for t in summ.get("jump_2020_07_06", {}).get("unexplained_tickers", []):
        f = P[(P["ticker"] == t) & P["date"].isin([pd.Timestamp("2020-07-02"), pd.Timestamp("2020-07-06")])]
        y = load_yahoo(t)
        yy = y[y["Date"].isin([pd.Timestamp("2020-07-02"), pd.Timestamp("2020-07-06")])] if y is not None else pd.DataFrame()
        ex[t] = {"fnspid": f[["date", "close", "volume"]].astype(str).to_dict("records"),
                 "yahoo": yy[["Date", "Close", "Volume"]].astype(str).to_dict("records") if len(yy) else []}
    summ["unexplained_examples"] = ex
    (common.RESULTS / "splice.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))
    print(J[J.group == "jump"].sort_values("discontinuity").head(10).to_string())


if __name__ == "__main__":
    main()
