#!/usr/bin/env python3
"""Corrected price release candidate: every FNSPID price row with vintage flags and, where justified,
repair factors that put earlier segments on the adjustment basis of each series' latest segment.

Per series and detected boundary b (first row on/after 2015-01-02, 2020-04-02, 2020-07-02, 2020-07-06):
  r_b  = close_b / close_prev,   dq_b = (adj/close)_b / (adj/close)_prev
  confirmed  (Yahoo sample): disc_b = r_b / yahoo_move_b equals a product (or inverse) of consecutive
             Yahoo split ratios dated after the previous row (<= 2023-12-28) within 2%, and the
             FNSPID/Yahoo ratio is stable (log-IQR < 2%) over 60 rows on each side (same security).
             Close factor f_b = that exact split product; adj factor = f_b * dq_b.
  assumed    (no independent data): |log r_b| < 0.2, so no split is assumed (f_b = 1); the dividend
             vintage change is taken from the data itself (adj factor = dq_b). Validated on the Yahoo
             control sample (see splice.json) but not independently confirmed for these series.
  unconfirmed: split-like move without confirmation -> earlier rows get no factor (NaN).
Row factors are cumulative products over the boundaries after the row. Repaired values are
  open/high/low/close * close_factor, adj_close * adj_factor, volume / close_factor
(NaN factor = not repairable without an independent source).

Outputs: data/derived/release/prices_rc.parquet, prices_series_flags.parquet, prices_rc_summary.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

REL = common.DERIVED / "release"
YDIR = common.RAW / "yahoo"
BOUNDS = pd.to_datetime(["2015-01-02", "2020-04-02", "2020-07-02", "2020-07-06"])
V_LAST = pd.Timestamp("2023-12-28")
CLUSTER_ENDS = pd.to_datetime(["2020-04-01", "2020-06-19", "2020-07-02"])


def yahoo(t):
    f = YDIR / f"{t}.csv"
    if not f.exists():
        return None
    y = pd.read_csv(f)
    if len(y) < 100:
        return None
    y["Date"] = pd.to_datetime(y["Date"])
    return y.set_index("Date").sort_index()


def split_products(splits: pd.Series):
    """All products of consecutive split events (including the empty product 1.0)."""
    r = list(splits.to_numpy())
    out = {1.0}
    for i, j in itertools.combinations(range(len(r) + 1), 2):
        out.add(float(np.prod(r[i:j])))
    return out


def confirm(f: pd.DataFrame, y: pd.DataFrame, i_prev: int, i_b: int):
    dp, db = f["date"].iat[i_prev], f["date"].iat[i_b]
    if dp not in y.index or db not in y.index:
        return None, "yahoo_missing_dates"
    disc = (f["close"].iat[i_b] / f["close"].iat[i_prev]) / (y.at[db, "Close"] / y.at[dp, "Close"])
    sp = y.loc[(y.index > dp) & (y.index <= V_LAST) & (y["Stock Splits"] > 0), "Stock Splits"]
    cands = split_products(sp)
    best = min(list(cands) + [1 / c for c in cands], key=lambda c: abs(np.log(disc / c)))
    if abs(np.log(disc / best)) > 0.02:
        return None, "no_split_product_match"
    # same security on both sides: FNSPID/Yahoo ratio stable within each side
    for lo, hi in ((max(0, i_prev - 59), i_prev + 1), (i_b, min(len(f), i_b + 60))):
        seg = f.iloc[lo:hi]
        rr = seg["close"].to_numpy() / y["Close"].reindex(seg["date"]).to_numpy()
        lr = np.log(rr[np.isfinite(rr) & (rr > 0)])
        if len(lr) < 10 or np.quantile(lr, .75) - np.quantile(lr, .25) > 0.02:
            return None, "ratio_unstable"
    return best, "confirmed"


def process(P: pd.DataFrame, symdir: set):
    """Factors/status for a (ticker, date)-sorted chunk of whole series."""
    fac_c = np.ones(len(P))
    fac_a = np.ones(len(P))
    seg = np.zeros(len(P), np.int8)
    status = np.full(len(P), "single_vintage", dtype=object)
    flags, brk_rows = [], []
    starts = np.r_[0, np.nonzero(P["ticker"].to_numpy()[1:] != P["ticker"].to_numpy()[:-1])[0] + 1, len(P)]
    order = {"confirmed": 0, "assumed_no_split": 1, "unconfirmed": 2}
    for k in range(len(starts) - 1):
        a, e = starts[k], starts[k + 1]
        f = P.iloc[a:e]
        t = f["ticker"].iat[0]
        dates = f["date"]
        bi = [int(np.searchsorted(dates.to_numpy(), b.to_datetime64())) for b in BOUNDS]
        bi = sorted({i for i in bi if 0 < i < len(f)})
        y = yahoo(t) if bi else None
        cf, af = 1.0, 1.0
        worst = "confirmed"
        nb = len(bi)
        for j, ib in enumerate(reversed(bi)):          # latest boundary first
            ip = ib - 1
            r = f["close"].iat[ib] / f["close"].iat[ip]
            q = (f["adj_close"].iat[ib] / f["close"].iat[ib]) / (f["adj_close"].iat[ip] / f["close"].iat[ip])
            fac, how = (None, "no_yahoo")
            if y is not None:
                fac, how = confirm(f, y, ip, ib)
            if fac is not None:
                bstat = "confirmed"
            elif abs(np.log(r)) < 0.2:
                fac, bstat = 1.0, "assumed_no_split"
            else:
                fac, bstat = np.nan, "unconfirmed"
            brk_rows.append({"ticker": t, "boundary_date": dates.iat[ib].date(), "prev_date": dates.iat[ip].date(),
                             "r": r, "dq": q, "close_factor": fac, "adj_factor": fac * q if np.isfinite(fac) else np.nan,
                             "status": bstat, "detail": how})
            if order[bstat] > order[worst]:
                worst = bstat
            cf = cf * fac
            af = af * fac * q if np.isfinite(fac) else np.nan
            lo = bi[nb - j - 2] if nb - j - 2 >= 0 else 0
            sl = slice(a + lo, a + ib)
            fac_c[sl] = cf
            fac_a[sl] = af
            seg[sl] = j + 1
            status[sl] = {"confirmed": "repaired_confirmed", "assumed_no_split": "repaired_assumed_no_split",
                          "unconfirmed": "unrepaired_break"}[worst]
        if bi:
            status[a + bi[-1]:e] = "latest_vintage"
        flags.append({"ticker": t, "first": dates.iat[0].date(), "last": dates.iat[-1].date(), "rows": len(f),
                      "layout": int(f["layout"].iat[0]), "n_boundaries": nb,
                      "ends_on_cluster_date": bool(dates.iat[-1] in set(CLUSTER_ENDS)),
                      "ends_before_2023_12": bool(dates.iat[-1] < pd.Timestamp("2023-12-01")),
                      "ticker_listed_2026_11": bool(t in symdir or t.replace(".", "-") in symdir or t.replace("-", ".") in symdir),
                      "series_status": "single_vintage" if not bi else worst})
    P = P.copy()
    P["segment_from_latest"] = seg
    P["close_factor"] = fac_c
    P["adj_factor"] = fac_a
    P["rc_status"] = status
    for c in ("open", "high", "low", "close"):
        P[c + "_rc"] = P[c] * P["close_factor"]
    P["adj_close_rc"] = P["adj_close"] * P["adj_factor"]
    P["volume_rc"] = P["volume"] / P["close_factor"]
    B = pd.DataFrame(brk_rows)
    if len(B):
        key = pd.MultiIndex.from_arrays([B["ticker"], pd.to_datetime(B["boundary_date"])])
        P["vintage_break"] = pd.MultiIndex.from_arrays([P["ticker"], P["date"]]).isin(key)
    else:
        P["vintage_break"] = False
    return P, pd.DataFrame(flags), B


def main():
    REL.mkdir(parents=True, exist_ok=True)
    symdir = set(pd.read_csv(common.RAW / "symdir" / "nasdaqlisted.txt", sep="|")["Symbol"].dropna()) | \
        set(pd.read_csv(common.RAW / "symdir" / "otherlisted.txt", sep="|")["ACT Symbol"].dropna())
    src = common.DERIVED / "prices.parquet"
    tickers = sorted(pd.read_parquet(src, columns=["ticker"])["ticker"].astype(str).unique())
    writer = None
    Fs, Bs = [], []
    counts = {}
    ext_before = ext_after = ext_unrep = 0
    for c0 in range(0, len(tickers), 400):
        chunk = tickers[c0:c0 + 400]
        P = pd.read_parquet(src, filters=[("ticker", "in", chunk)])
        P["ticker"] = P["ticker"].astype(str)
        P["date"] = pd.to_datetime(P["date"])
        P = P.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
        P, F, B = process(P, symdir)
        on_b = P["date"].isin(BOUNDS)
        r0 = P.groupby("ticker")["close"].pct_change() + 1
        r1 = P.groupby("ticker")["close_rc"].pct_change() + 1
        ext_before += int((on_b & ((r0 > 2) | (r0 < 0.5))).sum())
        ext_after += int((on_b & ((r1 > 2) | (r1 < 0.5))).sum())
        ext_unrep += int((on_b & ((r0 > 2) | (r0 < 0.5)) & P["close_rc"].isna()).sum())
        # the *_rc columns are not stored (close_rc = close * close_factor, etc.) to keep the file compact
        P = P.drop(columns=["open_rc", "high_rc", "low_rc", "close_rc", "adj_close_rc", "volume_rc"])
        for k_, v_ in P["rc_status"].value_counts().items():
            counts[k_] = counts.get(k_, 0) + int(v_)
        P["date"] = P["date"].dt.date
        P["rc_status"] = P["rc_status"].astype(str)
        tbl = pa.Table.from_pandas(P, preserve_index=False)
        tbl = tbl.cast(pa.schema([f.with_type(pa.dictionary(pa.int32(), pa.string())) if f.name in ("ticker", "rc_status")
                                  else f for f in tbl.schema]))
        if writer is None:
            writer = pq.ParquetWriter(REL / "prices_rc.parquet", tbl.schema, compression="zstd", compression_level=9)
        writer.write_table(tbl)
        Fs.append(F)
        Bs.append(B)
        print(f"{c0 + len(chunk)}/{len(tickers)} tickers", flush=True)
    writer.close()
    F = pd.concat(Fs, ignore_index=True)
    B = pd.concat(Bs, ignore_index=True)
    F.to_parquet(REL / "prices_series_flags.parquet", index=False)
    B.to_csv(common.RESULTS / "prices_rc_boundaries.csv", index=False)
    summ = {"rows": int(sum(counts.values())), "rows_by_status": counts,
            "boundaries": int(len(B)), "boundaries_by_status": B["status"].value_counts().to_dict(),
            "boundaries_by_date_status": B.groupby(["boundary_date", "status"]).size().rename("n").reset_index().astype(str).to_dict("records"),
            "confirmed_nontrivial_split_factor": int(((B["status"] == "confirmed") & (np.abs(np.log(B["close_factor"])) > 1e-9)).sum()),
            "series": int(len(F)), "series_by_status": F["series_status"].value_counts().to_dict(),
            "series_end_cluster": int(F["ends_on_cluster_date"].sum()),
            "series_end_cluster_listed_2026": int((F["ends_on_cluster_date"] & F["ticker_listed_2026_11"]).sum()),
            "unconfirmed_detail": B.loc[B["status"] == "unconfirmed", "detail"].value_counts().to_dict(),
            "extreme_moves_on_boundary_dates_before": ext_before,
            "extreme_moves_on_boundary_dates_after_repair_finite": ext_after,
            "extreme_moves_on_boundary_dates_left_unrepaired": ext_unrep}
    (REL / "prices_rc_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
