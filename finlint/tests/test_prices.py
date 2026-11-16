import numpy as np
import pandas as pd

from finlint import prices


def by_check(findings):
    return {f.check: f for f in findings}


def test_nearest_split():
    k, res = prices.nearest_split(np.array([0.25, 4.02, 0.5, 1.03, 0.6667, 0.0833]))
    assert k[0] == 4 and abs(k[1] - 0.25) < 1e-12 and k[2] == 2
    assert np.isnan(k[3])                       # ordinary move, not a split
    assert abs(k[4] - 1.5) < 1e-9               # 3:2 split
    assert k[5] == 12


def test_splice_detected(splice_panel):
    p = prices.prepare(splice_panel)
    s = prices.detect_splice_dates(p)
    assert len(s) > 0
    splice_day = pd.bdate_range("2019-01-01", periods=300)[200]
    assert set(s["date"]) == {splice_day}
    sig = dict(zip(s["signal"], s["count"]))
    assert sig["jumps"] == 15                   # the 15 'split' tickers
    assert sig["adj_factor_reset"] == 30        # every series changes adjustment vintage
    j = prices.detect_jumps(p)
    assert (j["kind"] == "split_like").sum() == 15
    assert set(j["split_k"].dropna()) == {4.0}


def test_clean_panel_has_no_splice(clean_panel):
    f = by_check(prices.run_all(clean_panel))
    assert f["prices.splice_dates"].severity == "info"
    assert f["prices.split_like_jumps"].metrics["jumps"] == 0
    assert f["prices.duplicates"].metrics["dup_rows"] == 0


def test_spike_reverts_is_not_split(clean_panel):
    df = clean_panel.copy()
    i = df.index[(df.ticker == "C03")][100]
    df.loc[i, "close"] *= 10                    # one bad tick, next day back to normal
    j = prices.detect_jumps(prices.prepare(df))
    kinds = j.set_index("ticker")["kind"].to_dict()
    assert kinds == {"C03": "spike"}


def test_duplicates_weekend_stale(clean_panel):
    df = clean_panel.copy()
    df = pd.concat([df, df.iloc[[5]]])                                   # duplicate row
    sat = pd.Timestamp("2019-01-05")
    df = pd.concat([df, pd.DataFrame([{"ticker": "C00", "date": sat, "close": 1.0, "adj_close": 1.0, "volume": 0}])])
    m = df.ticker == "C07"
    idx = df.index[m][50:60]
    df.loc[idx, "close"] = df.loc[idx[0], "close"]                       # 10 identical closes
    f = by_check(prices.run_all(df.reset_index(drop=True)))
    assert f["prices.duplicates"].metrics["dup_rows"] == 1
    assert f["prices.weekend_rows"].metrics["rows"] == 1
    assert f["prices.stale_runs"].metrics["runs"] >= 1
    assert f["prices.stale_runs"].metrics["rows_in_runs"] >= 10


def test_coverage_end_cluster_and_survivorship():
    d = pd.bdate_range("2005-01-01", "2015-12-31")
    rows = []
    for i in range(60):
        end = d[-1] if i < 30 else pd.Timestamp("2012-06-29")        # 30 series stop on one shared date
        for x in d[d <= end][::20]:
            rows.append((f"S{i}", x, 10.0 + (x.day_of_year % 7) * 0.01))
    df = pd.DataFrame(rows, columns=["ticker", "date", "close"])
    f = by_check(prices.check_coverage(prices.prepare(df)))
    assert f["prices.coverage_end_clusters"].metrics["series_in_clusters"] == 30
    assert "prices.survivorship_hint" in f                                # no scattered delistings at all


def test_ohlc_violation(clean_panel):
    df = clean_panel.copy()
    df["open"] = df["close"]
    df["high"] = df["close"] * 1.01
    df["low"] = df["close"] * 0.99
    df.loc[3, "high"] = df.loc[3, "close"] * 0.5
    f = by_check(prices.run_all(df))
    assert f["prices.ohlc_consistency"].metrics["rows"] == 1


def test_chunked_equals_single_pass(splice_panel):
    df = splice_panel.copy()
    df.loc[df.index[:3], "close"] = -1.0                                    # some nonpositive rows
    one = {f.check: f.metrics for f in prices.run_all(df)}
    tickers = sorted(df["ticker"].unique())
    chunks = [df[df["ticker"].isin(tickers[i:i + 7])] for i in range(0, len(tickers), 7)]
    many = {f.check: f.metrics for f in prices.run_chunked(chunks)}
    assert one.keys() == many.keys()
    for k in one:
        assert one[k] == many[k], k


def test_zero_open_reported_separately(clean_panel):
    df = clean_panel.copy()
    df["open"] = df["close"]
    df["high"] = df["close"] * 1.01
    df["low"] = df["close"] * 0.99
    df.loc[:9, "open"] = 0.0
    f = by_check(prices.run_all(df))
    assert f["prices.zero_as_missing"].metrics["zero_open"] == 10
    assert f["prices.ohlc_consistency"].metrics["rows"] == 0


def test_level_blowup_from_cumulative_reverse_splits(clean_panel):
    """A serial reverse-splitter adjusted by its later split product: early prices ~1e6x the recent level and
    volumes divided down to 0. The clean panel must not trigger the check."""
    from finlint import prices
    clean = prices.run_all(clean_panel)
    assert not [f for f in clean if f.check == "prices.level_blowup"]
    df = clean_panel.copy()
    t0 = df["ticker"].iloc[0]
    m = df["ticker"] == t0
    idx = df.index[m][:100]                      # first 100 rows of one series, before 1:10^6 of reverse splits
    df.loc[idx, ["close", "adj_close"]] *= 1e6
    for c in ("open", "high", "low"):
        if c in df:
            df.loc[idx, c] *= 1e6
    df.loc[idx, "volume"] = 0
    out = [f for f in prices.run_all(df) if f.check == "prices.level_blowup"]
    assert len(out) == 1
    assert out[0].metrics["rows"] == 100 and out[0].metrics["zero_volume_rows"] == 100
    assert out[0].metrics["series"] == 1
