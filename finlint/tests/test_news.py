import numpy as np
import pandas as pd

from finlint import news


def make_session_news(n=400, label_shift_h=-8, seed=0):
    """Headlines published inside their named session (New York time); stamps written as 'UTC' after a
    wrong-sign conversion (label = true UTC + label_shift_h)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2019-03-01", "2019-10-31")
    kinds = [("Stocks Moving In Friday's Pre-Market Session", 7 * 60, 9 * 60),
             ("Mid-Day Market Update: Dow Rises", 12 * 60, 13 * 60),
             ("Stocks Moving In Thursday's After-Hours Session", 16 * 60 + 30, 19 * 60)]
    rows = []
    for i in range(n):
        title, a, b = kinds[i % 3]
        day = days[rng.integers(len(days))]
        local = (day + pd.Timedelta(minutes=int(rng.integers(a, b)))).tz_localize("America/New_York")
        utc = local.tz_convert("UTC").tz_localize(None)
        lab = utc + pd.Timedelta(hours=label_shift_h)
        rows.append((lab.strftime("%Y-%m-%d %H:%M:%S UTC"), title, "XYZ", f"u{i}"))
    return pd.DataFrame(rows, columns=["date", "title", "ticker", "url"])


def test_timezone_shift_detected():
    df = make_session_news()
    f = news.check_timezone(df)[0]
    assert f.severity == "error"
    assert f.metrics["best_shift_h"] == 8
    assert f.metrics["in_window_best"] > 0.95
    assert f.metrics["in_window_as_labelled"] < 0.2


def test_correct_timestamps_pass():
    df = make_session_news(label_shift_h=0)
    f = news.check_timezone(df)[0]
    assert f.severity == "info"
    assert f.metrics["in_window_as_labelled"] > 0.95


def test_date_only_share_and_disguise():
    d = ["2020-01-0%d 00:00:00 UTC" % (i % 9 + 1) for i in range(90)]
    d += ["2020-01-02 13:%02d:00 UTC" % m for m in (5, 17, 29, 41, 53)]
    d += ["2020-01-03 23:00:00 UTC"] * 20         # date-only rows shifted to 23:00
    df = pd.DataFrame({"date": d, "title": "x", "ticker": "A", "url": [f"u{i}" for i in range(len(d))]})
    f = {x.check: x for x in news.check_date_only(df)}
    assert f["news.date_only_share"].metrics["date_only"] == 90
    assert f["news.date_only_share"].severity == "error"
    assert f["news.constant_time_of_day"].metrics["rows"] == 20


def test_to_naive_offsets():
    s = news.to_naive(pd.Series(["2020-06-05 10:30:54-04:00", "2020-06-05 06:30:54 UTC", "2020-06-05"]))
    assert str(s[0]) == "2020-06-05 14:30:54"
    assert str(s[1]) == "2020-06-05 06:30:54"
    assert str(s[2]) == "2020-06-05 00:00:00"


def test_duplicates():
    df = pd.DataFrame({"date": ["2020-01-02 00:00:00 UTC"] * 4, "title": ["a", "a", "b", "c"],
                       "ticker": ["A", "A", "A", "B"], "url": ["u1", "u1", "u2", "u2"]})
    f = {x.check: x for x in news.check_duplicates(df)}
    assert f["news.exact_duplicate_rows"].metrics["rows"] == 1
    assert f["news.duplicate_ticker_url"].metrics["rows"] == 1
    assert f["news.duplicate_ticker_title_time"].metrics["rows"] == 1


def test_hour_profile_flags_night_heavy_stamps():
    df = make_session_news()
    f = news.check_hour_profile(df)[0]
    assert f.metrics["share_00_04"] > 0.15 or f.metrics["share_rth"] < 0.5
