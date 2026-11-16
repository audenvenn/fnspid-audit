import numpy as np
import pandas as pd

from finlint import lookahead


SESS = pd.bdate_range("2021-03-01", "2021-03-31")   # no holidays in this window


def utc(s):
    return pd.Timestamp(s, tz="America/New_York").tz_convert("UTC").tz_localize(None)


def test_leakage_free_session_rules():
    ts = pd.Series([utc("2021-03-10 10:00"),   # before close -> same session
                    utc("2021-03-10 16:30"),   # after close -> next session
                    utc("2021-03-12 18:00"),   # Friday evening -> Monday
                    utc("2021-03-13 11:00"),   # Saturday -> Monday
                    pd.Timestamp("2021-03-10")])  # date-only -> next session
    do = [False, False, False, False, True]
    s = lookahead.leakage_free_session(ts, do, SESS)
    assert [str(x.date()) for x in s] == ["2021-03-10", "2021-03-11", "2021-03-15", "2021-03-15", "2021-03-11"]


def test_report_counts_leaks_of_same_date_join():
    # true publication times (UTC); stamps written 8 h early (wrong-sign conversion)
    true = pd.Series([utc("2021-03-10 10:00"), utc("2021-03-10 17:00"), utc("2021-03-10 21:30"),
                      utc("2021-03-11 01:30")])
    label = true - pd.Timedelta(hours=8)
    label = pd.concat([label, pd.Series([pd.Timestamp("2021-03-10")] * 2)], ignore_index=True)
    assigned = lookahead.same_date_join(label, SESS)
    f = {x.check: x for x in lookahead.lookahead_report(label, assigned, SESS, shift_h=8)}
    # row0 fine; rows 1,2 after close on 03-10 but joined to 03-10; row3 (01:30 on 03-11) stamped 03-10 17:30 UTC -> 03-10
    assert f["lookahead.timed_after_cutoff"].metrics["leaking"] == 3
    assert f["lookahead.date_only_same_day"].metrics["same_day"] == 2
    assert f["lookahead.summary"].metrics["certain_lookahead"] == 3
    assert f["lookahead.summary"].metrics["unverifiable"] == 2


def test_no_leak_when_aligned():
    true = pd.Series([utc("2021-03-10 10:00"), utc("2021-03-10 17:00")])
    s = lookahead.leakage_free_session(true, [False, False], SESS)
    f = {x.check: x for x in lookahead.lookahead_report(true, s, SESS, date_only=[False, False])}
    assert f["lookahead.summary"].metrics["certain_lookahead"] == 0
