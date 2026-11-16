import json

import pandas as pd

from finlint import cli
from test_news import make_session_news


def test_cli_prices_and_news(tmp_path, splice_panel, capsys):
    pp = tmp_path / "p.csv"
    splice_panel.to_csv(pp, index=False)
    rc = cli.main(["prices", str(pp), "--json", str(tmp_path / "p.json")])
    out = capsys.readouterr().out
    assert rc == 1 and "prices.splice_dates" in out
    js = json.loads((tmp_path / "p.json").read_text())
    assert any(x["check"] == "prices.splice_dates" and x["severity"] == "error" for x in js)

    nn = tmp_path / "n.csv"
    make_session_news().to_csv(nn, index=False)
    rc = cli.main(["news", str(nn)])
    out = capsys.readouterr().out
    assert rc == 1 and "best_shift_h=8" in out


def test_cli_join(tmp_path, capsys):
    n = make_session_news()
    d = pd.to_datetime(n["date"].str[:10])
    sess = pd.bdate_range(d.min() - pd.Timedelta(days=3), d.max() + pd.Timedelta(days=3))
    pr = pd.DataFrame({"ticker": "XYZ", "date": sess, "close": 1.0})
    n.to_csv(tmp_path / "n.csv", index=False)
    pr.to_csv(tmp_path / "p.csv", index=False)
    rc = cli.main(["join", str(tmp_path / "n.csv"), str(tmp_path / "p.csv")])
    out = capsys.readouterr().out
    assert "lookahead.timed_after_cutoff" in out and rc == 1
