import numpy as np
import pandas as pd
import pytest


def bdays(start, n):
    return pd.bdate_range(start, periods=n)


@pytest.fixture
def splice_panel():
    """30 tickers over 300 business days; on day 200 the source switches from a download adjusted as of
    day 199 to one adjusted later, so tickers 0-14 (which 'split' 4:1 after day 199) jump by 1/4 and the
    dividend factor adj/close resets for every dividend payer."""
    rng = np.random.default_rng(0)
    d = bdays("2019-01-01", 300)
    rows = []
    for i in range(30):
        lr = rng.normal(0, 0.01, len(d))
        px = 50 * np.exp(np.cumsum(lr))
        close = px.copy()
        close[200:] /= 4 if i < 15 else 1          # new segment adjusted for a later 4:1 split
        q = np.ones(len(d))
        q[:200] = np.linspace(0.97, 1.0, 200)        # old vintage: factor -> 1 at its download date
        q[200:] = np.linspace(0.95, 0.99, 100)       # new vintage: later dividends included
        for k in range(len(d)):
            rows.append((f"T{i:02d}", d[k], close[k], close[k] * q[k], 1e6))
    return pd.DataFrame(rows, columns=["ticker", "date", "close", "adj_close", "volume"])


@pytest.fixture
def clean_panel():
    rng = np.random.default_rng(1)
    d = bdays("2019-01-01", 300)
    rows = []
    for i in range(30):
        px = 20 * np.exp(np.cumsum(rng.normal(0, 0.01, len(d))))
        q = np.linspace(0.9, 1.0, len(d))
        for k in range(len(d)):
            rows.append((f"C{i:02d}", d[k], px[k], px[k] * q[k], 1e6))
    return pd.DataFrame(rows, columns=["ticker", "date", "close", "adj_close", "volume"])
