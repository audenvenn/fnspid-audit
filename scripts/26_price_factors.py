"""Project the local repaired price panel to the distributable factor table (no price values).

Input:  data/derived/release/prices_rc.parquet  (written by 15_prices_release.py; not distributed)
Output: data/derived/release/price_factors.parquet (ticker, date, layout, segment_from_latest,
        close_factor, adj_factor, rc_status, vintage_break)
"""
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I does not add the script dir
import common  # noqa: E402

COLS = ["ticker", "date", "layout", "segment_from_latest", "close_factor", "adj_factor", "rc_status", "vintage_break"]

if __name__ == "__main__":
    src = common.DERIVED / "release" / "prices_rc.parquet"
    dst = common.DERIVED / "release" / "price_factors.parquet"
    t = pq.read_table(src, columns=COLS)
    pq.write_table(t, dst, compression="zstd")
    print(f"wrote {dst} ({t.num_rows} rows)")
