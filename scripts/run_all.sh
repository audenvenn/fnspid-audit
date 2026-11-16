#!/bin/sh
# Regenerate every number in AUDIT.md from scratch.  CPU only; peak RSS ~3-4 GB; ~1.5-2 h, dominated by
# streaming the two news CSVs (29 GB) from Hugging Face.  Run from anywhere:  sh scripts/run_all.sh
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
export ARROW_DEFAULT_MEMORY_POOL=system      # return freed Arrow memory to the OS (keeps peak RSS low)
run() { echo "== $*"; nice -n 10 "$PY" -I "$@"; }
cd "$ROOT"

sh scripts/01_fetch.sh                                   # raw inputs (pinned revisions, sha256-checked)
run scripts/10_prices_build.py                           # full_history.zip -> data/derived/prices.parquet
run scripts/11_prices_boundaries.py                      # splice dates, boundary jumps, series spans
run scripts/12_prices_reference_fetch.py                 # Yahoo reference sample (cached; polite)
run scripts/12_prices_reference_fetch.py --retry-short
run scripts/13_kaggle_mirror_convert.py && rm -f data/raw/kaggle_mirror/raw_analyst_ratings.jsonl
run scripts/14_prices_splice.py                          # root cause vs Yahoo
run scripts/15_prices_release.py                         # release/prices_rc.parquet (+ flags)
run scripts/26_price_factors.py                          # release/price_factors.parquet (distributable; no prices)
run scripts/16_prices_release_validate.py                # repaired columns vs Yahoo
run scripts/17_prices_quality.py                         # placeholder runs, NaN/zero rows, layouts
run scripts/20_news_stream.py --file 0                   # stream All_external.csv (5.7 GB)
run scripts/20_news_stream.py --file 1                   # stream nasdaq_exteral_data.csv (23 GB)
run scripts/21_news_audit.py                             # composition, duplicates, tickers, encoding
run scripts/22_news_timestamps.py                        # timezone defects and corrections
run scripts/25_news_dates.py                             # what calendar date-only stamps use
run scripts/23_news_release.py                           # release/news_patch.parquet
run scripts/24_news_materialize.py --file 1 --tickers AAPL,MSFT,TSLA --drop-summaries
run scripts/24_news_materialize.py --file 0 --tickers AAPL,MSFT,TSLA
run scripts/30_impact_experiment.py                      # downstream impact with block-bootstrap CIs
run scripts/40_finlint_runs.py                           # finlint reports on FNSPID and references
"$PY" -m pytest -q finlint                               # finlint test-suite
run scripts/50_numbers.py                                # numbers quoted in AUDIT.md -> results/numbers.md
