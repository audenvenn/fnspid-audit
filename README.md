# fnspid-audit

An independent audit of [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) (Dong et al., KDD 2024), a free financial news and stock-price dataset that many people use to train and evaluate news-aware stock predictors. As of 2026-11-16 it had about 12,000 Hugging Face downloads in the previous 30 days and 118 citing works. Its README says the original repository is no longer maintained.

**What is here:** the full report ([AUDIT.md](AUDIT.md)), a small data linter ([finlint](finlint/)), the scripts that regenerate every number in the report ([scripts/](scripts/)), and sidecar correction files.

The correction files contain **no original news text and no prices**. You join them to your own copy of FNSPID.

## Findings

All counts come from the full dataset (28.6M news records, 29.7M price rows), checked against independent references. The report has the evidence for each one.

1. **Timestamps are 8 hours early.**
   - Cause: `preprocess.py` applies the US-Eastern offset with the wrong sign (`offset = timedelta(hours=-4)`, then `dt + offset`).
   - Benzinga: all 55,986 timed rows that match the upstream Kaggle file sit exactly 8 h before the true UTC time.
   - Reuters: from 2009-11-16 on, rows are New York local time labelled "UTC".
2. **Date-only news is joined to the same day's close.**
   - 98.6% of ticker-tagged rows have no time of day.
   - The reference pipeline merges on the calendar date, so news published after the 16:00 close feeds a close it could not have influenced.
3. **The price panel splices two Yahoo downloads adjusted at different dates.**
   - The splice is at 2020-07-06. Examples: AAPL 364.11 → 93.46, TSLA 1208.66 → 91.44.
   - 329 series jump or fall by more than 2× on that day.
   - About 3,000 series stop in 2020.
   - This explains the "abnormal prices" reported in the original repository's issue #13.
4. **Duplicated and off-topic rows.**
   - `nasdaq_exteral_data.csv` contains every record of `All_external.csv`, so concatenating the two files double-counts 13.0M rows.
   - About 0.8M rows are Russian-language Lenta.ru news with no ticker.
   - 16.6% of nasdaq.com titles name only other tickers.
5. **Price-file quality.** 1.08M opens stored as 0, frozen placeholder prices, ticker-reuse splices, two column layouts, and mojibake.

## Why it matters

We ran a transparent headline-tone signal (Loughran–McDonald) against the next-session return. Each row shows FNSPID's own alignment against a leakage-free one; intervals are 95% block-bootstrap CIs:

| Metric | FNSPID alignment | Leakage-free | Difference [95% CI] |
|---|---|---|---|
| Long–short Sharpe | 3.63 | 0.35 | +3.28 [2.58, 3.96] |
| Rank IC | 0.0195 | 0.0064 | +0.0131 [0.0080, 0.0181] |
| Directional accuracy | 50.60% | 50.23% | +0.37 pp [0.08, 0.67] |

- The inflation comes almost entirely from the overnight leg, i.e. after-close news predicting the next open.
- On the rows with explicit times, a careful user who trusts the "UTC" labels and applies a correct 16:00 cut-off still gets a Sharpe of 2.21 instead of −0.18. The 8-hour shift defeats an otherwise correct pipeline.
- Scope, caveats, and why the leakage-free alignment is conservative for date-only rows: [AUDIT.md §4](AUDIT.md#4-downstream-impact).

## Using the corrections

The two large sidecar files are attached to the [GitHub release](../../releases):
- `news_patch.parquet`: 28.6M rows keyed by (`file`, `row`). Contains corrected UTC times (`ts_utc`, `ts_status`), the leakage-safe session (`session_no_leak`), and duplicate, ticker, language and encoding flags.
- `price_factors.parquet`: 29.7M rows keyed by (`ticker`, `date`). Contains repair factors and status, but no prices.

The per-series flags are in `data/flags/`. Columns and semantics are documented in [AUDIT.md §6](AUDIT.md#6-corrections-sidecar-files).

```python
import pandas as pd
patch = pd.read_parquet("news_patch.parquet")   # file: 0 = All_external.csv, 1 = nasdaq_exteral_data.csv
# row = 0-based record index in file order at HF revision bf9189c; join it to your copy of FNSPID read in file order
news_f1 = patch[patch.file == 1]
safe = news_f1[news_f1.has_ticker & news_f1.ticker_in_prices]   # use `session_no_leak` as the trading date
```

## finlint

A small linter for free financial news and price data. It checks for:
- splices from mixed adjustment vintages, and split-like jumps vs. bad ticks;
- coverage breaks and survivorship;
- stale, zero or NaN prices, and OHLC violations;
- duplicates;
- the share of date-only news;
- timezone sanity, using headlines anchored to a trading session;
- look-ahead risk in a news → price join.

```sh
pip install "./finlint[parquet]"
finlint prices path/to/prices.parquet --json prices_report.json
finlint news   path/to/news.parquet
finlint join   path/to/news.parquet path/to/prices.parquet
```

It flags every defect class above on FNSPID, and is clean on the upstream timestamps and on the Yahoo reference data. Run the tests with `pytest finlint`.

## Reproducing

```sh
sh scripts/run_all.sh        # CPU only; ~1.5–2 h; streams ~29 GB from Hugging Face; peak memory ~3–4 GB
```

- Inputs are pinned (HF revision `bf9189c`, GitHub commit `4054842e`) and sha256-checked.
- Python ≥ 3.10 with numpy, pandas, pyarrow, scipy, requests and yfinance.
- `data/results/` holds the outputs of our run, including `numbers.md`, which traces every number in the report to the script that produced it.
- Raw price columns are removed from the committed copies. Running the scripts locally regenerates the full versions under `data/derived/results/`.

## Licensing and attribution

- **Code** (finlint and scripts): Apache-2.0. **Report and sidecar files**: CC BY 4.0.
- **FNSPID itself is not redistributed here.** Its README states that the authors stopped maintenance and released the rights for commercial and research use. Its prices originate from Yahoo Finance, so this repository contains no price values.
- This is an independent audit, not affiliated with the FNSPID authors. We thank them for releasing the dataset and its code.

## How this was made

The audit was carried out with substantial help from an AI assistant (Anthropic's Claude), which wrote most of the code and text under human direction. Every number in the report is produced by the scripts in this repository, and the checks can be re-run end to end. Please report errors via GitHub issues.

## Citation

If you use FNSPID, cite the original paper (citation as given in its README):

```bibtex
@misc{dong2024fnspid,
      title={FNSPID: A Comprehensive Financial News Dataset in Time Series},
      author={Zihan Dong and Xinyu Fan and Zhiyuan Peng},
      year={2024},
      eprint={2402.06698},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST}
}
```
