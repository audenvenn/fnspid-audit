#!/bin/sh
# Fetch the raw inputs (pinned revisions, checksums verified). Downloads are untrusted data: they go to
# data/raw/<name>/ and are only ever read by our scripts with `python -I`.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
REV=bf9189c41527198897d1af3e17b1a0095279fc45
HF="https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/$REV"
mkdir -p "$RAW/prices" "$RAW/kaggle_mirror" "$RAW/lm_dict" "$RAW/symdir" "$RAW/github_subset"

# 1. FNSPID prices (590 MB). The two news CSVs (5.7 GB, 23 GB) are streamed by 20_news_stream.py, not stored.
[ -f "$RAW/prices/full_history.zip" ] || curl -sSL -o "$RAW/prices/full_history.zip" "$HF/Stock_price/full_history.zip"
echo "03da4fce7ebea90d5715ba3501773d410ae663b617027b338ef000a9955dab91  $RAW/prices/full_history.zip" | shasum -a 256 -c

# 2. Upstream of FNSPID's timed Benzinga rows: third-party HF mirror of Kaggle raw_analyst_ratings (440 MB,
#    converted to parquet by 13_kaggle_mirror_convert.py and then deleted)
MREV=3d2752ad7a2cdec1834174eb5277ea637afea5f0
[ -f "$ROOT/data/derived/kaggle_analyst_ratings.parquet" ] || {
  curl -sSL -o "$RAW/kaggle_mirror/raw_analyst_ratings.jsonl" \
    "https://huggingface.co/datasets/khaihernlow/massive-stock-news-analysis-db-for-nlpbacktests/resolve/$MREV/data/raw_analyst_ratings.jsonl"
  echo "9fa9cde24faa0ece65ca747e60732f49863318924c9d6b51008180039f579a1a  $RAW/kaggle_mirror/raw_analyst_ratings.jsonl" | shasum -a 256 -c; }

# 3. Loughran-McDonald master dictionary as shipped in pysentiment2 0.1.1 (MIT); only LM.csv is extracted
[ -f "$RAW/lm_dict/LM.csv" ] || {
  curl -sSL -o "$RAW/lm_dict/pysentiment2-0.1.1-py3-none-any.whl" \
    https://files.pythonhosted.org/packages/d1/7c/596f3028260310d6206b7b88fe7d37fdb367913bfac9195912b27ab3cadb/pysentiment2-0.1.1-py3-none-any.whl
  echo "d0ff7a0b702654baeff6a6bec231e56ba4c4c9bc9ad4730856e8ca69335450c7  $RAW/lm_dict/pysentiment2-0.1.1-py3-none-any.whl" | shasum -a 256 -c
  (cd "$RAW/lm_dict" && unzip -o -q -j pysentiment2-0.1.1-py3-none-any.whl pysentiment2/static/LM.csv); }

# 4. Current listings (used only to ask whether a truncated ticker still trades today)
for f in nasdaqlisted otherlisted; do
  [ -f "$RAW/symdir/$f.txt" ] || curl -sS -A "Mozilla/5.0" -o "$RAW/symdir/$f.txt" "https://www.nasdaqtrader.com/dynamic/SymDir/$f.txt"
done

# 5. FNSPID code at the audited commit (only the files we cite are kept)
C=4054842ec476953b30ee874d4b7e8eea786a21fa
for f in README.md data_processor/preprocess.py data_processor/price_news_integrate.py data_processor/data_processor.md \
         data_scraper/stock_price_scraper/get_price_from_yahoo.py; do
  mkdir -p "$RAW/github_subset/$(dirname "$f")"
  [ -f "$RAW/github_subset/$f" ] || curl -sSL -o "$RAW/github_subset/$f" \
    "https://raw.githubusercontent.com/Zdong104/FNSPID_Financial_News_Dataset/$C/$f"
done
echo "$C" > "$RAW/github_subset/COMMIT"
echo fetch ok
