"""Shared paths and constants for the FNSPID audit scripts.

All scripts import this module from the scripts/ directory (never from data/).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../projects/fnspid
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
RESULTS = DERIVED / "results"                        # small JSON/CSV outputs quoted in AUDIT.md

HF_REPO = "Zihan1004/FNSPID"
HF_REV = "bf9189c41527198897d1af3e17b1a0095279fc45"   # dataset revision audited (lastModified 2024-04-09)
GH_REPO = "https://github.com/Zdong104/FNSPID_Financial_News_Dataset.git"
GH_COMMIT = "4054842ec476953b30ee874d4b7e8eea786a21fa"  # 2025-07-24, last commit

NEWS_FILES = {
    0: "Stock_news/All_external.csv",
    1: "Stock_news/nasdaq_exteral_data.csv",
}
NEWS_SIZES = {0: 5731397037, 1: 23232979597}
PRICE_ZIP = RAW / "prices" / "full_history.zip"
PRICE_ZIP_SHA256 = "03da4fce7ebea90d5715ba3501773d410ae663b617027b338ef000a9955dab91"

# Third-party mirror of the Kaggle "Massive Stock News Analysis DB" (miguelaenlle), raw_analyst_ratings
KAGGLE_MIRROR_REPO = "khaihernlow/massive-stock-news-analysis-db-for-nlpbacktests"
KAGGLE_MIRROR_REV = "3d2752ad7a2cdec1834174eb5277ea637afea5f0"
KAGGLE_MIRROR_SHA256 = "9fa9cde24faa0ece65ca747e60732f49863318924c9d6b51008180039f579a1a"

LM_CSV = RAW / "lm_dict" / "LM.csv"   # Loughran-McDonald master dictionary as shipped in pysentiment2 0.1.1


def hf_url(path: str, repo: str = HF_REPO, rev: str = HF_REV) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


def ensure_dirs() -> None:
    for p in (DERIVED, RESULTS):
        p.mkdir(parents=True, exist_ok=True)


def load_lm_sets():
    """Return (positive, negative) LM word sets (upper-case)."""
    import csv
    pos, neg = set(), set()
    with open(LM_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            w = r["Word"].strip().upper()
            if not w:
                continue
            if float(r["Positive"] or 0) > 0:
                pos.add(w)
            if float(r["Negative"] or 0) > 0:
                neg.add(w)
    return pos, neg


def nice_env() -> None:
    """Lower our own priority (shared machine)."""
    try:
        os.nice(10)
    except OSError:
        pass
