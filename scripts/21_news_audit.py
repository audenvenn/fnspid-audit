#!/usr/bin/env python3
"""News-side audit over all FNSPID news rows (both CSVs, parsed by 20_news_stream.py).

Composition by source, ticker presence, date-only share, impossible dates, encoding problems,
duplicates (within and across files), news that cannot be joined to prices, and news-ticker mismatches.
Memory is bounded by processing one file at a time and keeping only sorted 64-bit key sets across files.

Outputs: data/derived/results/news_composition.csv, news_audit.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import newslib as nl  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

COLS = ["file", "row", "date_raw", "title", "ticker", "url", "art_len", "art_hash",
        "enc_t", "enc_a", "bad_utf8", "tk_title", "tk_art", "tk_art_strict"]
SEP = "\x1f"
PAREN_TK = re.compile(r"\((?:(?:NYSE|NASDAQ|Nasdaq|NYSEMKT|NYSEAMERICAN|AMEX|NYSEARCA)\s*:\s*)?([A-Z]{1,5}(?:\.[A-Z])?)\)")


def h64(s: pd.Series) -> np.ndarray:
    """Deterministic 64-bit hash (pandas' fixed-key siphash) of each string."""
    return pd.util.hash_array(s.astype(str).to_numpy(dtype=object)).view(np.int64)


def dup_mask(x: np.ndarray) -> np.ndarray:
    """True for every occurrence after the first (sort-based; lighter than a hash table on 15M rows)."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    m = np.zeros(len(x), bool)
    m[order[1:]] = xs[1:] == xs[:-1]
    return m


def n_dups(x: np.ndarray) -> int:
    return int(len(x) - len(np.unique(x)))


def title_names_other_ticker(title: pd.Series, ticker: pd.Series) -> np.ndarray:
    """Title carries parenthesised ticker(s), e.g. "Biogen's (BIIB) ...", none of which is the assigned ticker."""
    out = np.zeros(len(title), bool)
    for i, (t, k) in enumerate(zip(title.to_numpy(), ticker.to_numpy())):
        if "(" in t:
            found = PAREN_TK.findall(t)
            if found and k not in found:
                out[i] = True
    return out


def main():
    common.ensure_dirs()
    span = pd.read_csv(common.RESULTS / "price_series_span.csv", parse_dates=["first", "last"]).set_index("ticker")
    price_tickers = set(span.index)
    empty_h = h64(pd.Series([""]))[0]
    orphan = {"after_series_end": 0, "before_series_start": 0, "ticker_not_in_prices": 0, "with_ticker": 0}
    orphan_by_tk, comp, enc_rows, mism, tickers_seen = {}, {}, [], [], set()
    res = {"rows_by_file": {}}
    sets = {}

    class SrcCode(dict):
        def setdefault_all(self, s):
            return self.setdefault(s, len(self))
    src_code = SrcCode()
    for fid in (0, 1):
        K = []
        for path in nl.parts(fid):
            d = pq.read_table(path, columns=COLS).to_pandas()
            d["src"] = nl.source_of(d["url"])
            ts = nl.stamp(d["date_raw"])
            d["day"] = ts.dt.normalize()
            d["date_only"] = nl.is_date_only(ts)
            d["has_tk"] = d["ticker"].str.len() > 0
            d["cyr"] = d["title"].str.contains("[А-Яа-яЁё]", regex=True)
            # "AT&T;", "A&B; Properties", "S&P; 500": an ampersand turned into a bogus entity by a bad HTML round-trip
            d["amp_mangled"] = d["title"].str.contains(r"\w&(?!(?:amp|lt|gt|quot|apos|nbsp|#\d+);)[A-Za-z]{1,6};", regex=True)
            d["in_prices"] = d["ticker"].isin(price_tickers)
            tickers_seen |= set(d.loc[d["has_tk"], "ticker"].unique())
            # news that cannot be joined to any price row because the series ends early, starts late or is absent
            if fid == 1:
                tkd_ = d[d["has_tk"]]
                last_ = tkd_["ticker"].map(span["last"])
                first_ = tkd_["ticker"].map(span["first"])
                orphan["with_ticker"] += len(tkd_)
                orphan["ticker_not_in_prices"] += int(last_.isna().sum())
                aft = tkd_["day"] > last_
                orphan["after_series_end"] += int(aft.sum())
                orphan["before_series_start"] += int((tkd_["day"] < first_).sum())
                for t_, n_ in tkd_.loc[aft | last_.isna(), "ticker"].value_counts().head(50).items():
                    orphan_by_tk[t_] = orphan_by_tk.get(t_, 0) + int(n_)
            for (f, s), x in d.groupby(["file", "src"]):
                c = comp.setdefault((f, s), {"rows": 0, "with_ticker": 0, "date_only": 0, "has_article": 0, "cyrillic_title": 0,
                                             "ticker_in_prices": 0, "min_date": "9999", "max_date": "0000", "pre1990": 0,
                                             "epoch_1969_1970": 0, "bad_utf8": 0, "enc_title_any": 0, "enc_art_any": 0,
                                             "empty_title": 0, "amp_mangled_title": 0})
                c["rows"] += len(x)
                c["with_ticker"] += int(x["has_tk"].sum())
                c["date_only"] += int(x["date_only"].sum())
                c["has_article"] += int((x["art_len"] > 0).sum())
                c["cyrillic_title"] += int(x["cyr"].sum())
                c["ticker_in_prices"] += int((x["has_tk"] & x["in_prices"]).sum())
                c["min_date"] = min(c["min_date"], str(x["date_raw"].min()))
                c["max_date"] = max(c["max_date"], str(x["date_raw"].max()))
                c["pre1990"] += int((x["day"] < "1990-01-01").sum())
                c["epoch_1969_1970"] += int(x["date_raw"].str.startswith(("1969-12-31", "1970-01-01")).sum())
                c["bad_utf8"] += int(x["bad_utf8"].sum())
                c["enc_title_any"] += int((x["enc_t"] > 0).sum())
                c["enc_art_any"] += int((x["enc_a"] > 0).sum())
                c["empty_title"] += int((x["title"].str.len() == 0).sum())
                c["amp_mangled_title"] += int(x["amp_mangled"].sum())
            for bit, name in [(1, "fffd"), (2, "mojibake"), (4, "html_entity"), (8, "ctrl"), (32, "letter_spaced")]:
                for fld in ("enc_t", "enc_a"):
                    m = (d[fld] & bit) > 0
                    if m.any():
                        for s, n in d.loc[m, "src"].value_counts().items():
                            enc_rows.append((fid, s, fld, name, int(n)))
            if fid == 1:
                x = d[(d["src"] == "nasdaq") & d["has_tk"]]
                long_ = (x["art_len"] >= 1000).to_numpy()
                mism.append(pd.DataFrame({
                    "long": long_, "tklen": x["ticker"].str.len().clip(upper=4).to_numpy(),
                    "no_mention": ((x["tk_art"] == 0) & (x["tk_title"] == 0)).to_numpy(),
                    "no_strict": (x["tk_art_strict"] == 0).to_numpy(),
                    "title_other_ticker": title_names_other_ticker(x["title"], x["ticker"])}))
            tk_url = d["ticker"] + SEP + d["url"]
            K.append(pd.DataFrame({
                "src": d["src"].map(src_code.setdefault_all).to_numpy(np.int16), "has_tk": d["has_tk"].to_numpy(),
                "url_h": h64(d["url"]), "rec_h": h64(tk_url + SEP + d["date_raw"] + SEP + d["title"]) ^ d["art_hash"].to_numpy(np.int64),
                "tkurl_h": h64(tk_url), "tktd_h": h64(d["ticker"] + SEP + d["title"] + SEP + d["date_raw"].str.slice(0, 10)),
                "tk_h": h64(d["ticker"]), "art_h": d["art_hash"].to_numpy(np.int64), "art_len": d["art_len"].to_numpy(np.int32),
                "ts_s": (ts.astype("int64") // 10**9).to_numpy(np.int64)}))
            print(path, len(d), flush=True)
        K = pd.concat(K, ignore_index=True)
        inv_src = {v: k for k, v in src_code.items()}
        res["rows_by_file"][fid] = int(len(K))
        has_url = (K["url_h"] != empty_h).to_numpy()
        has_tk = K["has_tk"].to_numpy()
        F = {}
        F["dup_records"] = n_dups(K["rec_h"].to_numpy())
        F["dup_ticker_url"] = n_dups(K["tkurl_h"].to_numpy()[has_url])
        tktd = K["tktd_h"].to_numpy()
        F["dup_ticker_title_day_with_ticker"] = n_dups(tktd[has_tk])
        dm = dup_mask(np.where(has_tk, tktd, np.arange(len(tktd), dtype=np.int64) * -1 - 1))
        srcv = K["src"].to_numpy()
        F["dup_ticker_title_day_by_source"] = {
            inv_src[c]: {"dups": int((dm & (srcv == c)).sum()), "rows_with_ticker": int((has_tk & (srcv == c)).sum())}
            for c in np.unique(srcv[has_tk])}
        del dm, tktd
        al = K["art_len"].to_numpy() >= 200
        art = pd.DataFrame({"art_h": K["art_h"].to_numpy()[al], "url_h": K["url_h"].to_numpy()[al]}).drop_duplicates()
        vc = art["art_h"].value_counts()
        F["articles_reused_under_multiple_urls"] = {"article_texts": int((vc > 1).sum()), "urls": int(vc[vc > 1].sum())}
        del art, vc
        sel = has_url & has_tk
        pairs = pd.DataFrame({"url_h": K["url_h"].to_numpy()[sel], "tk_h": K["tk_h"].to_numpy()[sel]}).drop_duplicates()
        per_url = pairs["url_h"].value_counts()
        F["urls_with_ticker"] = {"urls": int(len(per_url)), "share_multi_ticker": float((per_url > 1).mean()),
                                 "max_tickers_per_url": int(per_url.max()) if len(per_url) else 0}
        del pairs, per_url
        res[f"file{fid}"] = F
        # keep compact key sets for the cross-file comparison
        uh = K["url_h"].to_numpy()[has_url]
        uu, first = np.unique(uh, return_index=True)
        sets[fid] = {"rec": np.unique(K["rec_h"].to_numpy()), "tkurl": np.unique(K["tkurl_h"].to_numpy()[has_url]),
                     "url": uu, "url_ts": K["ts_s"].to_numpy()[has_url][first], "url_src": K["src"].to_numpy()[has_url][first]}
        del K, uh, first
    # ---- across files
    a, b = sets[0], sets[1]
    res["records_of_file0_also_in_file1"] = int(np.isin(a["rec"], b["rec"], assume_unique=True).sum())
    res["distinct_records"] = {"file0": int(len(a["rec"])), "file1": int(len(b["rec"]))}
    res["ticker_url_pairs_in_both_files"] = int(np.isin(a["tkurl"], b["tkurl"], assume_unique=True).sum())
    cu, ia, ib = np.intersect1d(a["url"], b["url"], assume_unique=True, return_indices=True)
    dh = (b["url_ts"][ib] - a["url_ts"][ia]) / 3600
    inv_src = {v: k for k, v in src_code.items()}
    res["urls_in_both_files"] = {"urls": int(len(cu)), "urls_file0": int(len(a["url"])),
                                 "by_source": {inv_src[k]: int(v) for k, v in pd.Series(a["url_src"][ia]).value_counts().head(8).items()},
                                 "stamp_identical_share": float((dh == 0).mean()) if len(dh) else None}
    del sets
    # ---- composition totals and joinability (file 1 holds every row; see records_of_file0_also_in_file1)
    comp_df = pd.DataFrame([{"file": f, "source": s, **v} for (f, s), v in comp.items()]).sort_values(["file", "rows"], ascending=[True, False])
    comp_df.to_csv(common.RESULTS / "news_composition.csv", index=False)
    for fid in (0, 1):
        t = comp_df[comp_df["file"] == fid].sum(numeric_only=True)
        res[f"file{fid}"].update({k: int(t[k]) for k in ("rows", "with_ticker", "date_only", "ticker_in_prices", "has_article",
                                                         "cyrillic_title", "pre1990", "epoch_1969_1970", "amp_mangled_title")})
    res["unjoinable_news_file1"] = orphan
    res["unjoinable_top_tickers"] = dict(sorted(orphan_by_tk.items(), key=lambda kv: -kv[1])[:15])
    res["unique_tickers_in_news"] = len(tickers_seen)
    res["unique_news_tickers_with_price_file"] = len(tickers_seen & price_tickers)
    M = pd.concat(mism, ignore_index=True)
    L = M[M["long"]]
    res["nasdaq_rows_ticker_checks"] = {
        "rows": int(len(M)), "title_names_other_ticker_only": int(M["title_other_ticker"].sum()),
        "title_names_other_ticker_share": float(M["title_other_ticker"].mean()),
        "long_articles": int(len(L)), "long_no_symbol_mention_share": float(L["no_mention"].mean()),
        "long_no_symbol_mention_by_ticker_len": L.groupby("tklen")["no_mention"].agg(["size", "mean"]).round(4).to_dict("index")}
    enc = pd.DataFrame(enc_rows, columns=["file", "source", "field", "flag", "rows"]).groupby(["file", "source", "field", "flag"])["rows"].sum()
    res["encoding_flags"] = {f"{a_}|{b_}|{c_}|{d_}": int(v) for (a_, b_, c_, d_), v in enc.items()}
    (common.RESULTS / "news_audit.json").write_text(json.dumps(res, indent=1, default=str))
    print(comp_df.to_string())
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
