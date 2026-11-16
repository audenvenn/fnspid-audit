#!/usr/bin/env python3
"""Stream one FNSPID news CSV from Hugging Face (pinned revision) and write compact per-row features.

The 5.7 GB / 23 GB CSVs are never stored: bytes are fetched with resumable HTTP range requests,
split into lines, parsed with the csv module, and reduced to one compact row per record:

  file, row, off            record id: file id, 0-based data-record index, byte offset of record start
  idx                       the nasdaq file's leading "Unnamed: 0" column (raw string)
  date_raw                  the Date field exactly as published
  title, ticker, url, publisher, author (author truncated)
  n_fields                  number of CSV fields in the record
  art_len, art_hash         length (chars) and 64-bit blake2b hash of the Article field
  sum_len                   length of Lsa_summary
  lm_t_pos/neg/n            Loughran-McDonald positive/negative/total word counts in the title
  lm_a_pos/neg/n            ... in the article body
  tk_title, tk_art, tk_art_strict   ticker mentions (word-boundary / "(TICK)", "NASDAQ:TICK", "$TICK", "Symbol: TICK")
  txt_time, txt_time_kind, txt_time_ctx
                            first explicit clock time with a US-Eastern zone label in the body ("10:32 a.m. ET");
                            kind 2 = introduced by "as of" / sentence-initial "At" (time of writing), 1 = any other
  art_head                  first 100 characters of the body (provider identification)
  enc_t, enc_a              encoding-problem bitmasks for title / article (see ENC_* below)
  bad_utf8                  1 if any physical line of the record failed strict UTF-8 decoding

Usage:  python -I scripts/20_news_stream.py --file 0 [--limit-bytes N] [--local PATH]
Output: data/derived/news_rows/f{file}/part-XXXXX.parquet  and  progress.json (resumable)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import requests  # noqa: E402

csv.field_size_limit(sys.maxsize)

ENC_FFFD = 1        # U+FFFD replacement character present
ENC_MOJIBAKE = 2    # typical UTF-8-read-as-cp1252 sequences (â€™, Ã©, Â ...)
ENC_ENTITY = 4      # undecoded HTML entities (&amp; &#39; &quot; ...)
ENC_CTRL = 8        # C0 control characters other than \t \n \r
ENC_NUL = 16        # NUL bytes
ENC_SPACED = 32     # letter-spaced text ("B y   A d a m")

MOJIBAKE_RE = re.compile(r"(?:â€[™œ\x9d˜“”¦¢]|Ã[\x80-\xbf©¨¢¡³±¶¼¤§]|Â[\xa0-\xbf ]|ï¿½)")
ENTITY_RE = re.compile(r"&(?:amp|lt|gt|quot|apos|nbsp|#\d{2,5}|#x[0-9a-fA-F]{2,4});")
CTRL_RE = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACED_RE = re.compile(r"(?:\b[A-Za-z] ){5,}")
WORD_RE = re.compile(r"[A-Za-z]+")
_T = r"(1[0-2]|0?[1-9]):([0-5]\d)\s*([AaPp])\.?\s?[Mm]\.?,?\s*(?:\(?\s*)(ET|EDT|EST|Eastern(?: Time)?)\b"
TIME_RE = re.compile(r"\b" + _T)
# "as of 11:34 a.m. ET" / sentence-initial "At 12:16 p.m. ET": phrases that usually describe the time of writing
TIME_ASOF_RE = re.compile(r"(?:\b[Aa]s of|(?:^|[.!?]\s+)At)\s+" + _T)

POS, NEG = common.load_lm_sets()


def enc_flags(s: str) -> int:
    if not s:
        return 0
    f = 0
    if "�" in s:
        f |= ENC_FFFD
    if MOJIBAKE_RE.search(s):
        f |= ENC_MOJIBAKE
    if "&" in s and ENTITY_RE.search(s):
        f |= ENC_ENTITY
    if CTRL_RE.search(s):
        f |= ENC_CTRL
    if SPACED_RE.search(s):
        f |= ENC_SPACED
    return f


def lm_counts(s: str):
    if not s:
        return 0, 0, 0
    words = WORD_RE.findall(s)
    p = n = 0
    for w in words:
        u = w.upper()
        if u in POS:
            p += 1
        elif u in NEG:
            n += 1
    return p, n, len(words)


def h64(s: str) -> int:
    if not s:
        return 0
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little", signed=True)


class LineSource:
    """Yield decoded text lines (with trailing '\n') from a local file or a resumable HTTP range stream.

    `next_offset` is the absolute byte offset of the next line to be yielded; after csv.reader returns a
    record, it is the offset where the next record starts (csv.reader pulls lines lazily).
    """

    def __init__(self, url=None, local=None, start=0, limit_bytes=None, block=1 << 22):
        self.url, self.local, self.block = url, local, block
        self.next_offset = start
        self.end = None if limit_bytes is None else start + limit_bytes
        self.bad_utf8_lines = 0
        self.nul_lines = 0
        self.last_bad = False
        self.reconnects = 0
        self.bytes_read = 0

    def _blocks(self, pos):
        if self.local:
            with open(self.local, "rb") as f:
                f.seek(pos)
                while True:
                    b = f.read(self.block)
                    if not b:
                        return
                    yield b
        else:
            while True:
                try:
                    hdr = {"Range": f"bytes={pos}-", "User-Agent": "fnspid-audit/0.1 (research data audit)"}
                    with requests.get(self.url, headers=hdr, stream=True, timeout=(30, 60)) as r:
                        if r.status_code not in (200, 206):
                            raise IOError(f"HTTP {r.status_code}")
                        if r.status_code == 200 and pos != 0:
                            raise IOError("server ignored Range header")
                        for b in r.iter_content(chunk_size=1 << 20):
                            if b:
                                pos += len(b)
                                if pos // (1 << 28) != (pos - len(b)) // (1 << 28):
                                    print(f"[stream] {pos:,} bytes", file=sys.stderr, flush=True)
                                yield b
                        return
                except (requests.RequestException, IOError) as e:  # resume from the exact byte offset
                    self.reconnects += 1
                    if self.reconnects > 200:
                        raise
                    print(f"[stream] reconnect #{self.reconnects} at {pos}: {e!r}", file=sys.stderr, flush=True)
                    time.sleep(min(60, 2 * self.reconnects))

    def __iter__(self):
        pos = self.next_offset
        buf = b""
        for b in self._blocks(pos):
            self.bytes_read += len(b)
            buf += b
            start = 0
            while True:
                j = buf.find(b"\n", start)
                if j < 0:
                    break
                line = buf[start:j + 1]
                start = j + 1
                yield self._decode(line)
                if self.end is not None and self.next_offset >= self.end:
                    return
            buf = buf[start:]
        if buf:
            yield self._decode(buf)

    def _decode(self, line: bytes) -> str:
        self.next_offset += len(line)
        try:
            s = line.decode("utf-8")
            self.last_bad = False
        except UnicodeDecodeError:
            s = line.decode("utf-8", "replace")
            self.bad_utf8_lines += 1
            self.last_bad = True
        if "\x00" in s:
            self.nul_lines += 1
            s = s.replace("\x00", "")
        return s


SCHEMA = pa.schema([
    ("file", pa.int8()), ("row", pa.int64()), ("off", pa.int64()), ("idx", pa.string()),
    ("date_raw", pa.string()), ("title", pa.string()), ("ticker", pa.string()), ("url", pa.string()),
    ("publisher", pa.string()), ("author", pa.string()), ("n_fields", pa.int16()),
    ("art_len", pa.int32()), ("art_hash", pa.int64()), ("sum_len", pa.int32()),
    ("lm_t_pos", pa.int16()), ("lm_t_neg", pa.int16()), ("lm_t_n", pa.int16()),
    ("lm_a_pos", pa.int32()), ("lm_a_neg", pa.int32()), ("lm_a_n", pa.int32()),
    ("tk_title", pa.int8()), ("tk_art", pa.int32()), ("tk_art_strict", pa.int32()),
    ("txt_time", pa.string()), ("txt_time_kind", pa.int8()), ("txt_time_ctx", pa.string()),
    ("art_head", pa.string()),
    ("enc_t", pa.int16()), ("enc_a", pa.int16()), ("bad_utf8", pa.int8()),
])


def ticker_counts(tk: str, title: str, art: str):
    if not tk or not tk.isascii() or len(tk) > 8:
        return 0, 0, 0
    e = re.escape(tk)
    wb = re.compile(r"(?<![A-Za-z0-9])" + e + r"(?![A-Za-z0-9])")
    strict = re.compile(r"(?:\(\s*(?:(?:NYSE|NASDAQ|Nasdaq|NYSEAMERICAN|AMEX|NYSEARCA|OTC)\s*:\s*)?" + e +
                        r"\s*\)|(?:NYSE|NASDAQ|Nasdaq|AMEX|OTC)\s*:\s*" + e + r"\b|\$" + e +
                        r"\b|Symbol:\s*" + e + r"\b)")
    t = 1 if (title and wb.search(title)) else 0
    if not art:
        return t, 0, 0
    return t, len(wb.findall(art)), len(strict.findall(art))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=int, required=True, choices=[0, 1])
    ap.add_argument("--local", default=None, help="read a local copy/sample instead of HTTP")
    ap.add_argument("--limit-bytes", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=200_000)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()

    out = Path(a.out) if a.out else common.DERIVED / "news_rows" / f"f{a.file}"
    out.mkdir(parents=True, exist_ok=True)
    prog_path = out / "progress.json"
    prog = {"next_offset": 0, "next_row": 0, "parts": 0, "done": False, "header": None,
            "bad_utf8_lines": 0, "nul_lines": 0, "reconnects": 0, "bytes": 0, "seconds": 0.0}
    if prog_path.exists() and not a.fresh:
        prog = json.loads(prog_path.read_text())
        if prog.get("done"):
            print("already done", prog_path)
            return
    url = None if a.local else common.hf_url(common.NEWS_FILES[a.file])
    src = LineSource(url=url, local=a.local, start=prog["next_offset"], limit_bytes=a.limit_bytes)
    reader = csv.reader(iter(src))
    header = prog["header"]
    if header is None:
        header = next(reader)
        prog["header"] = header
    col = {c: i for i, c in enumerate(header)}
    need = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Author", "Article", "Lsa_summary"]
    ci = [col[c] for c in need]
    ci_idx = col.get("Unnamed: 0")
    nh = len(header)

    rows = {k: [] for k in SCHEMA.names}
    row = prog["next_row"]
    t0 = time.time()
    part = prog["parts"]

    def flush(final=False):
        nonlocal rows, part
        if rows["row"]:
            tbl = pa.Table.from_pydict(rows, schema=SCHEMA)
            pq.write_table(tbl, out / f"part-{part:05d}.parquet", compression="zstd", compression_level=6)
            part += 1
            rows = {k: [] for k in SCHEMA.names}
        prog.update(next_offset=src.next_offset, next_row=row, parts=part, done=final,
                    bad_utf8_lines=prog["bad_utf8_lines"] + src.bad_utf8_lines,
                    nul_lines=prog["nul_lines"] + src.nul_lines,
                    reconnects=prog["reconnects"] + src.reconnects,
                    bytes=prog["bytes"] + src.bytes_read, seconds=prog["seconds"] + (time.time() - t0))
        src.bad_utf8_lines = src.nul_lines = src.reconnects = src.bytes_read = 0
        prog_path.write_text(json.dumps(prog, indent=1))

    bad_in_rec = 0
    while True:
        off = src.next_offset
        bad_before = src.bad_utf8_lines
        try:
            rec = next(reader)
        except StopIteration:
            break
        bad_in_rec = 1 if src.bad_utf8_lines > bad_before else 0
        nf = len(rec)
        if nf < nh:
            rec = rec + [""] * (nh - nf)
        date, title, tk, urlv, pub, auth, art, lsa = (rec[i] for i in ci)
        tp, tn, tt = lm_counts(title)
        apos, aneg, an = lm_counts(art)
        tkt, tka, tks = ticker_counts(tk, title, art)
        kind = 0
        m = TIME_ASOF_RE.search(art) if art else None
        if m:
            kind = 2
        else:
            m = TIME_RE.search(art) if art else None
            kind = 1 if m else 0
        if m:
            hh = int(m.group(1)) % 12 + (12 if m.group(3) in "Pp" else 0)
            ttime = f"{hh:02d}:{m.group(2)} {m.group(4)}"
            ctx = art[max(0, m.start() - 80): m.end() + 20].replace("\n", " ")
        else:
            ttime, ctx = None, None
        rows["file"].append(a.file); rows["row"].append(row); rows["off"].append(off)
        rows["idx"].append(rec[ci_idx] if ci_idx is not None else None)
        rows["date_raw"].append(date); rows["title"].append(title); rows["ticker"].append(tk)
        rows["url"].append(urlv[:400]); rows["publisher"].append(pub[:80]); rows["author"].append(auth[:80])
        rows["n_fields"].append(nf); rows["art_len"].append(len(art)); rows["art_hash"].append(h64(art.strip()))
        rows["sum_len"].append(len(lsa))
        rows["lm_t_pos"].append(min(tp, 32767)); rows["lm_t_neg"].append(min(tn, 32767)); rows["lm_t_n"].append(min(tt, 32767))
        rows["lm_a_pos"].append(apos); rows["lm_a_neg"].append(aneg); rows["lm_a_n"].append(an)
        rows["tk_title"].append(tkt); rows["tk_art"].append(tka); rows["tk_art_strict"].append(tks)
        rows["txt_time"].append(ttime); rows["txt_time_kind"].append(kind); rows["txt_time_ctx"].append(ctx)
        rows["art_head"].append(art[:100].replace("\n", " ") if art else None)
        rows["enc_t"].append(enc_flags(title)); rows["enc_a"].append(enc_flags(art[:20000]))
        rows["bad_utf8"].append(bad_in_rec)
        row += 1
        if len(rows["row"]) >= a.batch:
            flush()
            el = prog["seconds"]
            print(f"[f{a.file}] rows={row:,} off={src.next_offset:,} ({src.next_offset / common.NEWS_SIZES[a.file]:.1%}) "
                  f"{prog['bytes'] / max(el, 1e-9) / 1e6:.1f} MB/s", flush=True)
            t0 = time.time()
    flush(final=a.limit_bytes is None)
    print("done", json.dumps({k: v for k, v in prog.items() if k != "header"}))


if __name__ == "__main__":
    main()
