"""Grow data/corpus.csv from RSS feeds and user-supplied CSV files.

Usage:
    python -m hoax_detector.data_update fetch
    python -m hoax_detector.data_update import some.csv --text-col content --label-col label

Feeds:
    turnbackhoax.id                                   -> label 1 (hoax debunk articles)
    Antara (7 topics), Tempo, CNN Indonesia, Republika, detikNews -> label 0 (regular news)

Sources leak their own identity through boilerplate ("[SALAH]", "(ANTARA)",
"REPUBLIKA.CO.ID, JAKARTA -", trailing "..."), so those markers are stripped
before anything is stored.
"""
import argparse
import hashlib
import html
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ANTARA_TOPICS = ["terkini", "ekonomi", "politik", "hukum", "humaniora", "tekno", "olahraga"]
FEEDS = [
    {"name": "turnbackhoax", "url": "https://turnbackhoax.id/feed", "label": 1},
    *[
        {"name": "antara", "url": f"https://www.antaranews.com/rss/{t}.xml", "label": 0}
        for t in _ANTARA_TOPICS
    ],
    {"name": "tempo", "url": "https://rss.tempo.co/nasional", "label": 0},
    {"name": "cnnindonesia", "url": "https://www.cnnindonesia.com/nasional/rss", "label": 0},
    {"name": "republika", "url": "https://www.republika.co.id/rss", "label": 0},
    {"name": "detik", "url": "https://news.detik.com/berita/rss", "label": 0},
]
MAX_PER_FEED = 50  # keep any single outlet from dominating the valid class
COLUMNS = ["id", "text", "label", "source", "fetched_at"]
MAX_CHARS = 600  # store short excerpts only, not full articles
USER_AGENT = "hoax-detector-research/0.1 (student project)"

_HTML = re.compile(r"<[^>]+>")
_VERDICT_TAG = re.compile(r"\[[^\]]{2,40}\]")
_ANTARA = re.compile(r"^[^()]{0,40}\(ANTARA\)\s*-?\s*|\(?ANTARA\)?", re.I)
_REPUBLIKA = re.compile(r"REPUBLIKA\.CO\.ID,\s*[A-Za-z .]{2,30}?\s*\S\s+", re.I)
_TRUNCATED = re.compile(r"\s*(\.{3}|…)\s*$")
_SPACES = re.compile(r"\s+")
_NS = {"content": "http://purl.org/rss/1.0/modules/content/"}


def normalize(raw: str) -> str:
    text = html.unescape(_HTML.sub(" ", raw))
    text = _VERDICT_TAG.sub(" ", text)
    text = _ANTARA.sub(" ", text)
    text = _REPUBLIKA.sub(" ", text)
    text = _TRUNCATED.sub("", _SPACES.sub(" ", text).strip()[:MAX_CHARS])
    return text.strip()


def parse_feed(xml_bytes: bytes) -> list[str]:
    root = ET.fromstring(xml_bytes)
    texts = []
    for item in root.iter("item"):
        title = item.findtext("title") or ""
        body = item.findtext("content:encoded", namespaces=_NS) or item.findtext("description") or ""
        text = normalize(f"{title}. {body}")
        if len(text) >= 30:
            texts.append(text)
    return texts[:MAX_PER_FEED]


def _row_id(text: str) -> str:
    return hashlib.sha1(text.lower().encode()).hexdigest()[:16]


def load_corpus(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)


def merge(corpus: pd.DataFrame, texts: list[str], label: int, source: str) -> tuple[pd.DataFrame, int]:
    seen = set(corpus["id"])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for t in texts:
        rid = _row_id(t)
        if rid not in seen:
            seen.add(rid)
            rows.append({"id": rid, "text": t, "label": label, "source": source, "fetched_at": now})
    if not rows:
        return corpus, 0
    return pd.concat([corpus, pd.DataFrame(rows)], ignore_index=True), len(rows)


def fetch_feed(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def cmd_fetch(corpus_path: Path) -> None:
    corpus = load_corpus(corpus_path)
    for feed in FEEDS:
        try:
            texts = parse_feed(fetch_feed(feed["url"]))
        except Exception as exc:  # one broken feed must not stop the others
            print(f"[skip] {feed['name']}: {exc}")
            continue
        corpus, added = merge(corpus, texts, feed["label"], feed["name"])
        print(f"[ok] {feed['name']}: {len(texts)} items, {added} new")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(corpus_path, index=False)
    print(f"Corpus: {len(corpus)} rows ({int(corpus['label'].sum())} hoax) -> {corpus_path}")


def cmd_import(corpus_path: Path, csv_path: Path, text_col: str, label_col: str, source: str) -> None:
    df = pd.read_csv(csv_path).dropna(subset=[text_col, label_col])
    corpus = load_corpus(corpus_path)
    for label, group in df.groupby(df[label_col].astype(int)):
        texts = [normalize(str(t)) for t in group[text_col]]
        corpus, added = merge(corpus, [t for t in texts if len(t) >= 30], int(label), source)
        print(f"[ok] label={label}: {added} new rows")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(corpus_path, index=False)
    print(f"Corpus: {len(corpus)} rows -> {corpus_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=Path("data/corpus.csv"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="pull new items from the configured RSS feeds")
    imp = sub.add_parser("import", help="merge a labelled CSV (label: 1 = hoax, 0 = valid)")
    imp.add_argument("csv", type=Path)
    imp.add_argument("--text-col", default="text")
    imp.add_argument("--label-col", default="label")
    imp.add_argument("--source", default="import")
    args = p.parse_args()
    if args.cmd == "fetch":
        cmd_fetch(args.corpus)
    else:
        cmd_import(args.corpus, args.csv, args.text_col, args.label_col, args.source)


if __name__ == "__main__":
    main()
