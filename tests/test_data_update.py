from hoax_detector.data_update import COLUMNS, merge, normalize, parse_feed
import pandas as pd

FEED = b"""<?xml version="1.0"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel>
<item><title>[SALAH] Prabowo Resmi Bekukan DPR RI</title>
<description>&lt;p&gt;Beredar pesan berantai yang mengklaim DPR dibekukan.&lt;/p&gt;</description></item>
<item><title>Jakarta (ANTARA) - Harga emas naik</title>
<content:encoded>&lt;img src="x"/&gt; Harga emas Antam naik tipis pada perdagangan hari ini.</content:encoded></item>
<item><title>pendek</title><description>x</description></item>
</channel></rss>"""


def test_normalize_strips_label_leaks():
    out = normalize("[SALAH] Judul &amp; isi <b>tebal</b> (ANTARA) berita")
    assert "SALAH" not in out and "ANTARA" not in out and "<" not in out and "&amp;" not in out


def test_parse_feed_skips_short_items():
    texts = parse_feed(FEED)
    assert len(texts) == 2
    assert texts[0].startswith("Prabowo Resmi Bekukan DPR RI")
    assert texts[1].startswith("Harga emas")


def test_merge_deduplicates():
    corpus = pd.DataFrame(columns=COLUMNS)
    texts = parse_feed(FEED)
    corpus, added = merge(corpus, texts, 1, "t")
    assert added == 2
    corpus, added = merge(corpus, texts, 1, "t")
    assert added == 0 and len(corpus) == 2


def test_normalize_strips_republika_and_truncation():
    out = normalize("Judul. REPUBLIKA.CO.ID, TANGERANG SELATAN \u2013 Potensi laut besar bagi rakyat ...")
    assert "REPUBLIKA" not in out and "TANGERANG" not in out and not out.endswith(".")
    assert "Potensi laut besar bagi rakyat" in out
