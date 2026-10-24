"""
web.py: component 1 of a search engine, the crawler and document processor.

The "web" is simulated: data/web.json holds 37 pages on 6 fictional sites. `fetch()` plays the role of an
HTTP server. It wraps each HTML page in realistic boilerplate (navigation, footer, a tracking script) so that the
parser has something to strip, returns PDF and PPT files as already-extracted text, and answers 404 for a broken
link. Everything after `fetch()` is what a real crawler does:

  crawl()            breadth-first traversal from seed URLs with a frontier and a visited set
  parse_html()       HTML parsing: title, main text and out-links, boilerplate removed
  detect_language()  stopword-count language identification (English vs Indonesian)
  find_duplicates()  exact duplicates (same fingerprint) and near duplicates (Jaccard over word 3-shingles);
                     the copy with the most in-links is kept as the canonical page

This module does not depend on Streamlit. Try it from the terminal:
    python web.py
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import deque
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent / "data"

EN_WORDS = set("""the a an and or of to in is are was were be by for with on as that this it from at which can
their its they not but has have than so into each also when only every""".split())
ID_WORDS = set("""yang dan di ke dari untuk dengan pada adalah ini itu dalam tidak atau juga akan oleh agar lalu
karena seperti bagian ketika sama setiap""".split())


def load_web() -> dict:
    return json.loads((DATA_DIR / "web.json").read_text(encoding="utf-8"))


WEB = load_web()
PAGES = {p["url"]: p for p in WEB["pages"]}


# ---------------------------------------------------------------------------
# The simulated server
# ---------------------------------------------------------------------------
@dataclass
class Response:
    url: str
    status: int
    content_type: str = ""
    body: str = ""


def render_html(page: dict) -> str:
    host = urlparse(page["url"]).netloc
    links = "\n".join(f'      <li><a href="{html.escape(u)}">{html.escape(PAGES[u]["title"]) if u in PAGES else "Old syllabus (PDF)"}</a></li>'
                      for u in page["links"])
    return f"""<!DOCTYPE html>
<html lang="{'id' if host.endswith('.id') else 'en'}">
<head>
  <meta charset="utf-8">
  <title>{html.escape(page['title'])}</title>
  <script>window.dataLayer = []; function trackPageview() {{ dataLayer.push('view'); }}</script>
</head>
<body>
  <nav class="site-menu"><a href="/">Home</a> | <a href="/about">About</a> | <a href="/contact">Contact</a></nav>
  <main>
    <h1>{html.escape(page['title'])}</h1>
    <p>{html.escape(page['body'])}</p>
  </main>
  <aside class="related">
    <h2>Related pages</h2>
    <ul>
{links}
    </ul>
  </aside>
  <footer>© 2026 {host} · Privacy · Cookies</footer>
  <script>trackPageview();</script>
</body>
</html>"""


def fetch(url: str) -> Response:
    page = PAGES.get(url)
    if page is None:
        return Response(url, 404)
    if page["type"] == "html":
        return Response(url, 200, "text/html", render_html(page))
    mime = {"pdf": "application/pdf", "ppt": "application/vnd.ms-powerpoint"}[page["type"]]
    return Response(url, 200, mime, page["body"])        # a real crawler would run a PDF/PPT text extractor here


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------
class _Extractor(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "aside", "head"}

    def __init__(self):
        super().__init__()
        self.title, self.text, self.links = "", [], []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            if href.startswith("http"):
                self.links.append(href)
        if tag == "title":
            self._in_title = True
        elif tag in self.SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag in self.SKIP:
            self._skip -= 1

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip and data.strip():
            self.text.append(data.strip())


def parse_html(raw: str) -> tuple[str, str, list[str]]:
    """Return (title, main text without boilerplate, absolute out-links)."""
    p = _Extractor()
    p.feed(raw)
    text = " ".join(p.text)
    title = p.title.strip()
    if text.startswith(title):                 # the <h1> repeats the title; keep the body only
        text = text[len(title):].strip()
    return title, text, p.links


def detect_language(text: str) -> tuple[str, int, int]:
    words = re.findall(r"[a-z]+", text.lower())
    en = sum(w in EN_WORDS for w in words)
    idn = sum(w in ID_WORDS for w in words)
    return ("id" if idn > en else "en"), en, idn


@dataclass
class Document:
    url: str
    title: str
    text: str
    type: str
    lang: str
    links: list[str] = field(default_factory=list)
    lang_counts: tuple[int, int] = (0, 0)

    @property
    def host(self) -> str:
        return urlparse(self.url).netloc

    @property
    def short(self) -> str:
        u = urlparse(self.url)
        return u.netloc + (u.path if u.path != "/" else "")


def process(resp: Response) -> Document:
    page = PAGES[resp.url]
    if resp.content_type == "text/html":
        title, text, links = parse_html(resp.body)
    else:
        title, text, links = page["title"], resp.body, []
    lang, en, idn = detect_language(title + " " + text)
    return Document(resp.url, title, text, page["type"], lang, links, (en, idn))


# ---------------------------------------------------------------------------
# Crawling
# ---------------------------------------------------------------------------
@dataclass
class CrawlStep:
    order: int
    url: str
    status: int
    content_type: str
    links_found: int
    new_links: int
    frontier_after: int


def crawl(seeds: list[str], max_pages: int = 100) -> tuple[list[CrawlStep], list[Document], list[str]]:
    """Breadth-first crawl. Returns (log, fetched documents, URLs still waiting in the frontier)."""
    frontier = deque(seeds)
    seen = set(seeds)
    log, docs = [], []
    while frontier and len(log) < max_pages:
        url = frontier.popleft()
        resp = fetch(url)
        new = 0
        found = 0
        if resp.status == 200:
            doc = process(resp)
            docs.append(doc)
            found = len(doc.links)
            for link in doc.links:
                if link not in seen:
                    seen.add(link)
                    frontier.append(link)
                    new += 1
        log.append(CrawlStep(len(log) + 1, url, resp.status, resp.content_type, found, new, len(frontier)))
    return log, docs, list(frontier)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------
def shingles(text: str, k: int = 3) -> set[tuple[str, ...]]:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def fingerprint(text: str) -> str:
    return hashlib.md5(" ".join(re.findall(r"[a-z0-9]+", text.lower())).encode()).hexdigest()[:12]


@dataclass
class Duplicate:
    url: str
    kept: str
    kind: str          # "exact" or "near"
    similarity: float


def inlink_counts(docs: list[Document]) -> dict[str, int]:
    urls = {d.url for d in docs}
    count = dict.fromkeys(urls, 0)
    for d in docs:
        for link in set(d.links):
            if link in urls and link != d.url:
                count[link] += 1
    return count


def find_duplicates(docs: list[Document], threshold: float = 0.8) -> tuple[list[Document], list[Duplicate]]:
    """Keep one copy of every group of exact or near-duplicate documents.

    The copy that is kept (the canonical page) is the one with the most in-links from other crawled pages,
    a simple stand-in for link authority; ties go to the page crawled first.
    """
    inlinks = inlink_counts(docs)
    order = sorted(range(len(docs)), key=lambda i: (-inlinks[docs[i].url], i))
    kept_idx: list[int] = []
    sh: dict[str, set] = {}
    fp: dict[str, str] = {}
    dups: list[Duplicate] = []
    for i in order:
        d = docs[i]
        f = fingerprint(d.title + " " + d.text)
        s = shingles(d.text)
        if f in fp:
            dups.append(Duplicate(d.url, fp[f], "exact", 1.0))
            continue
        best = max(((jaccard(s, sh[docs[k].url]), docs[k].url) for k in kept_idx), default=(0.0, ""))
        if best[0] >= threshold:
            dups.append(Duplicate(d.url, best[1], "near", best[0]))
            continue
        kept_idx.append(i)
        sh[d.url] = s
        fp[f] = d.url
    return [docs[i] for i in sorted(kept_idx)], dups


if __name__ == "__main__":
    log, docs, left = crawl(WEB["seeds"])
    print(f"crawled {len(log)} URLs, fetched {len(docs)} documents, frontier left: {len(left)}")
    for s in log[:8]:
        print(f"  {s.order:>2}  {s.status}  +{s.new_links:<2} {s.url}")
    kept, dups = find_duplicates(docs)
    print("duplicates:", [(d.url, d.kind, round(d.similarity, 3)) for d in dups])
    print("never discovered:", sorted(set(PAGES) - {d.url for d in docs}))
    print("languages:", {l: sum(d.lang == l for d in kept) for l in ("en", "id")})
