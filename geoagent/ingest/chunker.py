"""Pages → chunks: section-aware, token-limited prose chunks plus one chunk per table.

The parser hands over prose (Page.text) and structured tables (Page.tables) separately.
Prose is split at numbered headings and packed under a token limit with word overlap;
each table becomes its own chunk (caption + Markdown table + notes), split between rows
with the caption and header repeated when it is too long. Running headers and footers
are dropped before anything else.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import tiktoken

from geoagent.ingest.parse import Page, Table

# cl100k is a stand-in for the embedding model's tokenizer; the limit leaves a margin
# below the embedding input limit (2,048 tokens for gemini-embedding-001).
_ENCODING = tiktoken.get_encoding("cl100k_base")

# "7.1 Regional Geology", "14 MINERAL RESOURCE ESTIMATES": a 1-2 digit section number,
# then a capitalised title without digits (excludes "10 Mt at 2.3 g/t", years, and
# table-of-contents lines ending in page numbers).
_HEADING = re.compile(r"^\d{1,2}(?:\.\d{1,2})*\s+[A-Z][^\d]*$")
_HEADING_MAX_CHARS = 100

Kind = Literal["text", "table"]


@dataclass(frozen=True)
class Chunk:
    ordinal: int  # 0..n-1 in document order
    kind: Kind
    section: str | None  # heading text, e.g. "7.1 Regional Geology"
    page_start: int  # PDF pages (unambiguous)
    page_end: int
    label_start: str | None  # printed pages, for citations
    label_end: str | None
    text: str
    token_count: int


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


@lru_cache(maxsize=65536)
def _word_tokens(word: str) -> int:
    return count_tokens(" " + word)  # words are mostly encoded with their leading space


def is_heading(line: str) -> bool:
    """A numbered section heading, not a sentence or a quantity that starts with a number."""
    text = line.strip()
    return (
        len(text) <= _HEADING_MAX_CHARS
        and bool(_HEADING.match(text))
        and not text.endswith((".", ",", ";", ":"))
    )


# --- running headers and footers -----------------------------------------------------


def _normalise(line: str) -> str:
    return re.sub(r"\d", "#", line).strip()


def _boilerplate(pages: Sequence[Page]) -> set[str]:
    """Lines (digits masked) repeated on at least half the pages, and on at least 2.

    Counts distinct pages, not occurrences: "DH-001" and "DH-002" both mask to "DH-###"
    but sit on one page, which makes them content, not a running header.
    """
    if len(pages) < 2:
        return set()
    seen: Counter[str] = Counter()
    for page in pages:
        seen.update({_normalise(ln) for ln in page.text.splitlines() if ln.strip()})
    threshold = max(2, math.ceil(len(pages) / 2))
    return {line for line, count in seen.items() if count >= threshold}


# --- document stream -----------------------------------------------------------------


@dataclass(frozen=True)
class _Word:
    text: str
    starts_line: bool
    page: int
    label: str | None


@dataclass(frozen=True)
class _TableAt:
    table: Table
    page: int
    label: str | None


@dataclass(frozen=True)
class _HeadingAt:
    text: str


Item = str | tuple[int, str | None] | _TableAt | _HeadingAt


def _stream(pages: Sequence[Page], skip: set[str], numbered_fallback: bool) -> list[Item]:
    """Prose lines, headings and tables in document order.

    A ``(page, label)`` marker precedes each page's items. Headings come from the
    parser (typography); only when it found none in the whole document are numbered
    lines ("7.1 Regional Geology") used instead. Tables are inserted at their
    ``line_index`` (an index into the page's original lines, before boilerplate removal).
    """
    items: list[Item] = []
    for page in pages:
        items.append((page.number, page.label))
        lines = page.text.splitlines()
        tables = sorted(page.tables, key=lambda t: t.line_index)
        heading_at = {h.start: h for h in page.headings}
        in_heading = {i for h in page.headings for i in range(h.start, h.end)}
        for i in range(len(lines) + 1):
            items.extend(
                _TableAt(t, page.number, page.label)
                for t in tables
                if min(t.line_index, len(lines)) == i
            )
            if i == len(lines) or not lines[i].strip() or _normalise(lines[i]) in skip:
                continue
            if i in heading_at:
                items.append(_HeadingAt(heading_at[i].text))
            elif i in in_heading:
                continue  # second line of a heading ("11" + title)
            elif numbered_fallback and is_heading(lines[i]):
                items.append(_HeadingAt(lines[i].strip()))
            else:
                items.append(lines[i])
    return items


# --- packing ------------------------------------------------------------------------


@dataclass(frozen=True)
class _Draft:
    kind: Kind
    section: str | None
    words: tuple[_Word, ...]


def _join(words: Sequence[_Word]) -> str:
    parts: list[str] = []
    for i, w in enumerate(words):
        if i:
            parts.append("\n" if w.starts_line else " ")
        parts.append(w.text)
    return "".join(parts)


def _split_long_word(word: _Word, limit: int) -> list[_Word]:
    """A single "word" longer than the limit (a long URL, a hash) is cut by tokens."""
    ids = _ENCODING.encode(word.text)
    if len(ids) <= limit:
        return [word]
    return [
        _Word(_ENCODING.decode(ids[i : i + limit]), word.starts_line and i == 0, word.page, word.label)
        for i in range(0, len(ids), limit)
    ]


def _pack(words: Sequence[_Word], max_tokens: int, overlap_tokens: int) -> list[tuple[_Word, ...]]:
    """Greedy word packing under max_tokens; consecutive pieces share ~overlap_tokens."""
    words = [piece for w in words for piece in _split_long_word(w, max_tokens // 2)]
    pieces: list[tuple[_Word, ...]] = []
    start = 0
    while start < len(words):
        end, total = start, 0
        while end < len(words):
            cost = _word_tokens(words[end].text)
            if end > start and total + cost > max_tokens:
                break
            total += cost
            end += 1
        while end - start > 1 and count_tokens(_join(words[start:end])) > max_tokens:
            end -= 1  # the per-word estimate can be slightly off: trim to the exact limit
        pieces.append(tuple(words[start:end]))
        if end >= len(words):
            break
        back, k = 0, end
        while k - 1 > start and back + _word_tokens(words[k - 1].text) <= overlap_tokens:
            k -= 1
            back += _word_tokens(words[k].text)
        start = k
    return pieces


def _md_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"


def _table_texts(table: Table, max_tokens: int) -> list[str]:
    """Caption + Markdown table + notes, split between rows when too long.

    Every piece repeats the caption and header row so it stands alone; notes go on the
    last piece, or on a piece of their own (still with caption and header) if they do
    not fit.
    """
    prefix = [table.caption] if table.caption else []
    prefix += [_md_row(table.header), _md_row(["---"] * len(table.header))]
    rows = [_md_row(r) for r in table.rows]
    notes = [table.notes] if table.notes else []

    def fits(lines: list[str]) -> bool:
        return count_tokens("\n".join(lines)) <= max_tokens

    if fits(prefix + rows + notes):
        return ["\n".join(prefix + rows + notes)]
    groups: list[list[str]] = [[]]
    for row in rows:
        if groups[-1] and not fits(prefix + groups[-1] + [row]):
            groups.append([])
        groups[-1].append(row)
    texts = ["\n".join(prefix + g) for g in groups]
    if notes:
        if fits(prefix + groups[-1] + notes):
            texts[-1] = "\n".join(prefix + groups[-1] + notes)
        else:
            texts.append("\n".join(prefix + notes))
    return texts


def _words_of(text: str, page: int, label: str | None) -> list[_Word]:
    return [
        _Word(word, j == 0, page, label)
        for line in text.splitlines()
        for j, word in enumerate(line.split())
    ]


def chunk_pages(
    pages: Sequence[Page],
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    strip_headers: bool = True,
) -> list[Chunk]:
    """Split parsed pages into retrieval chunks (see module docstring for the rules)."""
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")
    skip = _boilerplate(pages) if strip_headers else set()

    drafts: list[_Draft] = []
    section: str | None = None
    buffer: list[_Word] = []
    page, label = 0, None

    def flush() -> None:
        drafts.extend(_Draft("text", section, p) for p in _pack(buffer, max_tokens, overlap_tokens))
        buffer.clear()

    numbered_fallback = not any(p.headings for p in pages)
    for item in _stream(pages, skip, numbered_fallback):
        if isinstance(item, tuple):
            page, label = item
        elif isinstance(item, _TableAt):
            flush()
            for text in _table_texts(item.table, max_tokens):
                words = _words_of(text, item.page, item.label)
                for piece in _pack(words, max_tokens, 0):  # no-op unless a row is huge
                    drafts.append(_Draft("table", section, piece))
        elif isinstance(item, _HeadingAt):
            flush()
            section = item.text
        else:
            buffer.extend(_words_of(item, page, label))
    flush()

    chunks = []
    for ordinal, draft in enumerate(d for d in drafts if d.words):
        text = _join(draft.words)
        chunks.append(
            Chunk(
                ordinal=ordinal,
                kind=draft.kind,
                section=draft.section,
                page_start=min(w.page for w in draft.words),
                page_end=max(w.page for w in draft.words),
                label_start=draft.words[0].label,
                label_end=draft.words[-1].label,
                text=text,
                token_count=count_tokens(text),
            )
        )
    return chunks
