"""PDF → pages: prose text plus structured tables (caption, header, rows, notes)."""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

import pymupdf

log = logging.getLogger(__name__)

_NUMERIC = re.compile(r"^[-+]?\d[\d,]*(\.\d+)?%?$")
_MIN_OVERLAP = 0.5  # a header label owns a column when it covers > half its width


@dataclass(frozen=True)
class GridCell:
    """One cell of PyMuPDF's table grid: its text and horizontal extent on the page."""

    text: str
    x0: float
    x1: float


Grid = Sequence[Sequence[GridCell | None]]  # None = covered by a merged neighbour


@dataclass(frozen=True)
class Table:
    caption: str | None
    header: tuple[str, ...]  # one name per column, e.g. "Indicated · Au (g/t)"
    rows: tuple[tuple[str, ...], ...]  # data rows, same width as header
    notes: str | None
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Page:
    number: int  # PDF page, 1-based: always unambiguous
    label: str | None  # printed page number, when detectable
    text: str  # prose only: tables, captions and notes removed
    tables: tuple[Table, ...]


def _clean(text: str | None) -> str:
    """Collapse whitespace; wrapped cell text ("Coronation\nNorth") becomes one line."""
    return " ".join(text.split()) if text else ""


def _text(cell: GridCell | None) -> str:
    return cell.text if cell else ""


def _dedupe(parts: list[str]) -> list[str]:
    out: list[str] = []
    for part in parts:
        if part and (not out or out[-1] != part):
            out.append(part)
    return out


def _header_row_count(rows: Grid) -> int:
    """Row 0, plus following rows whose first cell is blank and that hold no numbers.

    Multi-level headers leave the first column empty under the top-left label
    ("Gold", "Area"); the first data row has a name there ("Macraes", "MOP").
    """
    count = 1
    for row in rows[1:]:
        if _text(row[0]) or any(_NUMERIC.match(_text(c)) for c in row):
            break
        count += 1
    return count


def _real_columns(head: Grid, body: Grid) -> list[int]:
    """Grid columns that hold a cell in at least half the full data rows.

    Double borders and merged cells add grid columns that are never a cell of their
    own in data rows (slivers); those are layout artefacts, not columns. Only "full"
    rows (2+ cells) are counted: rows holding a single cell are a cell or label wrapped
    over several grid rows, and would dilute the count.
    """
    width = len(head[0])
    full = [row for row in body if sum(c is not None for c in row) >= 2]
    rows = full or list(body) or [head[-1]]
    keep = [c for c in range(width) if 2 * sum(row[c] is not None for row in rows) >= len(rows)]
    return keep or [c for c in range(width) if any(row[c] is not None for row in rows)] or [0]


def _column_span(rows: Grid, col: int) -> tuple[float, float]:
    cells = [row[col] for row in rows if row[col] is not None]
    return median(c.x0 for c in cells), median(c.x1 for c in cells)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _column_names(head: Grid, spans: list[tuple[float, float]]) -> tuple[str, ...]:
    """Assign each header label to the columns it physically covers.

    A label owns a column when they overlap by more than half of the narrower of the
    two: a wide group label owns each column under it, and a narrow label centred over
    a wide merged column owns that column.

    A header row where some label covers more than one column is a group row
    ("Measured", "Indicated"): its labels are joined with " · ". Other rows are leaf
    rows whose labels were wrapped over lines ("Contained" / "Ozs (Moz)"): joined
    with a space.
    """
    groups: list[list[str]] = [[] for _ in spans]
    leaves: list[list[str]] = [[] for _ in spans]
    for row in head:
        placed = []
        for cell in row:
            if cell is None or not cell.text:
                continue
            covered = [
                i
                for i, (x0, x1) in enumerate(spans)
                if _overlap(cell.x0, cell.x1, x0, x1)
                > _MIN_OVERLAP * min(x1 - x0, cell.x1 - cell.x0)
            ]
            placed.append((cell.text, covered))
        target = groups if any(len(covered) > 1 for _, covered in placed) else leaves
        for text, covered in placed:
            for i in covered:
                target[i].append(text)
    return tuple(
        " · ".join(_dedupe([" · ".join(_dedupe(g)), " ".join(_dedupe(leaf))]))
        for g, leaf in zip(groups, leaves, strict=True)
    )


def _nearest_kept(col: int, keep: list[int]) -> int:
    """Index (into keep) of the closest real column at or left of col, else the first one."""
    left = [i for i, k in enumerate(keep) if k <= col]
    return left[-1] if left else 0


def _fold_rows(body: Grid, keep: list[int]) -> tuple[tuple[str, ...], ...]:
    """Map data rows onto the real columns.

    Text in a dropped column moves to the nearest real column on its left. A row with a
    single filled cell continues the row above (text wrapped over several grid rows):
    outside the first column always ("design, ongoing"), in the first column only after
    a row with numbers ("Total" / "Mineral" / "Resources"). A first-column-only row
    anywhere else is a group label ("Macraes") and is kept.
    """
    out: list[list[str]] = []
    for row in body:
        cells = [""] * len(keep)
        for col, cell in enumerate(row):
            if cell is not None and cell.text:
                i = _nearest_kept(col, keep)
                cells[i] = f"{cells[i]} {cell.text}".strip()
        filled = [i for i, v in enumerate(cells) if v]
        if not filled:
            continue
        if out and len(filled) == 1:
            (i,) = filled
            if i > 0 or any(_NUMERIC.match(v) for v in out[-1][1:]):
                out[-1][i] = f"{out[-1][i]} {cells[i]}".strip()
                continue
        out.append(cells)
    return tuple(tuple(r) for r in out)


def reconstruct_table(grid: Grid) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Turn a PyMuPDF table grid into (column names, data rows) using cell geometry."""
    if not grid:
        return (), ()
    rows = [[GridCell(_clean(c.text), c.x0, c.x1) if c else None for c in row] for row in grid]
    n_header = _header_row_count(rows)
    head, body = rows[:n_header], rows[n_header:]
    keep = _real_columns(head, body)
    spans = [_column_span(body or head, c) for c in keep]
    return _column_names(head, spans), _fold_rows(body, keep)


# --- captions, notes, page labels -------------------------------------------------
# Thresholds measured on the source reports: captions sit ~10 pt above their table,
# note lines are 7-10 pt apart, the footer is 50+ pt below the last note.

_CAPTION = re.compile(r"^Table\s+[A-Z]?\d+(?:[-–.]\d+)*\b", re.IGNORECASE)
_CAPTION_MAX_GAP = 30.0  # pt between caption and table top
_NOTES_MAX_GAP = 15.0  # pt between consecutive note lines
_BULLET = re.compile(r"[▪•●◦‣∙·\-–*]")
_PAGE_OF = re.compile(r"\bPage\s+(\d+)\s+of\s+\d+\b", re.IGNORECASE)
_MARGIN = 0.12  # header/footer zone: top and bottom 12% of the page


@dataclass(frozen=True)
class Line:
    """One text line with its position (PDF points, y grows downward) and font size."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float


Bbox = tuple[float, float, float, float]


def find_caption(lines: Sequence[Line], bbox: Bbox) -> tuple[str | None, set[int]]:
    """The "Table N..." line just above the table, plus its wrapped continuation lines.

    Returns the caption (one line of text) and the indices of the lines it used.
    """
    top = bbox[1]
    window = [
        (i, ln) for i, ln in enumerate(lines) if ln.y1 <= top + 2 and top - ln.y1 <= _CAPTION_MAX_GAP
    ]
    window.sort(key=lambda item: item[1].y0)
    starts = [pos for pos, (_, ln) in enumerate(window) if _CAPTION.match(ln.text.strip())]
    if not starts:
        return None, set()
    used = window[starts[-1] :]
    return _clean(" ".join(ln.text for _, ln in used)), {i for i, _ in used}


def find_notes(lines: Sequence[Line], bbox: Bbox, body_size: float) -> tuple[str | None, set[int]]:
    """Contiguous smaller-font lines directly below the table (sources, footnotes).

    Stops at the first line that is not smaller than body text (a heading, the next
    paragraph) or that starts after a vertical gap (the page footer).
    """
    previous_bottom = bbox[3]
    below = sorted(
        ((i, ln) for i, ln in enumerate(lines) if ln.y0 >= bbox[3] - 2),
        key=lambda item: (item[1].y0, item[1].x0),  # same baseline: bullet glyph before its text
    )
    used: list[tuple[int, Line]] = []
    for i, ln in below:
        if ln.y0 - previous_bottom > _NOTES_MAX_GAP or ln.size >= body_size - 0.5:
            break
        used.append((i, ln))
        previous_bottom = max(previous_bottom, ln.y1)
    if not used:
        return None, set()
    texts: list[str] = []
    bullet = ""
    for _, ln in used:
        text = _clean(ln.text)
        if _BULLET.fullmatch(text):  # PDFs often draw the bullet as its own line
            bullet = f"{text} "
            continue
        texts.append(bullet + text)
        bullet = ""
    return "\n".join(texts), {i for i, _ in used}


def body_font_size(lines: Sequence[Line]) -> float:
    """Most common font size on the page, weighted by characters."""
    weights: dict[float, int] = {}
    for ln in lines:
        size = round(ln.size, 1)
        weights[size] = weights.get(size, 0) + len(ln.text.strip())
    return max(weights, key=weights.__getitem__) if weights else 0.0


def page_label(pdf_label: str, lines: Sequence[Line], page_height: float) -> str | None:
    """Printed page number: the PDF's own page label, else "Page N of M" in the header/footer zone.

    Only the margins are searched, so "see Page 5 of 12" in body text is not mistaken
    for the page number.
    """
    if pdf_label.strip():
        return pdf_label.strip()
    for ln in lines:
        if ln.y1 <= _MARGIN * page_height or ln.y0 >= (1 - _MARGIN) * page_height:
            match = _PAGE_OF.search(ln.text)
            if match:
                return match.group(1)
    return None


# --- PDF → pages ------------------------------------------------------------------

_MIN_PAGE_CHARS = 50  # below this, a page is probably an image (scan) and needs OCR


def _open(data: bytes) -> pymupdf.Document:
    """Open PDF bytes; every unreadable input becomes ValueError (callers never see PyMuPDF errors)."""
    if not data:
        raise ValueError("empty PDF data")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ValueError(f"not a readable PDF: {exc}") from exc
    if doc.needs_pass:
        doc.close()
        raise ValueError("encrypted PDF: a password is required")
    return doc


def _page_lines(page: pymupdf.Page) -> list[Line]:
    """Text lines in content order, with position and median font size."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for raw in block.get("lines", []):
            spans = [s for s in raw["spans"] if s["text"].strip()]
            if spans:
                x0, y0, x1, y1 = raw["bbox"]
                text = _clean("".join(s["text"] for s in raw["spans"]))
                lines.append(Line(text, x0, y0, x1, y1, median(s["size"] for s in spans)))
    return lines


def _grid(table: Any) -> list[list[GridCell | None]]:
    """PyMuPDF table → grid of cells with their horizontal extent (None = merged)."""
    return [
        [
            GridCell(text or "", bbox[0], bbox[2]) if bbox else None
            for text, bbox in zip(texts, row.cells, strict=True)
        ]
        for texts, row in zip(table.extract(), table.rows, strict=True)
    ]


def _inside(line: Line, bbox: Bbox) -> bool:
    cx, cy = (line.x0 + line.x1) / 2, (line.y0 + line.y1) / 2
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def _build_page(
    page: pymupdf.Page, found: list[Any], lines: list[Line], in_table: set[int], body: float
) -> Page:
    prose_idx = [i for i in range(len(lines)) if i not in in_table]
    prose = [lines[i] for i in prose_idx]
    used = set(in_table)
    tables = []
    for table in found:
        bbox = tuple(table.bbox)
        caption, caption_used = find_caption(prose, bbox)
        notes, notes_used = find_notes(prose, bbox, body)
        used |= {prose_idx[i] for i in caption_used | notes_used}
        header, rows = reconstruct_table(_grid(table))
        tables.append(Table(caption, header, rows, notes, bbox))

    number = page.number + 1
    text = "\n".join(ln.text for i, ln in enumerate(lines) if i not in used)
    if len(text) < _MIN_PAGE_CHARS and not tables:
        log.warning("page %d has almost no text; it may be a scanned image that needs OCR", number)
    return Page(number, page_label(page.get_label(), lines, page.rect.height), text, tuple(tables))


def parse_pdf(data: bytes) -> list[Page]:
    """Parse PDF bytes into pages of prose plus structured tables.

    Two passes: the first collects every page's lines and tables so the body font size
    can be measured over the whole document (per page it fails on table-heavy pages);
    the second assigns captions and notes to tables and keeps the rest as prose.
    Raises ValueError for empty, corrupt or encrypted input.
    """
    with _open(data) as doc:
        scanned = []
        for page in doc:
            found = page.find_tables().tables
            lines = _page_lines(page)
            in_table = {i for i, ln in enumerate(lines) if any(_inside(ln, t.bbox) for t in found)}
            scanned.append((page, found, lines, in_table))
        body = body_font_size(
            [ln for _, _, lines, in_table in scanned for i, ln in enumerate(lines) if i not in in_table]
        )
        return [_build_page(page, found, lines, in_table, body) for page, found, lines, in_table in scanned]
