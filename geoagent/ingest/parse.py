"""PDF → pages: prose text plus structured tables (caption, header, rows, notes)."""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from statistics import median
from typing import Any

import pymupdf

log = logging.getLogger(__name__)

_NUMERIC = re.compile(r"^[-+]?\d[\d,]*(\.\d+)?%?$")
_MIN_OVERLAP = 0.5  # a header label owns a column when it covers > half its width
_MAX_HEADER_ROWS = 4  # Macraes resource header: group, spacer, label, units


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
    line_index: int = 0  # number of Page.text lines before the table


@dataclass(frozen=True)
class Heading:
    start: int  # first Page.text line of the heading
    end: int  # one past the last line (a heading can span lines: "11" + title)
    text: str  # the heading as one line, e.g. "11 SAMPLE PREPARATION, ANALYSES, AND SECURITY"


@dataclass(frozen=True)
class Page:
    number: int  # PDF page, 1-based: always unambiguous
    label: str | None  # printed page number, when detectable
    text: str  # prose only: tables, captions and notes removed
    tables: tuple[Table, ...]
    headings: tuple[Heading, ...] = ()  # detected from typography (size), over Page.text lines


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
    """Row 0, plus up to three following rows whose first cell is merged and that hold no numbers.

    In a multi-level header the top-left label ("Gold", "Area") is one cell merged down
    over the header rows, so their first cell comes back as None. A real empty cell ("")
    or a name ("Macraes", "MOP") starts the data.
    """
    count = 1
    for row in rows[1:_MAX_HEADER_ROWS]:
        if row[0] is not None or any(_NUMERIC.match(_text(c)) for c in row):
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
    if not cells:
        return 0.0, 0.0
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


def _merge_text(existing: str, new: str) -> str:
    """Append a folded fragment to a cell, dropping copies from doubled cell borders.

    Double-bordered cells appear twice in the grid (outer and inner cell): "0.2" + "0.2"
    stays "0.2", "Coronation" + "Coronation Dom1" becomes "Coronation Dom1". Comparison
    is by whole tokens, so "1" + "10" stays "1 10".
    """
    old, add = existing.split(), new.split()
    if old[: len(add)] == add:
        return existing
    if add[: len(old)] == old:
        return new
    return f"{existing} {new}"


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
    anywhere else is a group label ("Macraes") and is kept. A lone number under a
    number is a new row: joining them would corrupt the value ("864,072 0").
    """
    out: list[list[str]] = []
    for row in body:
        cells = [""] * len(keep)
        for col, cell in enumerate(row):
            if cell is not None and cell.text:
                i = _nearest_kept(col, keep)
                cells[i] = _merge_text(cells[i], cell.text)
        filled = [i for i, v in enumerate(cells) if v]
        if not filled:
            continue
        if out and len(filled) == 1:
            (i,) = filled
            above = out[-1][i]
            fuses_numbers = bool(above) and _NUMERIC.match(above) and _NUMERIC.match(cells[i])
            wraps = i > 0 or any(_NUMERIC.match(v) for v in out[-1][1:])
            if wraps and not fuses_numbers:
                out[-1][i] = _merge_text(above, cells[i])
                continue
        out.append(cells)
    return tuple(tuple(r) for r in out)


def reconstruct_table(grid: Grid) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Turn a PyMuPDF table grid into (column names, data rows) using cell geometry."""
    width = max((len(row) for row in grid), default=0)
    if width == 0 or not any(cell for row in grid for cell in row):
        return (), ()
    rows = [
        [GridCell(_clean(c.text), c.x0, c.x1) if c else None for c in row] + [None] * (width - len(row))
        for row in grid  # ragged grids are padded so every row has the same width
    ]
    while rows and not any(_text(c) for c in rows[0]):  # blank space inside the table bbox
        rows.pop(0)
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
_CAPTION_PUNCT = re.compile(r"^Table\s+[A-Z]?\d+(?:[-–.]\d+)*\s*[:.\-–—]", re.IGNORECASE)
_BULLET = re.compile(r"[▪•●◦‣∙·\-–*]")
_PAGE_OF = re.compile(r"\bPage\s+(\d+)\s+of\s+\d+\b", re.IGNORECASE)
_MARGIN = 0.12  # header/footer zone: top and bottom 12% of the page


@dataclass(frozen=True)
class Line:
    """One text line with its position (PDF points, y grows downward) and font."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool = False


Bbox = tuple[float, float, float, float]


def _is_caption(line: Line) -> bool:
    """A "Table N" line that is styled or punctuated like a caption.

    "Table 3 shows the grades..." is prose that merely mentions a table: real captions
    are bold in the source reports, or put punctuation after the number ("Table 1-1:").
    """
    text = line.text.strip()
    return bool(_CAPTION.match(text)) and (line.bold or bool(_CAPTION_PUNCT.match(text)))


def _over(line: Line, bbox: Bbox) -> bool:
    """The line sits horizontally over/under the table (side-by-side tables stay apart)."""
    overlap = _overlap(line.x0, line.x1, bbox[0], bbox[2])
    return overlap > _MIN_OVERLAP * min(line.x1 - line.x0, bbox[2] - bbox[0])


def find_caption(lines: Sequence[Line], bbox: Bbox) -> tuple[str | None, set[int]]:
    """The "Table N..." line just above the table, plus its wrapped continuation lines.

    Returns the caption (one line of text) and the indices of the lines it used.
    """
    top = bbox[1]
    window = [
        (i, ln)
        for i, ln in enumerate(lines)
        if ln.y1 <= top + 2 and top - ln.y1 <= _CAPTION_MAX_GAP and _over(ln, bbox)
    ]
    window.sort(key=lambda item: item[1].y0)
    starts = [pos for pos, (_, ln) in enumerate(window) if _is_caption(ln)]
    if not starts:
        return None, set()
    used = window[starts[-1] :]
    return _clean(" ".join(ln.text for _, ln in used)), {i for i, _ in used}


def find_notes(
    lines: Sequence[Line], bbox: Bbox, body_size: float, page_height: float | None = None
) -> tuple[str | None, set[int]]:
    """Contiguous smaller-font lines directly below the table (sources, footnotes).

    Stops at the first line that is not smaller than body text (a heading, the next
    paragraph), that starts after a vertical gap, that is another table's caption, or
    that lies in the footer zone (small, close footers look exactly like notes).
    """
    footer_top = (1 - _MARGIN) * page_height if page_height else float("inf")
    previous_bottom = bbox[3]
    below = sorted(
        ((i, ln) for i, ln in enumerate(lines) if ln.y0 >= bbox[3] - 2 and _over(ln, bbox)),
        key=lambda item: (item[1].y0, item[1].x0),  # same baseline: bullet glyph before its text
    )
    used: list[tuple[int, Line]] = []
    for i, ln in below:
        if (
            ln.y0 - previous_bottom > _NOTES_MAX_GAP
            or ln.size >= body_size - 0.5
            or ln.y0 >= footer_top
            or _is_caption(ln)
        ):
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


_SECTION_NUMBER = re.compile(r"\d{1,2}(?:\.\d{1,2})*\.?")
_HEADING_MIN_EXTRA_SIZE = 1.0  # pt above body size
_HEADING_MAX_CHARS = 120


_DOT_LEADER = re.compile(r"(\.\s?){4,}|…{2,}")  # table-of-contents "Summary ....... 14"


def _heading_styled(line: Line, body_size: float) -> bool:
    return (
        line.size >= body_size + _HEADING_MIN_EXTRA_SIZE
        and len(line.text) <= _HEADING_MAX_CHARS
        and not _DOT_LEADER.search(line.text)
    )


def find_headings(lines: Sequence[Line], body_size: float) -> tuple[Heading, ...]:
    """Section headings from typography: lines at least 1 pt larger than body text.

    Some reports number headings in the text ("1.10 Mineral Reserve Estimate"), others
    draw the number as a separate line ("11" / "SAMPLE PREPARATION...") or not at all;
    size is the signal they share. A bare number joins the title that follows; a title
    wrapped over tightly spaced lines of the same size is merged; lines that end like a
    sentence are emphasis, not headings.
    """
    out: list[Heading] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not _heading_styled(line, body_size):
            i += 1
            continue
        start, parts = i, [line.text]
        if _SECTION_NUMBER.fullmatch(line.text):
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            if nxt is None or not _heading_styled(nxt, body_size):
                i += 1  # a lone big number (a page or figure number), not a heading
                continue
            i += 1
            parts.append(nxt.text)
        while (
            i + 1 < len(lines)
            and abs(lines[i + 1].size - lines[i].size) <= 0.5
            and lines[i + 1].y0 - lines[i].y1 < 0.5 * lines[i].size
            and not _SECTION_NUMBER.match(lines[i + 1].text)
        ):
            i += 1
            parts.append(lines[i].text)
        text = _clean(" ".join(parts))
        if not text.endswith((".", ",", ";")) and not _DOT_LEADER.search(text):
            out.append(Heading(start, i + 1, text))
        i += 1
    return tuple(out)


RawTable = tuple[Bbox, tuple[str, ...], tuple[tuple[str, ...], ...]]  # bbox, header, rows


def attach_captions_and_notes(
    lines: Sequence[Line], tables: Sequence[RawTable], body_size: float, page_height: float
) -> tuple[list[Table], set[int]]:
    """Give each table its caption and notes; every line is claimed by at most one table.

    Tables are processed top-to-bottom, left-to-right, and a line taken by one table is
    no longer a candidate for the next (stacked tables: A's notes cannot swallow B's
    caption). Returns the tables and the indices of all lines they claimed, including
    the lines inside them.
    """
    claimed = {i for i, ln in enumerate(lines) if any(_inside(ln, bbox) for bbox, _, _ in tables)}

    def unclaimed() -> tuple[list[int], list[Line]]:
        free = [i for i in range(len(lines)) if i not in claimed]
        return free, [lines[i] for i in free]

    out = []
    for bbox, header, rows in sorted(tables, key=lambda t: (t[0][1], t[0][0])):
        free, candidates = unclaimed()
        caption, used = find_caption(candidates, bbox)
        claimed |= {free[i] for i in used}
        free, candidates = unclaimed()
        notes, used = find_notes(candidates, bbox, body_size, page_height)
        claimed |= {free[i] for i in used}
        out.append(Table(caption, header, rows, notes, bbox))
    return out, claimed


# --- PDF → pages ------------------------------------------------------------------

_MIN_PAGE_CHARS = 50  # below this, a page is probably an image (scan) and needs OCR


# MuPDF's own errors (FzErrorFormat, ...) derive from FzErrorBase, which is NOT a
# RuntimeError: both families must be caught to keep the ValueError-only contract.
_MUPDF_ERRORS = (pymupdf.mupdf.FzErrorBase, RuntimeError)


def _open(data: bytes) -> pymupdf.Document:
    """Open PDF bytes; every unreadable input becomes ValueError (callers never see PyMuPDF errors)."""
    if not data:
        raise ValueError("empty PDF data")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except _MUPDF_ERRORS as exc:
        raise ValueError(f"not a readable PDF: {exc}") from exc
    problem = None
    if not doc.is_pdf:  # MuPDF sniffs content: HTML or PNG bytes "open" as other formats
        problem = "not a PDF"
    elif doc.needs_pass:
        problem = "encrypted PDF: a password is required"
    elif doc.page_count == 0:
        problem = "PDF has no pages"
    if problem:
        doc.close()
        raise ValueError(problem)
    if doc.is_repaired:
        log.warning("PDF was damaged and repaired on open; some content may be missing")
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
                size = median(s["size"] for s in spans)
                bold = all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
                lines.append(Line(text, x0, y0, x1, y1, size, bold))
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


@dataclass
class _Scan:
    """Pass-1 result for one page: plain data only, so no PyMuPDF objects outlive pass 1."""

    number: int
    label: str | None
    height: float
    lines: list[Line]
    tables: list[RawTable]
    in_table: set[int]


def _scan_page(page: pymupdf.Page) -> _Scan:
    lines = _page_lines(page)
    tables = []
    for found in page.find_tables().tables:
        header, rows = reconstruct_table(_grid(found))
        if len(header) <= 1 and not rows:  # a bordered text box, not a table: keep it as prose
            continue
        tables.append((tuple(found.bbox), header, rows))
    in_table = {i for i, ln in enumerate(lines) if any(_inside(ln, bbox) for bbox, _, _ in tables)}
    label = page_label(page.get_label(), lines, page.rect.height)
    return _Scan(page.number + 1, label, page.rect.height, lines, tables, in_table)


def _build_page(scan: _Scan, body: float) -> Page:
    tables, used = attach_captions_and_notes(scan.lines, scan.tables, body, scan.height)
    kept = [ln for i, ln in enumerate(scan.lines) if i not in used]
    text = "\n".join(ln.text for ln in kept)
    # Where each table sat in the prose: the number of kept lines above it.
    tables = [replace(t, line_index=sum(ln.y1 <= t.bbox[1] + 2 for ln in kept)) for t in tables]
    if len(text) < _MIN_PAGE_CHARS and not tables:
        log.warning("page %d has almost no text; it may be a scanned image that needs OCR", scan.number)
    return Page(scan.number, scan.label, text, tuple(tables), find_headings(kept, body))


def parse_pdf(data: bytes) -> list[Page]:
    """Parse PDF bytes into pages of prose plus structured tables.

    Two passes: the first extracts every page's lines and tables as plain data, so the
    body font size can be measured over the whole document (per page it fails on
    table-heavy pages); the second assigns captions and notes to tables and keeps the
    rest as prose. Raises ValueError for empty, corrupt, encrypted or non-PDF input.
    """
    with _open(data) as doc:
        try:
            scans = [_scan_page(page) for page in doc]
            body = body_font_size(
                [ln for s in scans for i, ln in enumerate(s.lines) if i not in s.in_table]
            )
            return [_build_page(scan, body) for scan in scans]
        except _MUPDF_ERRORS as exc:  # e.g. a malformed page tree discovered mid-document
            raise ValueError(f"PDF could not be processed: {exc}") from exc
