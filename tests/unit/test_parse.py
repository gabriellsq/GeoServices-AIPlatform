import logging

import pymupdf
import pytest

from geoagent.ingest.parse import (
    GridCell,
    Line,
    Page,
    Table,
    attach_captions_and_notes,
    body_font_size,
    find_caption,
    find_notes,
    page_label,
    parse_pdf,
    reconstruct_table,
)
from tests.fixtures.fixture_pdf import HEADER, TABLE_CAPTION, TABLE_NOTES, build_fixture_pdf


@pytest.fixture(scope="module")
def pages() -> list[Page]:
    return parse_pdf(build_fixture_pdf())


def test_one_page_object_per_pdf_page_numbered_from_one(pages):
    assert [p.number for p in pages] == [1, 2, 3, 4]


def test_label_is_the_printed_page_number(pages):
    assert [p.label for p in pages] == ["1", "2", "3", "4"]


def test_prose_keeps_body_header_and_footer(pages):
    # Header/footer stripping is the chunker's job; the parser extracts faithfully.
    assert "1 SUMMARY" in pages[0].text
    assert HEADER in pages[1].text
    assert "Page 3 of 4" in pages[2].text
    assert "Significant intercepts are listed in the table below." in pages[2].text


def test_table_is_extracted_as_structure(pages):
    (table,) = pages[2].tables
    assert isinstance(table, Table)
    assert table.header == ("Hole", "From (m)", "To (m)", "Au (g/t)")
    assert table.rows == (
        ("DH-001", "45.0", "52.0", "3.10"),
        ("DH-002", "112.0", "118.5", "4.70"),
    )


def test_table_carries_its_caption_and_notes(pages):
    table = pages[2].tables[0]
    assert table.caption == TABLE_CAPTION
    assert table.notes == TABLE_NOTES


def test_table_content_is_not_duplicated_in_prose(pages):
    text = pages[2].text
    for fragment in ("DH-002", "112.0", TABLE_CAPTION, TABLE_NOTES):
        assert fragment not in text


def test_pages_without_tables_have_none(pages):
    assert [len(p.tables) for p in pages] == [0, 0, 1, 0]


def test_page_is_frozen(pages):
    with pytest.raises(AttributeError):
        pages[0].text = "x"  # type: ignore[misc]


@pytest.mark.parametrize("data", [b"definitely not a pdf", b""])
def test_unreadable_input_raises_value_error(data):
    with pytest.raises(ValueError):
        parse_pdf(data)


def grid(*rows):
    """Build a table grid: each cell is None (covered by a merged cell) or (text, x0, x1)."""
    return [[GridCell(*c) if c else None for c in row] for row in rows]


def test_reconstruct_multi_level_header():
    # Macraes resource-statement shape: merged group labels ("Measured" spans 3 columns,
    # covered cells come back as None), units on a second leaf row, a group-label data row
    # ("Macraes"), a wrapped cell, and an empty spacer row.
    raw = grid(
        [("Gold", 0, 50), ("Measured", 50, 80), None, None, ("Indicated", 80, 110), None, None],
        [None, ("", 50, 80), None, None, ("", 80, 110), None, None],
        [None, ("Tonnes", 50, 60), ("Au", 60, 70), ("Contained", 70, 80),
         ("Tonnes", 80, 90), ("Au", 90, 100), ("Contained", 100, 110)],
        [None, ("(Mt)", 50, 60), ("(g/t)", 60, 70), ("Ozs (Moz)", 70, 80),
         ("(Mt)", 80, 90), ("(g/t)", 90, 100), ("Ozs (Moz)", 100, 110)],
        [("Macraes", 0, 50), None, None, None, None, None, None],
        [("Coronation\nNorth", 0, 50), ("0.14", 50, 60), ("1.23", 60, 70), ("0.01", 70, 80),
         ("3.06", 80, 90), ("0.63", 90, 100), ("0.06", 100, 110)],
        [("", 0, 50), ("", 50, 60), ("", 60, 70), ("", 70, 80), ("", 80, 90), ("", 90, 100), ("", 100, 110)],
    )
    header, rows = reconstruct_table(raw)
    assert header == (
        "Gold",
        "Measured · Tonnes (Mt)",
        "Measured · Au (g/t)",
        "Measured · Contained Ozs (Moz)",
        "Indicated · Tonnes (Mt)",
        "Indicated · Au (g/t)",
        "Indicated · Contained Ozs (Moz)",
    )
    assert rows == (
        ("Macraes", "", "", "", "", "", ""),
        ("Coronation North", "0.14", "1.23", "0.01", "3.06", "0.63", "0.06"),
    )


def test_reconstruct_uses_geometry_not_grid_position():
    # Waihi resource-statement shape: double borders create sliver columns (60-66, 130-136,
    # 170-176) that never hold data; the "Indicated" label cell starts at x=72 while the
    # Tonnes column under it starts at x=66 (different grid columns, same position on the
    # page); the "Total Mineral Resources" label is wrapped over three grid rows.
    def data(label, t, au, inf):
        return [(label, 0, 60), None, (t, 66, 100), (au, 100, 130), None, (inf, 136, 170), None]

    raw = grid(
        [("Area", 0, 60), None, ("Indicated", 72, 130), None, ("", 130, 136), ("Inferred", 136, 176), None],
        [None, None, ("Tonnes", 66, 100), ("Au", 100, 130), None, ("Tonnes", 136, 170), None],
        [None, None, ("(Mt)", 66, 100), ("(Moz)", 100, 130), None, ("(Mt)", 136, 170), None],
        data("MOP", "6.50", "0.41", "2.3"),
        data("GOP", "3.22", "0.15", "0.8"),
        data("MUG", "6.42", "1.09", "2.7"),
        data("WUG", "2.39", "1.37", "1.3"),
        [("", 0, 60), ("Total", 60, 66), ("18.5", 66, 100), ("3.02", 100, 130), None, ("7.1", 136, 170), None],
        [None, ("Mineral", 60, 66), None, None, None, None, None],
        [None, ("Resources", 60, 66), None, None, None, None, None],
    )
    header, rows = reconstruct_table(raw)
    assert header == ("Area", "Indicated · Tonnes (Mt)", "Indicated · Au (Moz)", "Inferred · Tonnes (Mt)")
    assert rows == (
        ("MOP", "6.50", "0.41", "2.3"),
        ("GOP", "3.22", "0.15", "0.8"),
        ("MUG", "6.42", "1.09", "2.7"),
        ("WUG", "2.39", "1.37", "1.3"),
        ("Total Mineral Resources", "18.5", "3.02", "7.1"),
    )



def test_reconstruct_text_table_with_wrapped_cells():
    # Macraes p168 shape (tailings dams): each data column is one merged cell three grid
    # columns wide, the header label sits only over the middle third, and a long cell
    # wraps over several grid rows that hold nothing else.
    raw = grid(
        [("", 0, 30), ("", 30, 40), ("MTI", 40, 50), ("", 50, 60)],
        [("Status", 0, 30), ("Active closure", 30, 60), None, None],
        [("Notes", 0, 30), ("Embankment to", 30, 60), None, None],
        [None, ("design, ongoing", 30, 60), None, None],
        [None, ("deposition", 30, 60), None, None],
        [("Impact", 0, 30), ("High", 30, 60), None, None],
    )
    header, rows = reconstruct_table(raw)
    assert header == ("", "MTI")
    assert rows == (
        ("Status", "Active closure"),
        ("Notes", "Embankment to design, ongoing deposition"),
        ("Impact", "High"),
    )

# --- caption / notes / label detectors (pure functions over positioned lines) ---------

TABLE_BBOX = (50.0, 200.0, 500.0, 300.0)


def line(text, y0, y1, size=10.0):
    return Line(text=text, x0=50.0, y0=y0, x1=500.0, y1=y1, size=size)


def test_caption_is_the_table_line_just_above_including_its_wrapped_lines():
    lines = [
        line("Some prose about the deposit.", 150, 162),
        line("Table 14-3: Mineral Resource Statement", 175, 187),
        line("as at December 31, 2025", 188, 197),
    ]
    assert find_caption(lines, TABLE_BBOX) == (
        "Table 14-3: Mineral Resource Statement as at December 31, 2025",
        {1, 2},
    )


def test_no_caption_when_no_table_line_or_too_far_above():
    assert find_caption([line("Significant intercepts are listed below.", 180, 192)], TABLE_BBOX) == (None, set())
    assert find_caption([line("Table 1-1 Too far away", 100, 112)], TABLE_BBOX) == (None, set())


def test_notes_are_contiguous_smaller_font_lines_below():
    lines = [
        line("Notes:", 305, 314, size=8),
        line("- All figures are rounded.", 318, 327, size=8),
        line("1.10 Mineral Reserve Estimate", 335, 347, size=12),  # heading: bigger font, stop
        line("Released: March 27, 2026   Page 21 of 213", 780, 790, size=8),
    ]
    assert find_notes(lines, TABLE_BBOX, body_size=10) == ("Notes:\n- All figures are rounded.", {0, 1})


def test_notes_stop_at_a_vertical_gap():
    lines = [line("Note: uncut grades.", 305, 314, size=8), line("small footer text", 360, 370, size=8)]
    assert find_notes(lines, TABLE_BBOX, body_size=10) == ("Note: uncut grades.", {0})


def test_no_notes_when_body_text_follows_the_table():
    assert find_notes([line("The table shows...", 305, 317, size=10)], TABLE_BBOX, body_size=10) == (None, set())


def test_body_font_size_is_the_most_common_size_by_characters():
    lines = [line("x" * 200, 0, 10, size=11), line("Heading", 20, 30, size=14), line("y" * 50, 40, 50, size=9)]
    assert body_font_size(lines) == 11


@pytest.mark.parametrize(
    ("pdf_label", "footer", "expected"),
    [
        ("iv", None, "iv"),  # a label defined in the PDF wins
        ("", "Released: March 27, 2026   Page 21 of 213", "21"),
        ("", "Document ID: STU-063   Page  17  of  215", "17"),
        ("", "no page number here", None),
    ],
)
def test_page_label(pdf_label, footer, expected):
    lines = [line(footer, 780, 790)] if footer else []
    assert page_label(pdf_label, lines, page_height=842) == expected


def test_page_label_ignores_page_references_in_body_text():
    lines = [line("as discussed on Page 5 of 12 in Section 7", 400, 412)]
    assert page_label("", lines, page_height=842) is None


def test_notes_join_a_lone_bullet_glyph_with_its_text():
    lines = [
        line("Notes:", 305, 314, size=9),
        line("▪", 318, 326, size=9),
        line("Reported inclusive of Mineral Reserves.", 318, 327, size=9),
    ]
    notes, _ = find_notes(lines, TABLE_BBOX, body_size=10)
    assert notes == "Notes:\n▪ Reported inclusive of Mineral Reserves."


def test_encrypted_pdf_raises_value_error():
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 80), "secret")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    with pytest.raises(ValueError, match="encrypted"):
        parse_pdf(data)


def test_page_with_almost_no_text_is_logged_as_possibly_scanned(caplog):
    doc = pymupdf.open()
    doc.new_page()  # blank page: what a scanned image-only page looks like to a text parser
    with caplog.at_level(logging.WARNING):
        pages = parse_pdf(doc.tobytes())
    assert pages[0].text == ""
    assert "page 1" in caplog.text and "OCR" in caplog.text


# --- robustness (review findings #3, #4, #5) ---------------------------------------

ZERO_PAGE_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF"
)


def png_bytes() -> bytes:
    return pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 4, 4), False).tobytes("png")


@pytest.mark.parametrize(
    "make", [lambda: b"<html>x</html>", png_bytes, lambda: ZERO_PAGE_PDF], ids=["html", "png", "zero-pages"]
)
def test_non_pdf_and_empty_documents_raise_value_error(make):
    with pytest.raises(ValueError):
        parse_pdf(make())


def test_truncated_pdf_is_parsed_with_a_warning(caplog):
    data = build_fixture_pdf()
    with caplog.at_level(logging.WARNING):
        pages = parse_pdf(data[: len(data) // 2])
    assert pages  # MuPDF repairs what it can...
    assert "repaired" in caplog.text  # ...and we say so, because content may be missing


def test_errors_during_processing_become_value_error(monkeypatch):
    def broken(self, *args, **kwargs):
        raise RuntimeError("malformed page tree")

    monkeypatch.setattr(pymupdf.Page, "find_tables", broken)
    with pytest.raises(ValueError, match="malformed page tree"):
        parse_pdf(build_fixture_pdf())


@pytest.mark.parametrize(
    "raw",
    [
        [],
        [[]],
        [[None]],
        [[GridCell("A", 0, 10), GridCell("B", 10, 20)], [GridCell("x", 0, 10)]],  # ragged
        [[GridCell("A", 0, 10)], [None], [None]],  # body of merged-only cells
    ],
    ids=["empty", "empty-row", "single-none", "ragged", "all-none-body"],
)
def test_reconstruct_degenerate_grids_do_not_raise(raw):
    header, rows = reconstruct_table(raw)
    assert isinstance(header, tuple) and isinstance(rows, tuple)


# --- table content (review findings #2a, #8, #9) -----------------------------------


def test_lone_number_below_a_number_starts_a_new_row():
    # Waihi GeoMet shape: a single-value row under a numeric cell is a new row, not a
    # wrapped continuation ("864,072 0" would be a corrupted number).
    raw = grid(
        [("Op", 0, 50), ("Tonnes", 50, 90)],
        [("Pit A", 0, 50), ("864,072", 50, 90)],
        [None, ("0", 50, 90)],
    )
    _, rows = reconstruct_table(raw)
    assert rows == (("Pit A", "864,072"), ("", "0"))


def test_doubled_border_cells_are_not_duplicated():
    # Macraes p104 shape: some cells are drawn twice (outer + inner border); the inner
    # copy lands in an extra grid column that is folded back into the real column.
    raw = grid(
        [("Domain", 0, 50), None, ("Search", 50, 100), None],
        [("Coronation", 0, 50), ("Coronation Dom1", 4, 46), ("0.2", 50, 100), ("0.2", 56, 94)],
        [("Deepdell", 0, 50), None, ("0.3", 50, 100), None],
        [("Innes", 0, 50), None, ("0.4", 50, 100), None],
    )
    header, rows = reconstruct_table(raw)
    assert header == ("Domain", "Search")
    assert rows == (("Coronation Dom1", "0.2"), ("Deepdell", "0.3"), ("Innes", "0.4"))


def test_header_continuation_needs_a_merged_first_cell():
    # Only a first cell covered by the vertically merged top-left label (None) continues
    # the header; a real empty cell ("") starts the data, even with no numbers in the row.
    raw = grid(
        [("A", 0, 50), ("B", 50, 100), ("C", 100, 150)],
        [("", 0, 50), ("text one", 50, 100), ("note a", 100, 150)],
        [("", 0, 50), ("text two", 50, 100), ("note b", 100, 150)],
    )
    header, rows = reconstruct_table(raw)
    assert header == ("A", "B", "C")
    assert rows == (("", "text one", "note a"), ("", "text two", "note b"))


def test_header_is_at_most_four_rows():
    raw = grid(
        [("A", 0, 50), ("B", 50, 100)],
        *[[None, (f"h{i}", 50, 100)] for i in range(5)],
        [("x", 0, 50), ("1", 50, 100)],
    )
    header, rows = reconstruct_table(raw)
    assert header == ("A", "B h0 h1 h2")
    assert rows[0] == ("", "h3 h4") and rows[-1] == ("x", "1")


def test_bordered_text_box_stays_in_prose():
    # Waihi p165 shape: a paragraph inside a single bordered box is not a table.
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(50, 100, 500, 160), color=(0, 0, 0), width=0.5)
    page.insert_text((60, 130), "A paragraph drawn inside a bordered box, not a table.", fontsize=10)
    (parsed,) = parse_pdf(doc.tobytes())
    assert parsed.tables == ()
    assert "inside a bordered box" in parsed.text


def test_leading_empty_rows_are_ignored():
    # Waihi p86 shape: the table bbox includes blank grid rows above the real header.
    raw = grid(
        [None, None, ("", 90, 120)],
        [None, None, ("", 90, 120)],
        [("Ore Class", 0, 50), ("Au", 50, 90), ("Ag", 90, 120)],
        [("High", 0, 50), ("1.6", 50, 90), ("2.6", 90, 120)],
    )
    header, rows = reconstruct_table(raw)
    assert header == ("Ore Class", "Au", "Ag")
    assert rows == (("High", "1.6", "2.6"),)


# --- caption / notes ownership (review findings #6, #7) ------------------------------

LOW_TABLE = (50.0, 600.0, 500.0, 770.0)  # a table ending near the bottom of an 842 pt page


def test_footer_is_not_notes():
    footer = line("Released: 11th December 2024   Page 86 of 215", 780, 790, size=8)
    assert find_notes([footer], LOW_TABLE, body_size=10, page_height=842) == (None, set())


def test_notes_stop_at_the_next_tables_caption():
    lines = [line("Source: first survey", 305, 314, size=8), line("Table 2: Second table", 318, 327, size=8)]
    assert find_notes(lines, TABLE_BBOX, body_size=10) == ("Source: first survey", {0})


def test_caption_must_sit_over_the_table():
    side = Line("Table 4: Right-hand table", x0=600, y0=180, x1=900, y1=192, size=10, bold=True)
    assert find_caption([side], TABLE_BBOX) == (None, set())


def test_prose_mentioning_a_table_is_not_a_caption():
    prose = line("Table 3 shows the grades of the main intercepts", 180, 192)
    caption = Line("Table 3 Main intercepts", x0=50, y0=180, x1=500, y1=192, size=10, bold=True)
    assert find_caption([prose], TABLE_BBOX) == (None, set())
    assert find_caption([caption], TABLE_BBOX) == ("Table 3 Main intercepts", {0})


def test_each_line_belongs_to_one_table_only():
    # Two stacked tables: table A's notes must not swallow table B's caption.
    a, b = (50.0, 100.0, 500.0, 200.0), (50.0, 240.0, 500.0, 340.0)
    lines = [
        Line("Table 1: First", x0=50, y0=85, x1=300, y1=96, size=10, bold=True),
        line("Source: survey A", 204, 212, size=8),
        Line("Table 2: Second", x0=50, y0=222, x1=300, y1=233, size=8, bold=True),
        line("Body text after both tables.", 360, 372),
    ]
    tables, used = attach_captions_and_notes(
        lines, [(b, ("x",), ()), (a, ("y",), ())], body_size=10, page_height=842
    )
    by_top = {t.bbox[1]: t for t in tables}
    assert by_top[100.0].caption == "Table 1: First" and by_top[100.0].notes == "Source: survey A"
    assert by_top[240.0].caption == "Table 2: Second" and by_top[240.0].notes is None
    assert used == {0, 1, 2}


def test_side_by_side_tables_get_their_own_caption_and_notes():
    left, right = (50.0, 200.0, 280.0, 300.0), (320.0, 200.0, 550.0, 300.0)
    lines = [
        Line("Table 3: Left", x0=50, y0=185, x1=200, y1=196, size=10, bold=True),
        Line("Table 4: Right", x0=320, y0=185, x1=470, y1=196, size=10, bold=True),
        Line("Source: left", x0=50, y0=304, x1=200, y1=312, size=8),
        Line("Source: right", x0=320, y0=304, x1=470, y1=312, size=8),
    ]
    tables, _ = attach_captions_and_notes(
        lines, [(left, ("a",), ()), (right, ("b",), ())], body_size=10, page_height=842
    )
    by_left = {t.bbox[0]: t for t in tables}
    assert (by_left[50.0].caption, by_left[50.0].notes) == ("Table 3: Left", "Source: left")
    assert (by_left[320.0].caption, by_left[320.0].notes) == ("Table 4: Right", "Source: right")
