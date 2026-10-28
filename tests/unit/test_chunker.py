import re

import pytest

from geoagent.ingest.chunker import Chunk, chunk_pages, count_tokens, is_heading
from geoagent.ingest.parse import Heading, Page, Table, parse_pdf
from tests.fixtures.fixture_pdf import HEADER, TABLE_CAPTION, TABLE_NOTES, build_fixture_pdf

FIXTURE_SECTIONS = {
    "1 SUMMARY",
    "2 INTRODUCTION",
    "7.1 Regional Geology",
    "7.2 Mineralization",
    "14 MINERAL RESOURCE ESTIMATES",
}


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return chunk_pages(parse_pdf(build_fixture_pdf()))


def text_chunks(chunks):
    return [c for c in chunks if c.kind == "text"]


def table_chunks(chunks):
    return [c for c in chunks if c.kind == "table"]


# --- tokens and headings ---------------------------------------------------------------


def test_count_tokens_is_positive_and_monotonic():
    assert count_tokens("") == 0
    assert 0 < count_tokens("gold") < count_tokens("gold in quartz-sulphide veins")


@pytest.mark.parametrize(
    "line",
    ["1 SUMMARY", "7.1 Regional Geology", "1.10 Mineral Reserve Estimate", "14 MINERAL RESOURCE ESTIMATES"],
)
def test_numbered_headings_are_detected(line):
    assert is_heading(line)


@pytest.mark.parametrize(
    "line",
    [
        "10 Mt at 2.3 g/t Au",  # a quantity, not a heading
        "15 samples were assayed",  # a sentence starting with a number
        "2023 Drilling Program",  # a year
        "1 SUMMARY ........................ 15",  # a table-of-contents entry
        "3 The results are shown below.",  # ends like a sentence
    ],
)
def test_lines_that_only_look_like_headings_are_not(line):
    assert not is_heading(line)


# --- the fixture report ----------------------------------------------------------------


def test_text_sections_detected(chunks):
    assert {c.section for c in text_chunks(chunks)} == FIXTURE_SECTIONS


def test_ordinals_contiguous_and_text_non_empty(chunks):
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.text.strip() for c in chunks)


def test_token_counts_are_exact_and_bounded(chunks):
    for c in chunks:
        assert c.token_count == count_tokens(c.text)
        assert c.token_count <= 512


def test_no_text_chunk_mixes_sections(chunks):
    for c in text_chunks(chunks):
        assert not ("greywacke" in c.text and "quartz-sulphide" in c.text)


def test_section_crossing_a_page_break_tracks_both_pages(chunks):
    regional = [c for c in chunks if c.section == "7.1 Regional Geology"]
    assert regional[0].page_start == 2
    assert regional[-1].page_end == 3
    assert (regional[0].label_start, regional[-1].label_end) == ("2", "3")
    assert any("Weathering of the greywacke" in c.text for c in regional)


def test_headers_and_footers_are_stripped(chunks):
    for c in chunks:
        assert HEADER not in c.text
        assert not re.search(r"Page \d+ of \d+", c.text)


def test_headers_kept_when_stripping_disabled():
    chunks = chunk_pages(parse_pdf(build_fixture_pdf()), strip_headers=False)
    assert any(HEADER in c.text for c in chunks)


def test_table_becomes_one_chunk_with_caption_markdown_and_notes(chunks):
    (table,) = table_chunks(chunks)
    assert table.section == "7.2 Mineralization"
    assert (table.page_start, table.page_end, table.label_start) == (3, 3, "3")
    assert table.text.startswith(TABLE_CAPTION)
    assert "| Hole | From (m) | To (m) | Au (g/t) |" in table.text
    assert "| DH-002 | 112.0 | 118.5 | 4.70 |" in table.text
    assert table.text.rstrip().endswith(TABLE_NOTES)


def test_table_content_is_only_in_the_table_chunk(chunks):
    assert [c.kind for c in chunks if "DH-002" in c.text] == ["table"]


def test_table_sits_between_the_prose_around_it(chunks):
    kinds = [(c.section, c.kind) for c in chunks if c.section == "7.2 Mineralization"]
    assert kinds == [("7.2 Mineralization", "text"), ("7.2 Mineralization", "table")]


# --- synthetic pages -------------------------------------------------------------------


def page(number, text, tables=(), label=None):
    return Page(number=number, label=label, text=text, tables=tuple(tables))


def test_long_section_split_with_overlap():
    words = " ".join(f"w{i}" for i in range(3000))
    chunks = chunk_pages([page(1, "9 DRILLING\n" + words)], max_tokens=200, overlap_tokens=20)
    assert len(chunks) > 5
    assert all(c.section == "9 DRILLING" and c.token_count <= 200 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.text.split()[0] in prev.text.split(), "next chunk should start inside previous"
    covered = set(" ".join(c.text for c in chunks).split())
    assert {f"w{i}" for i in range(3000)} <= covered


def test_text_before_first_heading_has_no_section():
    chunks = chunk_pages([page(1, "Preface text.\n1 SUMMARY\nBody text.")])
    assert [(c.section, c.text) for c in chunks] == [(None, "Preface text."), ("1 SUMMARY", "Body text.")]


def test_boilerplate_counts_distinct_pages_not_occurrences():
    # "DH-001" and "DH-002" normalise to the same line ("DH-###") but appear on ONE page
    # of three: that is content, not a running header.
    pages = [
        page(1, "Running header 2026\n1 SUMMARY\nAlpha."),
        page(2, "Running header 2026\nDH-001\nDH-002"),
        page(3, "Running header 2026\nGamma."),
    ]
    text = " ".join(c.text for c in chunk_pages(pages))
    assert "Running header" not in text
    assert "DH-001" in text and "DH-002" in text


def test_large_table_is_split_by_rows_with_caption_and_header_repeated():
    rows = tuple((f"DH-{i:03d}", f"{i}.0", f"{i + 5}.0", f"{i / 10:.2f}") for i in range(80))
    table = Table(
        caption="Table 10-2: All intercepts",
        header=("Hole", "From (m)", "To (m)", "Au (g/t)"),
        rows=rows,
        notes="Notes: downhole lengths.",
        bbox=(0.0, 0.0, 1.0, 1.0),
        line_index=1,
    )
    chunks = chunk_pages([page(5, "10 DRILLING\nIntro.", [table], label="5")], max_tokens=150)
    pieces = table_chunks(chunks)
    assert len(pieces) > 1
    for piece in pieces:
        assert piece.token_count <= 150
        assert piece.text.startswith("Table 10-2: All intercepts")
        assert "| Hole | From (m) | To (m) | Au (g/t) |" in piece.text
        assert piece.section == "10 DRILLING"
    body_rows = [ln for p in pieces for ln in p.text.splitlines() if ln.startswith("| DH-")]
    assert body_rows == [f"| {' | '.join(r)} |" for r in rows]  # every row exactly once, in order
    assert pieces[-1].text.rstrip().endswith("Notes: downhole lengths.")


def test_pipes_in_cells_are_escaped():
    table = Table(None, ("A", "B"), (("x|y", "1"),), None, (0.0, 0.0, 1.0, 1.0), line_index=0)
    (chunk,) = table_chunks(chunk_pages([page(1, "", [table])]))
    assert "| x\\|y | 1 |" in chunk.text


@pytest.mark.parametrize(("max_tokens", "overlap"), [(100, 100), (1, 0), (100, -1)])
def test_invalid_limits_raise(max_tokens, overlap):
    with pytest.raises(ValueError):
        chunk_pages([page(1, "x")], max_tokens=max_tokens, overlap_tokens=overlap)


# --- review findings ---------------------------------------------------------------


def test_footer_with_varying_digit_count_is_boilerplate():
    # "Page 5 of 120" and "Page 105 of 120" must mask to the same line.
    pages = [page(n, f"Section body {'x' * n}\nPage {n} of 120") for n in range(1, 121)]
    assert "Page" not in " ".join(c.text for c in chunk_pages(pages))


def test_headings_win_over_boilerplate():
    # Bare page numbers in the footer mask to "#"; a heading whose first line is "11"
    # also masks to "#" and must not be erased with them.
    pages = [page(n, f"Body text {'y' * n}.\n{n}") for n in range(10, 30)]
    pages[5] = Page(15, None, "11\nSAMPLE PREP\nBody of section.\n15", (), (Heading(0, 2, "11 SAMPLE PREP"),))
    assert "11 SAMPLE PREP" in {c.section for c in chunk_pages(pages)}


def test_a_lone_cover_heading_does_not_disable_numbered_sections():
    pages = [
        Page(1, None, "Technical Report", (), (Heading(0, 1, "Technical Report"),)),
        page(2, "1 SUMMARY\nSummary text."),
        page(3, "2 INTRODUCTION\nIntro text."),
    ]
    sections = [c.section for c in chunk_pages(pages)]
    assert sections == ["1 SUMMARY", "2 INTRODUCTION"]


def test_numbered_heading_may_contain_digits_in_its_title():
    assert is_heading("2.1 Zone 3 Geology")
    assert is_heading("7.4 NI 43-101 Compliance")
    assert not is_heading("Signed by Qualified Persons:")


def test_repeated_mid_page_lines_are_content_in_short_documents():
    # Only the first/last 6 lines of a page can be running headers/footers.
    def body(tag):
        return "\n".join([f"{tag}{i}" for i in range(6)] + ["Notes:"] + [f"{tag}z{i}" for i in range(6)])

    pages = [page(1, body("a")), page(2, body("b"))]
    assert "Notes:" in " ".join(c.text for c in chunk_pages(pages))


def test_empty_input():
    assert chunk_pages([]) == []


def test_sections_come_from_the_parsers_headings():
    text = "11\nSAMPLE PREPARATION\nBody one.\nSampling Methods\nBody two."
    headings = (Heading(0, 2, "11 SAMPLE PREPARATION"), Heading(3, 4, "Sampling Methods"))
    chunks = chunk_pages([Page(1, None, text, (), headings)])
    assert [(c.section, c.text) for c in chunks] == [
        ("11 SAMPLE PREPARATION", "Body one."),
        ("Sampling Methods", "Body two."),
    ]


def test_numbering_regex_is_only_a_fallback():
    # Once the parser found headings in the document, numbered lines are just text
    # (Waihi: "10 Dilution domains" is a list item, not a section).
    headings = (Heading(0, 1, "Intro"), Heading(2, 3, "Scope"), Heading(4, 5, "Methods"))
    pages = [
        Page(1, None, "Intro\nBody.\nScope\nBody.\nMethods\nBody.", (), headings),
        Page(2, None, "10 Dilution domains\nMore.", ()),
    ]
    assert {c.section for c in chunk_pages(pages)} == {"Intro", "Scope", "Methods"}
