import pymupdf

from tests.fixtures.fixture_pdf import HEADER, build_fixture_pdf


def test_fixture_has_four_pages_with_header_and_footer():
    doc = pymupdf.open(stream=build_fixture_pdf(), filetype="pdf")
    assert doc.page_count == 4
    for i, page in enumerate(doc, start=1):
        text = page.get_text()
        assert HEADER in text
        assert f"Page {i} of 4" in text


def test_variants_differ():
    assert build_fixture_pdf("a") != build_fixture_pdf("b")


def test_page_three_has_one_detectable_table():
    doc = pymupdf.open(stream=build_fixture_pdf(), filetype="pdf")
    tables = doc[2].find_tables().tables
    assert len(tables) == 1
    assert tables[0].extract()[2][0].strip() == "DH-002"


def test_same_variant_is_byte_identical():
    assert build_fixture_pdf("a") == build_fixture_pdf("a")
