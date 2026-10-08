"""Fictional NI 43-101-style report used by parser/chunker/pipeline tests."""

import pymupdf

HEADER = "Kiwi Ridge Project - NI 43-101 Technical Report"

TABLE_ROWS = [
    ["Hole", "From (m)", "To (m)", "Au (g/t)"],
    ["DH-001", "45.0", "52.0", "3.10"],
    ["DH-002", "112.0", "118.5", "4.70"],
]


def _pages(variant: str) -> list[list[str]]:
    tonnes = "1.2" if variant == "a" else "1.5"
    return [
        [
            "1 SUMMARY",
            "The Kiwi Ridge Project is a fictional epithermal gold prospect.",
            "This report exists only to exercise automated tests.",
            "2 INTRODUCTION",
            "The report follows the structure of an NI 43-101 technical report.",
            "Drilling and sampling were completed between 2021 and 2023.",
        ],
        [
            "7 GEOLOGICAL SETTING AND MINERALIZATION",
            "7.1 Regional Geology",
            "The project area is underlain by greywacke basement rocks.",
            "The greywacke is overlain by andesitic volcanic flows and tuffs.",
            "Regional faults trend north-east across the greywacke terrane.",
        ],
        [
            "Weathering of the greywacke extends to a depth of thirty metres.",
            "7.2 Mineralization",
            "Gold occurs in quartz-sulphide veins hosted by the andesite.",
            "The quartz-sulphide veins dip steeply to the north-west.",
            "Significant intercepts are listed in the table below.",
        ],
        [
            "14 MINERAL RESOURCE ESTIMATES",
            f"The Indicated Mineral Resource is {tonnes} Mt at 2.3 g/t Au.",
            "The Inferred Mineral Resource is 0.4 Mt at 1.8 g/t Au.",
        ],
    ]


def _draw_table(page: pymupdf.Page, x: float, y: float) -> None:
    col_w, row_h = 110.0, 18.0
    for r, row in enumerate(TABLE_ROWS):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(x + c * col_w, y + r * row_h, x + (c + 1) * col_w, y + (r + 1) * row_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.5)
            page.insert_text((rect.x0 + 3, rect.y1 - 5), cell, fontsize=9)


def build_fixture_pdf(variant: str = "a") -> bytes:
    pages = _pages(variant)
    doc = pymupdf.open()
    for number, lines in enumerate(pages, start=1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 40), HEADER, fontsize=9)
        y = 80.0
        for line in lines:
            page.insert_text((50, y), line, fontsize=10)
            y += 16
        if number == 3:
            _draw_table(page, 50, y + 10)
        page.insert_text((260, 815), f"Page {number} of {len(pages)}", fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data
