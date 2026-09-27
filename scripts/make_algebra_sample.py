"""Reproduce examples/sample_curricula/algebra_unit/ from the OpenStax textbook.

Source: *Elementary Algebra* (OpenStax, 1st edition, 2017), Lynn Marecek and MaryAnne Anthony-Smith,
licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The later 2e is CC BY-NC-SA, so
the 1st edition is used deliberately. Pages 205-256 (Chapter 2, sections 2.1-2.4) are copied
unchanged into a smaller PDF, and the book's bookmarks for those sections are recreated, as a real
textbook PDF would have them.

    python scripts/make_algebra_sample.py /path/to/ElementaryAlgebra-OP_Ft8T0ij.pdf

(The book PDF is 78 MB: https://assets.openstax.org/oscms-prodcms/media/documents/ElementaryAlgebra-OP_Ft8T0ij.pdf)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter

SOURCE_SHA256 = "e4e3fd30a1dcabee9fb120b64b93717ee8b70295ed196cc215eeb3c427564ae9"
FIRST, LAST = 205, 256            # 1-based page numbers in the source PDF (inclusive)
CHAPTER = "Chapter 2. Solving Linear Equations and Inequalities"
OUT = Path(__file__).resolve().parents[1] / "examples" / "sample_curricula" / "algebra_unit" / \
    "Elementary_Algebra_Ch2_Linear_Equations.pdf"


def main(src: str) -> None:
    data = Path(src).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCE_SHA256:
        sys.exit(f"unexpected source PDF (sha256 {digest}); expected the OpenStax 1st-edition file")
    reader = PdfReader(src)
    writer = PdfWriter()
    for i in range(FIRST - 1, LAST):
        writer.add_page(reader.pages[i])
    # recreate the book's bookmarks for the extracted sections (page numbers relative to the new PDF)
    wanted = {}

    def walk(outline):
        for item in outline:
            if isinstance(item, list):
                walk(item)
                continue
            n = reader.get_destination_page_number(item)
            if n is not None and FIRST - 1 <= n < LAST and item.title.startswith("2."):
                wanted[item.title.rstrip("*")] = n - (FIRST - 1)
    walk(reader.outline)
    parent = writer.add_outline_item(CHAPTER, 0)
    for title, page in sorted(wanted.items(), key=lambda kv: kv[1]):
        writer.add_outline_item(title, page, parent=parent)
    writer.add_metadata({"/Title": "Elementary Algebra, Chapter 2 (sections 2.1-2.4)",
                         "/Author": "Lynn Marecek, MaryAnne Anthony-Smith (OpenStax)",
                         "/Subject": "Excerpt, CC BY 4.0, unmodified pages"})
    # pages copied out of a book reference the whole book's shared resources: keep one copy of each
    # identical object and drop the ones no remaining page uses
    writer.compress_identical_objects(remove_duplicates=True, remove_unreferenced=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as f:
        writer.write(f)
    print(f"wrote {OUT} ({LAST - FIRST + 1} pages; sections: {', '.join(wanted)})")
    print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ElementaryAlgebra-OP_Ft8T0ij.pdf")
