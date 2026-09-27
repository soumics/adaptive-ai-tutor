"""Curriculum ingestion: any subject's material (PDF, Markdown, plain text) -> sections + citable chunks.

Built on Project 3's ingestion (``rag_assistant.ingest``, imported, not copied): its text cleaning,
running-header/footer removal and page chunking are reused as they are. What a tutor needs on top
is **structure**. The sections of the material become the topics that the diagnostic, the practice
questions, the review schedule and the progress dashboard are organised by, so this module finds
sections and keeps every chunk inside one section:

* **PDF**: the document's bookmarks (outline), which textbooks normally have; otherwise numbered
  headings such as ``2.3 Solve Equations ...`` at the start of a line; otherwise one section per file.
  Text is extracted in pypdf's layout mode, which keeps the word spacing that plain extraction loses
  on some typeset PDFs. A section that starts part-way down a page is split at its heading.
* **Markdown**: ``#`` is the unit title, ``##`` headings are topics (deeper headings stay inside their
  topic). A line ``Prerequisites: <topic>; <topic>`` records explicit prerequisites.
* **Plain text**: numbered headings, otherwise one section per file.

A chunk never crosses a section or page boundary, so every citation points to one section (and, for
PDFs, one page).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from rag_assistant.ingest import Chunk, Page, chunk_page, clean_text, remove_running_lines

SUFFIXES = (".pdf", ".md", ".markdown", ".txt")
NUMBERED_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2})+)\.?\s+([A-Z][^\n]{3,100})$")
PREREQ_LINE = re.compile(r"^\s*Prerequisites?\s*:\s*(.+)$", re.I | re.M)


@dataclass
class Section:
    section_id: str          # "algebra:s03" -- unique within a curriculum
    number: str              # "2.3" when the material numbers its sections, else ""
    title: str               # "Solve Equations with Variables and Constants on Both Sides"
    doc_id: str
    doc_title: str
    order: int               # position in the curriculum (reading order)
    first_page: int = 0      # PDF pages (1-based) the section spans; 0 for Markdown/text
    last_page: int = 0
    explicit_prereqs: list[str] = field(default_factory=list)   # as written in the material

    @property
    def label(self) -> str:
        return f"{self.number} {self.title}".strip()


@dataclass
class TutorChunk(Chunk):
    """Project 3's chunk plus the section it belongs to."""

    section_id: str = ""
    section_label: str = ""

    def citation(self) -> str:
        page = f", p. {self.page}" if self.page else ""
        return f"{self.title}, §{self.section_label}{page}"


@dataclass
class Curriculum:
    name: str
    sections: list[Section]
    chunks: list[TutorChunk]
    documents: list[str]

    def section(self, section_id: str) -> Section:
        return next(s for s in self.sections if s.section_id == section_id)

    def chunks_for(self, section_id: str) -> list[TutorChunk]:
        return [c for c in self.chunks if c.section_id == section_id]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(name=self.name, documents=self.documents,
                                        sections=[asdict(s) for s in self.sections],
                                        chunks=[asdict(c) for c in self.chunks]), indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Curriculum":
        d = json.loads(Path(path).read_text())
        return cls(d["name"], [Section(**s) for s in d["sections"]], [TutorChunk(**c) for c in d["chunks"]],
                   d["documents"])


# ------------------------------------------------------------------ helpers


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def split_number(title: str) -> tuple[str, str]:
    """'2.3. Solve Equations ...' -> ('2.3', 'Solve Equations ...'); '3. Stability' -> ('3', 'Stability')."""
    m = re.match(r"^\s*(\d{1,2}(?:\.\d{1,2})*)\.?\s+(.+?)\s*\*?$", title)
    return (m.group(1), m.group(2).strip()) if m else ("", title.strip().rstrip("*").strip())


def find_heading(text: str, number: str, title: str) -> int:
    """Character offset where a section heading starts in a page's text, or -1.

    Matches the first words of the title (layout extraction may break a long heading across lines
    or put the number on its own line), preferring a match preceded by the section number.
    """
    words = _norm(title).split()[:5]
    if not words:
        return -1
    pattern = r"[^a-z0-9]+".join(re.escape(w) for w in words)
    low = text.lower()
    best = -1
    for m in re.finditer(pattern, low):
        start = m.start()
        if number:
            before = low[max(0, start - 12):start]
            if re.search(re.escape(number) + r"\.?\s*$", before):
                return low.rfind(number, 0, start)
        if best < 0:
            best = start
    return best


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "doc"


# ------------------------------------------------------------------ PDF


def pdf_outline(reader) -> list[tuple[int, str, int]]:
    """Flattened bookmarks: [(depth, title, 0-based page)]."""
    out: list[tuple[int, str, int]] = []

    def walk(items, depth):
        for it in items:
            if isinstance(it, list):
                walk(it, depth + 1)
                continue
            n = reader.get_destination_page_number(it)
            if n is not None:
                out.append((depth, it.title.strip(), n))
    try:
        walk(reader.outline, 0)
    except Exception:  # malformed outlines exist in the wild: fall back to headings
        return []
    return out


def read_pdf_pages(path: Path) -> tuple[list[str], object]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    texts = []
    for p in reader.pages:
        try:
            raw = p.extract_text(extraction_mode="layout")
        except Exception:  # layout mode can fail on unusual pages; plain mode still gives the words
            raw = p.extract_text() or ""
        texts.append(clean_text(raw or ""))
    return remove_running_lines(texts), reader


def pdf_title(reader, path: Path) -> str:
    meta = getattr(reader, "metadata", None) or {}
    title = (meta.get("/Title") or "").strip() if hasattr(meta, "get") else ""
    return title or path.stem.replace("_", " ")


def pdf_sections(pages: list[str], outline: list[tuple[int, str, int]]) -> list[tuple[str, int, int]]:
    """Topics as (bookmark title, start page index, start char offset) from the outline's leaves.

    Leaves are bookmarks with no deeper bookmark after them before the next bookmark of the same or
    lower depth, i.e. the most detailed level (sections, not chapters). Glossaries, answer keys and
    front matter are skipped.
    """
    skip = re.compile(r"^(glossary|index|answer key|preface|contents|review exercises|practice test|"
                      r"chapter review|key terms|key concepts)", re.I)
    leaves = []
    for i, (depth, title, page) in enumerate(outline):
        nxt = outline[i + 1][0] if i + 1 < len(outline) else -1
        if nxt > depth or skip.match(title):
            continue
        leaves.append((title, page))
    out = []
    for title, page in leaves:
        number, name = split_number(title)
        off = find_heading(pages[page], number, name) if page < len(pages) else -1
        out.append((title, page, max(0, off)))
    return out


def heading_sections(pages: list[str]) -> list[tuple[str, int, int]]:
    """Fallback without bookmarks: numbered headings at the start of a line, in increasing order."""
    out, last = [], None
    for pi, text in enumerate(pages):
        for m in re.finditer(r"(?m)^.*$", text):
            line = m.group(0).strip()
            h = NUMBERED_HEADING.match(line)
            if not h:
                continue
            key = tuple(int(x) for x in h.group(1).split("."))
            if last is not None and key <= last:
                continue            # cross-references and repeated headers go backwards: skip
            last = key
            out.append((line, pi, m.start()))
    return out


def ingest_pdf(path: Path, subject: str, order0: int, chunk_words: int, overlap_words: int):
    pages, reader = read_pdf_pages(path)
    doc_id, doc_title = _slug(path.stem), pdf_title(reader, path)
    marks = pdf_sections(pages, pdf_outline(reader)) or heading_sections(pages) or [(doc_title, 0, 0)]
    sections, chunks = [], []
    for i, (title, page, off) in enumerate(marks):
        # the section runs from its heading to the next section's heading (or the end of the document)
        nxt_page, nxt_off = (marks[i + 1][1], marks[i + 1][2]) if i + 1 < len(marks) else (len(pages) - 1, None)
        number, name = split_number(title)
        sid = f"{subject}:s{order0 + i + 1:02d}"
        last = nxt_page if (nxt_off is None or nxt_off > 0) else max(page, nxt_page - 1)
        sec = Section(sid, number, name, doc_id, doc_title, order0 + i, page + 1, last + 1)
        sections.append(sec)
        for p in range(page, last + 1):
            a = off if p == page else 0
            b = nxt_off if (p == nxt_page and nxt_off is not None) else len(pages[p])
            chunks += _chunk_segment(pages[p][a:b].strip(), sec, p + 1, chunk_words, overlap_words)
    return sections, chunks, doc_title


# ------------------------------------------------------------------ Markdown / text


def ingest_markdown(path: Path, subject: str, order0: int, chunk_words: int, overlap_words: int):
    text = path.read_text(encoding="utf-8", errors="replace")
    doc_id = _slug(path.stem)
    h1 = re.search(r"^#\s+(.+)$", text, re.M)
    doc_title = h1.group(1).strip() if h1 else path.stem.replace("_", " ")
    parts = re.split(r"(?m)^##\s+(.+)$", text)
    # parts = [preamble, title1, body1, title2, body2, ...]
    heads = list(zip(parts[1::2], parts[2::2])) or [(doc_title, text)]
    sections, chunks = [], []
    for i, (title, body) in enumerate(heads):
        number, name = split_number(title.strip())
        prereqs = [p.strip().rstrip(".") for m in PREREQ_LINE.finditer(body) for p in m.group(1).split(";")
                   if p.strip()]
        body = PREREQ_LINE.sub("", body)
        body = re.sub(r"(?m)^#{3,6}\s+", "", body)                 # sub-headings stay as text
        body = re.sub(r"\*\*|__|`", "", body)
        sid = f"{subject}:s{order0 + i + 1:02d}"
        sec = Section(sid, number, name, doc_id, doc_title, order0 + i, explicit_prereqs=prereqs)
        sections.append(sec)
        chunks += _chunk_segment(body.strip(), sec, 0, chunk_words, overlap_words)
    return sections, chunks, doc_title


def ingest_text(path: Path, subject: str, order0: int, chunk_words: int, overlap_words: int):
    text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
    doc_id, doc_title = _slug(path.stem), path.stem.replace("_", " ")
    marks = heading_sections([text]) or [(doc_title, 0, 0)]
    sections, chunks = [], []
    for i, (title, _, off) in enumerate(marks):
        end = marks[i + 1][2] if i + 1 < len(marks) else len(text)
        number, name = split_number(title)
        sec = Section(f"{subject}:s{order0 + i + 1:02d}", number, name, doc_id, doc_title, order0 + i)
        sections.append(sec)
        chunks += _chunk_segment(text[off:end].strip(), sec, 0, chunk_words, overlap_words)
    return sections, chunks, doc_title


def _chunk_segment(text: str, sec: Section, page: int, chunk_words: int, overlap_words: int) -> list[TutorChunk]:
    """Project 3's page chunker applied to one section's text on one page."""
    out = []
    for c in chunk_page(Page(sec.doc_id, sec.doc_title, page, text), chunk_words, overlap_words, min_words=12):
        k = c.chunk_id.rsplit(":c", 1)[1]
        out.append(TutorChunk(f"{sec.section_id}:p{page}:c{k}", c.doc_id, c.title, page, c.text, c.n_words,
                              section_id=sec.section_id, section_label=sec.label))
    return out


# ------------------------------------------------------------------ entry point


def ingest_curriculum(paths: Iterable[str | Path], name: str | None = None, chunk_words: int = 180,
                      overlap_words: int = 30) -> Curriculum:
    """Ingest files (or folders of files) into one :class:`Curriculum`. Files are read in name order."""
    files: list[Path] = []
    for p in (Path(x) for x in paths):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUFFIXES) if p.is_dir() else [p]
    files = [f for f in files if f.suffix.lower() in SUFFIXES and f.name.upper() not in ("SOURCE.MD", "README.MD")]
    if not files:
        raise ValueError(f"no curriculum files found (supported: {', '.join(SUFFIXES)})")
    subject = _slug(name or files[0].parent.name or files[0].stem)
    sections: list[Section] = []
    chunks: list[TutorChunk] = []
    titles = []
    for f in files:
        reader = {".pdf": ingest_pdf, ".md": ingest_markdown, ".markdown": ingest_markdown}.get(f.suffix.lower(),
                                                                                              ingest_text)
        s, c, t = reader(f, subject, len(sections), chunk_words, overlap_words)
        sections += s
        chunks += c
        titles.append(t)
    # drop sections that produced no text at all (e.g. a bookmark to a figure-only page)
    keep = {c.section_id for c in chunks}
    sections = [s for s in sections if s.section_id in keep]
    return Curriculum(name or titles[0], sections, chunks, titles)
