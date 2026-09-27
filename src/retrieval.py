"""Retrieval over a curriculum: Project 3's hybrid retriever, plus per-topic filtering and persistence.

``rag_assistant.retrieve.Retriever`` (BM25 + dense FAISS search merged with Reciprocal Rank Fusion)
is used unchanged. The tutor adds:

* **topic filtering**: explanations for a topic, practice questions and Socratic hints search only
  that topic's chunks, so a question about section 2.3 is answered from section 2.3;
* **an index per curriculum** saved next to the curriculum (chunks with their sections, vectors,
  and a fingerprint of the source files, so a changed file triggers a rebuild).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from rag_assistant.embeddings import Embedder, make_embedder
from rag_assistant.ingest import file_fingerprint
from rag_assistant.retrieve import Hit, Retriever

from curriculum_ingest import Curriculum, TutorChunk, ingest_curriculum

DEFAULT_EMBEDDER = "ollama:nomic-embed-text"


def embed_text(c: TutorChunk) -> str:
    """What gets embedded: document and section headings give short chunks their context."""
    return f"{c.title} | {c.section_label}\n{c.text}"


class CurriculumIndex:
    def __init__(self, cur: Curriculum, vectors: np.ndarray | None, embedder: Embedder | None, meta: dict | None = None):
        self.cur, self.vectors, self.embedder = cur, vectors, embedder
        self.meta = meta or {}
        self.retriever = Retriever(cur.chunks, vectors, embedder)
        self._by_topic: dict[tuple[str, ...], Retriever] = {}

    def _subset(self, topics: tuple[str, ...]) -> Retriever:
        if topics not in self._by_topic:
            idx = [i for i, c in enumerate(self.cur.chunks) if c.section_id in topics]
            if not idx:
                raise ValueError(f"no material for topics {topics}")
            vec = None if self.vectors is None else self.vectors[idx]
            self._by_topic[topics] = Retriever([self.cur.chunks[i] for i in idx], vec, self.embedder)
        return self._by_topic[topics]

    def search(self, query: str, k: int = 5, topics: list[str] | None = None, method: str = "hybrid") -> list[Hit]:
        r = self._subset(tuple(sorted(topics))) if topics else self.retriever
        if method != "bm25" and r.index is None:
            method = "bm25"
        return r.search(query, k=k, method=method)

    def best_score(self, query: str, topics: list[str] | None = None) -> float:
        r = self._subset(tuple(sorted(topics))) if topics else self.retriever
        return r.best_dense_score(query) if r.index is not None else 0.0

    def topic_passages(self, section_id: str, max_words: int = 900) -> list[TutorChunk]:
        """A topic's chunks in reading order, up to ``max_words`` (spread across the topic) -- the material
        practice questions are written from."""
        chunks = self.cur.chunks_for(section_id)
        if sum(c.n_words for c in chunks) <= max_words:
            return chunks
        n = max(1, round(len(chunks) * max_words / sum(c.n_words for c in chunks)))
        picks = sorted({round(i * (len(chunks) - 1) / max(1, n - 1)) for i in range(n)})
        return [chunks[i] for i in picks]

    # -------------------------------------------------------------- persistence
    def save(self, index_dir: str | Path) -> Path:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        self.cur.save(index_dir / "curriculum.json")
        if self.vectors is not None:
            np.save(index_dir / "vectors.npy", self.vectors)
        (index_dir / "meta.json").write_text(json.dumps(self.meta, indent=2))
        return index_dir

    @classmethod
    def load(cls, index_dir: str | Path, base_url: str = "http://localhost:11434") -> "CurriculumIndex":
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text())
        cur = Curriculum.load(index_dir / "curriculum.json")
        vec_path = index_dir / "vectors.npy"
        embedder = make_embedder(meta["embedder_spec"], base_url) if meta.get("embedder_spec") else None
        return cls(cur, np.load(vec_path) if vec_path.exists() else None, embedder, meta)


def build_index(paths: list[str | Path], index_dir: str | Path, name: str | None = None,
                embedder_spec: str | None = DEFAULT_EMBEDDER, base_url: str = "http://localhost:11434") -> CurriculumIndex:
    """Ingest curriculum files, embed every chunk and save the index. ``embedder_spec=None`` builds a
    keyword-only (BM25) index, which needs no embedding model."""
    files = [Path(p) for p in paths]
    t0 = time.perf_counter()
    cur = ingest_curriculum(files, name=name)
    t1 = time.perf_counter()
    embedder = make_embedder(embedder_spec, base_url) if embedder_spec else None
    vectors = embedder.embed_documents([embed_text(c) for c in cur.chunks]) if embedder else None
    t2 = time.perf_counter()
    sources = [f for p in files for f in (sorted(p.rglob("*")) if p.is_dir() else [p]) if f.is_file()]
    meta = dict(name=cur.name, embedder_spec=embedder_spec, n_chunks=len(cur.chunks), n_topics=len(cur.sections),
                documents=cur.documents, source_fingerprint=file_fingerprint(sources),
                ingest_seconds=round(t1 - t0, 2), embed_seconds=round(t2 - t1, 2))
    idx = CurriculumIndex(cur, vectors, embedder, meta)
    idx.save(index_dir)
    return idx


def load_or_build(paths: list[str | Path], index_dir: str | Path, **kw) -> CurriculumIndex:
    """Reuse a saved index if its source files are unchanged, otherwise rebuild it."""
    index_dir = Path(index_dir)
    files = [Path(p) for p in paths]
    sources = [f for p in files for f in (sorted(p.rglob("*")) if p.is_dir() else [p]) if f.is_file()]
    meta_path = index_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("source_fingerprint") == file_fingerprint(sources) and \
                meta.get("embedder_spec") == kw.get("embedder_spec", DEFAULT_EMBEDDER):
            return CurriculumIndex.load(index_dir, kw.get("base_url", "http://localhost:11434"))
    return build_index(files, index_dir, **kw)
