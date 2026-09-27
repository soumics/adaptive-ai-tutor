# Architecture

## Flow

```
curriculum files (PDF / Markdown / text)
   │  curriculum_ingest.py   sections from PDF bookmarks / numbered headings / Markdown headings;
   │                         chunks never cross a section or page       (text cleaning, header removal and
   │                                                                    chunking: Project 3's rag_assistant.ingest)
   ▼
Curriculum ──► topic_graph.py      topics + prerequisites (explicit > "see Section x.y" > reading order)
   │
   ├─► retrieval.py        Project 3's hybrid retriever (BM25 + FAISS dense, Reciprocal Rank Fusion),
   │                       per-topic filtering, one saved index per curriculum
   │
   ├─► question_gen.py     LLM writes candidate questions from a topic's passages → code checks them
   │                       (structure, grounding, exact maths, independent re-answer) → question bank (JSON)
   │
   ├─► diagnostic.py       staircase per topic in prerequisite order → placement + starting topic
   ├─► tutor_engine.py     cited explanations (Project 3's citation checker) with retry / quoted-passage fallback
   ├─► socratic.py         leading questions anchored on a hidden cited reference answer
   ├─► grading.py          exact (MCQ, equation solutions) or rubric keywords + LLM quotes → score → SM-2 quality
   ├─► spaced_repetition.py  SM-2 scheduling
   └─► progress_tracker.py   SQLite: attempts, review cards, placement, mastery → dashboard
                             app.py (Gradio) and examples/run_demo.py drive the whole flow
```

## What the LLM does, and what it doesn't

| Decision | Made by | Why |
|---|---|---|
| Which sections exist, what depends on what | code (document structure) | deterministic and inspectable |
| Explanation wording, Socratic questions | LLM | natural language is what LLMs are for |
| Whether an explanation is allowed to be shown | code (citation check) | the model can't vouch for itself |
| Practice question drafts, rubric drafts | LLM | |
| Whether a question is kept | code (+ an independent LLM re-answer for MCQs) | a wrong key teaches something wrong |
| Correctness of equation answers | code (exact rational solver) | LLMs make arithmetic slips |
| Credit for a free-text rubric point the keywords missed | LLM, **only with a verified quote** | the quote must really be in the learner's answer |
| Scores, SM-2 quality, review intervals, mastery, placement | code | must be reproducible and testable |

## Reuse of Project 3

Project 3 ([llm-rag-assistant](https://github.com/soumics/llm-rag-assistant)) is installed as a package
pinned to commit `109044d` (see `requirements.txt`) and imported, not copied:

| Project 3 module | Used for |
|---|---|
| `rag_assistant.ingest` | `clean_text`, `remove_running_lines`, `chunk_page`, `Chunk`, `file_fingerprint` |
| `rag_assistant.embeddings` | `make_embedder` (Ollama embedders; hashing embedder) |
| `rag_assistant.retrieve` | `Retriever` (BM25 + FAISS + RRF), `Hit` |
| `rag_assistant.generate` | `OllamaLLM`, `ClaudeLLM`, `make_llm`, `parse_citations`, `is_abstention`, `NOT_FOUND` |

What this project adds on top: section structure and per-section chunks (`TutorChunk` extends
Project 3's `Chunk`), per-topic retrieval, structured (JSON) generation, and everything pedagogical.

## Design decisions

**Sections are the unit of everything.** Topics, prerequisites, question banks, placement, review
cards and mastery are all per section, so the tutor's structure is the curriculum's own structure,
and a new subject needs no code or configuration: only its files.

**Citations are enforced, not requested.** The prompt asks for `[S#]` citations, but an answer is only
shown if the parser finds valid ones. After one failed retry the tutor shows the retrieved passages
themselves, quoted with their citations, instead of an unsourced answer. Out-of-scope questions are
declined with pointers to the nearest sections. `tests/test_citations_present.py` covers every path.

**Question quality is checked by code, not trusted.** In the demo builds the checks rejected 19 of
84 candidates, about one in four (reasons in `examples/eval_results.json`): questions that weren't
self-contained, answer keys that were wrong (caught by the exact solver or by the independent
re-answer), ungrounded rubrics and near-duplicates.

**The diagnostic is short because the graph is used.** A missing prerequisite turns a dependent topic's
assessment into a single probe question, so a beginner on a 5-topic unit answers 6 questions, not 10+.

**Local first.** Everything runs on one laptop with Ollama (`qwen3.5:4b`, `nomic-embed-text`); the
Anthropic back-end is a configuration option for better language quality at a cost.

## Limits

- **Text only.** Equations drawn as graphics in a PDF (common in maths textbooks, including the
  OpenStax sample) aren't extracted; the tutor sees the surrounding words, not the formulas. OCR or a
  math-aware PDF parser would be needed.
- **Structure detection needs structure.** Without bookmarks, numbered headings or Markdown headings,
  a document becomes one topic.
- **Prerequisites are simple.** Explicit lists, cross-references and reading order; no inference from
  the content itself.
- **Explanations aren't fact-checked beyond citations and equation steps.** A cited sentence can still
  paraphrase its source imperfectly. Linear-equation steps in worked examples are checked exactly;
  other arithmetic the small model adds (e.g. a percentage from a formula) is not.
- **Socratic quality depends on the model.** With a 4B local model the leading questions are
  reasonable but its judgement of a learner's partly-right reply is inconsistent.
- **Simulated learners.** The diagnostic's placement accuracy is measured with simulated response
  patterns; no learning outcomes with real students have been measured.
