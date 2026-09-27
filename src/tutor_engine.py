"""Grounded explanations and Q&A: every answer cites the curriculum.

The retrieved passages are numbered sources ``[S1]..[Sk]``, each labelled with its section and page.
The model must use only them and cite a source for every claim. Citations are then parsed and
verified with Project 3's checker (``rag_assistant.generate.parse_citations``): a citation to a source
that wasn't provided is invalid.

**Every answer carries at least one valid citation**, enforced in code, not just in the prompt:

1. the model's answer is accepted if it has valid citations, no invalid ones, and no arithmetic error in
   its worked examples (``check_arithmetic``: every linear equation it writes must appear in the material
   or keep the solution of the equation before it -- solving steps can't change the solution);
2. otherwise the model gets one retry, told exactly what was wrong;
3. if that also fails, the tutor doesn't show an unsourced answer: it shows the relevant passages
   themselves, quoted with their citations (``fallback=True``);
4. if the material doesn't cover the question, the tutor says so and points to the closest
   sections it did find, again with citations.

``tests/test_citations_present.py`` checks all four paths and fails the build if any answer lacks a
citation.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from rag_assistant.generate import NOT_FOUND, is_abstention, parse_citations
from rag_assistant.retrieve import Hit

from grading import normalise, solve_linear
from model_provider import ModelProvider
from retrieval import CurriculumIndex

LEVELS = {
    "beginner": "The learner is new to this topic: use plain words, define every term, go one step at a time, "
                "and include one small worked example if the sources have one.",
    "intermediate": "The learner knows the basics: be concise, focus on the reasoning, and connect ideas.",
    "advanced": "The learner has mastered the prerequisites: be brief and precise, and point out subtleties or "
                "common mistakes the sources mention.",
}

TUTOR_SYSTEM = """You are a patient tutor. You teach ONLY from the numbered sources in the user message, which come \
from the learner's own course material.

Rules:
- Every sentence that states a fact, definition, rule or step must end with the source it comes from, written \
exactly like [S1] or [S2][S3].
- Use only what the sources say. Do not add facts, formulas or examples from general knowledge.
- If the sources do not cover the question, reply with exactly: "{not_found}" and nothing else.
- {level}
- Keep it short: a few sentences or a short list of steps."""

RETRY_NOTE = ("\n\nYour previous reply was rejected because {problem}. Answer again following the rules exactly: "
              "cite only the sources listed, as [S1], [S2], ... at the end of each factual sentence.")


@dataclass
class CitedSource:
    label: str            # "S2"
    citation: str         # "Elementary Algebra ..., §2.3 Solve Equations ..., p. 32"
    section_id: str
    chunk_id: str


@dataclass
class TutorAnswer:
    question: str
    text: str
    citations: list[CitedSource]
    sources: list[Hit]
    grounded: bool         # the model's own answer passed the citation check
    abstained: bool        # the material doesn't cover the question
    fallback: bool         # the model's answers failed the check twice: passages shown instead
    attempts: int
    seconds: float = 0.0
    uncited_sentences: int = 0
    notes: list[str] = field(default_factory=list)

    def formatted(self) -> str:
        refs = "\n".join(f"[{c.label}] {c.citation}" for c in self.citations)
        return f"{self.text}\n\nSources:\n{refs}"


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(f"[S{i}] ({h.chunk.citation()})\n{h.chunk.text}" for i, h in enumerate(hits, 1))


def resolve(text: str, hits: list[Hit]) -> tuple[list[CitedSource], list[str]]:
    """Project 3's citation parser, mapped back to the tutor's chunks (with their sections)."""
    valid, invalid = parse_citations(text, hits)
    by_id = {h.chunk.chunk_id: h.chunk for h in hits}
    return [CitedSource(c.label, by_id[c.chunk_id].citation(), by_id[c.chunk_id].section_id, c.chunk_id)
            for c in valid], invalid


def strip_invalid(text: str, invalid: list[str]) -> str:
    for label in invalid:
        text = text.replace(f"[{label}]", "")
    return re.sub(r"[ \t]+([.,;:])", r"\1", text).strip()


def count_uncited_sentences(text: str) -> int:
    """Sentences (outside lists' lead-ins) that end without a [S#] citation -- reported, not enforced."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 25]
    return sum(1 for s in sentences if not re.search(r"\[S\d+\]\W*$", s) and not s.endswith(":"))


EQUATION = re.compile(r"[-\w\s().+*/\u2212]+=[-\w\s().+*/\u2212]+")


def equations_in(text: str) -> list[tuple[str, object]]:
    """(equation, exact solution) for every one-variable linear equation written in a text, in order."""
    t = text.replace("$", " ").replace("\\cdot", "*").replace("\\times", "*")
    out = []
    for m in EQUATION.finditer(t):
        cand = m.group(0)
        if cand.count("=") != 1:
            continue
        left, right = cand.split("=")
        left = re.split(r"[A-Za-z]{2,}[\s,:]", left)[-1]           # drop words before the maths
        right = re.split(r"[\s,]+[A-Za-z]{2,}", right)[0]          # ... and after it
        eq = f"{left.strip()} = {right.strip().rstrip('.,')}"
        sol = solve_linear(eq)
        if sol is not None:
            out.append((eq, sol))
    return out


def check_arithmetic(answer: str, sources_text: str) -> list[str]:
    """Worked-example check: each linear equation in the answer must either appear in the course material
    or have the same solution as an earlier equation in the answer (solving steps preserve the solution).
    Returns the equations that break this (e.g. "9x = 8x - 6" followed by "17x = -6")."""
    src = normalise(sources_text).replace(" ", "")
    seen: dict[str, list] = {}                       # variable -> [(equation, solution), ...] in order
    problems = []
    for eq, sol in equations_in(answer):
        var = re.findall(r"(?<![a-z])[a-z](?![a-z])", eq.lower())[0]
        earlier = seen.setdefault(var, [])
        given = re.fullmatch(r"\s*[a-z]\s*=\s*-?[\d.]+(?:\s*/\s*\d+)?\s*", eq, re.I) is not None
        if given and not earlier:
            continue                                  # "K = 2" states a value; it isn't a solving step
        in_sources = normalise(eq).replace(" ", "") in src
        if not in_sources and earlier and all(sol != s for _, s in earlier):
            problems.append(f"'{eq}' (solution {sol}) does not follow from '{earlier[-1][0]}' "
                            f"(solution {earlier[-1][1]})")
        earlier.append((eq, sol))
    return problems


def quote_passages(hits: list[Hit], n: int = 2, words: int = 60) -> tuple[str, list[CitedSource]]:
    """Fallback answer: the top passages themselves, shortened, each with its citation."""
    parts, cites = [], []
    for i, h in enumerate(hits[:n], 1):
        w = h.chunk.text.split()
        excerpt = " ".join(w[:words]) + (" ..." if len(w) > words else "")
        parts.append(f"> {excerpt} [S{i}]")
        cites.append(CitedSource(f"S{i}", h.chunk.citation(), h.chunk.section_id, h.chunk.chunk_id))
    return "\n\n".join(parts), cites


def ask(question: str, index: CurriculumIndex, provider: ModelProvider, level: str = "intermediate",
        topics: list[str] | None = None, k: int = 5, min_score: float | None = None) -> TutorAnswer:
    """Answer a learner's question from the curriculum, always with citations.

    ``topics`` restricts retrieval to those sections; ``min_score`` (cosine similarity of the best
    match) short-circuits clearly off-topic questions without calling the model.
    """
    t0 = time.perf_counter()
    hits = index.search(question, k=k, topics=topics)
    if not hits:
        hits = index.search(question, k=k)            # nothing in the chosen topic: widen to the whole course
    if not hits:
        raise ValueError("the curriculum index is empty")
    if min_score is not None and index.best_score(question, topics) < min_score:
        return _not_covered(question, hits, t0, attempts=0)
    system = TUTOR_SYSTEM.format(not_found=NOT_FOUND, level=LEVELS.get(level, LEVELS["intermediate"]))
    user = f"Sources:\n\n{format_sources(hits)}\n\nLearner's question: {question}"
    problem = None
    for attempt in (1, 2):
        result = provider.text(system, user + (RETRY_NOTE.format(problem=problem) if problem else ""))
        text = result.text.strip()
        if is_abstention(text):
            return _not_covered(question, hits, t0, attempts=attempt)
        cites, invalid = resolve(text, hits)
        maths = check_arithmetic(text, " ".join(h.chunk.text for h in hits))
        if cites and not invalid and not maths:
            return TutorAnswer(question, text, cites, hits, True, False, False, attempt, time.perf_counter() - t0,
                               count_uncited_sentences(text))
        problem = ("it cited sources that do not exist: " + ", ".join(invalid)) if invalid else \
            "it contained no [S#] citations" if not cites else \
            "its worked example has an arithmetic error: " + "; ".join(maths)
    text, cites = quote_passages(hits)
    text = ("I couldn't write an explanation I can back up with your material, so here is what it says "
            f"(read the passage and ask me about any step):\n\n{text}")
    return TutorAnswer(question, text, cites, hits, False, False, True, 2, time.perf_counter() - t0,
                       notes=[f"model answers rejected: {problem}"])


def _not_covered(question: str, hits: list[Hit], t0: float, attempts: int) -> TutorAnswer:
    """The material doesn't cover it: say so, and point to the closest sections found (with citations)."""
    seen, cites = set(), []
    for h in hits:
        if h.chunk.section_id in seen:
            continue
        seen.add(h.chunk.section_id)
        cites.append(CitedSource(f"S{len(cites) + 1}", h.chunk.citation(), h.chunk.section_id, h.chunk.chunk_id))
        if len(cites) == 2:
            break
    pointers = " and ".join(f"[{c.label}]" for c in cites)
    text = (f"Your course material doesn't cover this question, so I won't answer it from general knowledge. "
            f"The closest parts of the material are {pointers}.")
    return TutorAnswer(question, text, cites, hits, False, True, False, attempts, time.perf_counter() - t0)
