"""Practice questions written from the curriculum, then checked before a learner ever sees them.

Two kinds, per topic and difficulty (1 = recall, 2 = understanding, 3 = application):

* **Multiple choice**: fully deterministic to grade.
* **Short answer** with a rubric: 2-4 points, each with keywords, and a model answer. If the answer
  is a single linear equation's solution (algebra), it is graded exactly instead (see ``grading.py``).

The LLM writes candidates as JSON from the topic's passages. Code then decides what is kept:

1. **Structure checks**: four distinct options, a valid correct index, no "all/none of the above",
   a question that isn't a copy of an option, a source label that exists.
2. **Grounding checks**: every rubric point keeps only keywords that occur in the source passage, and
   only points that the question's own model answer meets are kept (a rubric the reference answer
   fails would be unfair to the learner).
3. **Independent answer check** (``verify=True``): the model answers each multiple-choice question
   again, from the passages, without seeing the key; a question is kept only if it agrees. For
   short answers that are equation solutions, the solution is checked exactly by ``solve_linear``.
4. **Option order** is shuffled with a seeded RNG, so the correct letter isn't biased (LLMs favour "A").

Kept questions go into a JSON **question bank** per curriculum, so the diagnostic, quizzes and
reviews reuse them and a demo is repeatable.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from grading import contains_phrase, normalise, parse_number, solve_linear
from model_provider import ModelProvider, StructuredOutputError
from retrieval import CurriculumIndex

DIFFICULTY = {1: "recall: a fact, definition or rule stated in the passages",
              2: "understanding: why or when something holds, or what a term means in context",
              3: "application: use a rule or method from the passages on a small new case"}

MCQ_SCHEMA = {"type": "object", "required": ["questions"], "properties": {"questions": {"type": "array", "items": {
    "type": "object", "required": ["question", "options", "correct_index", "explanation", "source"],
    "properties": {"question": {"type": "string"},
                   "options": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
                   "correct_index": {"type": "integer", "minimum": 0, "maximum": 3},
                   "explanation": {"type": "string"}, "source": {"type": "string"}}}}}}

SHORT_SCHEMA = {"type": "object", "required": ["questions"], "properties": {"questions": {"type": "array", "items": {
    "type": "object", "required": ["question", "model_answer", "rubric", "source"],
    "properties": {"question": {"type": "string"}, "model_answer": {"type": "string"},
                   "rubric": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
                       "type": "object", "required": ["point", "keywords"],
                       "properties": {"point": {"type": "string"},
                                      "keywords": {"type": "array", "items": {"type": "string"}}}}},
                   "source": {"type": "string"}}}}}}

ANSWER_SCHEMA = {"type": "object", "required": ["answer_index"],
                 "properties": {"answer_index": {"type": "integer", "minimum": 0, "maximum": 3}}}

GEN_SYSTEM = """You write practice questions for a learner, using ONLY the numbered course passages given. \
Every question must be answerable from the passages alone, and its correct answer must be stated or directly \
implied by them. Name the passage each question is based on in "source" (e.g. "S2"). Write clear, self-contained \
questions (never refer to "the passage" or "the text"). Do not use "all of the above" or "none of the above"."""


@dataclass
class RubricPoint:
    point: str
    keywords: list[str]
    weight: float = 1.0


@dataclass
class Question:
    qid: str
    section_id: str
    kind: str                     # "mcq" | "short"
    difficulty: int
    prompt: str
    options: list[str] = field(default_factory=list)
    answer_index: int = -1
    explanation: str = ""
    model_answer: str = ""
    rubric: list[RubricPoint] = field(default_factory=list)
    exact_answer: str = ""        # e.g. "3/2" when the answer is an equation's solution, checked exactly
    source_chunk_id: str = ""
    citation: str = ""
    verified: bool = False


def passages_block(chunks) -> str:
    return "\n\n".join(f"[S{i}] ({c.citation()})\n{c.text}" for i, c in enumerate(chunks, 1))


def _qid(section_id: str, text: str) -> str:
    return section_id + ":" + hashlib.sha1(normalise(text).encode()).hexdigest()[:10]


def _source(label: str, chunks):
    m = re.fullmatch(r"\[?S(\d+)\]?", label.strip())
    if not m or not 1 <= int(m.group(1)) <= len(chunks):
        return None
    return chunks[int(m.group(1)) - 1]


BANNED = re.compile(r"\b(all|none|both) of the (above|options|answers)\b", re.I)
META = re.compile(r"\b(the|this) (passage|text|excerpt|source)s?\b|\bS\d+\b|\[S\d+\]", re.I)
VISUAL = re.compile(r"\b(?:figure|fig\.|table|diagram|illustration|picture|image)\s*\d|\bthe (?:figure|diagram|"
                    r"illustration|picture|image|graph) (?:above|below|shown)", re.I)
OPTION_PREFIX = re.compile(r"^\s*(?:\(?[A-Da-d]\)|[A-Da-d][.:]|\(?[1-4]\))\s+")
COMMON = set("""a an and are as at be by can for from has have how in is it its of on or that the this to was were
what when which with will would should could may might also more less most very usually often better best good
bad same other some any each every only just than then there these those they them their your you we our
""".split())


def clean_options(item: dict) -> None:
    """Remove letter prefixes the model sometimes adds ("B) ...") -- the UI adds its own letters."""
    item["options"] = [OPTION_PREFIX.sub("", str(o)).strip() for o in item.get("options", [])]


META_PHRASES = [
    re.compile(r"^\s*(?:according to|based on|as (?:described|stated|explained) in|in|from) (?:(?:the|this) "
               r"(?:passage|text|excerpt|source|material)s?|\[?S\d+\]?)\s*,?\s*", re.I),
    re.compile(r"\s*,?\s*(?:as|that is|which is)? ?(?:described|stated|explained|mentioned|given|shown) in "
               r"(?:the|this) (?:passage|text|excerpt|source|material)s?", re.I),
    re.compile(r"\s+(?:from|in|see|per|of) (?:the )?(?:\[?S\d+\]?|source|passage|text)(?![a-z])", re.I),
    re.compile(r"\s*\(?\[?S\d+\]?\)?(?![a-z])"),
]


def repair_meta(question: str) -> str:
    """Remove "According to the text," / "described in the passage" / "[S2]" from a question, so a good
    question isn't thrown away for how it was phrased; the checks then run on the repaired text."""
    q = question
    for pat in META_PHRASES:
        q = pat.sub(" " if pat is not META_PHRASES[0] else "", q)
    q = re.sub(r"\s+([?.,])", r"\1", re.sub(r"\s{2,}", " ", q)).strip()
    return q[:1].upper() + q[1:] if q else q


def clean_explanation(text: str) -> str:
    """Explanations are shown with the question's citation, so the model's own "[S2]" / "Source S2 states:"
    references are removed rather than shown as dangling labels."""
    t = re.sub(r"\b(?:the )?source \[?S\d+\]?\s*(?:states|says|explains|notes|shows)\s*:?\s*", "The material says: ",
               text, flags=re.I)
    t = re.sub(r"\s*\(?\[?\bS\d+\b\]?\)?", "", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def check_mcq(item: dict, chunks) -> str | None:
    """Reason the candidate is rejected, or None if it passes the structure checks."""
    clean_options(item)
    item["question"] = repair_meta(str(item.get("question", "")))
    item["explanation"] = clean_explanation(str(item.get("explanation", "")))
    opts = [str(o).strip() for o in item.get("options", [])]
    if len(opts) != 4 or any(not o for o in opts):
        return "needs four non-empty options"
    if len({normalise(o) for o in opts}) != 4:
        return "options are not distinct"
    if not 0 <= int(item.get("correct_index", -1)) <= 3:
        return "invalid correct_index"
    q = str(item.get("question", "")).strip()
    if len(q) < 12:
        return "question too short"
    if any(BANNED.search(o) for o in opts):
        return "uses all/none of the above"
    if META.search(q) or any(META.search(o) for o in opts):
        return "refers to 'the passage' or a source label instead of being self-contained"
    if VISUAL.search(q) or any(VISUAL.search(o) for o in opts):
        return "depends on a figure or table the learner can't see"
    key = normalise(opts[int(item["correct_index"])])
    if len(key) >= 8 and re.search(rf"(?<!\w){re.escape(key)}(?!\w)", normalise(q)):
        return "the correct option is written in the question"
    if _source(str(item.get("source", "")), chunks) is None:
        return "unknown source label"
    eq = find_equation(q)
    values = [parse_number(o) for o in opts]
    if eq is not None and sum(v is not None for v in values) >= 3:   # a numeric answer to an equation: check it
        sol = solve_linear(eq)
        if sol is not None:
            if values[int(item["correct_index"])] != sol:
                return "the keyed option is not the equation's solution"
            if sum(v == sol for v in values) != 1:
                return "more than one option equals the solution"
    return None


SOLVE_INTENT = re.compile(r"\b(solve|solution|solving|find the value|value of [a-z]\b|what is [a-z]\b|"
                          r"satisf|root of)", re.I)
TRIVIAL = re.compile(r"^\s*(?:[a-z]\s*=\s*-?[\d.]+(?:/\d+)?|-?[\d.]+(?:/\d+)?\s*=\s*[a-z])\s*$", re.I)


def find_equation(text: str) -> str | None:
    """A one-variable linear equation the question asks to solve ('Solve: 3x + 4 = 10'), if there is one.

    Given values such as "K = 2" or "at t = 0" are not equations to solve: a question only counts if it
    asks for a solution and the equation is more than a variable set to a number.
    """
    if not SOLVE_INTENT.search(text):
        return None
    for m in re.finditer(r"[-\w\s().+*/\u2212]*=[-\w\s().+*/\u2212]*", text):
        cand = m.group(0).strip(" .?").lower()
        # keep only the part that looks like maths: cut off words before and sentences after the equation
        left, right = cand.split("=", 1)
        left = re.split(r"[a-z]{2,}\s", left)[-1]
        right = re.split(r"\s[a-z]{2,}|\.\s", right)[0].rstrip(".")
        eq = f"{left.strip()} = {right.strip()}"
        if not TRIVIAL.match(eq) and solve_linear(eq) is not None:
            return eq
    return None


def useful_keyword(k: str) -> bool:
    """A keyword must carry meaning: not a common word, and at least 4 letters if it is a single word."""
    words = normalise(k).split()
    content = [w for w in words if w not in COMMON]
    return bool(content) and (len(words) > 1 or len(content[0]) >= 4)


def grounded_rubric(rubric: list[dict], source_text: str) -> list[RubricPoint]:
    """Keep only meaningful keywords that occur in the source passage; drop points left with none."""
    src = normalise(source_text)
    out = []
    for r in rubric:
        point = re.sub(r'^[\s:;,"\'-]+|[\s"\']+$', "", str(r.get("point", "")))
        kws = [k.strip() for k in r.get("keywords", []) if k.strip() and useful_keyword(k) and normalise(k) in src]
        if kws and len(point) > 8:
            out.append(RubricPoint(point, sorted(set(kws), key=kws.index)))
    return out


def _shuffle(item: dict, rng: random.Random) -> tuple[list[str], int]:
    opts = [str(o).strip() for o in item["options"]]
    correct = opts[int(item["correct_index"])]
    rng.shuffle(opts)
    return opts, opts.index(correct)


def verify_mcq(q: Question, chunks, provider: ModelProvider, seed: int = 0) -> bool:
    """Ask the model the question again, from the passages, without the key; keep it only if it agrees."""
    letters = "ABCD"
    user = (f"Passages:\n\n{passages_block(chunks)}\n\nQuestion: {q.prompt}\n" +
            "\n".join(f"{letters[i]}) {o}" for i, o in enumerate(q.options)) +
            "\n\nUsing only the passages, which option is correct? Give its index (0 for A, 1 for B, ...).")
    try:
        d = provider.json("You answer multiple-choice questions strictly from the given passages.", user,
                          ANSWER_SCHEMA, temperature=0.0, seed=seed)
    except (StructuredOutputError, ValueError):
        return False
    return int(d.get("answer_index", -1)) == q.answer_index


def generate_mcq(index: CurriculumIndex, section_id: str, provider: ModelProvider, difficulty: int, n: int = 2,
                 seed: int = 0, verify: bool = True, log: list | None = None) -> list[Question]:
    chunks = index.topic_passages(section_id)
    topic = index.cur.section(section_id).label
    user = (f"Topic: {topic}\n\nPassages:\n\n{passages_block(chunks)}\n\nWrite {n} multiple-choice questions at "
            f"difficulty {difficulty} ({DIFFICULTY[difficulty]}). Each has exactly 4 options, one correct.")
    data = provider.json(GEN_SYSTEM, user, MCQ_SCHEMA, temperature=0.7, seed=seed)
    rng = random.Random(f"{section_id}:{difficulty}:{seed}")
    out = []
    for item in data.get("questions", []):
        why = check_mcq(item, chunks)
        if why:
            (log if log is not None else []).append(("rejected", "mcq", why, item.get("question", "")))
            continue
        src = _source(item["source"], chunks)
        opts, ans = _shuffle(item, rng)
        q = Question(_qid(section_id, item["question"]), section_id, "mcq", difficulty, item["question"].strip(),
                     opts, ans, item.get("explanation", "").strip(), source_chunk_id=src.chunk_id,
                     citation=src.citation())
        if verify:
            q.verified = verify_mcq(q, chunks, provider, seed)
            if not q.verified:
                (log if log is not None else []).append(("rejected", "mcq", "failed the independent answer check",
                                                         q.prompt))
                continue
        out.append(q)
    return out


def generate_short(index: CurriculumIndex, section_id: str, provider: ModelProvider, difficulty: int, n: int = 1,
                   seed: int = 0, log: list | None = None) -> list[Question]:
    chunks = index.topic_passages(section_id)
    topic = index.cur.section(section_id).label
    user = (f"Topic: {topic}\n\nPassages:\n\n{passages_block(chunks)}\n\nWrite {n} short-answer questions at "
            f"difficulty {difficulty} ({DIFFICULTY[difficulty]}). For each, give a model answer of one or two "
            "sentences and a rubric of 2-4 points a good answer must contain. For each rubric point list 1-4 "
            "keywords or short phrases copied exactly from the passages that show the point was made. If the "
            "question asks to solve an equation, the model answer must be the solution, e.g. \"x = 5\".")
    data = provider.json(GEN_SYSTEM, user, SHORT_SCHEMA, temperature=0.7, seed=seed)
    out = []
    for item in data.get("questions", []):
        src = _source(str(item.get("source", "")), chunks)
        q_text = repair_meta(str(item.get("question", "")).strip())
        if src is None or len(q_text) < 12 or META.search(q_text) or VISUAL.search(q_text):
            (log if log is not None else []).append(("rejected", "short", "bad source or question", q_text))
            continue
        exact = ""
        eq = find_equation(q_text)
        if eq is not None and parse_number(item["model_answer"]) is not None:   # a numeric solution: check it
            sol, given = solve_linear(eq), parse_number(item["model_answer"])
            if given is None or given != sol:
                (log if log is not None else []).append(("rejected", "short", "model answer is not the solution",
                                                         q_text))
                continue
            exact = str(sol)
        rubric = grounded_rubric(item.get("rubric", []), " ".join(c.text for c in chunks))
        # self-consistency: the model answer must earn every point of its own rubric, otherwise the rubric
        # asks for something the reference answer doesn't say -- those points are dropped
        rubric = [r for r in rubric if any(contains_phrase(item["model_answer"], k) for k in r.keywords)]
        if not rubric and not exact:
            (log if log is not None else []).append(("rejected", "short", "rubric not grounded in the passages "
                                                     "or not met by the model answer", q_text))
            continue
        out.append(Question(_qid(section_id, q_text), section_id, "short", difficulty, q_text,
                            model_answer=item["model_answer"].strip(), rubric=rubric, exact_answer=exact,
                            source_chunk_id=src.chunk_id, citation=src.citation(), verified=bool(exact)))
    return out


# ------------------------------------------------------------------ question bank


def similar(a: str, b: str, threshold: float = 0.6) -> bool:
    """Near-duplicate questions: Jaccard overlap of their content words."""
    wa = {w for w in normalise(a).split() if w not in COMMON and len(w) > 2}
    wb = {w for w in normalise(b).split() if w not in COMMON and len(w) > 2}
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= threshold


@dataclass
class QuestionBank:
    curriculum: str
    questions: list[Question]
    generator: str = ""
    log: list = field(default_factory=list)

    def for_topic(self, section_id: str, kind: str | None = None, difficulty: int | None = None) -> list[Question]:
        return [q for q in self.questions if q.section_id == section_id and (kind is None or q.kind == kind)
                and (difficulty is None or q.difficulty == difficulty)]

    def get(self, qid: str) -> Question:
        return next(q for q in self.questions if q.qid == qid)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(curriculum=self.curriculum, generator=self.generator,
                                        questions=[asdict(q) for q in self.questions], log=self.log), indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "QuestionBank":
        d = json.loads(Path(path).read_text())
        qs = []
        for q in d["questions"]:
            q["rubric"] = [RubricPoint(**r) for r in q.get("rubric", [])]
            qs.append(Question(**q))
        return cls(d["curriculum"], qs, d.get("generator", ""), d.get("log", []))


def build_bank(index: CurriculumIndex, provider: ModelProvider, mcq_per_level: int = 2, short_per_topic: int = 2,
               seed: int = 7, verify: bool = True, progress=None) -> QuestionBank:
    """Questions for every topic: ``mcq_per_level`` multiple-choice at each difficulty, plus short answers."""
    bank = QuestionBank(index.cur.name, [], provider.name)
    seen = set()
    topics = [s.section_id for s in index.cur.sections]
    for ti, sid in enumerate(topics):
        if progress:
            progress(ti / len(topics), f"Writing questions for {index.cur.section(sid).label}")
        for level in (1, 2, 3):
            for attempt in range(2):          # a second try if checks rejected too many
                have = len(bank.for_topic(sid, "mcq", level))
                if have >= mcq_per_level:
                    break
                try:
                    new = generate_mcq(index, sid, provider, level, mcq_per_level - have + 1, seed + attempt,
                                       verify, bank.log)
                except (StructuredOutputError, ValueError) as e:
                    bank.log.append(("error", "mcq", str(e)[:200], sid))
                    continue
                for q in new:
                    if q.qid in seen or len(bank.for_topic(sid, "mcq", level)) >= mcq_per_level:
                        continue
                    if any(similar(q.prompt, o.prompt) for o in bank.for_topic(sid)):
                        bank.log.append(("rejected", "mcq", "near-duplicate of a kept question", q.prompt))
                        continue
                    seen.add(q.qid)
                    bank.questions.append(q)
        try:
            for q in generate_short(index, sid, provider, 2, short_per_topic, seed, bank.log):
                if q.qid not in seen and not any(similar(q.prompt, o.prompt) for o in bank.for_topic(sid, "short")):
                    seen.add(q.qid)
                    bank.questions.append(q)
        except (StructuredOutputError, ValueError) as e:
            bank.log.append(("error", "short", str(e)[:200], sid))
    return bank
