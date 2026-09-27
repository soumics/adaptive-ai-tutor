"""Adaptive diagnostic: a short pretest that places the learner on the topic graph. No LLM involved.

Topics are visited in prerequisite order. Within a topic, a **staircase** picks the questions
(multiple choice, so grading is exact): start at difficulty 2; after a correct answer go up, after a
wrong one go down. The decision table (``classify``) is:

    d2 right, d3 right                 -> mastered        (2 questions)
    d2 wrong, d1 wrong                 -> not yet         (2 questions)
    d2 right, d3 wrong, then d2 again  -> developing      (3 questions, either way)
    d2 wrong, d1 right, then d2 again  -> developing if right, else not yet

**Prerequisite gaps shorten the test**: if a topic's prerequisite came out "not yet", the topic gets
a single probe question at difficulty 2. A wrong probe marks it "not yet" straight away (the gap
explains it); a right probe means the learner may know it anyway, so the normal staircase continues.

The placement says, per topic, mastered / developing / not yet, the explanation level the tutor
should use (advanced / intermediate / beginner), and the **starting topic**: the first topic in
learning order that isn't mastered but whose prerequisites are.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from question_gen import Question, QuestionBank
from topic_graph import TopicGraph

STATUS_LEVEL = {"mastered": "advanced", "developing": "intermediate", "not yet": "beginner", "untested": "beginner"}


@dataclass
class TopicResult:
    section_id: str
    answers: list[tuple[str, int, bool]] = field(default_factory=list)   # (qid, difficulty, correct)
    status: str = ""                                                      # "" while in progress
    probe: bool = False                                                   # prerequisite gap: probe first

    @property
    def score(self) -> float:
        w = {1: 1, 2: 2, 3: 3}
        total = sum(w[d] for _, d, _ in self.answers)
        return round(sum(w[d] for _, d, ok in self.answers if ok) / total, 3) if total else 0.0


def classify(answers: list[tuple[int, bool]], probe: bool = False) -> str | None:
    """Status after these (difficulty, correct) answers, or None if another question is needed."""
    seq = [(d, ok) for d, ok in answers]
    if probe:
        if not seq:
            return None
        if not seq[0][1]:
            return "not yet"
        seq = seq[1:]                    # the probe was right: it counts as the staircase's first d2 answer
        seq = [(2, True)] + seq
    if len(seq) < 2:
        return None
    (d1, a), (d2, b) = seq[0], seq[1]
    if a and b:
        return "mastered"
    if not a and not b:
        return "not yet"
    if len(seq) < 3:
        return None
    if a and not b:                      # right at 2, wrong at 3, then another 2
        return "developing"
    return "developing" if seq[2][1] else "not yet"


def next_difficulty(answers: list[tuple[int, bool]], probe: bool = False) -> int:
    seq = ([(2, True)] + answers[1:]) if (probe and answers and answers[0][1]) else answers
    if not seq:
        return 2
    if len(seq) == 1:
        return 3 if seq[0][1] else 1
    return 2                             # the third question is always back at difficulty 2


@dataclass
class Placement:
    topics: dict[str, TopicResult]
    start_topic: str | None
    overall: str
    questions_asked: int

    def level_for(self, section_id: str) -> str:
        return STATUS_LEVEL.get(self.topics[section_id].status, "intermediate")


class Diagnostic:
    def __init__(self, graph: TopicGraph, bank: QuestionBank):
        self.graph, self.bank = graph, bank
        self.order = graph.order()
        self.results = {sid: TopicResult(sid) for sid in self.order}
        self.used: set[str] = set()
        self.current: Question | None = None

    def _topic_in_progress(self) -> str | None:
        return next((sid for sid in self.order if not self.results[sid].status), None)

    def _pick(self, sid: str, difficulty: int) -> Question | None:
        """An unused multiple-choice question for the topic, at the wanted difficulty or the nearest one."""
        for d in sorted((1, 2, 3), key=lambda x: (abs(x - difficulty), -x)):
            for q in self.bank.for_topic(sid, "mcq", d):
                if q.qid not in self.used:
                    return q
        return None

    def next_question(self) -> Question | None:
        """The next question to ask, or None when the diagnostic is complete."""
        while True:
            sid = self._topic_in_progress()
            if sid is None:
                self.current = None
                return None
            r = self.results[sid]
            if not r.answers and not r.probe:
                r.probe = any(self.results[p].status == "not yet" for p in self.graph.prerequisites(sid))
            q = self._pick(sid, next_difficulty([(d, ok) for _, d, ok in r.answers], r.probe))
            if q is None:                 # the bank has run out for this topic
                r.status = classify([(d, ok) for _, d, ok in r.answers], r.probe) or \
                    ("untested" if not r.answers else "developing")
                continue
            self.current = q
            return q

    def answer(self, correct: bool) -> None:
        """Record the result for the question last returned by :meth:`next_question`."""
        q = self.current
        if q is None:
            raise RuntimeError("no question is waiting for an answer")
        self.used.add(q.qid)
        r = self.results[q.section_id]
        r.answers.append((q.qid, q.difficulty, bool(correct)))
        status = classify([(d, ok) for _, d, ok in r.answers], r.probe)
        if status:
            r.status = status
        self.current = None

    def placement(self) -> Placement:
        mastered = {sid for sid, r in self.results.items() if r.status == "mastered"}
        ready = self.graph.ready_topics(mastered)
        frac = len(mastered) / max(1, len(self.order))
        overall = "advanced" if frac >= 2 / 3 else "intermediate" if frac >= 1 / 3 else "beginner"
        return Placement(dict(self.results), ready[0] if ready else None, overall,
                         sum(len(r.answers) for r in self.results.values()))


def run_diagnostic(graph: TopicGraph, bank: QuestionBank, answer_fn) -> Placement:
    """Run a whole diagnostic with ``answer_fn(question) -> bool`` (used by tests and simulations)."""
    d = Diagnostic(graph, bank)
    while (q := d.next_question()) is not None:
        d.answer(answer_fn(q))
    return d.placement()
