"""Socratic mode: lead the learner to the answer with questions instead of handing it over.

Each turn, the model (given the same numbered course passages as the tutor) reacts briefly to the
learner's last reply and asks **one** leading question. It also reports, as JSON, whether the learner
has now reached the answer and which passage the turn is based on.

The pedagogy's hard rules are enforced in code, not left to the prompt:

* **Every turn cites the material.** If the model's text lacks a valid ``[S#]``, the turn's source is
  appended as a pointer ("Look at [S2].") and the turn is flagged ``cite_added``.
* **The learner isn't left stuck.** After ``max_turns`` exchanges, or when the learner asks for the
  answer twice ("I don't know", "just tell me"), the session switches to a full grounded explanation
  from ``tutor_engine.ask`` (with citations), and ends.
* **Success ends the session** with a short grounded summary, so the learner leaves with the correct,
  cited statement of what they worked out.
* **The model is anchored**: at the start, the tutor writes a grounded, cited reference answer
  (``tutor_engine.ask``) that the learner doesn't see. The Socratic prompt steers towards it and judges
  the learner's replies against it, which keeps a small local model on the learner's actual question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_assistant.retrieve import Hit

from model_provider import ModelProvider, StructuredOutputError
from retrieval import CurriculumIndex
from tutor_engine import CitedSource, ask, format_sources, resolve

SOCRATIC_SYSTEM = """You are a Socratic tutor. Help the learner work out the answer to THEIR question \
themselves, using ONLY the numbered course sources. You are also given a reference answer that the learner cannot \
see: guide the learner towards exactly that answer, and never state it. Each turn:
1. In "feedback", react to the learner's last reply in one or two sentences. If it is right, say clearly that it \
is right. If part of it is wrong, gently say which part, citing the source like [S2]. On the first turn, write a \
one-sentence encouraging start.
2. In "question", ask ONE short leading question that moves the learner one step closer to the reference answer.
3. In "source", name the source the question is based on, e.g. "S2".
4. Set "reached_answer" to true if the learner's last reply contains the main point of the reference answer, \
even in their own words."""

TURN_SCHEMA = {"type": "object", "required": ["feedback", "question", "source", "reached_answer"],
               "properties": {"feedback": {"type": "string"}, "question": {"type": "string"},
                              "source": {"type": "string"}, "reached_answer": {"type": "boolean"}}}

GIVE_UP = re.compile(r"\b(i ?don'?t know|no idea|not sure|give up|just tell me|tell me the answer|what is the answer|"
                     r"show me|i'?m stuck)\b", re.I)


@dataclass
class Turn:
    text: str
    citations: list[CitedSource]
    done: bool = False
    revealed: bool = False           # the session ended with a full explanation instead of the learner's own answer
    cite_added: bool = False         # the model forgot to cite and a pointer to its source was appended


@dataclass
class SocraticSession:
    question: str
    index: CurriculumIndex
    provider: ModelProvider
    topics: list[str] | None = None
    level: str = "intermediate"
    max_turns: int = 4
    k: int = 4
    hits: list[Hit] = field(default_factory=list)
    history: list[tuple[str, str]] = field(default_factory=list)   # (speaker, text)
    give_ups: int = 0
    done: bool = False
    reference: object = None         # the hidden grounded answer (a TutorAnswer)

    def start(self) -> Turn:
        self.reference = ask(self.question, self.index, self.provider, level=self.level, topics=self.topics, k=self.k)
        self.hits = self.reference.sources
        if self.reference.abstained:  # nothing to lead towards: say so, with pointers to the closest material
            self.done = True
            return Turn(self.reference.text, self.reference.citations, done=True, revealed=True)
        return self._turn(None)

    def reply(self, learner_text: str) -> Turn:
        if self.done:
            raise RuntimeError("the session has ended")
        self.history.append(("learner", learner_text))
        if GIVE_UP.search(learner_text):
            self.give_ups += 1
        tutor_turns = sum(1 for s, _ in self.history if s == "tutor")
        if self.give_ups >= 2 or tutor_turns >= self.max_turns:
            return self._explain()
        return self._turn(learner_text)

    # -------------------------------------------------------------- internals
    def _transcript(self) -> str:
        return "\n".join(f"{'Tutor' if s == 'tutor' else 'Learner'}: {t}" for s, t in self.history) or "(none yet)"

    def _turn(self, learner_text: str | None) -> Turn:
        user = (f"Sources:\n\n{format_sources(self.hits)}\n\nLearner's question: {self.question}\n\n"
                f"Reference answer (hidden from the learner):\n{self.reference.text}\n\n"
                f"Conversation so far:\n{self._transcript()}")
        try:
            d = self.provider.json(SOCRATIC_SYSTEM, user, TURN_SCHEMA, temperature=0.3, seed=len(self.history))
        except (StructuredOutputError, ValueError):
            return self._explain()
        if learner_text is not None and d.get("reached_answer"):
            return self._finish(str(d.get("feedback", "")).strip())
        text = f"{str(d.get('feedback', '')).strip()}\n\n{str(d.get('question', '')).strip()}".strip()
        cites, invalid = resolve(text, self.hits)
        added = False
        if not cites or invalid:
            for label in invalid:
                text = text.replace(f"[{label}]", "")
            src = re.fullmatch(r"\[?S(\d+)\]?", str(d.get("source", "")).strip())
            n = int(src.group(1)) if src and 1 <= int(src.group(1)) <= len(self.hits) else 1
            text += f" (Look at [S{n}].)"
            cites, _ = resolve(text, self.hits)
            added = True
        self.history.append(("tutor", text))
        return Turn(text, cites, cite_added=added)

    def _explain(self) -> Turn:
        a = self.reference or ask(self.question, self.index, self.provider, level=self.level, topics=self.topics,
                                  k=self.k)
        text = "Let's go through it together.\n\n" + a.text
        self.history.append(("tutor", text))
        self.done = True
        return Turn(text, a.citations, done=True, revealed=True)

    def _finish(self, feedback: str) -> Turn:
        a = self.reference
        text = (f"{feedback}\n\nYou've got it. Here is the idea in full, from your material:\n\n{a.text}").strip()
        self.history.append(("tutor", text))
        self.done = True
        return Turn(text, a.citations, done=True)
