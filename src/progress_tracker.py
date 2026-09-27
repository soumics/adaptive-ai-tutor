"""Per-learner progress, stored locally in SQLite: attempts, review cards, per-topic mastery.

* Every graded answer (diagnostic, practice, review) is an **attempt**.
* Every question a learner has answered becomes an SM-2 **card**; the answer's quality schedules its
  next review (``spaced_repetition.review``).
* **Mastery** of a topic is a recency-weighted success rate with a neutral prior:
  ``(1 + Σ wᵢ·scoreᵢ) / (2 + Σ wᵢ)`` over the topic's attempts, newest first, with wᵢ = 0.85^i.
  It starts at 0.5 with no evidence and moves towards the learner's recent performance, so one lucky
  or unlucky answer doesn't swing it far. Status: *mastered* at >= 0.8 with at least 3 attempts,
  *developing* at >= 0.5, otherwise *needs work*; *not started* without attempts.

The dashboard reads everything from here, so progress survives restarts of the app.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from spaced_repetition import Card, review

DECAY = 0.85
SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, learner TEXT, curriculum TEXT, section_id TEXT, qid TEXT,
    kind TEXT, score REAL, quality INTEGER, source TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS cards (
    learner TEXT, curriculum TEXT, qid TEXT, section_id TEXT, ef REAL, repetitions INTEGER,
    interval INTEGER, due TEXT, PRIMARY KEY (learner, curriculum, qid));
CREATE TABLE IF NOT EXISTS placements (
    learner TEXT, curriculum TEXT, section_id TEXT, status TEXT, ts TEXT,
    PRIMARY KEY (learner, curriculum, section_id));
"""


@dataclass
class TopicProgress:
    section_id: str
    label: str
    mastery: float
    attempts: int
    status: str
    placement: str
    due_reviews: int
    next_review: date | None


def mastery_from(scores_newest_first: list[float], decay: float = DECAY) -> float:
    w = [decay ** i for i in range(len(scores_newest_first))]
    return round((1 + sum(wi * s for wi, s in zip(w, scores_newest_first))) / (2 + sum(w)), 3)


def status_from(mastery: float, attempts: int) -> str:
    if attempts == 0:
        return "not started"
    if mastery >= 0.8 and attempts >= 3:
        return "mastered"
    return "developing" if mastery >= 0.5 else "needs work"


class ProgressTracker:
    def __init__(self, db_path: str | Path = ":memory:"):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.executescript(SCHEMA)

    # -------------------------------------------------------------- writes
    def record(self, learner: str, curriculum: str, q, grade, source: str = "practice",
               now: datetime | None = None) -> tuple[Card, bool]:
        """Store an attempt and update the question's review card. Returns (card, again_today)."""
        now = now or datetime.now()
        self.db.execute("INSERT INTO attempts (learner, curriculum, section_id, qid, kind, score, quality, source, ts)"
                        " VALUES (?,?,?,?,?,?,?,?,?)", (learner, curriculum, q.section_id, q.qid, q.kind,
                                                         grade.score, grade.quality, source, now.isoformat()))
        card, again = review(self.card(learner, curriculum, q.qid), grade.quality, now.date())
        self.db.execute("INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?)",
                        (learner, curriculum, q.qid, q.section_id, card.ef, card.repetitions, card.interval,
                         card.due.isoformat()))
        self.db.commit()
        return card, again

    def set_placement(self, learner: str, curriculum: str, statuses: dict[str, str], now: datetime | None = None):
        ts = (now or datetime.now()).isoformat()
        self.db.executemany("INSERT OR REPLACE INTO placements VALUES (?,?,?,?,?)",
                            [(learner, curriculum, sid, st, ts) for sid, st in statuses.items()])
        self.db.commit()

    # -------------------------------------------------------------- reads
    def card(self, learner: str, curriculum: str, qid: str) -> Card:
        row = self.db.execute("SELECT ef, repetitions, interval, due FROM cards WHERE learner=? AND curriculum=? "
                              "AND qid=?", (learner, curriculum, qid)).fetchone()
        if row is None:
            return Card(qid)
        return Card(qid, row[0], row[1], row[2], date.fromisoformat(row[3]) if row[3] else None)

    def due_items(self, learner: str, curriculum: str, today: date) -> list[str]:
        rows = self.db.execute("SELECT qid FROM cards WHERE learner=? AND curriculum=? AND due<=? ORDER BY due, qid",
                               (learner, curriculum, today.isoformat())).fetchall()
        return [r[0] for r in rows]

    def scores(self, learner: str, curriculum: str, section_id: str) -> list[float]:
        rows = self.db.execute("SELECT score FROM attempts WHERE learner=? AND curriculum=? AND section_id=? "
                               "ORDER BY ts DESC, id DESC", (learner, curriculum, section_id)).fetchall()
        return [r[0] for r in rows]

    def mastery(self, learner: str, curriculum: str, section_id: str) -> float:
        return mastery_from(self.scores(learner, curriculum, section_id))

    def placement(self, learner: str, curriculum: str) -> dict[str, str]:
        rows = self.db.execute("SELECT section_id, status FROM placements WHERE learner=? AND curriculum=?",
                               (learner, curriculum)).fetchall()
        return dict(rows)

    def dashboard(self, learner: str, curriculum: str, topics: list[tuple[str, str]], today: date) -> list[TopicProgress]:
        """One row per (section_id, label) topic, in the given order."""
        place = self.placement(learner, curriculum)
        out = []
        for sid, label in topics:
            s = self.scores(learner, curriculum, sid)
            m = mastery_from(s)
            due_rows = self.db.execute("SELECT due FROM cards WHERE learner=? AND curriculum=? AND section_id=?",
                                       (learner, curriculum, sid)).fetchall()
            dues = sorted(date.fromisoformat(r[0]) for r in due_rows)
            out.append(TopicProgress(sid, label, m, len(s), status_from(m, len(s)), place.get(sid, "-"),
                                     sum(d <= today for d in dues), next((d for d in dues if d > today), None)))
        return out
