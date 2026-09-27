"""Topics and their prerequisites, derived from the curriculum's own structure.

Each section is a topic. Prerequisite edges come from, in order of trust:

1. **explicit**: the material says so (a ``Prerequisites:`` line in Markdown notes);
2. **reference**: the text of a section refers to an earlier section by number
   ("see Section 2.1", "as in section 2.2");
3. **sequence**: otherwise, a numbered section builds on the previous one in the same chapter
   (2.3 after 2.2), which is how textbooks are normally written.

Edges only ever point from an earlier section to a later one, so the graph is acyclic and its
reading order is a valid learning order. The diagnostic uses the graph to place a learner (a topic
counts as "ready" when its prerequisites are mastered) and the tutor uses it to recommend what to
study next.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from curriculum_ingest import Curriculum, Section, _norm, split_number


@dataclass
class Edge:
    before: str      # section_id that must come first
    after: str
    kind: str        # "explicit" | "reference" | "sequence"


@dataclass
class TopicGraph:
    topics: list[Section]
    edges: list[Edge] = field(default_factory=list)

    def ids(self) -> list[str]:
        return [t.section_id for t in self.topics]

    def topic(self, sid: str) -> Section:
        return next(t for t in self.topics if t.section_id == sid)

    def prerequisites(self, sid: str) -> list[str]:
        return [e.before for e in self.edges if e.after == sid]

    def dependents(self, sid: str) -> list[str]:
        return [e.after for e in self.edges if e.before == sid]

    def order(self) -> list[str]:
        """A topological order (Kahn's algorithm), ties broken by reading order."""
        pos = {t.section_id: t.order for t in self.topics}
        indeg = {sid: 0 for sid in pos}
        for e in self.edges:
            indeg[e.after] += 1
        ready = sorted((s for s, d in indeg.items() if d == 0), key=pos.get)
        out = []
        while ready:
            s = ready.pop(0)
            out.append(s)
            for t in self.dependents(s):
                indeg[t] -= 1
                if indeg[t] == 0:
                    ready.append(t)
                    ready.sort(key=pos.get)
        if len(out) != len(pos):
            raise ValueError("prerequisite graph has a cycle")
        return out

    def ready_topics(self, mastered: set[str]) -> list[str]:
        """Topics not yet mastered whose prerequisites all are, in learning order."""
        return [s for s in self.order() if s not in mastered and all(p in mastered for p in self.prerequisites(s))]

    def as_rows(self) -> list[tuple[str, str, str]]:
        """(topic, requires, how the requirement was found) for display."""
        rows = []
        for t in sorted(self.topics, key=lambda t: t.order):
            pre = [e for e in self.edges if e.after == t.section_id]
            rows.append((t.label, "; ".join(self.topic(e.before).label for e in pre) or "-",
                         ", ".join(sorted({e.kind for e in pre})) or "-"))
        return rows


def _match_section(text: str, topics: list[Section]) -> Section | None:
    """Resolve a prerequisite written by an author ('3. Stability and Poles', '2.1', 'PID control')."""
    number, name = split_number(text)
    if number:
        hit = [t for t in topics if t.number == number]
        if hit:
            return hit[0]
    key = _norm(name or text)
    for t in topics:
        if _norm(t.title) == key or (key and (key in _norm(t.title) or _norm(t.title) in key)):
            return t
    return None


def build_topic_graph(cur: Curriculum) -> TopicGraph:
    topics = sorted(cur.sections, key=lambda s: s.order)
    pos = {t.section_id: t.order for t in topics}
    edges: dict[tuple[str, str], Edge] = {}

    def add(before: Section, after: Section, kind: str) -> None:
        if pos[before.section_id] < pos[after.section_id] and (before.section_id, after.section_id) not in edges:
            edges[(before.section_id, after.section_id)] = Edge(before.section_id, after.section_id, kind)

    for t in topics:                                                   # 1. explicit
        for text in t.explicit_prereqs:
            s = _match_section(text, topics)
            if s is not None:
                add(s, t, "explicit")
    by_number = {t.number: t for t in topics if t.number}
    ref = re.compile(r"\bsections?\s+(\d{1,2}(?:\.\d{1,2})+)", re.I)
    for t in topics:                                                   # 2. references in the text
        text = " ".join(c.text for c in cur.chunks_for(t.section_id))
        for num in ref.findall(text):
            if num in by_number:
                add(by_number[num], t, "reference")
    for t in topics:                                                   # 3. sequence (numbered, same chapter)
        if t.explicit_prereqs or not t.number:           # the author's own list takes precedence
            continue
        if "." not in t.number:
            continue
        chapter, n = t.number.rsplit(".", 1)
        prev = by_number.get(f"{chapter}.{int(n) - 1}")
        if prev is not None:
            add(prev, t, "sequence")
    return TopicGraph(topics, list(edges.values()))
