"""Diagnostic placement on synthetic response patterns (no LLM: multiple-choice only)."""

import pytest

from diagnostic import classify, run_diagnostic
from question_gen import Question, QuestionBank
from topic_graph import build_topic_graph


@pytest.fixture(scope="module")
def graph(controls_cur):
    return build_topic_graph(controls_cur)          # 5 topics: 1 -> 2 -> 3 -> {4 (also needs 1), 5}


@pytest.fixture(scope="module")
def bank(graph):
    qs = [Question(f"{sid}:d{d}:{i}", sid, "mcq", d, f"Q {sid} {d} {i}?", ["a", "b", "c", "d"], 0)
          for sid in graph.ids() for d in (1, 2, 3) for i in range(2)]
    return QuestionBank("controls", qs)


def learner(knows: dict[str, int]):
    """Answers correctly iff it knows the topic up to the question's difficulty."""
    return lambda q: q.difficulty <= knows.get(q.section_id, 0)


def test_decision_table():
    assert classify([(2, True), (3, True)]) == "mastered"
    assert classify([(2, False), (1, False)]) == "not yet"
    assert classify([(2, True), (3, False)]) is None                # a third question is needed
    assert classify([(2, True), (3, False), (2, False)]) == "developing"
    assert classify([(2, False), (1, True), (2, True)]) == "developing"
    assert classify([(2, False), (1, True), (2, False)]) == "not yet"
    assert classify([(2, False)], probe=True) == "not yet"           # prerequisite gap confirmed at once


def test_expert_is_placed_at_the_end_with_two_questions_per_topic(graph, bank):
    p = run_diagnostic(graph, bank, learner({sid: 3 for sid in graph.ids()}))
    assert all(r.status == "mastered" for r in p.topics.values())
    assert p.start_topic is None and p.overall == "advanced" and p.questions_asked == 10


def test_novice_is_placed_at_the_first_topic_and_the_test_is_short(graph, bank):
    p = run_diagnostic(graph, bank, learner({}))
    ids = graph.ids()
    assert p.start_topic == ids[0] and p.overall == "beginner"
    assert all(r.status == "not yet" for r in p.topics.values())
    # 2 questions on the first topic, then a single probe for each topic whose prerequisite is missing
    assert p.questions_asked == 2 + 4


def test_partial_knowledge_places_at_the_first_gap(graph, bank):
    ids = graph.ids()
    p = run_diagnostic(graph, bank, learner({ids[0]: 3, ids[1]: 3, ids[2]: 2}))
    assert [p.topics[s].status for s in ids[:3]] == ["mastered", "mastered", "developing"]
    assert p.start_topic == ids[2]                                   # first topic not mastered
    assert p.level_for(ids[2]) == "intermediate" and p.level_for(ids[0]) == "advanced"


def test_a_right_probe_still_gets_a_full_assessment(graph, bank):
    ids = graph.ids()
    # knows topic 5 (consensus) although its prerequisite (3, stability) is weak
    p = run_diagnostic(graph, bank, learner({ids[0]: 3, ids[1]: 3, ids[4]: 3}))
    assert p.topics[ids[2]].status == "not yet"
    assert p.topics[ids[4]].probe and p.topics[ids[4]].status == "mastered"
    assert p.start_topic == ids[2]                                   # the gap is what to study next
