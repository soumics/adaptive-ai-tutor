"""The web UI builds with every stage of the flow, and drives a diagnostic and a quiz end to end
(keyword-only index and a scripted model, so no Ollama is needed)."""

import pytest

gr = pytest.importorskip("gradio")

import app  # noqa: E402
from progress_tracker import ProgressTracker  # noqa: E402
from question_gen import Question, QuestionBank  # noqa: E402
from topic_graph import build_topic_graph  # noqa: E402

from .conftest import scripted  # noqa: E402


@pytest.fixture()
def course(controls_index, tmp_path, monkeypatch):
    graph = build_topic_graph(controls_index.cur)
    qs = [Question(f"{sid}:d{d}", sid, "mcq", d, f"Question {sid} {d}?", ["w", "x", "y", "z"], 1,
                   citation=f"Controls §{sid}") for sid in graph.ids() for d in (1, 2, 3)]
    monkeypatch.setattr(app, "PROVIDER", scripted("Answer [S1]."))
    monkeypatch.setattr(app, "TRACKER", ProgressTracker(tmp_path / "p.sqlite"))
    monkeypatch.setitem(app._COURSES, "controls", dict(index=controls_index, graph=graph, bank=QuestionBank("c", qs),
                                                       bank_path=tmp_path / "bank.json"))
    return dict(course="controls", learner="ana")


def test_ui_has_every_stage():
    app.PROVIDER = scripted("x")
    demo = app.build_ui()
    labels = {getattr(b, "label", None) for b in demo.blocks.values()}
    for needed in ("Sample curriculum", "...or upload your own (PDF, Markdown, text)", "Tutor", "Mode",
                   "Mastery by topic"):
        assert needed in labels, needed


def test_diagnostic_quiz_and_dashboard_through_the_ui_handlers(course):
    state = course
    state, *_ = app.diag_start(state)
    n = 0
    while state["diag"].current is not None:                  # always answer B (correct)
        state, text, *_ = app.diag_answer("B) x", state)
        n += 1
    assert "Placement: advanced" in text and n == 10          # 5 topics x 2 questions
    topic = app.topic_choices(app._COURSES["controls"])[0]
    state, prompt, *_ = app.quiz_start(topic, state)
    state, _, _, _, feedback = app.submit("A) w", "", state)  # wrong answer
    assert "Not quite" in feedback and "Next review of this question" in feedback
    table, df = app.dashboard(state)
    assert len(table) == 5 and df["mastery"].between(0, 1).all()


def test_tutor_answers_carry_sources(course):
    history, _, _ = app.tutor_send("What is a time constant?", [], app.ALL_TOPICS, "beginner", "Explain", course)
    assert history[-1]["role"] == "assistant" and "**[S1]**" in history[-1]["content"]
