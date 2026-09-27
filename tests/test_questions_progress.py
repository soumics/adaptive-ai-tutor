"""Question checks (what gets rejected, and why) and the progress tracker's mastery + review schedule."""

from datetime import date, datetime, timedelta

import pytest

from grading import Grade
from progress_tracker import ProgressTracker, mastery_from, status_from
from question_gen import Question, QuestionBank, check_mcq, generate_mcq, generate_short, grounded_rubric

from .conftest import scripted

GOOD = {"question": "What is the steady-state error of P control on a first-order plant?",
        "options": ["1/(1 + Kp K)", "Zero", "Kp K", "Infinite"], "correct_index": 0,
        "explanation": "Stated in the PID section.", "source": "S1"}


@pytest.mark.parametrize("change, reason", [
    ({"options": ["A", "A", "B", "C"]}, "not distinct"),
    ({"options": ["1/(1 + Kp K)", "Zero", "Kp K", "All of the above"]}, "all/none"),
    ({"question": "What does the passage say about the error?"}, "self-contained"),
    ({"options": ["1/(1 + Kp K)", "Zero", "See S2", "Infinite"]}, "self-contained"),
    ({"question": "In Figure 2.4, what expression is on the right side?"}, "figure"),
    ({"source": "S9"}, "unknown source"),
    ({"question": "Solve: 3x + 4 = 10. What is x?", "options": ["3", "2", "4", "6"], "correct_index": 0}, "not the equation"),
])
def test_bad_multiple_choice_candidates_are_rejected(controls_index, change, reason):
    chunks = controls_index.topic_passages(controls_index.cur.sections[3].section_id)
    assert check_mcq({**GOOD, **change}, chunks) is not None and reason.split()[0] in check_mcq({**GOOD, **change}, chunks)
    assert check_mcq(dict(GOOD), chunks) is None


def test_generated_questions_are_shuffled_verified_and_cited(controls_index):
    sid = controls_index.cur.sections[3].section_id
    gen = {"questions": [GOOD, {**GOOD, "question": "Which term reacts to accumulated past error?",
                                "options": ["Integral", "Derivative", "Proportional", "Feedforward"],
                                "correct_index": 0}]}
    # generation reply, then the independent check answers: agrees with the first, disagrees with the second
    provider = scripted(gen)
    qs = generate_mcq(controls_index, sid, provider, 1, n=2, verify=False)
    assert len(qs) == 2 and all(q.citation.endswith("§4 PID Control") for q in qs)
    keyed = [q.options[q.answer_index] for q in qs]
    assert keyed == ["1/(1 + Kp K)", "Integral"]                      # the key follows the shuffled option
    log = []
    checked = generate_mcq(controls_index, sid, scripted(gen, {"answer_index": qs[0].answer_index},
                                                         {"answer_index": (qs[1].answer_index + 1) % 4}),
                           1, n=2, verify=True, log=log)
    assert [q.prompt for q in checked] == [qs[0].prompt] and checked[0].verified
    assert log[-1][2] == "failed the independent answer check"


def test_short_answers_exact_check_and_grounded_rubric(algebra_index):
    sid = algebra_index.cur.sections[0].section_id
    reply = {"questions": [
        {"question": "Solve: x + 3 = 8.", "model_answer": "x = 5", "rubric": [], "source": "S1"},
        {"question": "Solve: x - 4 = 10.", "model_answer": "x = 6", "rubric": [], "source": "S1"},  # wrong key
    ]}
    log = []
    qs = generate_short(algebra_index, sid, scripted(reply), 2, n=2, log=log)
    assert [(q.prompt, q.exact_answer, q.verified) for q in qs] == [("Solve: x + 3 = 8.", "5", True)]
    assert log[-1][2] == "model answer is not the solution"
    src = "The integral keeps growing as long as any error remains."
    rub = grounded_rubric([{"point": "Integral grows", "keywords": ["keeps growing", "usually", "explodes"]},
                           {"point": "Invented point", "keywords": ["quantum tunnelling"]}], src)
    assert [(r.point, r.keywords) for r in rub] == [("Integral grows", ["keeps growing"])]


def test_mastery_status_and_review_schedule(tmp_path):
    assert mastery_from([]) == 0.5 and status_from(0.5, 0) == "not started"
    assert mastery_from([1, 1, 1, 1, 1]) > 0.8 > mastery_from([1, 1, 0, 0, 0]) > 0.3
    t = ProgressTracker(tmp_path / "p.sqlite")
    q = Question("s1:q", "s1", "mcq", 2, "Q?", ["a", "b", "c", "d"], 0)
    now = datetime(2026, 3, 1, 9)
    for i in range(4):
        card, again = t.record("ana", "ctl", q, Grade(1.0, True, 4, ""), "practice", now + timedelta(days=i))
    assert (card.repetitions, card.interval) == (4, 38)            # 1, 6, 15, round(15 * 2.5) = 38
    rows = t.dashboard("ana", "ctl", [("s1", "Topic 1"), ("s2", "Topic 2")], date(2026, 3, 4))
    assert rows[0].status == "mastered" and rows[0].attempts == 4 and rows[0].due_reviews == 0
    assert rows[1].status == "not started"
    _, again = t.record("ana", "ctl", q, Grade(0.0, False, 1, ""), "review", now + timedelta(days=40))
    assert again and t.card("ana", "ctl", "s1:q").interval == 1
    assert t.due_items("ana", "ctl", date(2026, 4, 11)) == ["s1:q"]
    # progress persists across restarts (a new tracker on the same file)
    assert ProgressTracker(tmp_path / "p.sqlite").mastery("ana", "ctl", "s1") == t.mastery("ana", "ctl", "s1")
    bank = QuestionBank("ctl", [q])
    bank.save(tmp_path / "bank.json")
    assert QuestionBank.load(tmp_path / "bank.json").get("s1:q").options == ["a", "b", "c", "d"]


def test_given_values_are_not_mistaken_for_equations_and_near_duplicates_are_caught():
    from question_gen import find_equation, similar

    # regression: "K = 2" and "t = 0" are data, not equations to solve (these questions were wrongly rejected)
    assert find_equation("A first-order plant with gain K = 2 is controlled with Kp = 3. What is the error?") is None
    assert find_equation("What is the step response when the input jumps from 0 to U at t = 0?") is None
    assert find_equation("What value of y satisfies 2y + 1 = 7?") == "2y + 1 = 7"
    assert similar("What characteristic do poles with a positive real part cause in a system response?",
                   "What happens to the response of a system when one of its poles has a positive real part?")
    assert not similar("What is a time constant?", "What is a pole?")


def test_rubric_points_the_model_answer_misses_are_dropped(controls_index):
    sid = controls_index.cur.sections[2].section_id                  # 3 Stability and Poles
    reply = {"questions": [{"question": "Where are the closed-loop poles found?", "model_answer": "1 + C(s)G(s) = 0",
                            "rubric": [{"point": "Gives the characteristic equation", "keywords": ["1 + C(s)G(s) = 0"]},
                                       {"point": "Says the poles are its roots", "keywords": ["roots"]}],
                            "source": "S1"}]}
    q = generate_short(controls_index, sid, scripted(reply), 2, n=1)[0]
    assert [p.point for p in q.rubric] == ["Gives the characteristic equation"]


def test_meta_phrasing_is_repaired_and_text_options_skip_the_equation_check(controls_index):
    from question_gen import repair_meta

    assert repair_meta("According to the text, what is a pole?") == "What is a pole?"
    assert repair_meta("What is the rule described in the passage for stability?") == "What is the rule for stability?"
    assert repair_meta("In S2, what does feedback do?") == "What does feedback do?"
    assert repair_meta("According to the passage, what is the error?") == "What is the error?"
    assert repair_meta("Using the process shown in Example 2.25 from Source, what is the cost?") == \
        "Using the process shown in Example 2.25, what is the cost?"
    assert repair_meta("In Example 2.14 from S1, which property is used?") == "In Example 2.14, which property is used?"
    assert repair_meta("What does Sam solve?") == "What does Sam solve?"                  # names are untouched
    chunks = controls_index.topic_passages(controls_index.cur.sections[0].section_id)
    item = {"question": "When solving an equation such as x + 3 = 8, what is the first step?",
            "options": ["Subtract 3 from both sides", "Add 8", "Divide by 3", "Multiply by x"], "correct_index": 0,
            "explanation": "", "source": "S1"}
    assert check_mcq(item, chunks) is None          # asks for a step, not a number: no false rejection
