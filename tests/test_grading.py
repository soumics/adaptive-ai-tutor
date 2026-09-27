"""Grading against known right / wrong / partial answers; the LLM can only add credit it can quote."""

from fractions import Fraction

import pytest

from grading import contains_phrase, grade, parse_number, sm2_quality, solve_linear
from question_gen import Question, RubricPoint

from .conftest import scripted

MCQ = Question("t:1", "t", "mcq", 1, "What is the first step?", ["Guess", "Simplify each side", "Divide", "Stop"], 1,
               "Simplify first.", citation="Algebra, §2.4, p. 41")
EXACT = Question("t:2", "t", "short", 3, "Solve: 5(n − 4) − 4n = −8.", model_answer="n = 12", exact_answer="12",
                 citation="Algebra, §2.1, p. 6")
RUBRIC = Question("t:3", "t", "short", 2, "Why does integral action remove steady-state error?",
                  model_answer="The integral keeps growing while any error remains, so the error is driven to zero.",
                  rubric=[RubricPoint("The integral accumulates past error", ["accumulated past error", "accumulates"]),
                          RubricPoint("It keeps growing while error remains", ["keeps growing"]),
                          RubricPoint("Result: zero steady-state error", ["zero steady-state error"])],
                  citation="Controls, §4 PID Control")


def test_multiple_choice_is_exact():
    right, wrong, blank = grade(MCQ, 1), grade(MCQ, 3), grade(MCQ, None)
    assert (right.score, right.correct, right.quality) == (1.0, True, 4)
    assert (wrong.score, wrong.correct, wrong.quality) == (0.0, False, 1)
    assert blank.score == 0.0
    assert "B) Simplify each side" in wrong.feedback and "§2.4" in wrong.feedback


@pytest.mark.parametrize("answer, ok", [("n = 12", True), ("12", True), ("n=12.0", True), ("24/2", True),
                                        ("n = 11", False), ("-12", False), ("twelve", False), ("", False)])
def test_exact_answers(answer, ok):
    g = grade(EXACT, answer)
    assert g.correct is ok and g.score == (1.0 if ok else 0.0)


def test_equation_solver_and_number_parsing():
    assert solve_linear("4x − 2 = 2x + 1") == Fraction(3, 2)
    assert solve_linear("Solve: 2(m − 4) + 3 = −1") == 2
    assert solve_linear("2/3(6m - 3) = 8 - m") == 2
    assert solve_linear("3x = 3x + 1") is None                 # no solution
    assert solve_linear("x*x = 4") is None                     # not linear
    assert solve_linear("import os = 1") is None               # not maths: refused, never evaluated
    assert parse_number("x = −3/4") == Fraction(-3, 4) and parse_number("1.25") == Fraction(5, 4)


def test_rubric_full_partial_and_zero_by_keywords():
    full = grade(RUBRIC, "Because the integral accumulates the error; it keeps growing until there is zero "
                         "steady-state error.")
    partial = grade(RUBRIC, "The integral term accumulates error over time.")
    zero = grade(RUBRIC, "Derivative action adds damping.")
    assert (full.score, full.correct, full.quality) == (1.0, True, 5)
    assert partial.score == pytest.approx(1 / 3, abs=1e-3) and partial.quality == 2
    assert zero.score == 0.0 and zero.quality == 0
    assert "✓ The integral accumulates past error" in partial.feedback and "✗" in partial.feedback


def test_llm_credit_requires_a_real_quote_from_the_answer():
    answer = "The integral term adds up the errors, and while an error is left it continues to rise."
    honest = scripted({"points": [{"index": 1, "met": True, "quote": "while an error is left it continues to rise"}]})
    invented = scripted({"points": [{"index": 1, "met": True, "quote": "keeps growing while error remains"},
                                    {"index": 2, "met": True, "quote": "zero steady-state error"}]})
    g1 = grade(RUBRIC, answer, honest)
    g2 = grade(RUBRIC, answer, invented)
    assert g1.points[1].met and g1.points[1].method == "llm-quote"
    assert g1.score == pytest.approx(1 / 3, abs=1e-3)                    # only the quoted point is credited
    assert g2.score == 0.0                                     # quotes not in the answer earn nothing


def test_quality_mapping_and_word_matching():
    assert [sm2_quality(s, "short") for s in (1, 0.8, 0.5, 0.3, 0.1, 0)] == [5, 4, 3, 2, 1, 0]
    assert contains_phrase("The loops are closed.", "loop") and not contains_phrase("loophole", "loop")
