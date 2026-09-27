"""Every tutor answer must cite the curriculum -- on every code path, in both subjects. Fails the build if not.

The LLM is scripted, including a badly behaved one, so the guarantee is tested against the model
*not* following instructions: uncited answers, invented source numbers, and failures in Socratic mode.
"""

import re

import pytest

from socratic import SocraticSession
from tutor_engine import TutorAnswer, ask

from .conftest import scripted

CITE = re.compile(r"\[S(\d+)\]")


def assert_cited(a):
    """At least one citation, every [S#] in the text resolvable, and each citation names a real section."""
    assert a.citations, f"answer has no citation: {a.text!r}"
    labels = {c.label for c in a.citations}
    assert set(f"S{n}" for n in CITE.findall(a.text)) <= labels, "text cites a source that isn't listed"
    for c in a.citations:
        assert "§" in c.citation and c.section_id


SUBJECTS = [("controls_index", "Why does integral action remove steady-state error?"),
            ("algebra_index", "How do I check whether a number is a solution of an equation?")]


@pytest.mark.parametrize("index_name, question", SUBJECTS)
def test_well_behaved_model_answer_is_accepted_with_its_citations(request, index_name, question):
    idx = request.getfixturevalue(index_name)
    a = ask(question, idx, scripted("Start from the definition [S1]. Then apply it step by step [S2]."))
    assert isinstance(a, TutorAnswer) and a.grounded and not a.fallback and a.attempts == 1
    assert_cited(a)


@pytest.mark.parametrize("index_name, question", SUBJECTS)
def test_uncited_answer_is_retried_then_replaced_by_cited_passages(request, index_name, question):
    idx = request.getfixturevalue(index_name)
    llm = scripted("Here is the answer from general knowledge, with no sources at all.")
    a = ask(question, idx, llm)
    assert a.fallback and not a.grounded and a.attempts == 2
    assert "rejected" in llm.llm.prompts[1][1]                    # the retry says what was wrong
    assert_cited(a)
    assert "general knowledge" not in a.text                    # the unsourced text is never shown


def test_invented_source_numbers_are_rejected(controls_index):
    a = ask("What is a time constant?", controls_index, scripted("It is tau [S9].", "Tau sets the speed [S1]."))
    assert a.grounded and a.attempts == 2 and [c.label for c in a.citations] == ["S1"]
    assert_cited(a)


def test_out_of_scope_questions_still_point_to_the_material(controls_index):
    a = ask("How do I solve 3x + 4 = 10?", controls_index,
            scripted("The provided documents do not contain this information."))
    assert a.abstained and not a.grounded
    assert "doesn't cover" in a.text
    assert_cited(a)


def test_socratic_turns_always_cite_even_when_the_model_forgets(controls_index):
    forgetful = scripted({"feedback": "Good start.", "question": "What happens to the integral while error remains?",
                          "source": "S2", "reached_answer": False})
    s = SocraticSession("Why does integral action remove steady-state error?", controls_index, forgetful)
    t = s.start()
    assert t.cite_added and "[S2]" in t.text and not t.done
    assert_cited(t)


def test_socratic_gives_a_cited_explanation_when_the_learner_is_stuck(controls_index):
    turn = {"feedback": "Think about it [S1].", "question": "What does the integral add up?", "source": "S1",
            "reached_answer": False}
    # 1st call: the hidden reference answer; then one JSON turn per exchange
    llm = scripted("The integral accumulates the error, so it removes steady-state error [S1].", turn, turn)
    s = SocraticSession("Why does integral action remove steady-state error?", controls_index, llm, max_turns=4)
    s.start()
    first = s.reply("I don't know")                              # one "I don't know": keep guiding
    assert not first.done
    t = s.reply("just tell me")                                  # the second: explain, with citations
    assert t.done and t.revealed and "accumulates the error" in t.text
    assert_cited(t)


def test_worked_examples_with_arithmetic_errors_are_not_shown(algebra_index):
    """A cited but wrong worked example (9x = 8x - 6 -> '17x = -6') is retried, then replaced by passages."""
    from tutor_engine import check_arithmetic

    bad = "Choose the left side for $9x = 8x - 6$ [S1]. Add $8x$ to both sides, resulting in $17x = -6$ [S1]."
    good = "For $12x = -x + 26$, add $x$ to both sides to get $13x = 26$, so $x = 2$ [S1]."
    assert check_arithmetic(bad, "Solve 9x = 8x - 6.") and not check_arithmetic(good, "Solve 12x = -x + 26.")
    assert not check_arithmetic("With K = 2 and Kp = 1 the error is e = 1/(1 + 2) [S1].", "")   # given values
    llm = scripted(bad)
    a = ask("Which side should be the variable side?", algebra_index, llm)
    assert a.fallback and "17x" not in a.text and "arithmetic error" in llm.llm.prompts[1][1]
    assert_cited(a)
    fixed = ask("Which side should be the variable side?", algebra_index, scripted(bad, good))
    assert fixed.grounded and fixed.attempts == 2
