"""SM-2 against hand-computed cases: intervals, easiness factor, lapses, floor, due dates."""

from datetime import date, timedelta

import pytest

from spaced_repetition import Card, due_cards, next_ef, review

DAY = date(2026, 1, 1)


def run(qualities, card=None):
    card = card or Card("q")
    today, history = DAY, []
    for q in qualities:
        card, again = review(card, q, today)
        history.append((card.interval, card.ef, card.repetitions, again))
        today = card.due
    return card, history


def test_perfect_recall_intervals_1_6_16_45():
    # EF: 2.5 -> 2.6 -> 2.7 -> 2.8 -> 2.9; intervals use the EF from before each review:
    # 1, 6, round(6 * 2.7) = 16, round(16 * 2.8) = 45
    _, h = run([5, 5, 5, 5])
    assert [x[0] for x in h] == [1, 6, 16, 45]
    assert [x[1] for x in h] == pytest.approx([2.6, 2.7, 2.8, 2.9])
    assert [x[2] for x in h] == [1, 2, 3, 4]


def test_easiness_changes_per_quality():
    deltas = {q: round(next_ef(2.5, q) - 2.5, 4) for q in range(6)}
    assert deltas == {5: 0.1, 4: 0.0, 3: -0.14, 2: -0.32, 1: -0.54, 0: -0.8}


def test_quality_4_keeps_ef_and_grows_interval():
    _, h = run([4, 4, 4])
    assert [x[0] for x in h] == [1, 6, 15]            # round(6 * 2.5) = 15
    assert all(x[1] == pytest.approx(2.5) for x in h)


def test_lapse_restarts_repetitions_but_keeps_lowered_ef():
    card, h = run([5, 5, 5, 1])
    assert card.repetitions == 0 and card.interval == 1
    assert card.ef == pytest.approx(2.8 - 0.54)
    assert h[-1][3] is True                            # shown again in the same session
    card, h2 = run([4, 4, 4], card)                     # relearning: 1, 6, then round(6 * 2.26)
    assert [x[0] for x in h2] == [1, 6, 14]


def test_ef_never_below_1_3():
    card, _ = run([3, 0, 0, 0, 0, 0])
    assert card.ef == pytest.approx(1.3)
    card, h = run([3, 3, 3, 3, 3, 3])                  # EF falls by 0.14 per review, floored at 1.3
    assert [x[1] for x in h] == pytest.approx([2.36, 2.22, 2.08, 1.94, 1.8, 1.66])
    assert all(x[3] for x in h)                        # q = 3 (< 4) is repeated the same day


def test_due_dates_and_ordering():
    card, _ = run([5])
    assert card.due == DAY + timedelta(days=1)
    assert not card.is_due(DAY) and card.is_due(DAY + timedelta(days=1))
    new, overdue = Card("new"), Card("old", due=DAY - timedelta(days=3))
    later = Card("later", due=DAY + timedelta(days=9))
    assert [c.item_id for c in due_cards([new, later, overdue], DAY)] == ["old", "new"]
    with pytest.raises(ValueError):
        review(Card("x"), 6, DAY)
