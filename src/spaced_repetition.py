"""SM-2 spaced repetition (SuperMemo 2, P. A. Wozniak, 1987), deterministic and unit-tested.

For each review the learner's answer gets a quality q from 0 (blackout) to 5 (perfect), which
``grading.sm2_quality`` derives from the grade. Then, following the widely used formulation of the
algorithm (as given, e.g., in the Wikipedia article on SuperMemo):

    if q >= 3:                       # recalled
        interval = 1                 if repetitions == 0
                   6                 if repetitions == 1
                   round(interval * EF) otherwise
        repetitions += 1
    else:                            # forgotten: start the repetitions again
        repetitions = 0
        interval = 1
    EF = max(1.3, EF + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))

The interval uses the easiness factor EF from *before* this review; EF then changes by +0.10 for
q = 5, 0 for q = 4, -0.14 for q = 3, -0.32 for q = 2, -0.54 for q = 1 and -0.80 for q = 0, never
below 1.3. New items start at EF = 2.5. As in the original method, an item answered with q < 4
is shown again later in the same session (``again_today``) until it is answered well.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta

INITIAL_EF = 2.5
MIN_EF = 1.3


@dataclass(frozen=True)
class Card:
    item_id: str
    ef: float = INITIAL_EF
    repetitions: int = 0
    interval: int = 0              # days
    due: date | None = None        # None: new, due now

    def is_due(self, today: date) -> bool:
        return self.due is None or self.due <= today


def next_ef(ef: float, q: int) -> float:
    return max(MIN_EF, round(ef + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02), 4))


def review(card: Card, quality: int, today: date) -> tuple[Card, bool]:
    """Apply one review. Returns (updated card, again_today)."""
    if not 0 <= quality <= 5:
        raise ValueError("quality must be 0..5")
    if quality >= 3:
        if card.repetitions == 0:
            interval = 1
        elif card.repetitions == 1:
            interval = 6
        else:
            interval = round(card.interval * card.ef)
        reps = card.repetitions + 1
    else:
        reps, interval = 0, 1
    new = replace(card, ef=next_ef(card.ef, quality), repetitions=reps, interval=interval,
                  due=today + timedelta(days=interval))
    return new, quality < 4


def due_cards(cards: list[Card], today: date) -> list[Card]:
    """Cards due today, most overdue first, new cards last."""
    due = [c for c in cards if c.is_due(today)]
    return sorted(due, key=lambda c: (c.due is None, c.due or today))
