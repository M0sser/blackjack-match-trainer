"""Fixed ways of playing a hand: Book (basic strategy), Cautious, and Dealer-copy.

Each style is a function of a HandView. style_table() turns one into a lookup table
for the fast engine by calling the function, so the two forms cannot disagree.
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Callable

import numpy as np

from trainer.engine import HandView
from trainer.rules import Rules

STYLE_NAMES = ("book", "cautious", "dealer_copy")
STYLE_INDEX = {name: index for index, name in enumerate(STYLE_NAMES)}
ACTIONS = ("stand", "hit", "double", "split")   # the index is the action code


def _book(rules: Rules) -> Callable[[HandView], str]:
    """Basic strategy for six decks, dealer standing on soft 17, no surrender."""
    das = rules.double_after_split

    def splits(pair: int, up: int) -> bool:
        if pair in (1, 8):
            return True
        if pair == 9:
            return up in (2, 3, 4, 5, 6, 8, 9)
        if pair == 7:
            return 2 <= up <= 7
        if pair == 6:
            return (2 if das else 3) <= up <= 6
        if pair == 4:
            return das and up in (5, 6)
        if pair in (2, 3):
            return (2 if das else 4) <= up <= 7
        return False            # fives and tens are never split

    def style(view: HandView) -> str:
        up, total = view.upcard, view.total
        if view.can_split and view.pair_value and splits(view.pair_value, up):
            return "split"
        if view.soft:
            double = ((total in (13, 14) and up in (5, 6))
                      or (total in (15, 16) and 4 <= up <= 6)
                      or (total in (17, 18) and 3 <= up <= 6))
            if double and view.can_double:
                return "double"
            if total >= 19:
                return "stand"
            if total == 18:
                return "stand" if double or up in (2, 7, 8) else "hit"
            return "hit"
        if total >= 17:
            return "stand"
        if total >= 13:
            return "stand" if 2 <= up <= 6 else "hit"
        if total == 12:
            return "stand" if 4 <= up <= 6 else "hit"
        double = ((total == 11 and 2 <= up <= 10)
                  or (total == 10 and 2 <= up <= 9)
                  or (total == 9 and 3 <= up <= 6))
        return "double" if double and view.can_double else "hit"

    return style


def _cautious(view: HandView) -> str:
    """A common casual player: never risks busting, doubles and splits only the obvious hands."""
    if view.can_split and view.pair_value in (1, 8):
        return "split"
    if view.soft:
        return "hit" if view.total <= 17 else "stand"
    if view.total >= 12:
        return "stand"
    if view.can_double and view.total in (10, 11) and 2 <= view.upcard <= 9:
        return "double"
    return "hit"


def _dealer_copy(view: HandView) -> str:
    """Plays like the dealer: hits under 17, never doubles or splits."""
    return "hit" if view.total < 17 else "stand"


def make_style(name: str, rules: Rules) -> Callable[[HandView], str]:
    if name == "book":
        return _book(rules)
    if name == "cautious":
        return _cautious
    if name == "dealer_copy":
        return _dealer_copy
    raise ValueError(f"unknown play style: {name}")


@lru_cache(maxsize=None)
def style_table(name: str, rules: Rules) -> np.ndarray:
    """Action codes indexed [can_double][can_split][pair_value][soft][total][upcard]."""
    style = make_style(name, rules)
    table = np.zeros((2, 2, 11, 2, 22, 11), dtype=np.int8)
    for dbl, spl, pair, soft, total, up in itertools.product((0, 1), (0, 1), range(11), (0, 1), range(22), range(1, 11)):
        view = HandView(total=total, soft=bool(soft), pair_value=pair, upcard=up,
                        can_double=bool(dbl), can_split=bool(spl))
        table[dbl, spl, pair, soft, total, up] = ACTIONS.index(style(view))
    table.setflags(write=False)
    return table
