import itertools

import pytest

from trainer.engine import HandView
from trainer.rules import Rules
from trainer.styles import ACTIONS, STYLE_NAMES, make_style, style_table

R = Rules()


def V(total, up, soft=False, pair=0, dbl=True, spl=False):
    return HandView(total=total, soft=soft, pair_value=pair, upcard=up, can_double=dbl, can_split=spl)


BOOK = [
    (V(16, 10), "hit"), (V(16, 6), "stand"), (V(12, 3), "hit"), (V(12, 4), "stand"),
    (V(11, 10), "double"), (V(11, 10, dbl=False), "hit"), (V(11, 1), "hit"),
    (V(10, 10), "hit"), (V(10, 9), "double"), (V(9, 3), "double"), (V(9, 2), "hit"), (V(8, 6), "hit"),
    (V(17, 1), "stand"),
    (V(18, 9, soft=True), "hit"), (V(18, 2, soft=True), "stand"), (V(18, 5, soft=True), "double"),
    (V(18, 5, soft=True, dbl=False), "stand"), (V(17, 3, soft=True), "double"),
    (V(17, 3, soft=True, dbl=False), "hit"), (V(13, 5, soft=True), "double"), (V(13, 4, soft=True), "hit"),
    (V(19, 6, soft=True), "stand"),
    (V(16, 10, pair=8, spl=True), "split"), (V(16, 10, pair=8), "hit"),
    (V(20, 6, pair=10, spl=True), "stand"), (V(18, 7, pair=9, spl=True), "stand"),
    (V(18, 8, pair=9, spl=True), "split"), (V(10, 9, pair=5, spl=True), "double"),
    (V(8, 5, pair=4, spl=True), "split"), (V(8, 4, pair=4, spl=True), "hit"),
    (V(12, 1, soft=True, pair=1, spl=True), "split"), (V(12, 1, soft=True, pair=1), "hit"),
    (V(4, 7, pair=2, spl=True), "split"), (V(4, 8, pair=2, spl=True), "hit"),
]


@pytest.mark.parametrize("view, action", BOOK)
def test_book(view, action):
    assert make_style("book", R)(view) == action


def test_book_without_double_after_split():
    book = make_style("book", Rules(double_after_split=False))
    assert book(V(8, 5, pair=4, spl=True)) == "hit"
    assert book(V(4, 3, pair=2, spl=True)) == "hit" and book(V(4, 4, pair=2, spl=True)) == "split"
    assert book(V(12, 2, pair=6, spl=True)) == "hit" and book(V(12, 3, pair=6, spl=True)) == "split"


def test_cautious():
    c = make_style("cautious", R)
    assert c(V(12, 10)) == "stand" and c(V(11, 5)) == "double" and c(V(11, 10)) == "hit"
    assert c(V(9, 5)) == "hit" and c(V(17, 6, soft=True)) == "hit" and c(V(18, 9, soft=True)) == "stand"
    assert c(V(16, 10, pair=8, spl=True)) == "split" and c(V(18, 6, pair=9, spl=True)) == "stand"


def test_dealer_copy():
    d = make_style("dealer_copy", R)
    assert d(V(16, 6)) == "hit" and d(V(17, 10)) == "stand" and d(V(17, 10, soft=True)) == "stand"
    assert d(V(11, 6)) == "hit" and d(V(16, 10, pair=8, spl=True)) == "hit"


@pytest.mark.parametrize("name", STYLE_NAMES)
def test_table_equals_function(name):
    style, table = make_style(name, R), style_table(name, R)
    assert table.shape == (2, 2, 11, 2, 22, 11)
    for dbl, spl, pair, soft, total, up in itertools.product((0, 1), (0, 1), range(11), (0, 1), range(4, 21), range(1, 11)):
        view = HandView(total=total, soft=bool(soft), pair_value=pair, upcard=up, can_double=bool(dbl), can_split=bool(spl))
        assert ACTIONS[table[dbl, spl, pair, soft, total, up]] == style(view)


@pytest.mark.parametrize("name", STYLE_NAMES)
def test_a_style_never_asks_for_what_it_cannot_do(name):
    style = make_style(name, R)
    for pair, soft, total, up in itertools.product(range(11), (False, True), range(4, 21), range(1, 11)):
        view = HandView(total=total, soft=soft, pair_value=pair, upcard=up, can_double=False, can_split=False)
        assert style(view) in ("hit", "stand")
