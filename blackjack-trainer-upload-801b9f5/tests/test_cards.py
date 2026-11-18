import numpy as np
import pytest

from trainer.cards import VALUES, card_str, hand_total, is_red, new_shoe, parse_card, value_of
from trainer.errors import TrainerError


def test_parse_and_format():
    for text in ("AS", "10H", "KD", "2C"):
        assert card_str(parse_card(text)) == text
    assert card_str(parse_card("10")) == "10S"
    for bad in ("1", "11", "XS", "", "10Z"):
        with pytest.raises(TrainerError):
            parse_card(bad)


def test_values_and_colours():
    assert [value_of(parse_card(t)) for t in ("AS", "5H", "10D", "JC", "QS", "KH")] == [1, 5, 10, 10, 10, 10]
    assert all(VALUES[c] == value_of(c) for c in range(52))
    assert is_red(parse_card("KH")) and is_red(parse_card("KD"))
    assert not is_red(parse_card("KS")) and not is_red(parse_card("KC"))


def test_hand_total():
    T = lambda *t: hand_total([parse_card(x) for x in t])
    assert T("AS", "6H") == (17, True)
    assert T("AS", "6H", "10D") == (17, False)
    assert T("AS", "AH") == (12, True)
    assert T("10S", "9H", "5D") == (24, False)


def test_new_shoe_is_six_decks_and_seeded():
    a, b = new_shoe(np.random.default_rng(7)), new_shoe(np.random.default_rng(7))
    assert len(a) == 312 and (a == b).all() and (np.bincount(a, minlength=52) == 6).all()
