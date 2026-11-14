import numpy as np
import pytest

from trainer.cards import VALUES, parse_card
from trainer.engine import Round, play_round
from trainer.fast import WIDTH, play_rounds, sample_completions, sample_sequences
from trainer.rules import Rules
from trainer.sidebets import pp_multipliers
from trainer.styles import ACTIONS, STYLE_NAMES, make_style, style_table

R = Rules()


def test_sequences_are_real_shoe_prefixes():
    seq = sample_sequences(np.random.default_rng(1), 2000)
    assert seq.shape == (2000, WIDTH) and seq.dtype == np.int8
    assert max(np.bincount(row, minlength=52).max() for row in seq) <= 6
    assert not (seq[0] == seq[1]).all()


def test_completions_keep_known_cards_and_respect_the_shoe():
    known = {0: parse_card("KS"), 1: parse_card("KS"), 2: parse_card("KS"), 4: parse_card("KS")}
    seq = sample_completions(np.random.default_rng(2), 3000, known)
    for pos, card in known.items():
        assert (seq[:, pos] == card).all()
    assert max((row == parse_card("KS")).sum() for row in seq) <= 6


@pytest.mark.parametrize("style", STYLE_NAMES)
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_fast_engine_equals_reference(style, k):
    seq = sample_sequences(np.random.default_rng(100 + k), 3000)
    fast = play_rounds(R, seq, style_table(style, R), k)
    fn = make_style(style, R)
    slow = np.array([play_round(R, [int(c) for c in row], 100.0 * (1 + k), 100.0, fn).x for row in seq])
    assert np.array_equal(fast, slow)


def test_fast_engine_equals_reference_without_double_after_split():
    rules = Rules(double_after_split=False, split_by_value=False)
    seq = sample_sequences(np.random.default_rng(9), 3000)
    fn = make_style("book", rules)
    slow = np.array([play_round(rules, [int(c) for c in row], 400.0, 100.0, fn).x for row in seq])
    assert np.array_equal(play_rounds(rules, seq, style_table("book", rules), 3), slow)


def test_fast_engine_takes_a_different_chip_level_per_row():
    seq = sample_sequences(np.random.default_rng(4), 2000)
    levels = np.arange(2000) % 4
    mixed = play_rounds(R, seq, style_table("book", R), levels)
    for k in range(4):
        assert np.array_equal(mixed[levels == k], play_rounds(R, seq[levels == k], style_table("book", R), k))


@pytest.mark.parametrize("first", ["hit", "stand", "double"])
def test_script_forces_the_first_action_then_follows_the_style(first):
    known = {0: parse_card("9S"), 1: parse_card("10H"), 2: parse_card("7D")}
    seq = sample_completions(np.random.default_rng(5), 3000, known)
    seq = seq[VALUES[seq[:, 3]] != 1]                      # the dealer checked: no blackjack
    book = make_style("book", R)

    def slow_one(row):
        r = Round(R, [int(c) for c in row], 400.0, 100.0)
        r.act(first)
        while r.phase != "done":
            r.act(book(r.view()))
        return r.result.x

    fast = play_rounds(R, seq, style_table("book", R), 3, script=[ACTIONS.index(first)])
    assert np.array_equal(fast, np.array([slow_one(row) for row in seq]))


def test_script_can_reach_into_a_split():
    known = {0: parse_card("8S"), 1: parse_card("6H"), 2: parse_card("8D")}
    seq = sample_completions(np.random.default_rng(6), 2000, known)
    book = make_style("book", R)

    def slow_one(row):
        r = Round(R, [int(c) for c in row], 400.0, 100.0)
        r.act("split")
        r.act("stand")          # an eight cannot reach 21 with one more card, so this is always a real decision
        while r.phase != "done":
            r.act(book(r.view()))
        return r.result.x

    fast = play_rounds(R, seq, style_table("book", R), 3, script=[ACTIONS.index("split"), ACTIONS.index("stand")])
    assert np.array_equal(fast, np.array([slow_one(row) for row in seq]))


def test_book_loses_a_little_per_chip():
    total, n = 0.0, 0
    for seed in range(5):
        x = play_rounds(R, sample_sequences(np.random.default_rng(seed), 400_000), style_table("book", R), 3)
        total, n = total + x.sum(), n + len(x)
    assert -0.009 < total / n < -0.002


def test_perfect_pairs_come_up_at_the_six_deck_rates():
    m = pp_multipliers(sample_sequences(np.random.default_rng(3), 400_000), R.pp_pays)
    for mult, chances in ((25, 5), (12, 6), (6, 12)):
        assert abs((m == mult).mean() - chances / 311) < 0.001
