import numpy as np
import pytest

from trainer.opponents import OpponentType
from trainer.rules import Rules
from trainer.solver import (bet_menu, chip_grid, credit, final_distribution, history_index, load_solution,
                            propagate, save_solution, solve)
from trainer.tables import build_tables

R = Rules()
one = lambda rule, param=0.0: [OpponentType("x", "book", rule, param, 1.0)]
start = lambda sol: float(np.interp(1000.0, sol.grid, sol.value[0][0]))


@pytest.fixture(scope="module")
def tables():
    return build_tables(R, n_rounds=200_000, seed=3)


@pytest.fixture(scope="module")
def vs_turtle(tables):
    return solve(R, one("turtle"), tables, coarse=True)


def test_history_index_and_grid():
    assert history_index([]) == 0 and history_index([2]) == 2 and history_index([1, 0, 7]) == 71
    for coarse in (False, True):
        g = chip_grid(coarse)
        assert g[0] == 0.0 and (np.diff(g) > 0).all() and 1000.0 in g and 1005.0 in g and 20000.0 in g
        m = bet_menu(coarse)
        assert m[0] == 5.0 and (np.diff(m) > 0).all() and (m % 5 == 0).all()


def test_propagate_follows_the_betting_rule(tables):
    assert propagate(R, one("turtle")[0], tables, [2, 2, 4]) == {(990.0, 0.0): 1.0}
    assert propagate(R, one("chaser", 50)[0], tables, [2, 2]) == {(850.0, 200.0): 1.0}


def test_final_distribution_and_credit(tables):
    chips, weight = final_distribution(R, one("turtle"), tables, [2, 2, 2, 2], 2)
    assert chips.tolist() == [950.0] and abs(weight.sum() - 1.0) < 1e-12
    assert credit(chips, weight, np.array([940.0, 950.0, 960.0]), 0.5).tolist() == [0.0, 0.5, 1.0]


def test_beats_an_opponent_who_only_bets_the_minimum(vs_turtle):
    assert start(vs_turtle) >= 0.88


def test_beats_an_opponent_who_always_goes_all_in(tables):
    assert start(solve(R, one("all_in"), tables, coarse=True)) >= 0.90


def test_sitting_on_a_lead_is_certain(vs_turtle):
    h = history_index([2, 2, 2, 2])                       # the turtle is on 980
    assert np.interp(1000.0, vs_turtle.grid, vs_turtle.value[4][h]) >= 0.999
    assert np.interp(1000.0, vs_turtle.grid, vs_turtle.bet[4][h]) <= 10.0


def test_values_are_chances_and_a_huge_stack_cannot_be_caught(tables):
    sol = solve(R, one("flat", 100), tables, coarse=True)
    for v in sol.value:
        assert v.min() >= 0.0 and v.max() <= 1.0 + 1e-9
    live = [history_index(h) for h in ([2, 2, 2, 2], [4, 4, 4, 4], [2, 4, 3, 6])]
    assert sol.value[4][live, int(np.searchsorted(sol.grid, 20000.0))].min() >= 0.99
    assert [v.shape[0] for v in sol.value] == [1, 8, 64, 512, 4096]


def test_no_chips_means_no_bet(vs_turtle):
    assert (vs_turtle.bet[0][0][vs_turtle.grid < 5.0] == 0.0).all()
    assert vs_turtle.bet[0][0][int(np.searchsorted(vs_turtle.grid, 1000.0))] >= 5.0


def test_solution_round_trip(tmp_path, vs_turtle):
    save_solution(vs_turtle, tmp_path / "s.npz")
    again = load_solution(tmp_path / "s.npz")
    assert all(np.array_equal(a, b) for a, b in zip(again.value, vs_turtle.value))
    assert all(np.array_equal(a, b) for a, b in zip(again.bet, vs_turtle.bet))
    assert np.array_equal(again.grid, vs_turtle.grid)


def test_candidates_include_all_but_one_step(tables):
    from trainer.solver import _Candidates
    grid = np.array([0.0, 12.0, 20.0, 1007.5, 1160.0])
    cand = _Candidates(R, tables, grid, bet_menu(False))
    options = lambda chips: set(cand.bets[cand.chip_index == int(np.flatnonzero(grid == chips)[0])].tolist())
    assert {1155.0, 1160.0} <= options(1160.0) and {1000.0, 1005.0} <= options(1007.5)
    assert {15.0, 20.0} <= options(20.0)
    assert options(12.0) == {10.0} and options(0.0) == {0.0}
