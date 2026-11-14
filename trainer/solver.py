"""Solve the match backwards.

For every round, every history of round classes and every chip total, find the main bet
that gives the best chance of finishing ahead of a hidden opponent drawn from the
population (equal chips count `rules.tie_credit`).

Opponents go forward: their chips after any history follow from their betting rule and
the outcome tables. The player goes backward: round five is scored against the opponents'
final chips, round four against round five's answers, and so on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from trainer.opponents import OpponentType, bets_for, initial_memory, next_memory
from trainer.rules import Rules
from trainer.styles import STYLE_INDEX
from trainer.tables import LEVELS, N_CLASSES, Tables

_PRUNE = 1e-7          # opponent states rarer than this share are dropped
_MAX_STATES = 80       # beyond this many, an opponent's chip totals are merged into wider buckets
_BUCKETS = (5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0)


# ---- grids ---------------------------------------------------------------------------

def chip_grid(coarse: bool = False) -> np.ndarray:
    """Chip totals the solution is stored on. Values in between are interpolated."""
    if coarse:
        parts = [np.arange(0, 900, 25), np.arange(900, 1100, 5), np.arange(1100, 3000, 25),
                 np.arange(3000, 10000, 100), np.arange(10000, 40001, 1000), [100000]]
    else:
        parts = [np.arange(0, 2000, 5), np.arange(2000, 5000, 25), np.arange(5000, 10000, 100),
                 np.arange(10000, 40001, 1000), [100000]]
    return np.unique(np.concatenate(parts)).astype(np.float64)


def bet_menu(coarse: bool = False) -> np.ndarray:
    """Main bets the solver tries. All-in is always tried as well."""
    if coarse:
        menu = [5, 10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000,
                1500, 2000, 3000, 5000, 10000, 20000, 50000]
    else:
        menu = np.concatenate([np.arange(5, 101, 5), np.arange(125, 1001, 25), np.arange(1100, 5001, 100),
                               np.arange(5500, 20001, 500), np.arange(22500, 100001, 2500)])
    return np.asarray(menu, dtype=np.float64)


def history_index(classes: Sequence[int]) -> int:
    """A history as one number: base 8, first round most significant."""
    index = 0
    for g in classes:
        index = index * N_CLASSES + int(g)
    return index


# ---- opponents forward ---------------------------------------------------------------

def _outcome_lists(tables: Tables) -> list:
    """outcomes[style][level][class] = [(result per chip, chance), ...] as plain floats."""
    styles, levels = tables.vals.shape[0], tables.vals.shape[1]
    return [[[[(float(v), float(q)) for v, q in zip(tables.vals[s, k, g], tables.probs[s, k, g]) if q > 0]
              for g in range(N_CLASSES)] for k in range(levels)] for s in range(styles)]


def _level(chips: float, bet: float) -> int:
    """How many extra stakes are affordable after the bet, capped at 3."""
    return min(LEVELS - 1, int((chips - bet) // bet))


def _tidy(states: dict) -> dict:
    """Drop negligible states, merge chip totals if there are too many, and renormalise."""
    if not states:
        return states
    total = sum(states.values())
    floor = _PRUNE * total
    states = {key: p for key, p in states.items() if p >= floor}
    for width in _BUCKETS:
        if len(states) <= _MAX_STATES:
            break
        merged: dict = {}
        for (chips, memory), p in states.items():
            key = (round(chips / width) * width, memory)
            merged[key] = merged.get(key, 0.0) + p
        states = merged
    total = sum(states.values())
    if total != 1.0:
        states = {key: p / total for key, p in states.items()}
    return states


def _step(rules: Rules, opp: OpponentType, outcomes: list, states: dict, round_no: int, g: int) -> dict:
    """One round of class g for one opponent: (chips, memory) -> chance."""
    by_level = outcomes[STYLE_INDEX[opp.style]]
    out: dict = {}
    for (chips, memory), p in states.items():
        for bet, share in bets_for(opp, rules, round_no, chips, memory):
            weight = p * share
            if bet <= 0.0:
                out[(chips, memory)] = out.get((chips, memory), 0.0) + weight
                continue
            for v, q in by_level[_level(chips, bet)][g]:
                key = (chips + bet * v, next_memory(opp, memory, bet, v))
                out[key] = out.get(key, 0.0) + weight * q
    return _tidy(out)


def _start(rules: Rules, opp: OpponentType) -> dict:
    return {(float(rules.starting_chips), initial_memory(opp)): 1.0}


def propagate(rules: Rules, opp: OpponentType, tables: Tables, history: Sequence[int]) -> dict:
    """The opponent's (chips, memory) -> chance after the given history of round classes."""
    outcomes = _outcome_lists(tables)
    states = _start(rules, opp)
    for round_no, g in enumerate(history, start=1):
        states = _step(rules, opp, outcomes, states, round_no, int(g))
    return states


def _last_round_atoms(rules: Rules, population: Sequence[OpponentType], states_by_opp: Sequence[dict],
                      round_no: int):
    """Everything the population is about to do in the last round, as flat arrays."""
    chips, bets, styles, levels, weights = [], [], [], [], []
    for opp, states in zip(population, states_by_opp):
        style = STYLE_INDEX[opp.style]
        for (c, memory), p in states.items():
            for bet, share in bets_for(opp, rules, round_no, c, memory):
                chips.append(c)
                bets.append(bet)
                styles.append(style)
                levels.append(_level(c, bet) if bet > 0 else 0)
                weights.append(opp.weight * p * share)
    return (np.array(chips, dtype=np.float64), np.array(bets, dtype=np.float64),
            np.array(styles, dtype=np.intp), np.array(levels, dtype=np.intp), np.array(weights, dtype=np.float64))


def _final_from_atoms(atoms, tables: Tables, g: int) -> tuple[np.ndarray, np.ndarray]:
    """Distinct final chip totals (ascending) and the weight on each, for last-round class g."""
    chips, bets, styles, levels, weights = atoms
    if chips.size == 0:
        return np.zeros(0), np.zeros(0)
    final = chips[:, None] + bets[:, None] * tables.vals[styles, levels, g]
    weight = weights[:, None] * tables.probs[styles, levels, g]
    live = weight > 0
    final, weight = final[live], weight[live]
    if final.size == 0:
        return np.zeros(0), np.zeros(0)
    order = np.argsort(final, kind="stable")
    final, weight = final[order], weight[order]
    first = np.flatnonzero(np.concatenate(([True], final[1:] != final[:-1])))
    return final[first], np.add.reduceat(weight, first)


def final_distributions(rules: Rules, population: Sequence[OpponentType], tables: Tables,
                        history: Sequence[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each class of the last round: the population's final chip totals and weights,
    given the classes of all earlier rounds."""
    outcomes = _outcome_lists(tables)
    states_by_opp = []
    for opp in population:
        states = _start(rules, opp)
        for round_no, g in enumerate(history, start=1):
            states = _step(rules, opp, outcomes, states, round_no, int(g))
        states_by_opp.append(states)
    atoms = _last_round_atoms(rules, population, states_by_opp, len(history) + 1)
    return [_final_from_atoms(atoms, tables, g) for g in range(N_CLASSES)]


def final_distribution(rules: Rules, population: Sequence[OpponentType], tables: Tables,
                       history4: Sequence[int], g5: int) -> tuple[np.ndarray, np.ndarray]:
    return final_distributions(rules, population, tables, history4)[int(g5)]


def credit(final_chips: np.ndarray, final_weights: np.ndarray, my_chips: np.ndarray, tie_credit: float) -> np.ndarray:
    """Chance of finishing ahead with each of `my_chips`: the weight strictly below,
    plus tie_credit times the weight exactly level."""
    cumulative = np.concatenate(([0.0], np.cumsum(final_weights)))
    lo = np.searchsorted(final_chips, my_chips, side="left")
    hi = np.searchsorted(final_chips, my_chips, side="right")
    return cumulative[lo] + tie_credit * (cumulative[hi] - cumulative[lo])


def _credit_at_sorted(targets: np.ndarray, final_chips: np.ndarray, final_weights: np.ndarray,
                      tie_credit: float) -> np.ndarray:
    """credit() for a long sorted array of targets, by merging instead of searching per target."""
    n = targets.size
    lo = np.searchsorted(targets, final_chips, side="left")
    hi = np.searchsorted(targets, final_chips, side="right")
    steps = (np.bincount(hi, weights=final_weights * (1.0 - tie_credit), minlength=n + 1)
             + np.bincount(lo, weights=final_weights * tie_credit, minlength=n + 1))
    return np.cumsum(steps)[:n]


# ---- the player backward -------------------------------------------------------------

@dataclass
class Solution:
    grid: np.ndarray
    value: list     # value[r - 1][history, chip]: chance of finishing ahead at the start of round r
    bet: list       # bet[r - 1][history, chip]: best main bet there (0.0 where no bet is possible)


class _Candidates:
    """Every (chip total, bet) pair the solver scores, and where each lands for each class."""

    def __init__(self, rules: Rules, tables: Tables, grid: np.ndarray, menu: np.ndarray):
        chip_index, bets = [], []
        for i, chips in enumerate(grid):
            cap = rules.floor_bet(chips)
            if cap < rules.min_bet:
                chip_index.append(i)        # no bet possible: the round is skipped
                bets.append(0.0)
                continue
            options = [b for b in menu if rules.min_bet <= b <= cap]
            if not options or options[-1] != cap:
                options.append(cap)         # all-in
            chip_index.extend([i] * len(options))
            bets.extend(options)
        self.chip_index = np.array(chip_index, dtype=np.intp)
        self.bets = np.array(bets, dtype=np.float64)
        self.count = self.bets.size
        self.starts = np.flatnonzero(np.concatenate(([True], self.chip_index[1:] != self.chip_index[:-1])))
        chips = grid[self.chip_index]
        safe = np.where(self.bets > 0, self.bets, 1.0)
        level = np.where(self.bets > 0, np.minimum(LEVELS - 1, np.floor((chips - self.bets) / safe)), 0).astype(np.intp)

        book = STYLE_INDEX["book"]
        self.targets, self.pair, self.weight = [], [], []
        for g in range(N_CLASSES):
            targets, pair, weight = [], [], []
            for k in range(LEVELS):
                rows = np.flatnonzero(level == k)
                for v, q in zip(tables.vals[book, k, g], tables.probs[book, k, g]):
                    if q > 0 and rows.size:
                        targets.append(chips[rows] + self.bets[rows] * v)
                        pair.append(rows)
                        weight.append(np.full(rows.size, tables.p_class[g] * q))
            if targets:
                targets, pair, weight = np.concatenate(targets), np.concatenate(pair), np.concatenate(weight)
                order = np.argsort(targets, kind="stable")
                targets, pair, weight = targets[order], pair[order], weight[order]
            else:
                targets, pair, weight = np.zeros(0), np.zeros(0, dtype=np.intp), np.zeros(0)
            self.targets.append(targets)
            self.pair.append(pair)
            self.weight.append(weight)

    def best(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per chip total: the best score and the smallest bet that reaches it."""
        top = np.maximum.reduceat(scores, self.starts)
        reaches = scores >= top[self.chip_index] - 1e-9
        first = np.minimum.reduceat(np.where(reaches, np.arange(self.count), self.count), self.starts)
        return top, self.bets[first]


def solve(rules: Rules, population: Sequence[OpponentType], tables: Tables, coarse: bool = False,
          progress: Callable[[float, str], None] | None = None) -> Solution:
    grid = chip_grid(coarse)
    cand = _Candidates(rules, tables, grid, bet_menu(coarse))
    outcomes = _outcome_lists(tables)
    live_classes = [g for g in range(N_CLASSES) if tables.p_class[g] > 0 and cand.targets[g].size]
    rounds = rules.rounds
    value = [np.zeros((N_CLASSES ** r, grid.size)) for r in range(rounds)]
    bet = [np.zeros((N_CLASSES ** r, grid.size)) for r in range(rounds)]
    tie = rules.tie_credit
    leaves = N_CLASSES ** (rounds - 1)
    seen = [0]

    def last_round(index: int, states_by_opp: list) -> None:
        atoms = _last_round_atoms(rules, population, states_by_opp, rounds)
        scores = np.zeros(cand.count)
        for g in live_classes:
            chips, weights = _final_from_atoms(atoms, tables, g)
            ahead = _credit_at_sorted(cand.targets[g], chips, weights, tie)
            scores += np.bincount(cand.pair[g], weights=cand.weight[g] * ahead, minlength=cand.count)
        value[rounds - 1][index], bet[rounds - 1][index] = cand.best(scores)
        seen[0] += 1
        if progress and seen[0] % 64 == 0:
            progress(0.9 * seen[0] / leaves, "Working out the last round")

    def walk(depth: int, index: int, states_by_opp: list) -> None:
        if depth == rounds - 1:
            last_round(index, states_by_opp)
            return
        for g in range(N_CLASSES):
            if tables.p_class[g] <= 0:
                continue
            following = [_step(rules, opp, outcomes, states, depth + 1, g)
                         for opp, states in zip(population, states_by_opp)]
            walk(depth + 1, index * N_CLASSES + g, following)

    walk(0, 0, [_start(rules, opp) for opp in population])

    for r in range(rounds - 1, 0, -1):          # rounds 4, 3, 2, 1
        following = value[r]
        for index in range(N_CLASSES ** (r - 1)):
            scores = np.zeros(cand.count)
            for g in live_classes:
                ahead = np.interp(cand.targets[g], grid, following[index * N_CLASSES + g])
                scores += np.bincount(cand.pair[g], weights=cand.weight[g] * ahead, minlength=cand.count)
            value[r - 1][index], bet[r - 1][index] = cand.best(scores)
        if progress:
            progress(0.9 + 0.1 * (rounds - r) / (rounds - 1), "Working back to the first round")

    return Solution(grid=grid, value=value, bet=bet)


def save_solution(solution: Solution, path) -> None:
    arrays = {"grid": solution.grid}
    for r, (v, b) in enumerate(zip(solution.value, solution.bet), start=1):
        arrays[f"value_{r}"] = v
        arrays[f"bet_{r}"] = b
    np.savez_compressed(path, **arrays)


def load_solution(path) -> Solution:
    with np.load(path) as data:
        rounds = sum(1 for name in data.files if name.startswith("value_"))
        return Solution(grid=data["grid"],
                        value=[data[f"value_{r}"] for r in range(1, rounds + 1)],
                        bet=[data[f"bet_{r}"] for r in range(1, rounds + 1)])
