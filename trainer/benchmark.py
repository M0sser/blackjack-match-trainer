"""Benchmark: play many full matches and report how often each way of betting finishes ahead.

Within a match everyone gets the same cards. The players all use Book play and differ only
in how they bet: the Brain player bets the solved main bet, the others follow a simple rule.
Each opponent plays its own style and betting rule. The report doubles as the brain's
acceptance test.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from trainer.fast import play_rounds, sample_sequences
from trainer.opponents import bets_for, initial_memory, legal_bet, next_memory
from trainer.styles import STYLE_INDEX, STYLE_NAMES, style_table
from trainer.tables import LEVELS, class_of

PLAYERS = ("Brain", "Minimum only", "Flat 100", "Flat 500", "25% of chips", "All-in once, then minimum")
ALL = "All opponents"
FIELDS = ("player", "opponent", "ahead", "level", "credit", "low", "high")
_ROOT = Path(__file__).resolve().parent.parent


def _by_unique(func, *columns) -> np.ndarray:
    """func(*values) for every row, calling it once per distinct combination of values."""
    keys = np.stack([np.asarray(c, dtype=np.float64) for c in columns], axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    values = np.array([func(*(float(v) for v in key)) for key in unique], dtype=np.float64)
    return values[inverse.reshape(-1)]


def _levels(chips: np.ndarray, bets: np.ndarray) -> np.ndarray:
    """min(3, (chips - bet) // bet) per row; 0 where there is no bet."""
    safe = np.where(bets > 0, bets, 1.0)
    return np.where(bets > 0, np.minimum(LEVELS - 1, (chips - bets) // safe), 0).astype(np.intp)


def _settle(results: np.ndarray, style: int, chips: np.ndarray, bets: np.ndarray):
    """Chips after the round and the result per chip, for each match (x is 0 when sitting out)."""
    x = results[style, _levels(chips, bets), np.arange(chips.size)]
    x = np.where(bets > 0, x, 0.0)
    return chips + bets * x, x


def _simple_bets(name: str, rules, round_no: int, chips: np.ndarray, all_in: np.ndarray) -> np.ndarray:
    if name == "Minimum only":
        wanted = lambda c: rules.min_bet
    elif name == "Flat 100":
        wanted = lambda c: 100.0
    elif name == "Flat 500":
        wanted = lambda c: 500.0
    elif name == "25% of chips":
        wanted = lambda c: 0.25 * c
    else:  # "All-in once, then minimum"
        return _by_unique(lambda c, go: legal_bet(rules, c, c if go else rules.min_bet), chips, all_in)
    return _by_unique(lambda c: legal_bet(rules, c, wanted(c)), chips)


def _opponent_bets(opp, rules, round_no: int, chips: np.ndarray, memory: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """Each match's bet; where the rule offers several bets, each match draws one by its chance."""
    keys = np.stack([chips, memory], axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    options = [bets_for(opp, rules, round_no, float(c), float(m)) for c, m in unique]
    bets = np.array([pairs[0][0] for pairs in options], dtype=np.float64)[inverse]
    if any(len(pairs) > 1 for pairs in options):
        u = rng.random(chips.size)
        for i, pairs in enumerate(options):
            if len(pairs) == 1:
                continue
            rows = np.flatnonzero(inverse == i)
            edges = np.cumsum([p for _, p in pairs])
            pick = np.minimum(np.searchsorted(edges, u[rows], side="right"), len(pairs) - 1)
            bets[rows] = np.array([b for b, _ in pairs])[pick]
    return bets


def _brain_bets(brain, round_no: int, chips: np.ndarray, history: np.ndarray) -> np.ndarray:
    """brain.table_bet for each match; `history` holds the classes so far, base 8."""
    def bet(hist: float, c: float) -> float:
        h = int(hist)
        classes = [(h >> (3 * (round_no - 2 - i))) & 7 for i in range(round_no - 1)]
        return brain.table_bet(round_no, c, classes)
    return _by_unique(bet, history, chips)


def _row(player: str, opponent: str, ahead: float, level: float, values: np.ndarray, tie: float) -> dict:
    credit = ahead + tie * level
    se = float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
    return {"player": player, "opponent": opponent, "ahead": ahead, "level": level, "credit": credit,
            "low": max(0.0, credit - 1.96 * se), "high": min(1.0, credit + 1.96 * se)}


def run_benchmark(brain, matches: int = 20_000, seed: int = 1) -> list[dict]:
    rules, population = brain.rules, brain.population
    rng = np.random.default_rng(seed)
    tables = [style_table(name, rules) for name in STYLE_NAMES]
    book = STYLE_INDEX["book"]
    start = float(rules.starting_chips)

    players = {name: np.full(matches, start) for name in PLAYERS}
    all_in = np.ones(matches, dtype=bool)          # the all-in player is still going all-in
    opp_chips = [np.full(matches, start) for _ in population]
    opp_memory = [np.full(matches, initial_memory(opp)) for opp in population]
    history = np.zeros(matches, dtype=np.int64)

    for round_no in range(1, rules.rounds + 1):
        seq = sample_sequences(rng, matches, rules.decks)
        results = np.array([[play_rounds(rules, seq, tables[s], k) for k in range(LEVELS)]
                            for s in range(len(STYLE_NAMES))])
        classes = class_of(results[book, LEVELS - 1])

        for name in PLAYERS:
            chips = players[name]
            if name == "Brain":
                bets = _brain_bets(brain, round_no, chips, history)
            else:
                bets = _simple_bets(name, rules, round_no, chips, all_in)
            players[name], x = _settle(results, book, chips, bets)
            if name == PLAYERS[-1]:
                all_in &= (bets > 0) & (x == 0)

        for i, opp in enumerate(population):
            chips, memory = opp_chips[i], opp_memory[i]
            bets = _opponent_bets(opp, rules, round_no, chips, memory, rng)
            opp_chips[i], x = _settle(results, STYLE_INDEX[opp.style], chips, bets)
            opp_memory[i] = _by_unique(lambda m, b, v: next_memory(opp, m, b, v), memory, bets, x)

        history = history * 8 + classes

    tie = float(rules.tie_credit)
    weights = np.array([opp.weight for opp in population], dtype=np.float64)
    weights = weights / weights.sum()
    rows = []
    for name in PLAYERS:
        mine = players[name]
        mix_ahead = mix_level = 0.0
        mix_values = np.zeros(matches)
        for opp, theirs, w in zip(population, opp_chips, weights):
            ahead, level = (mine > theirs).astype(np.float64), (mine == theirs).astype(np.float64)
            values = ahead + tie * level
            rows.append(_row(name, opp.name, float(ahead.mean()), float(level.mean()), values, tie))
            mix_ahead += w * float(ahead.mean())
            mix_level += w * float(level.mean())
            mix_values += w * values
        rows.append(_row(name, ALL, mix_ahead, mix_level, mix_values, tie))
    return rows


def main(argv=None) -> None:
    from trainer.brain import Brain, BrainConfig
    from trainer.settings import load_settings, population_from

    parser = argparse.ArgumentParser(description="How often each way of betting finishes ahead.")
    parser.add_argument("--matches", type=int, default=20_000)
    args = parser.parse_args(argv)

    settings = load_settings(_ROOT / "data")        # the same rules and opponents the trainer is using
    brain = Brain.prepare(settings.rules, population_from(settings), _ROOT / "data" / "cache", BrainConfig(),
                          progress=lambda f, m: print(f"{m} ({f:.0%})"))
    rows = run_benchmark(brain, matches=args.matches)

    out = _ROOT / "data" / "benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Brain's own estimate at the start: {brain.start_chance():.3f}")
    print(f"{'Player':<28}{'Ahead':>8}{'Level':>8}{'Credit':>8}   95% range")
    for r in rows:
        if r["opponent"] == ALL:
            print(f"{r['player']:<28}{r['ahead']:>8.3f}{r['level']:>8.3f}{r['credit']:>8.3f}"
                  f"   {r['low']:.3f}-{r['high']:.3f}")
    print(f"Every row written to {out}")


if __name__ == "__main__":
    main()
