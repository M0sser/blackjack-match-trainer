"""The brain: grades a bet or a play by the chance of finishing ahead of the hidden opponent.

It holds the outcome tables and the solved value tables, and scores any single decision
with one exact step forward into them.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from trainer.cards import VALUES, value_of
from trainer.engine import play_round
from trainer.fast import play_rounds, sample_completions
from trainer.opponents import MIXED_SHARES, OpponentType
from trainer.rules import Rules
from trainer.sidebets import fct_multipliers, pp_multiplier
from trainer.solver import (Solution, credit, final_distributions, history_index, load_solution,
                            save_solution, solve)
from trainer.styles import ACTIONS, STYLE_INDEX, make_style, style_table
from trainer.tables import LEVELS, N_CLASSES, Tables, build_tables, class_of, load_tables, save_tables

TOLERANCE = 0.005                      # a choice this close to the best counts as fine
_SIDE_MENU = (0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
_CLASS_SAMPLES = 2000
_CACHE_VERSION = 1
_FINALS_KEPT = 256

History = list                         # one dict per past round: class -> probability


@dataclass(frozen=True)
class BrainConfig:
    table_rounds: int = 1_000_000      # practice rounds dealt to build the outcome tables
    samples: int = 40_000              # deals of the unseen cards used to grade one play
    coarse: bool = False               # a rougher, much quicker solve (tests)
    seed: int = 20261114


@dataclass
class BetGrade:
    yours: tuple                       # (main, Perfect Pairs, 4 Card Total)
    best: tuple
    your_chance: float
    best_chance: float
    fine: bool


@dataclass
class PlaySpot:
    """Everything the player could see at one decision."""
    round_no: int
    chips_start: float                 # chips before this round's bets
    main_bet: float
    pp_bet: float
    fct_bet: float
    insurance_bet: float               # 0 unless insurance was already taken
    player: list                       # the two dealt cards
    upcard: int
    drawn: list                        # cards drawn since the deal, in table order
    script: list                       # play actions already taken this round
    legal: list
    stage: str                         # "insurance" or "play"


@dataclass
class PlayGrade:
    yours: str
    best: str
    chances: dict
    your_chance: float
    best_chance: float
    fine: bool


def _combos(history: Sequence[dict]) -> list:
    """Every way the past rounds may have gone, with its probability."""
    combos = [((), 1.0)]
    for dist in history:
        combos = [(classes + (int(g),), p * float(q))
                  for classes, p in combos for g, q in dist.items() if q > 0]
    kept = [(classes, p) for classes, p in combos if p >= 0.001] or combos
    total = sum(p for _, p in kept)
    return [(classes, p / total) for classes, p in kept]


def _fingerprint(rules: Rules, population: Sequence[OpponentType], config: BrainConfig) -> str:
    payload = {
        "version": _CACHE_VERSION,
        "rules": rules.to_dict(),
        "population": [[o.label, o.style, o.rule, float(o.param), round(float(o.weight), 12)] for o in population],
        "config": [config.table_rounds, config.coarse, config.seed],
        "mixed_shares": list(MIXED_SHARES),     # a change to how Mixed bets must re-solve
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _load(path: Path, loader):
    if not path.exists():
        return None
    try:
        return loader(path)
    except Exception:
        return None                     # a damaged cache file is rebuilt


def _save(path: Path, saver) -> None:
    part = path.with_name(path.stem + ".part.npz")
    saver(part)
    os.replace(part, path)


class Brain:
    def __init__(self, rules: Rules, population: Sequence[OpponentType], tables: Tables,
                 solution: Solution, config: BrainConfig = BrainConfig()):
        self.rules = rules
        self.population = list(population)
        self.tables = tables
        self.solution = solution
        self.config = config
        self._book = make_style("book", rules)
        self._book_table = style_table("book", rules)
        self._finals: dict = {}

    @classmethod
    def prepare(cls, rules: Rules, population: Sequence[OpponentType], cache_dir,
                config: BrainConfig = BrainConfig(),
                progress: Callable[[float, str], None] | None = None) -> "Brain":
        """Load the tables and the solution from the cache, building whatever is missing."""
        say = progress or (lambda fraction, message: None)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fp = _fingerprint(rules, population, config)
        tables_path = cache_dir / f"tables-{fp}.npz"
        solution_path = cache_dir / f"solution-{fp}.npz"

        tables = _load(tables_path, load_tables)
        if tables is None:
            say(0.0, "Dealing practice rounds")
            tables = build_tables(rules, config.table_rounds, config.seed)
            _save(tables_path, lambda p: save_tables(tables, p))

        solution = _load(solution_path, load_solution)
        if solution is None:
            say(0.1, "Solving the match")
            solution = solve(rules, population, tables, coarse=config.coarse,
                             progress=lambda f, m: say(0.1 + 0.9 * f, f"Solving the match: {m.lower()}"))
            _save(solution_path, lambda p: save_solution(solution, p))

        say(1.0, "Ready")
        return cls(rules, population, tables, solution, config)

    # ---- what the tables say ----------------------------------------------------

    def start_chance(self) -> float:
        """Chance of finishing ahead at the start of a match, betting as the solution says."""
        return float(np.interp(self.rules.starting_chips, self.solution.grid, self.solution.value[0][0]))

    def table_bet(self, round_no: int, chips: float, classes: Sequence[int]) -> float:
        """The solved table's main bet at the nearest stored chip total not above `chips`."""
        cap = self.rules.floor_bet(chips)
        if cap < self.rules.min_bet:
            return 0.0
        column = max(0, int(np.searchsorted(self.solution.grid, chips, side="right")) - 1)
        bet = float(self.solution.bet[round_no - 1][history_index(classes)][column])
        return float(min(max(bet, self.rules.min_bet), cap))

    def round_class(self, shoe: Sequence[int]) -> int:
        """The class of a round whose whole card order is known."""
        bet = self.rules.min_bet
        result = play_round(self.rules, [int(c) for c in shoe], bet * LEVELS, bet, self._book)
        return int(class_of(result.x))

    def round_class_dist(self, known: dict) -> dict:
        """The class of a round whose card order is only partly known (position -> card),
        as class -> probability, by dealing out the unseen cards."""
        known = {int(position): int(card) for position, card in known.items()}
        seed = int.from_bytes(hashlib.sha256(json.dumps(sorted(known.items())).encode()).digest()[:8], "big")
        seq = sample_completions(np.random.default_rng(seed ^ self.config.seed), _CLASS_SAMPLES, known,
                                 self.rules.decks)
        if 3 not in known:              # the hand was played out, so the dealer had no blackjack
            up, hole = VALUES[seq[:, 1]], VALUES[seq[:, 3]]
            blackjack = (up + hole == 11) & ((up == 1) | (hole == 1))
            if (~blackjack).any():
                seq = seq[~blackjack]
        classes = class_of(play_rounds(self.rules, seq, self._book_table, LEVELS - 1))
        counts = np.bincount(classes, minlength=N_CLASSES)
        return {g: float(count) / len(classes) for g, count in enumerate(counts) if count}

    # ---- scoring ----------------------------------------------------------------

    def _final_chips(self, classes: tuple) -> list:
        """The opponents' final chips for each class of the last round, given the earlier rounds."""
        found = self._finals.get(classes)
        if found is None:
            if len(self._finals) >= _FINALS_KEPT:
                self._finals.pop(next(iter(self._finals)))
            found = final_distributions(self.rules, self.population, self.tables, classes)
            self._finals[classes] = found
        return found

    def _next(self, round_no: int, combos: list, g: int, chips: np.ndarray) -> np.ndarray:
        """Chance of finishing ahead after round `round_no` came up class g and left `chips`."""
        out = np.zeros(len(chips))
        if round_no < self.rules.rounds:
            following = self.solution.value[round_no]
            for classes, p in combos:
                out += p * np.interp(chips, self.solution.grid, following[history_index(classes + (g,))])
        else:
            for classes, p in combos:
                final_chips, final_weights = self._final_chips(classes)[g]
                out += p * credit(final_chips, final_weights, chips, self.rules.tie_credit)
        return out

    def _score(self, round_no: int, combos: list, chips: float, mains, pps=None, fcts=None) -> np.ndarray:
        """Chance of finishing ahead for each bet, playing Book this round."""
        tables = self.tables
        mains = np.asarray(mains, dtype=np.float64)
        out = np.zeros(mains.size)
        sides = pps is not None
        if sides:
            pps, fcts = np.asarray(pps, dtype=np.float64), np.asarray(fcts, dtype=np.float64)
            left = chips - mains - pps - fcts
        else:
            left = chips - mains
        level = np.clip(np.floor(left / mains + 1e-9), 0, LEVELS - 1).astype(np.intp)
        book = STYLE_INDEX["book"]
        for g in range(N_CLASSES):
            p_class = float(tables.p_class[g])
            if p_class <= 0:
                continue
            for k in range(LEVELS):
                rows = np.flatnonzero(level == k)
                if rows.size == 0:
                    continue
                if sides:
                    for (x, pp_mult, fct_mult), q in zip(tables.side_vals[k, g], tables.side_probs[k, g]):
                        if q > 0:
                            final = chips + mains[rows] * x + pps[rows] * pp_mult + fcts[rows] * fct_mult
                            out[rows] += p_class * q * self._next(round_no, combos, g, final)
                else:
                    for x, q in zip(tables.vals[book, k, g], tables.probs[book, k, g]):
                        if q > 0:
                            out[rows] += p_class * q * self._next(round_no, combos, g, chips + mains[rows] * x)
        return out

    def _side_candidates(self, chips: float) -> np.ndarray:
        """(main, Perfect Pairs, 4 Card Total) bets tried in the last round, smallest side stakes first."""
        rules = self.rules
        menu = [s for s in _SIDE_MENU if s == 0 or abs(s / rules.bet_step - round(s / rules.bet_step)) < 1e-9]
        rows = []
        for pp in menu:
            for fct in menu:
                if pp == 0 and fct == 0:
                    continue
                cap = rules.floor_bet(chips - pp - fct)
                if cap < rules.min_bet:
                    continue
                mains = {rules.min_bet, cap}
                mains.update(float(m) for m in np.arange(25, min(1000.0, cap) + 1, 25))
                mains.update(float(m) for m in np.arange(100, cap + 1, 100))
                rows.extend((m, pp, fct) for m in mains if rules.min_bet <= m <= cap)
        rows.sort(key=lambda t: (t[1] + t[2], t[0]))
        return np.array(rows, dtype=np.float64).reshape(-1, 3)

    def grade_bet(self, round_no: int, chips: float, history: Sequence[dict], main_bet: float,
                  pp_bet: float = 0.0, fct_bet: float = 0.0) -> BetGrade:
        rules = self.rules
        cap = rules.floor_bet(chips)
        if cap < rules.min_bet:
            raise ValueError("no bet is possible with these chips")
        combos = _combos(history)

        mains = np.arange(rules.min_bet, cap + rules.bet_step / 2, rules.bet_step)
        scores = self._score(round_no, combos, chips, mains)
        best_chance = float(scores.max())
        best = (float(mains[int(np.argmax(scores >= best_chance - 1e-9))]), 0.0, 0.0)

        yours = (float(main_bet), float(pp_bet), float(fct_bet))
        if pp_bet or fct_bet:
            your_chance = float(self._score(round_no, combos, chips, [yours[0]], [yours[1]], [yours[2]])[0])
        else:
            your_chance = float(self._score(round_no, combos, chips, [yours[0]])[0])

        if round_no == rules.rounds:        # only here is a side bet's score exact, so only here is one recommended
            triples = self._side_candidates(chips)
            if len(triples):
                side_scores = self._score(round_no, combos, chips, triples[:, 0], triples[:, 1], triples[:, 2])
                top = int(np.argmax(side_scores >= side_scores.max() - 1e-9))
                if side_scores[top] > best_chance + TOLERANCE:
                    best = tuple(float(v) for v in triples[top])
                    best_chance = float(side_scores[top])

        if your_chance > best_chance:
            best, best_chance = yours, your_chance
        return BetGrade(yours=yours, best=best, your_chance=your_chance, best_chance=best_chance,
                        fine=your_chance >= best_chance - TOLERANCE)

    def _spot_seed(self, spot: PlaySpot) -> int:
        text = json.dumps([spot.round_no, spot.chips_start, spot.main_bet, spot.pp_bet, spot.fct_bet,
                           spot.insurance_bet, [int(c) for c in spot.player], int(spot.upcard),
                           [int(c) for c in spot.drawn], list(spot.script), spot.stage])
        return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") ^ self.config.seed

    def grade_play(self, spot: PlaySpot, history: Sequence[dict], chosen: str) -> PlayGrade:
        """Score every legal choice at this decision by dealing out the cards the player could not see."""
        if chosen not in spot.legal:
            raise ValueError(f"{chosen} is not one of the legal choices {spot.legal}")
        rules = self.rules
        combos = _combos(history)
        known = {0: int(spot.player[0]), 1: int(spot.upcard), 2: int(spot.player[1])}
        for i, card in enumerate(spot.drawn):
            known[4 + i] = int(card)
        seq = sample_completions(np.random.default_rng(self._spot_seed(spot)), self.config.samples, known,
                                 rules.decks)
        hole = VALUES[seq[:, 3]]
        if spot.stage == "play":            # the dealer has already checked for blackjack
            up_value = value_of(spot.upcard)
            if up_value == 1:
                seq = seq[hole != 10]
            elif up_value == 10:
                seq = seq[hole != 1]
            hole = VALUES[seq[:, 3]]
        n = len(seq)
        classes = class_of(play_rounds(rules, seq, self._book_table, LEVELS - 1))

        main = float(spot.main_bet)
        side = np.zeros(n)
        if spot.pp_bet:
            side += spot.pp_bet * pp_multiplier(spot.player[0], spot.player[1], rules.pp_pays)
        if spot.fct_bet:
            side += spot.fct_bet * fct_multipliers(seq, rules.fct_pays)

        ahead = {}
        for choice in spot.legal:
            if spot.stage == "insurance":
                stake = main / 2 if choice == "insurance" else 0.0
                insurance = np.where(hole == 10, 2 * stake, -stake)
                script = ()
            else:
                stake = float(spot.insurance_bet)
                insurance = np.full(n, -stake)
                script = [ACTIONS.index(a) for a in list(spot.script) + [choice]]
            left = spot.chips_start - main - spot.pp_bet - spot.fct_bet - stake
            extra = max(0, min(LEVELS - 1, int(left // main)))
            x = play_rounds(rules, seq, self._book_table, extra, script=script)
            final = spot.chips_start + main * x + side + insurance
            chance = np.zeros(n)
            for g in range(N_CLASSES):
                rows = classes == g
                if rows.any():
                    chance[rows] = self._next(spot.round_no, combos, g, final[rows])
            ahead[choice] = chance

        chances = {choice: float(values.mean()) for choice, values in ahead.items()}
        best = max(spot.legal, key=lambda choice: chances[choice])
        gap = ahead[best] - ahead[chosen]
        shortfall = float(gap.mean())
        noise = float(gap.std() / np.sqrt(n)) if n > 1 else 0.0
        return PlayGrade(yours=chosen, best=best, chances=chances, your_chance=chances[chosen],
                         best_chance=chances[best], fine=shortfall <= TOLERANCE or shortfall <= 2 * noise)
