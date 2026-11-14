"""The practice table: one five-round match against a hidden opponent, step by step.

PracticeMatch deals the match, takes the player's bets and plays, keeps the clock and
reveals grades: after each round in Coach mode, only when the match is over in Match mode.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np

from trainer.brain import Brain, PlaySpot
from trainer.cards import card_str, hand_total, new_shoe, value_of
from trainer.engine import HandView, Round, play_round
from trainer.errors import TrainerError
from trainer.opponents import OpponentType, bets_for, initial_memory, next_memory
from trainer.rules import Rules
from trainer.styles import make_style

MODES = ("coach", "match")


def replay_opponent(rules: Rules, opp: OpponentType, shoes: Sequence, rng: np.random.Generator) -> float:
    """The opponent's final chips over these shoes, played with its own style and betting rule."""
    style = make_style(opp.style, rules)
    chips, memory = float(rules.starting_chips), initial_memory(opp)
    for round_no, shoe in enumerate(shoes, 1):
        bets = bets_for(opp, rules, round_no, chips, memory)
        if bets[0][0] == 0.0:
            break
        if opp.rule == "random":
            bet = bets[int(rng.choice(len(bets), p=[p for _, p in bets]))][0]
        else:
            bet = bets[0][0]
        result = play_round(rules, shoe, chips, bet, style)
        chips = result.chips_after
        memory = next_memory(opp, memory, bet, result.x)
    return float(chips)


# ---- text for grades ---------------------------------------------------------------

def _chips(amount: float) -> str:
    """1150.0 -> "1,150", 1007.5 -> "1,007.5"."""
    return f"{amount:,.1f}".rstrip("0").rstrip(".")


def _bet_text(main: float, pp: float, fct: float) -> str:
    text = _chips(main)
    if pp:
        text += f" + {_chips(pp)} on Perfect Pairs"
    if fct:
        text += f" + {_chips(fct)} on 4 Card Total"
    return text


def _action_name(action: str) -> str:
    return action.replace("_", " ").capitalize()


def _hand_text(total: int, soft: bool, upcard: int) -> str:
    up = "A" if value_of(upcard) == 1 else str(value_of(upcard))
    return f"{'soft ' if soft else ''}{total} against {up}"


def _bet_grade(record: dict, grade) -> dict:
    if grade.fine:
        reason = "Good bet."
    elif sum(grade.best) < sum(grade.yours):
        reason = "A smaller bet finishes ahead more often from here."
    else:
        reason = "A bigger bet finishes ahead more often from here."
    return {"kind": "bet", "hand": None,
            "spot": f"Round {record['round']}, {_chips(record['chips_start'])} chips",
            "yours": _bet_text(*grade.yours), "best": _bet_text(*grade.best),
            "your_chance": grade.your_chance, "best_chance": grade.best_chance,
            "fine": bool(grade.fine), "reason": reason}


def _play_grade(decision: dict, grade) -> dict:
    yours, best = _action_name(grade.yours), _action_name(grade.best)
    reason = "Good play." if grade.fine else f"{best} finishes ahead more often than {yours.lower()} here."
    return {"kind": decision["spot"].stage, "hand": decision["hand"], "spot": decision["text"],
            "yours": yours, "best": best, "your_chance": grade.your_chance,
            "best_chance": grade.best_chance, "fine": bool(grade.fine), "reason": reason}


# ---- the match -----------------------------------------------------------------------

class PracticeMatch:
    """One practice match. Every public method returns the new state; a call that does not
    fit the phase raises TrainerError and changes nothing."""

    def __init__(self, rules: Rules, brain: Brain, population: Sequence[OpponentType], mode: str,
                 seed: int, now: Callable[[], float] = time.monotonic, shoes=None, opponent=None):
        if mode not in MODES:
            raise ValueError(f"unknown practice mode: {mode}")
        self.rules, self.brain, self.mode = rules, brain, mode
        self._now = now
        rng = np.random.default_rng(seed)
        weights = np.array([o.weight for o in population], dtype=np.float64)
        drawn = population[int(rng.choice(len(population), p=weights / weights.sum()))]
        dealt = [new_shoe(rng, rules.decks) for _ in range(rules.rounds)]
        self._opponent = opponent if opponent is not None else drawn
        self._shoes = list(shoes) if shoes is not None else dealt
        self._opponent_chips = replay_opponent(rules, self._opponent, self._shoes, rng)

        self._phase = "bet"
        self._round_no = 1
        self._chips = float(rules.starting_chips)     # chips before the current round's bets
        self._round: Round | None = None
        self._record: dict | None = None               # the round on the table
        self._records: list[dict] = []                 # finished rounds
        self._classes: list[int] = []                  # brain.round_class of finished rounds
        self._last_bet: dict | None = None
        self._forfeit = False
        self._step = 0                                 # goes up with every change; names the screen a click was made on
        self._clock_started = False
        self._used = 0.0                               # running seconds banked while stopped
        self._since: float | None = None               # when the clock last started running

    # ---- public calls ---------------------------------------------------------------

    def state(self) -> dict:
        self._tick()
        return self._snapshot()

    def bet(self, main, pp=0.0, fct=0.0, step=None) -> dict:
        self._tick()
        self._check_step(step)
        self._require("bet")
        round_ = Round(self.rules, self._shoes[self._round_no - 1], self._chips, main, pp, fct)
        self._round = round_
        self._record = {"round": self._round_no, "chips_start": self._chips,
                        "player": list(round_.hands[0].cards), "decisions": [], "grades": None}
        self._last_bet = {"main": round_.main_bet, "pp": round_.pp_bet, "fct": round_.fct_bet}
        self._sync()
        self._step += 1
        return self._snapshot()

    def act(self, action: str, step=None) -> dict:
        self._tick()
        self._check_step(step)
        self._require("insurance", "play")
        round_ = self._round
        hand_index = round_.active
        total, soft = hand_total(round_.hands[hand_index].cards)
        spot = PlaySpot(round_no=self._round_no, chips_start=self._chips, main_bet=round_.main_bet,
                        pp_bet=round_.pp_bet, fct_bet=round_.fct_bet, insurance_bet=round_.insurance_bet,
                        player=list(self._record["player"]), upcard=round_.upcard,
                        drawn=list(round_.shoe[4:round_.pos]), script=list(round_.actions),
                        legal=round_.legal_actions(), stage=self._phase)
        round_.act(action)
        self._record["decisions"].append({"spot": spot, "chosen": action, "hand": hand_index + 1,
                                          "text": _hand_text(total, soft, round_.upcard)})
        self._sync()
        self._step += 1
        return self._snapshot()

    def next_round(self, step=None) -> dict:
        self._tick()
        self._check_step(step)
        self._require("round_over")
        if self._round_no >= self.rules.rounds or self._chips < self.rules.min_bet:
            self._end_match()
        else:
            self._round_no += 1
            self._round = self._record = None
            self._phase = "bet"
            self._run_clock()
        self._step += 1
        return self._snapshot()

    def current_view(self) -> HandView:
        if self._phase != "play":
            raise TrainerError("There is no hand to play right now.")
        return self._round.view()

    # ---- phases ----------------------------------------------------------------------

    def _check_step(self, step) -> None:
        """Refuse a click made on a screen that has since changed (the second half of a double-click)."""
        if step is None or self._phase == "match_over":
            return
        if step != self._step:
            raise TrainerError("That click came twice, so the second one was ignored.")

    def _require(self, *phases: str) -> None:
        if self._phase in phases:
            return
        if self._phase == "match_over":
            raise TrainerError("Time ran out, so the match is forfeit." if self._forfeit else "The match is over.")
        if self._phase == "bet":
            raise TrainerError("First place your bet.")
        if self._phase == "round_over":
            raise TrainerError("This round is over. Go on to the next round.")
        raise TrainerError("Finish this round first.")

    def _sync(self) -> None:
        """Follow the round's phase; book the result when it ends."""
        if self._round.phase != "done":
            self._phase = self._round.phase
            return
        self._phase = "round_over"
        result = self._round.result
        self._record.update(net=result.net, chips_after=result.chips_after,
                            bet=self._bet_dict(), history=[{g: 1.0} for g in self._classes])
        self._records.append(self._record)
        self._classes.append(self.brain.round_class(self._shoes[self._round_no - 1]))
        self._chips = result.chips_after
        if self.mode == "coach":
            self._stop_clock()

    def _end_match(self, forfeit: bool = False) -> None:
        self._forfeit = forfeit
        self._phase = "match_over"
        self._step += 1
        self._stop_clock()

    # ---- the clock -------------------------------------------------------------------

    def _elapsed(self) -> float:
        if self._since is None:
            return self._used
        return self._used + (self._now() - self._since)

    def _run_clock(self) -> None:
        if self._since is None and self._used < self.rules.clock_seconds:
            self._since = self._now()

    def _stop_clock(self) -> None:
        self._used = self._elapsed()
        self._since = None

    def _tick(self) -> None:
        """Start the clock on the first call; act on time running out."""
        if not self._clock_started:
            self._clock_started = True
            self._run_clock()
        if self._phase == "match_over" or self._elapsed() < self.rules.clock_seconds:
            return
        if self.mode == "match":
            self._end_match(forfeit=True)
        else:
            self._stop_clock()
            self._used = float(self.rules.clock_seconds)

    # ---- grades ----------------------------------------------------------------------

    def _grades(self, record: dict) -> list[dict]:
        """The grades of one finished round, computed the first time they are shown."""
        if record["grades"] is None:
            bet, history = record["bet"], record["history"]
            grades = [_bet_grade(record, self.brain.grade_bet(record["round"], record["chips_start"], history,
                                                              bet["main"], bet["pp"], bet["fct"]))]
            for decision in record["decisions"]:
                grades.append(_play_grade(decision, self.brain.grade_play(decision["spot"], history,
                                                                          decision["chosen"])))
            record["grades"] = grades
        return record["grades"]

    def _grades_shown(self) -> list[dict] | None:
        if self._round is None or self._round.phase != "done":
            return None
        if self._phase == "match_over" or (self.mode == "coach" and self._phase == "round_over"):
            return self._grades(self._record)
        return None

    # ---- the state -------------------------------------------------------------------

    def _bet_dict(self) -> dict | None:
        r = self._round
        if r is None:
            return None
        return {"main": r.main_bet, "pp": r.pp_bet, "fct": r.fct_bet, "insurance": r.insurance_bet}

    def _summary(self) -> dict | None:
        if self._phase != "match_over":
            return None
        you, them = self._chips, self._opponent_chips
        if self._forfeit:
            outcome = "forfeit"
        else:
            outcome = "ahead" if you > them else "behind" if you < them else "level"
        rounds = [{"round": r["round"], "bet": r["bet"], "net": r["net"], "chips_after": r["chips_after"],
                   "grades": self._grades(r)} for r in self._records]
        return {"you": you, "opponent": them, "opponent_name": self._opponent.name, "outcome": outcome,
                "start_chance": self.brain.start_chance(), "rounds": rounds}

    def _snapshot(self) -> dict:
        r = self._round
        done = r is not None and r.phase == "done"
        limit = self.rules.clock_seconds
        hands, dealer, result = [], [], None
        if r is not None:
            for i, hand in enumerate(r.hands):
                total, soft = hand_total(hand.cards)
                hands.append({"cards": [card_str(c) for c in hand.cards], "total": total, "soft": soft,
                              "stake": hand.stake, "active": self._phase == "play" and i == r.active,
                              "outcome": hand.outcome})
            dealer = [None if c is None else card_str(c) for c in r.visible_dealer()]
        if done:
            res = r.result
            result = {"main": res.main_net, "pp": res.pp_net, "fct": res.fct_net,
                      "insurance": res.insurance_net, "net": res.net}
        return {
            "mode": self.mode,
            "phase": self._phase,
            "step": self._step,
            "round": self._round_no,
            "rounds": self.rules.rounds,
            "chips": r.chips if self._phase in ("insurance", "play") else self._chips,
            "clock": {"left": max(0.0, limit - self._elapsed()), "limit": limit,
                      "running": self._since is not None},
            "bet": self._bet_dict(),
            "last_bet": self._last_bet,
            "dealer": dealer,
            "dealer_total": hand_total(r.result.dealer_cards)[0] if done else None,
            "hands": hands,
            "legal": r.legal_actions() if r is not None and self._phase in ("insurance", "play") else [],
            "result": result,
            "grades": self._grades_shown(),
            "summary": self._summary(),
        }
