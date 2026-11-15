"""Match review: playing rounds typed in by hand, check it adds up, then grade it.

`check_rounds` only replays and totals the chips; it never grades. `analyze_match`
grades.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from trainer.brain import Brain, PlaySpot
from trainer.cards import SUITS, hand_total, parse_card, rank_of, value_of
from trainer.engine import Round
from trainer.errors import TrainerError
from trainer.history import save_review_analysis
from trainer.rules import Rules

_STAND_IN_DOWN = parse_card("2")    # used for a down card never seen; it never makes a dealer blackjack

_DID = {"hit": "hit on", "stand": "stood on", "double": "doubled on", "split": "split on",
        "insurance": "took insurance with", "no_insurance": "turned down insurance with"}
_DOING = {"hit": "hitting", "stand": "standing", "double": "doubling", "split": "splitting",
          "insurance": "taking insurance", "no_insurance": "turning it down"}


@dataclass
class _Decision:
    """One choice the player made, with what could be seen at that moment."""
    stage: str
    hand: int                          # 1 or 2
    cards: list                        # the hand being played
    drawn: list
    script: list
    legal: list
    insurance_bet: float
    choice: str


@dataclass
class _Replay:
    round_: Round
    shoe: list
    down_seen: bool
    decisions: list = field(default_factory=list)


# ---- reading what was typed in ----------------------------------------------------

def _card(text) -> int:
    return parse_card(text if isinstance(text, str) else str(text))


def _player_cards(data: dict) -> list:
    player = data["player"]
    if not isinstance(player, list) or len(player) != 2:
        raise TrainerError("enter your two cards.")
    cards = [_card(c) for c in player]
    suits = data.get("suits")
    if suits is None:
        return cards
    if not isinstance(suits, list) or len(suits) != 2:
        raise TrainerError("enter a suit for each of your two cards.")
    out = []
    for card, suit in zip(cards, suits):
        letter = str(suit).strip().upper()
        if letter not in SUITS:
            raise TrainerError(f'"{suit}" is not a suit. Use S, H, D or C.')
        out.append(rank_of(card) * 4 + SUITS.index(letter))
    return out


def _list(data: dict, key: str) -> list:
    value = data.get(key) or []
    if not isinstance(value, list):
        raise TrainerError(f"\"{key}\" has to be a list.")
    return value


# ---- the replay -------------------------------------------------------------------

def _replay(rules: Rules, data: dict, chips: float) -> _Replay:
    """Play the round as entered on the reference engine. Raises TrainerError."""
    player = _player_cards(data)
    upcard = _card(data["upcard"])
    dealer = [_card(c) for c in _list(data, "dealer")]
    actions = _list(data, "actions")
    down_seen = bool(dealer)
    shoe = [player[0], upcard, player[1], dealer[0] if down_seen else _STAND_IN_DOWN] + dealer[1:]

    round_ = Round(rules, shoe, chips, data["main"], data.get("pp", 0), data.get("fct", 0))
    if round_.pp_bet and rank_of(player[0]) == rank_of(player[1]) and data.get("suits") is None:
        raise TrainerError("enter the suits of your pair for the Perfect Pairs bet.")
    replay = _Replay(round_, shoe, down_seen)
    try:
        if round_.phase == "insurance":
            _act(replay, "insurance" if data.get("insurance") else "no_insurance", [])
        elif data.get("insurance"):
            round_.act("insurance")            # refused with the engine's own message
        for entry in actions:
            _act(replay, str(entry["action"]).strip().lower(), _action_cards(entry))
    except IndexError:
        raise TrainerError(_short_dealer(replay, upcard, dealer)) from None
    _check_finish(replay)
    return replay


def _action_cards(entry: dict) -> list:
    action = str(entry["action"]).strip().lower()
    if action == "split":
        cards = entry["cards"]
        if not isinstance(cards, list) or len(cards) != 2:
            raise TrainerError("a split needs the second card of each hand.")
        return [_card(c) for c in cards]
    if action in ("hit", "double"):
        return [_card(entry["card"])]
    return []


def _act(replay: _Replay, action: str, cards: list) -> None:
    """Record what the player could see, put the action's cards next in the shoe, and act."""
    round_ = replay.round_
    if round_.phase == "done":
        raise TrainerError(f'the hand was already over, so there was no "{action}".')
    if round_.phase in ("insurance", "play"):
        hand = round_.hands[round_.active]
        replay.decisions.append(_Decision(
            stage=round_.phase, hand=round_.active + 1, cards=list(hand.cards),
            drawn=list(replay.shoe[4:round_.pos]), script=list(round_.actions),
            legal=round_.legal_actions(), insurance_bet=round_.insurance_bet, choice=action))
    replay.shoe[round_.pos:round_.pos] = cards
    round_.act(action)


def _short_dealer(replay: _Replay, upcard: int, dealer: list) -> str:
    if not replay.down_seen:
        return "enter the dealer's face-down card."
    total = hand_total([upcard] + dealer)[0]
    return f"the dealer had {total} and must draw again."


def _all_bust(round_: Round) -> bool:
    return all(hand_total(h.cards)[0] > 21 for h in round_.hands)


def _check_finish(replay: _Replay) -> None:
    """Once the round is over, every entered card must have been used and the down card known if needed."""
    round_ = replay.round_
    if round_.phase != "done":
        return
    if not replay.down_seen and (round_.fct_bet or not _all_bust(round_)):
        raise TrainerError("enter the dealer's face-down card.")
    if round_.result.cards_used < len(replay.shoe):
        if round_.actions and not _all_bust(round_):
            total = hand_total(round_.result.dealer_cards)[0]
            raise TrainerError(f"the dealer had {total} and would not draw.")
        raise TrainerError("the dealer would not draw: the round was already decided.")


def _replay_or_message(rules: Rules, n: int, data, chips: float):
    """(replay, None) or (None, a message starting "Round n:")."""
    try:
        if not isinstance(data, dict):
            raise TrainerError("this round is not filled in.")
        return _replay(rules, data, chips), None
    except TrainerError as error:
        return None, f"Round {n}: {error}"
    except KeyError as error:
        return None, f"Round {n}: \"{error.args[0]}\" is missing."
    except Exception:
        return None, f"Round {n}: something in this round is not filled in right."


def _replay_all(rules: Rules, rounds) -> list:
    """(check entry, replay or None) for each round given."""
    rounds = rounds if isinstance(rounds, list) else []
    chips, ended_after, out = float(rules.starting_chips), None, []
    for i, data in enumerate(rounds):
        n = i + 1
        entry = {"ok": True, "error": None, "complete": False, "net": None,
                 "chips_before": chips, "chips_after": None}
        if out and out[-1][0]["chips_after"] is None:
            out.append((entry, None))          # waiting for an earlier round
            continue
        if ended_after is not None or n > rules.rounds:
            reason = (f"the match was over after round {ended_after}, when the chips ran out."
                      if ended_after is not None else f"the match has only {rules.rounds} rounds.")
            out.append((entry | {"ok": False, "error": f"Round {n}: {reason}"}, None))
            continue
        replay, error = _replay_or_message(rules, n, data, chips)
        if error:
            out.append((entry | {"ok": False, "error": error}, None))
            continue
        result = replay.round_.result
        if result is not None:
            chips = float(result.chips_after)
            entry |= {"complete": True, "net": float(result.net), "chips_after": chips}
            if rules.floor_bet(chips) < rules.min_bet:
                ended_after = n
        out.append((entry, replay))
    return out


def check_rounds(rules: Rules, rounds: list) -> list:
    """Each round's result and chip totals, or a plain message naming the round. Never a grade."""
    return [entry for entry, _ in _replay_all(rules, rounds)]


# ---- the analysis -----------------------------------------------------------------

def _finished_replays(rules: Rules, rounds) -> list:
    """The replays of a complete finished match, or TrainerError naming the first problem."""
    replays = []
    for n, (entry, replay) in enumerate(_replay_all(rules, rounds), start=1):
        if not entry["ok"]:
            raise TrainerError(entry["error"])
        if not entry["complete"]:
            raise TrainerError(f"Round {n}: it is not finished yet.")
        replays.append((entry, replay))
        if rules.floor_bet(entry["chips_after"]) < rules.min_bet:
            return replays
    if len(replays) < rules.rounds:
        raise TrainerError(f"Round {len(replays) + 1}: it has not been entered yet.")
    return replays


def _known(replay: _Replay) -> dict:
    """Every seen card by its position in the round's card order."""
    used = replay.round_.result.cards_used
    return {i: card for i, card in enumerate(replay.shoe[:used]) if i != 3 or replay.down_seen}


def _chips(amount: float) -> str:
    return f"{amount:,.1f}".rstrip("0").rstrip(".")


def _bet_text(main: float, pp: float, fct: float) -> str:
    text = _chips(main)
    if pp:
        text += f" + {_chips(pp)} on Perfect Pairs"
    if fct:
        text += f" + {_chips(fct)} on 4 Card Total"
    return text


def _spot_text(cards: list, upcard: int) -> str:
    total, soft = hand_total(cards)
    up = "A" if value_of(upcard) == 1 else str(value_of(upcard))
    return f"{'soft ' if soft else ''}{total} against {up}"


def _grade_bet(brain: Brain, n: int, entry: dict, replay: _Replay, history: list) -> dict:
    r = replay.round_
    g = brain.grade_bet(n, entry["chips_before"], history, r.main_bet, r.pp_bet, r.fct_bet)
    return {"yours": g.yours[0], "best": g.best[0], "pp": g.yours[1], "fct": g.yours[2],
            "best_pp": g.best[1], "best_fct": g.best[2], "your_chance": g.your_chance,
            "best_chance": g.best_chance, "fine": g.fine}


def _prior_rounds(rules: Rules, brain: Brain, previous: list):
    if len(previous) >= rules.rounds:
        raise TrainerError("All rounds have already been played.")
    chips, history = float(rules.starting_chips), []
    for n, (entry, replay) in enumerate(_replay_all(rules, previous), 1):
        if not entry["ok"]:
            raise TrainerError(entry["error"])
        if not entry["complete"]:
            raise TrainerError(f"Round {n}: finish this round first.")
        chips = entry["chips_after"]
        history.append(brain.round_class_dist(_known(replay)))
    if rules.floor_bet(chips) < rules.min_bet:
        raise TrainerError("The match ended because there are not enough chips to bet.")
    return chips, history


def recommend_bet(rules: Rules, brain: Brain, previous: list) -> dict:
    """Recommend the next bet using only completed earlier rounds."""
    chips, history = _prior_rounds(rules, brain, previous)
    grade = brain.grade_bet(len(previous) + 1, chips, history, rules.min_bet)
    return {"best": grade.best[0], "best_pp": grade.best[1], "best_fct": grade.best[2],
            "best_chance": grade.best_chance}


def recommend_play(rules: Rules, brain: Brain, previous: list, current: dict) -> dict:
    """Score the next legal play using only cards visible to the player."""
    chips, history = _prior_rounds(rules, brain, previous)
    if not isinstance(current, dict):
        raise TrainerError("Enter the bet, your cards and the dealer's upcard first.")
    replay, error = _replay_or_message(rules, len(previous) + 1, current | {"dealer": []}, chips)
    if error:
        raise TrainerError(error)
    r = replay.round_
    legal = r.legal_actions() if r.phase == "play" else []
    if not legal:
        return {"best": None, "chances": {}}
    spot = PlaySpot(round_no=len(previous) + 1, chips_start=chips, main_bet=r.main_bet,
                    pp_bet=r.pp_bet, fct_bet=r.fct_bet, insurance_bet=r.insurance_bet,
                    player=replay.shoe[0:3:2], upcard=r.upcard, drawn=list(replay.shoe[4:r.pos]),
                    script=list(r.actions), legal=legal, stage=r.phase)
    grade = brain.grade_play(spot, history, legal[0])
    return {"best": grade.best, "chances": grade.chances, "hand": r.active + 1}


def _grade_plays(brain: Brain, n: int, entry: dict, replay: _Replay, history: list) -> list:
    r, plays = replay.round_, []
    for d in replay.decisions:
        if len(d.legal) < 2:
            continue                           # no real choice, e.g. insurance without the chips
        spot = PlaySpot(round_no=n, chips_start=entry["chips_before"], main_bet=r.main_bet,
                        pp_bet=r.pp_bet, fct_bet=r.fct_bet, insurance_bet=d.insurance_bet,
                        player=replay.shoe[0:3:2], upcard=r.upcard, drawn=d.drawn,
                        script=d.script, legal=d.legal, stage=d.stage)
        g = brain.grade_play(spot, history, d.choice)
        plays.append({"hand": d.hand, "spot": _spot_text(d.cards, r.upcard), "yours": g.yours,
                      "best": g.best, "your_chance": g.your_chance, "best_chance": g.best_chance,
                      "fine": g.fine})
    return plays


def _biggest_miss(rounds: list) -> dict | None:
    misses = []
    for r in rounds:
        bet = r["bet"]
        if not bet["fine"]:
            text = (f"Round {r['round']}: you bet {_bet_text(bet['yours'], bet['pp'], bet['fct'])} "
                    f"where {_bet_text(bet['best'], bet['best_pp'], bet['best_fct'])} was best.")
            misses.append((bet["best_chance"] - bet["your_chance"], r["round"], "bet", text))
        for p in r["plays"]:
            if not p["fine"]:
                kind = "insurance" if p["yours"] in ("insurance", "no_insurance") else "play"
                text = (f"Round {r['round']}: you {_DID[p['yours']]} {p['spot']} "
                        f"where {_DOING[p['best']]} was best.")
                misses.append((p["best_chance"] - p["your_chance"], r["round"], kind, text))
    if not misses:
        return None
    cost, n, kind, text = max(misses, key=lambda m: m[0])
    return {"round": n, "kind": kind, "text": text, "cost": cost}


def analyze_match(rules: Rules, brain: Brain, rounds: list) -> dict:
    """Grade every bet and play of a complete finished match."""
    replays = _finished_replays(rules, rounds)
    report_rounds, history = [], []
    for n, (entry, replay) in enumerate(replays, start=1):
        report_rounds.append({
            "round": n, "chips_before": entry["chips_before"], "chips_after": entry["chips_after"],
            "net": entry["net"],
            "bet": _grade_bet(brain, n, entry, replay, history),
            "plays": _grade_plays(brain, n, entry, replay, history),
        })
        history = history + [brain.round_class_dist(_known(replay))]
    return {
        "rounds": report_rounds,
        "final_chips": replays[-1][0]["chips_after"],
        "off_bets": sum(not r["bet"]["fine"] for r in report_rounds),
        "off_plays": sum(not p["fine"] for r in report_rounds for p in r["plays"]),
        "biggest_miss": _biggest_miss(report_rounds),
    }


def save_review(data_dir, payload: dict, report: dict) -> Path:
    """Save analysis alongside its autosaved review, or create a legacy timestamped record."""
    if payload.get("review_id"):
        return save_review_analysis(data_dir, payload, report)
    folder = Path(data_dir) / "reviews"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path, extra = folder / f"{stamp}.json", 1
    while path.exists():                       # two reviews saved in the same second
        extra += 1
        path = folder / f"{stamp}-{extra}.json"
    path.write_text(json.dumps({"payload": payload, "report": report}, indent=2), encoding="utf-8")
    return path
