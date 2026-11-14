"""Stand-in opponents: who they are, how common each one is, and how they bet."""

from __future__ import annotations

from dataclasses import dataclass

from trainer.rules import Rules

STYLE_SUFFIXES = {
    "book": ", plays by the book",
    "cautious": ", plays cautiously",
    "dealer_copy": ", copies the dealer",
}


@dataclass(frozen=True)
class OpponentType:
    label: str
    style: str
    rule: str
    param: float
    weight: float

    @property
    def name(self) -> str:
        return self.label + STYLE_SUFFIXES[self.style]


RULE_VARIANTS = [
    ("Turtle", "turtle", 0),
    ("Flat 25", "flat", 25),
    ("Flat 50", "flat", 50),
    ("Flat 100", "flat", 100),
    ("Flat 200", "flat", 200),
    ("Flat 500", "flat", 500),
    ("Share 10%", "share", 0.10),
    ("Share 25%", "share", 0.25),
    ("Share 50%", "share", 0.50),
    ("All-in", "all_in", 0),
    ("Double up and sit", "double_up_sit", 0),
    ("Late lunge", "late_lunge", 0),
    ("Chaser 50", "chaser", 50),
    ("Chaser 100", "chaser", 100),
    ("Random", "random", 0),
]

DEFAULT_RULE_WEIGHTS = dict(zip(
    (label for label, _, _ in RULE_VARIANTS),
    (5, 7, 7, 15, 11, 8, 5, 8, 5, 8, 6, 5, 3, 3, 4),
))

DEFAULT_STYLE_WEIGHTS = {"book": 50, "cautious": 40, "dealer_copy": 10}


def build_population(rule_weights=DEFAULT_RULE_WEIGHTS, style_weights=DEFAULT_STYLE_WEIGHTS) -> list[OpponentType]:
    """Cross every rule with every style; weight is the product, zeros dropped, normalised to sum 1."""
    raw = []
    for label, rule, param in RULE_VARIANTS:
        for style in STYLE_SUFFIXES:
            weight = rule_weights.get(label, 0) * style_weights.get(style, 0)
            if weight > 0:
                raw.append((label, style, rule, param, weight))
    total = sum(entry[4] for entry in raw)
    return [OpponentType(label, style, rule, float(param), weight / total)
            for label, style, rule, param, weight in raw]


def legal_bet(rules: Rules, chips: float, wanted: float) -> float:
    """Round a wanted bet down to a step, raise it to the minimum, cap it at the chips; 0.0 if no bet fits."""
    cap = rules.floor_bet(chips)
    if cap < rules.min_bet:
        return 0.0
    return min(max(rules.floor_bet(wanted), rules.min_bet), cap)


def _wanted(opp: OpponentType, rules: Rules, round_no: int, chips: float, memory: float) -> list[float]:
    """The amounts an opponent would like to bet, each equally likely."""
    rule = opp.rule
    if rule == "turtle":
        return [rules.min_bet]
    if rule == "flat":
        return [opp.param]
    if rule == "share":
        return [opp.param * chips]
    if rule == "all_in":
        return [chips]
    if rule == "double_up_sit":
        return [chips if chips <= rules.starting_chips else rules.min_bet]
    if rule == "late_lunge":
        return [chips if round_no >= rules.rounds else rules.min_bet]
    if rule == "chaser":
        return [memory]
    if rule == "random":
        return [rules.min_bet, 0.10 * chips, 0.25 * chips, 0.50 * chips, chips]
    raise ValueError(f"unknown betting rule: {rule}")


def bets_for(opp: OpponentType, rules: Rules, round_no: int, chips: float, memory: float) -> list[tuple[float, float]]:
    """(bet, probability) pairs for this round, equal bets merged and sorted; [(0.0, 1.0)] if no bet is possible."""
    if legal_bet(rules, chips, rules.min_bet) == 0.0:
        return [(0.0, 1.0)]
    wanted = _wanted(opp, rules, round_no, chips, memory)
    share = 1.0 / len(wanted)
    merged: dict[float, float] = {}
    for amount in wanted:
        bet = legal_bet(rules, chips, amount)
        merged[bet] = merged.get(bet, 0.0) + share
    return sorted(merged.items())


def initial_memory(opp: OpponentType) -> float:
    """Memory at the start of a match: the base bet for a chaser, 0.0 for everyone else."""
    return float(opp.param) if opp.rule == "chaser" else 0.0


def next_memory(opp: OpponentType, memory: float, bet: float, x: float) -> float:
    """Memory after a round with result x: a chaser doubles on a loss, resets on a win, keeps it on a push."""
    if opp.rule != "chaser":
        return 0.0
    if x < 0:
        return memory * 2
    if x > 0:
        return float(opp.param)
    return memory
