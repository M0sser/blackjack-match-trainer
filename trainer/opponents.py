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
    ("Flat 250", "flat", 250),
    ("Flat 300", "flat", 300),
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
    ("Mixed", "mixed", 0),
]

# A Mixed opponent bets one of these shares of its current chips each round, equally likely.
# First read off the 28 real matches below as (0.10, 0.18, 0.225, 0.375, 0.60): the middles
# of five equal-sized groups of 62 bet-to-chips ratios, with the bet taken as the hand's net.
# 2026-11-16: scaled by 1.4 to fit the opponent finals of the 47 logged matches. Scales from
# about 1.37 to 1.50 fit equally well; 1.4 also matches the share of opponents who went broke.
MIXED_SHARES = (0.14, 0.252, 0.315, 0.525, 0.84)

# The first guess at the mix (design doc, section 6.4), before any real matches were seen.
PRIOR_RULE_WEIGHTS = {
    "Turtle": 5, "Flat 25": 7, "Flat 50": 7, "Flat 100": 15, "Flat 200": 11, "Flat 500": 8,
    "Share 10%": 5, "Share 25%": 8, "Share 50%": 5, "All-in": 8, "Double up and sit": 6,
    "Late lunge": 5, "Chaser 50": 3, "Chaser 100": 3, "Random": 4,
}

# Opponents in 28 finished 21 Rivals matches (screenshots of match stats, 2026-11-15),
# each sorted to the betting rule its five hand results fit best. Opponents whose bets
# changed from hand to hand with no fixed rule are counted as Mixed.
OBSERVED_RULE_COUNTS = {
    "Turtle": 1, "Flat 25": 1, "Flat 50": 1, "Flat 250": 2, "Flat 300": 4, "Flat 500": 2,
    "All-in": 2, "Chaser 100": 1, "Mixed": 14,
}

# Opponents in 47 matches logged in the review (2026-11-16), where only the opponent's final
# chips are known. Each match is shared out over the rules by how likely each rule made that
# final total given the round results, with the weights above as the starting belief, so
# these are fractional counts that add up to 47. Real finals sat in the outer fifths of the
# predicted spread far more often than the old mix expected, which moves weight to big bets.
# Fitted together with the MIXED_SHARES above.
LOGGED_RULE_COUNTS = {
    "Turtle": 1.30, "Flat 25": 2.37, "Flat 50": 1.02, "Flat 100": 1.63, "Flat 200": 1.20,
    "Flat 250": 1.98, "Flat 300": 3.67, "Flat 500": 6.40, "Share 10%": 0.41, "Share 25%": 0.45,
    "Share 50%": 0.45, "All-in": 2.60, "Double up and sit": 1.08, "Late lunge": 0.58,
    "Chaser 50": 0.22, "Chaser 100": 1.88, "Random": 0.55, "Mixed": 19.21,
}

# The prior counts as this many matches, so a rule not seen in the real matches keeps a small weight.
PRIOR_MATCHES = 10

DEFAULT_RULE_WEIGHTS = {
    label: OBSERVED_RULE_COUNTS.get(label, 0) + LOGGED_RULE_COUNTS.get(label, 0)
    + PRIOR_MATCHES * PRIOR_RULE_WEIGHTS.get(label, 0) / sum(PRIOR_RULE_WEIGHTS.values())
    for label, _, _ in RULE_VARIANTS
}

# Play styles cannot be seen in match stats, so these are still the first guess.
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
    if rule == "mixed":
        return [share * chips for share in MIXED_SHARES]
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
