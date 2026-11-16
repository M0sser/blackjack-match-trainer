"""The rules of the match. Every field is a setting with a default from the design doc, section 5."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Rules:
    starting_chips: float = 1000.0
    rounds: int = 5
    decks: int = 6
    blackjack_pays: float = 1.5
    insurance_offered: bool = True
    double_after_split: bool = True
    double_on_first_ace: bool = False   # doubling is unavailable when a hand starts with an ace
    split_by_value: bool = True      # False: only two cards of the same rank can be split
    min_bet: float = 10.0
    bet_step: float = 5.0
    tie_credit: float = 0.5          # what equal final chips count for: 1 is a win, 0 a loss
    clock_seconds: int = 360
    pp_pays: tuple = (25, 12, 6)     # Perfect Pairs: same rank and suit, same rank and colour, same rank
    fct_pays: tuple = (2, 8, 10)     # 4 Card Total: 36-37, 38-39, 40 or more

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = list(value) if isinstance(value, tuple) else value
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Rules":
        """Build rules from a dict, ignoring unknown keys and keeping each field's type."""
        kwargs = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            kind = type(f.default)
            if kind is tuple:
                kwargs[f.name] = tuple(value)
            elif kind is bool:
                kwargs[f.name] = bool(value)
            else:
                kwargs[f.name] = kind(value)
        return cls(**kwargs)

    def fingerprint(self) -> str:
        text = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def floor_bet(self, amount: float) -> float:
        """Round an amount down to a whole number of bet steps."""
        return math.floor(amount / self.bet_step + 1e-9) * self.bet_step
