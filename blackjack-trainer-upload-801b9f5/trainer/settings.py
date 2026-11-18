"""What the user can change: a few rules, the mix of stand-in opponents, and the default mode."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from trainer.errors import TrainerError
from trainer.opponents import (DEFAULT_RULE_WEIGHTS, DEFAULT_STYLE_WEIGHTS, OpponentType, build_population)
from trainer.rules import Rules

MODES = ("coach", "match")
# The rules a user may change in this version; the rest are fixed (see the design doc, section 5).
ADJUSTABLE = ("min_bet", "tie_credit", "double_after_split", "split_by_value", "insurance_offered", "clock_seconds")


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TrainerError(f"{name} has to be a number.")
    return float(value)


def _rules_from(data) -> Rules:
    if not isinstance(data, dict):
        raise TrainerError("The rules were not understood.")
    rules = Rules()
    changes = {}
    if "min_bet" in data:
        changes["min_bet"] = _number(data["min_bet"], "The minimum bet")
    if "tie_credit" in data:
        changes["tie_credit"] = _number(data["tie_credit"], "What equal chips count for")
    if "clock_seconds" in data:
        changes["clock_seconds"] = int(_number(data["clock_seconds"], "The clock length"))
    for name in ("double_after_split", "split_by_value", "insurance_offered"):
        if name in data:
            changes[name] = bool(data[name])
    rules = replace(rules, **changes)

    steps = rules.min_bet / rules.bet_step
    if rules.min_bet <= 0 or abs(steps - round(steps)) > 1e-9:
        raise TrainerError(f"The minimum bet has to be a multiple of {rules.bet_step:g}.")
    if not 0 <= rules.tie_credit <= 1:
        raise TrainerError("Equal chips can count as a loss (0), half a win (0.5) or a win (1).")
    if not 30 <= rules.clock_seconds <= 3600:
        raise TrainerError("The clock has to be between 30 seconds and one hour.")
    return rules


def _weights_from(data, defaults: dict, what: str) -> dict:
    if not isinstance(data, dict):
        raise TrainerError(f"The {what} weights were not understood.")
    weights = {}
    for name in defaults:
        value = _number(data.get(name, 0), f"The weight for {name}")
        if value < 0:
            raise TrainerError(f"The weight for {name} cannot be below 0.")
        weights[name] = value
    if not any(value > 0 for value in weights.values()):
        raise TrainerError(f"At least one {what} needs a weight above 0.")
    return weights


@dataclass
class Settings:
    rules: Rules = field(default_factory=Rules)
    rule_weights: dict = field(default_factory=lambda: {k: float(v) for k, v in DEFAULT_RULE_WEIGHTS.items()})
    style_weights: dict = field(default_factory=lambda: {k: float(v) for k, v in DEFAULT_STYLE_WEIGHTS.items()})
    default_mode: str = "coach"

    def to_dict(self) -> dict:
        return {"rules": self.rules.to_dict(), "rule_weights": dict(self.rule_weights),
                "style_weights": dict(self.style_weights), "default_mode": self.default_mode}

    @classmethod
    def from_dict(cls, data) -> "Settings":
        """Build settings from a dict; anything missing keeps its default. Raises TrainerError
        in plain words for a value that cannot be used."""
        if not isinstance(data, dict):
            raise TrainerError("The settings were not understood.")
        settings = cls()
        if "rules" in data:
            settings.rules = _rules_from(data["rules"])
        if "rule_weights" in data:
            settings.rule_weights = _weights_from(data["rule_weights"], DEFAULT_RULE_WEIGHTS, "betting rule")
        if "style_weights" in data:
            settings.style_weights = _weights_from(data["style_weights"], DEFAULT_STYLE_WEIGHTS, "play style")
        if "default_mode" in data:
            if data["default_mode"] not in MODES:
                raise TrainerError("The mode has to be Coach or Match.")
            settings.default_mode = data["default_mode"]
        return settings

    def same_brain_as(self, other: "Settings") -> bool:
        """True when the two settings need the same solve."""
        return (self.rules == other.rules and self.rule_weights == other.rule_weights
                and self.style_weights == other.style_weights)


def load_settings(data_dir) -> Settings:
    """Saved settings, or the defaults when there are none or the file cannot be read."""
    try:
        return Settings.from_dict(json.loads((Path(data_dir) / "settings.json").read_text(encoding="utf-8")))
    except (OSError, ValueError, TrainerError):
        return Settings()


def save_settings(data_dir, settings: Settings) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "settings.json").write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")


def population_from(settings: Settings) -> list[OpponentType]:
    return build_population(settings.rule_weights, settings.style_weights)
