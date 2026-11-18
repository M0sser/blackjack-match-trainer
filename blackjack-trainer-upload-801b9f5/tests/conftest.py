from pathlib import Path

import pytest

from trainer.brain import Brain, BrainConfig
from trainer.opponents import OpponentType, build_population
from trainer.rules import Rules

# A small, quick brain for tests: fewer practice rounds, a coarse grid, a handful of opponents.
QUICK = BrainConfig(table_rounds=200_000, samples=8_000, coarse=True)
QUICK_RULE_WEIGHTS = {"Turtle": 1, "Flat 100": 3, "Flat 500": 1, "Share 25%": 2, "All-in": 1}
QUICK_STYLE_WEIGHTS = {"book": 2, "cautious": 1, "dealer_copy": 0}


@pytest.fixture(scope="session")
def quick_brain(tmp_path_factory):
    pop = build_population(rule_weights=QUICK_RULE_WEIGHTS, style_weights=QUICK_STYLE_WEIGHTS)
    return Brain.prepare(Rules(), pop, tmp_path_factory.mktemp("quick"), QUICK)


@pytest.fixture(scope="session")
def turtle_brain(tmp_path_factory):
    pop = [OpponentType("Turtle", "book", "turtle", 0.0, 1.0)]
    return Brain.prepare(Rules(), pop, tmp_path_factory.mktemp("turtle"), QUICK)


@pytest.fixture(scope="session")
def full_brain():
    """The real thing, cached where the app looks for it, so the first start of the app is instant."""
    cache = Path(__file__).resolve().parent.parent / "data" / "cache"
    return Brain.prepare(Rules(), build_population(), cache, BrainConfig())
