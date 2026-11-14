"""Cards as small integers: rank_index * 4 + suit_index, 0..51."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from trainer.errors import TrainerError

RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
SUITS = ("S", "H", "D", "C")   # spades, hearts, diamonds, clubs


def parse_card(text: str) -> int:
    """'10S' or 'KH'; a rank alone ('10') means spades."""
    shown = text if isinstance(text, str) else repr(text)
    t = text.strip().upper() if isinstance(text, str) else ""
    if t in RANKS:
        return RANKS.index(t) * 4
    if t and t[-1] in SUITS and t[:-1] in RANKS:
        return RANKS.index(t[:-1]) * 4 + SUITS.index(t[-1])
    raise TrainerError(f'"{shown}" is not a card. Use A, 2 to 10, J, Q or K, with an optional suit letter S, H, D or C.')


def card_str(card: int) -> str:
    return RANKS[int(card) // 4] + SUITS[int(card) % 4]


def rank_of(card: int) -> int:
    return int(card) // 4


def suit_of(card: int) -> int:
    return int(card) % 4


def value_of(card: int) -> int:
    """Blackjack value: ace 1, tens and faces 10."""
    rank = int(card) // 4
    return 1 if rank == 0 else min(rank + 1, 10)


def is_red(card: int) -> bool:
    return int(card) % 4 in (1, 2)


VALUES = np.array([value_of(c) for c in range(52)], dtype=np.int8)


def hand_total(cards: Sequence[int]) -> tuple[int, bool]:
    """Best total and whether an ace is being counted as 11."""
    hard = sum(value_of(c) for c in cards)
    if hard + 10 <= 21 and any(value_of(c) == 1 for c in cards):
        return hard + 10, True
    return hard, False


def new_shoe(rng: np.random.Generator, decks: int = 6) -> np.ndarray:
    return rng.permutation(np.repeat(np.arange(52, dtype=np.int8), decks))
