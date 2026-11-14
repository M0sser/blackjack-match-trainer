"""The two side bets. A multiplier is the payout per chip staked, or -1 when the bet loses."""

from __future__ import annotations

import numpy as np

from trainer.cards import VALUES, is_red, rank_of, suit_of, value_of


def pp_multiplier(c1: int, c2: int, pays=(25, 12, 6)) -> int:
    """Perfect Pairs on the player's first two cards."""
    if rank_of(c1) != rank_of(c2):
        return -1
    if suit_of(c1) == suit_of(c2):
        return int(pays[0])
    if is_red(c1) == is_red(c2):
        return int(pays[1])
    return int(pays[2])


def fct_multiplier(p1: int, p2: int, up: int, hole: int, pays=(2, 8, 10)) -> int:
    """4 Card Total: the player's two cards plus both dealer cards, an ace counting 11."""
    total = sum(11 if value_of(c) == 1 else value_of(c) for c in (p1, p2, up, hole))
    if total >= 40:
        return int(pays[2])
    if total >= 38:
        return int(pays[1])
    if total >= 36:
        return int(pays[0])
    return -1


def pp_multipliers(seq: np.ndarray, pays=(25, 12, 6)) -> np.ndarray:
    """Perfect Pairs for many rounds at once. Player cards are in columns 0 and 2."""
    c1 = seq[:, 0].astype(np.int16)
    c2 = seq[:, 2].astype(np.int16)
    same_rank = (c1 // 4) == (c2 // 4)
    same_suit = (c1 % 4) == (c2 % 4)
    same_colour = np.isin(c1 % 4, (1, 2)) == np.isin(c2 % 4, (1, 2))
    out = np.full(len(seq), -1, dtype=np.int16)
    out[same_rank] = pays[2]
    out[same_rank & same_colour] = pays[1]
    out[same_rank & same_suit] = pays[0]
    return out


def fct_multipliers(seq: np.ndarray, pays=(2, 8, 10)) -> np.ndarray:
    """4 Card Total for many rounds at once. The four dealt cards are columns 0 to 3."""
    values = VALUES[seq[:, :4].astype(np.int16)].astype(np.int16)
    total = np.where(values == 1, 11, values).sum(axis=1)
    out = np.full(len(seq), -1, dtype=np.int16)
    out[total >= 36] = pays[0]
    out[total >= 38] = pays[1]
    out[total >= 40] = pays[2]
    return out
