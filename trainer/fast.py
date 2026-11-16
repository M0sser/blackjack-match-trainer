"""The fast engine: many rounds at once with NumPy, for fixed play styles.

It follows the same rules as engine.Round (the reference) and must give exactly the
same result for the same cards, style and spare chips. tests/test_fast.py checks that.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from trainer.cards import VALUES
from trainer.rules import Rules

WIDTH = 40          # cards kept per round; far more than a round can use
_CHUNK = 100_000


def _first_cards(rng: np.random.Generator, pool: np.ndarray, n: int, take: int) -> np.ndarray:
    """n rows, each the first `take` cards of its own shuffle of `pool` (partial Fisher-Yates)."""
    out = np.empty((n, take), dtype=np.int8)
    size = len(pool)
    for start in range(0, n, _CHUNK):
        rows = min(_CHUNK, n - start)
        work = np.tile(pool, (rows, 1))
        index = np.arange(rows)
        for i in range(take):
            j = rng.integers(i, size, size=rows)
            picked = work[index, j]
            work[index, j] = work[index, i]
            work[index, i] = picked
        out[start:start + rows] = work[:, :take]
    return out


def sample_sequences(rng: np.random.Generator, n: int, decks: int = 6, width: int = WIDTH) -> np.ndarray:
    """(n, width) card codes: each row is the top of an independent freshly shuffled shoe."""
    pool = np.repeat(np.arange(52, dtype=np.int8), decks)
    return _first_cards(rng, pool, n, width)


def sample_completions(rng: np.random.Generator, n: int, known: dict[int, int], decks: int = 6,
                       width: int = WIDTH) -> np.ndarray:
    """Rows that hold the known cards at their positions; every other position is drawn
    without replacement from what is left of the shoe."""
    counts = np.full(52, decks, dtype=np.int64)
    for card in known.values():
        counts[int(card)] -= 1
    if (counts < 0).any():
        raise ValueError("more copies of one card than the shoe holds")
    pool = np.repeat(np.arange(52, dtype=np.int8), counts)
    unknown = [p for p in range(width) if p not in known]
    seq = np.empty((n, width), dtype=np.int8)
    for position, card in known.items():
        if position < width:
            seq[:, position] = card
    seq[:, unknown] = _first_cards(rng, pool, n, len(unknown))
    return seq


def play_rounds(rules: Rules, seq: np.ndarray, table: np.ndarray, extra_bets,
                script: Sequence[int] = ()) -> np.ndarray:
    """Main result per chip bet for every row of `seq`.

    table      -- a style's action table from styles.style_table
    extra_bets -- how many extra stakes (doubles, splits) the player can afford: an int or one per row
    script     -- action codes forced on the first len(script) player decisions of every row
    """
    seq = np.asarray(seq)
    n, width = seq.shape
    codes = seq.astype(np.int16)
    vals = VALUES[codes].astype(np.int16)
    last = width - 1
    extra = np.broadcast_to(np.asarray(extra_bets, dtype=np.int16), (n,))
    forced = np.asarray(script, dtype=np.int8)
    das = rules.double_after_split

    p1, up, p2, hole = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3]
    if rules.split_by_value:
        is_pair = p1 == p2
    else:
        is_pair = (codes[:, 0] // 4) == (codes[:, 2] // 4)

    # Blackjacks settle before any play (the dealer checks the down card).
    player_bj = (p1 + p2 == 11) & ((p1 == 1) | (p2 == 1))
    dealer_bj = (up + hole == 11) & ((up == 1) | (hole == 1))
    x = np.zeros(n, dtype=np.float64)
    x[player_bj & ~dealer_bj] = rules.blackjack_pays
    x[dealer_bj & ~player_bj] = -1.0
    live = ~(player_bj | dealer_bj)

    hard = np.zeros((2, n), dtype=np.int16)      # hand totals counting aces as 1
    hard[0] = p1 + p2
    ace = np.zeros((2, n), dtype=bool)
    ace[0] = (p1 == 1) | (p2 == 1)
    ncards = np.zeros((2, n), dtype=np.int8)
    ncards[0] = 2
    mult = np.ones((2, n), dtype=np.float64)     # 2 after a double
    done = np.zeros((2, n), dtype=bool)
    hands = np.ones(n, dtype=np.int8)
    cur = np.zeros(n, dtype=np.int8)             # hand being played
    used = np.zeros(n, dtype=np.int16)           # extra stakes spent
    decisions = np.zeros(n, dtype=np.int16)
    split_aces = np.zeros(n, dtype=bool)
    pos = np.full(n, 4, dtype=np.int64)          # next card to deal
    active = live.copy()

    def draw(rows: np.ndarray) -> np.ndarray:
        card = vals[rows, np.minimum(pos[rows], last)]
        pos[rows] += 1
        return card

    while True:
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        c = cur[idx]
        already = done[c, idx]
        h = hard[c, idx]
        soft = ace[c, idx] & (h + 10 <= 21)
        total = h + 10 * soft

        auto = ~already & (total >= 21)          # 21 or bust: finished without a decision
        done[c[auto], idx[auto]] = True

        need = ~already & ~auto
        if need.any():
            r, rc = idx[need], c[need]
            unsplit = hands[r] == 1
            two = ncards[rc, r] == 2
            spare = used[r] < extra[r]
            first_card = np.where(rc == 0, p1[r], p2[r])
            can_double = two & spare & (unsplit | das) & ((first_card != 1) | rules.double_on_first_ace)
            pair_value = np.where(unsplit & two & is_pair[r], p1[r], 0)
            can_split = (pair_value > 0) & spare
            action = table[can_double.astype(np.intp), can_split.astype(np.intp), pair_value,
                           soft[need].astype(np.intp), total[need], up[r]]
            if forced.size:
                count = decisions[r]
                action = np.where(count < forced.size, forced[np.minimum(count, forced.size - 1)], action)
            decisions[r] += 1

            stand = action == 0
            done[rc[stand], r[stand]] = True

            takes_card = (action == 1) | (action == 2)
            if takes_card.any():
                hr, hc = r[takes_card], rc[takes_card]
                card = draw(hr)
                hard[hc, hr] += card
                ace[hc, hr] |= card == 1
                ncards[hc, hr] += 1
                doubled = action[takes_card] == 2
                dr, dc = hr[doubled], hc[doubled]
                mult[dc, dr] = 2.0
                used[dr] += 1
                done[dc, dr] = True

            split = action == 3
            if split.any():
                sr = r[split]
                card = draw(sr)                  # the first hand's second card comes at once
                hard[0, sr] = p1[sr] + card
                ace[0, sr] = (p1[sr] == 1) | (card == 1)
                ncards[0, sr] = 2
                hard[1, sr] = p2[sr]
                ace[1, sr] = p2[sr] == 1
                ncards[1, sr] = 1
                hands[sr] = 2
                used[sr] += 1
                split_aces[sr] = p1[sr] == 1
                done[0, sr] = split_aces[sr]     # split aces get one card each

        finished = idx[done[cur[idx], idx]]
        if finished.size:
            more = cur[finished] + 1 < hands[finished]
            nxt = finished[more]
            if nxt.size:                         # the second hand gets its second card now
                cur[nxt] = 1
                card = draw(nxt)
                hard[1, nxt] += card
                ace[1, nxt] |= card == 1
                ncards[1, nxt] = 2
                done[1, nxt] = split_aces[nxt]
            active[finished[~more]] = False

    # Dealer: draws nothing if every player hand is bust, else draws to 17 (soft 17 stands).
    bust = hard > 21
    all_bust = bust[0] & ((hands == 1) | bust[1])
    dealer_plays = live & ~all_bust
    dhard = (up + hole).astype(np.int16)
    dace = (up == 1) | (hole == 1)
    while True:
        dtotal = dhard + 10 * (dace & (dhard + 10 <= 21))
        rows = np.flatnonzero(dealer_plays & (dtotal < 17))
        if rows.size == 0:
            break
        card = draw(rows)
        dhard[rows] += card
        dace[rows] |= card == 1
    dealer_bust = dhard > 21

    for k in (0, 1):
        present = live & (hands > k)
        total_k = hard[k] + 10 * (ace[k] & (hard[k] + 10 <= 21))
        outcome = np.where(bust[k], -1.0,
                           np.where(dealer_bust | (total_k > dtotal), 1.0,
                                    np.where(total_k < dtotal, -1.0, 0.0)))
        x[present] += (outcome * mult[k])[present]
    return x
