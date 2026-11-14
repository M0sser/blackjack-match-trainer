"""Simulated outcome tables.

A round's *class* is the result per chip that Book play gets with unlimited chips,
sorted into eight buckets. The tables say how often each class comes up and, within
each class, what every play style gets at every level of spare chips.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trainer.fast import play_rounds, sample_sequences
from trainer.rules import Rules
from trainer.sidebets import fct_multipliers, pp_multipliers
from trainer.styles import STYLE_INDEX, STYLE_NAMES, style_table

CLASS_VALUES = (-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0, 3.0)
N_CLASSES = 8
LEVELS = 4            # spare chips for 0, 1, 2 or 3+ extra stakes
_CHUNK = 100_000


def class_of(x):
    """Class index of a Book result per chip: -3 or worse is 0, then -2, -1, 0, +1,
    a blackjack payout, +2, and +3 or better is 7."""
    arr = np.asarray(x, dtype=np.float64)
    out = np.full(arr.shape, 5, dtype=np.int64)     # not a whole number: a blackjack payout
    out[arr <= -3] = 0
    out[arr == -2] = 1
    out[arr == -1] = 2
    out[arr == 0] = 3
    out[arr == 1] = 4
    out[arr == 2] = 6
    out[arr >= 3] = 7
    return out


@dataclass
class Tables:
    p_class: np.ndarray       # (8,) chance of each class
    vals: np.ndarray          # (styles, levels, 8, J) results per chip
    probs: np.ndarray         # (styles, levels, 8, J) their chances given the class
    side_vals: np.ndarray     # (levels, 8, Js, 3) Book: (result, Perfect Pairs multiplier, 4 Card Total multiplier)
    side_probs: np.ndarray    # (levels, 8, Js) their chances given the class


def build_tables(rules: Rules, n_rounds: int = 1_000_000, seed: int = 20261114) -> Tables:
    rng = np.random.default_rng(seed)
    slots = np.array(sorted({-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, float(rules.blackjack_pays)}))
    pp_values = np.array(sorted({-1.0, *(float(v) for v in rules.pp_pays)}))
    fct_values = np.array(sorted({-1.0, *(float(v) for v in rules.fct_pays)}))
    n_slots, n_pp, n_fct = len(slots), len(pp_values), len(fct_values)
    book = STYLE_INDEX["book"]
    style_tables = [style_table(name, rules) for name in STYLE_NAMES]

    counts = np.zeros((len(STYLE_NAMES), LEVELS, N_CLASSES, n_slots))
    side_counts = np.zeros((LEVELS, N_CLASSES * n_slots * n_pp * n_fct))
    class_counts = np.zeros(N_CLASSES)

    done = 0
    while done < n_rounds:
        rows = min(_CHUNK, n_rounds - done)
        seq = sample_sequences(rng, rows, rules.decks)
        results = {(s, k): play_rounds(rules, seq, style_tables[s], k)
                   for s in range(len(STYLE_NAMES)) for k in range(LEVELS)}
        g = class_of(results[book, LEVELS - 1])
        class_counts += np.bincount(g, minlength=N_CLASSES)
        pp_index = np.searchsorted(pp_values, pp_multipliers(seq, rules.pp_pays))
        fct_index = np.searchsorted(fct_values, fct_multipliers(seq, rules.fct_pays))
        for (s, k), x in results.items():
            slot = np.minimum(np.searchsorted(slots, x), n_slots - 1)
            if not np.array_equal(slots[slot], x):
                raise ValueError("a round gave a result the tables have no slot for")
            cell = g * n_slots + slot
            counts[s, k] += np.bincount(cell, minlength=N_CLASSES * n_slots).reshape(N_CLASSES, n_slots)
            if s == book:
                triple = (cell * n_pp + pp_index) * n_fct + fct_index
                side_counts[k] += np.bincount(triple, minlength=side_counts.shape[1])
        done += rows

    seen = np.where(class_counts > 0, class_counts, 1.0)
    probs = counts / seen[None, None, :, None]
    vals = np.broadcast_to(slots, counts.shape).copy()

    side = side_counts.reshape(LEVELS, N_CLASSES, n_slots * n_pp * n_fct) / seen[None, :, None]
    triples = np.stack(np.meshgrid(slots, pp_values, fct_values, indexing="ij"), axis=-1).reshape(-1, 3)
    width = max(1, int((side > 0).sum(axis=-1).max()))
    side_vals = np.zeros((LEVELS, N_CLASSES, width, 3))
    side_probs = np.zeros((LEVELS, N_CLASSES, width))
    for k in range(LEVELS):
        for cls in range(N_CLASSES):
            live = np.flatnonzero(side[k, cls] > 0)
            live = live[np.argsort(-side[k, cls, live], kind="stable")]
            side_vals[k, cls, :len(live)] = triples[live]
            side_probs[k, cls, :len(live)] = side[k, cls, live]

    return Tables(p_class=class_counts / class_counts.sum(), vals=vals, probs=probs,
                  side_vals=side_vals, side_probs=side_probs)


def save_tables(tables: Tables, path) -> None:
    np.savez_compressed(path, p_class=tables.p_class, vals=tables.vals, probs=tables.probs,
                        side_vals=tables.side_vals, side_probs=tables.side_probs)


def load_tables(path) -> Tables:
    with np.load(path) as data:
        return Tables(p_class=data["p_class"], vals=data["vals"], probs=data["probs"],
                      side_vals=data["side_vals"], side_probs=data["side_probs"])
