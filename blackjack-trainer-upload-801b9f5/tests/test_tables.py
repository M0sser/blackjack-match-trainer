import numpy as np
import pytest

from trainer.rules import Rules
from trainer.styles import STYLE_INDEX
from trainer.tables import CLASS_VALUES, build_tables, class_of, load_tables, save_tables


@pytest.fixture(scope="module")
def tables():
    return build_tables(Rules(), n_rounds=200_000, seed=3)


def test_class_of():
    assert [int(class_of(x)) for x in (-4, -3, -2, -1, 0, 1, 1.5, 2, 3, 4)] == [0, 0, 1, 2, 3, 4, 5, 6, 7, 7]
    assert class_of(np.array([-1.0, 1.5])).tolist() == [2, 5]


def test_class_rates_look_like_blackjack(tables):
    p = tables.p_class
    assert abs(p.sum() - 1.0) < 1e-12
    assert 0.38 < p[2] < 0.48 and 0.27 < p[4] < 0.36 and 0.07 < p[3] < 0.10 and 0.040 < p[5] < 0.050


def test_conditional_rows_sum_to_one(tables):
    assert np.allclose(tables.probs.sum(axis=-1)[:, :, tables.p_class > 0], 1.0)
    assert np.allclose(tables.side_probs.sum(axis=-1)[:, tables.p_class > 0], 1.0)


def test_book_with_full_chips_is_the_class_itself(tables):
    b = STYLE_INDEX["book"]
    for g in range(1, 7):
        j = int(tables.probs[b, 3, g].argmax())
        assert tables.probs[b, 3, g, j] == 1.0 and tables.vals[b, 3, g, j] == CLASS_VALUES[g]


def test_no_spare_chips_means_no_doubles_or_splits(tables):
    b = STYLE_INDEX["book"]
    live = tables.probs[b, 0] > 0
    assert tables.vals[b, 0][live].max() <= 1.5 and tables.vals[b, 0][live].min() >= -1.0


def test_side_table_uses_the_real_payouts(tables):
    live = tables.side_probs > 0
    assert set(np.unique(tables.side_vals[..., 1][live])) <= {-1.0, 6.0, 12.0, 25.0}
    assert set(np.unique(tables.side_vals[..., 2][live])) <= {-1.0, 2.0, 8.0, 10.0}


def test_side_table_agrees_with_the_main_table(tables):
    b = STYLE_INDEX["book"]
    for k in range(4):
        for g in np.flatnonzero(tables.p_class > 0):
            from_side = float((tables.side_vals[k, g, :, 0] * tables.side_probs[k, g]).sum())
            from_main = float((tables.vals[b, k, g] * tables.probs[b, k, g]).sum())
            assert abs(from_side - from_main) < 1e-9


def test_same_seed_same_tables_and_round_trip(tmp_path, tables):
    again = build_tables(Rules(), n_rounds=200_000, seed=3)
    assert np.array_equal(again.probs, tables.probs)
    save_tables(tables, tmp_path / "t.npz")
    loaded = load_tables(tmp_path / "t.npz")
    assert np.array_equal(loaded.vals, tables.vals) and np.array_equal(loaded.side_probs, tables.side_probs)
