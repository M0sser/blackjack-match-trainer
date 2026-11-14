# tests/test_benchmark.py
import pytest
from trainer.benchmark import PLAYERS, run_benchmark

def test_shape_of_the_report(quick_brain):
    rows = run_benchmark(quick_brain, matches=400, seed=2)
    opponents = {r["opponent"] for r in rows}
    assert "All opponents" in opponents and len(rows) == len(PLAYERS) * len(opponents)
    for r in rows:
        assert 0.0 <= r["low"] <= r["credit"] <= r["high"] <= 1.0
        assert abs(r["credit"] - (r["ahead"] + 0.5 * r["level"])) < 1e-9
    again = run_benchmark(quick_brain, matches=400, seed=2)
    assert [r["credit"] for r in again] == [r["credit"] for r in rows]

@pytest.mark.slow
def test_the_brain_is_as_good_as_it_claims(full_brain):
    rows = run_benchmark(full_brain, matches=20_000, seed=11)
    mix = {r["player"]: r["credit"] for r in rows if r["opponent"] == "All opponents"}
    assert abs(mix["Brain"] - full_brain.start_chance()) <= 0.02
    for name, value in mix.items():
        assert mix["Brain"] >= value - 0.01, name

@pytest.mark.slow
def test_the_brain_handles_the_extremes(full_brain):
    rows = {(r["player"], r["opponent"]): r["credit"] for r in run_benchmark(full_brain, matches=20_000, seed=12)}
    assert rows[("Brain", "All-in, plays by the book")] >= 0.80
