# tests/test_practice.py
import json, numpy as np, pytest
from trainer.cards import parse_card
from trainer.errors import TrainerError
from trainer.opponents import OpponentType
from trainer.practice import PracticeMatch, replay_opponent
from trainer.rules import Rules
from trainer.styles import make_style

R = Rules()
FLAT = OpponentType("Flat 100", "book", "flat", 100.0, 1.0)
def shoe(*cards, fill="2C"):
    return [parse_card(c) for c in cards] + [parse_card(fill)] * 30

def play_book(match, bet=100.0):
    book, s = make_style("book", R), match.state()
    while s["phase"] != "match_over":
        if s["phase"] == "bet":
            s = match.bet(min(bet, R.floor_bet(s["chips"])))
        elif s["phase"] == "insurance":
            s = match.act("no_insurance")
        elif s["phase"] == "play":
            s = match.act(book(match.current_view()))
        else:
            s = match.next_round()
    return s

def test_a_full_match_is_reproducible(quick_brain):
    a = play_book(PracticeMatch(R, quick_brain, quick_brain.population, "match", seed=5))
    b = play_book(PracticeMatch(R, quick_brain, quick_brain.population, "match", seed=5))
    assert a["summary"]["you"] == b["summary"]["you"] and a["summary"]["opponent"] == b["summary"]["opponent"]
    assert a["summary"]["outcome"] in ("ahead", "level", "behind") and len(a["summary"]["rounds"]) == 5
    json.dumps(a)

def test_coach_grades_each_round_and_match_waits(quick_brain):
    for mode, shown in (("coach", True), ("match", False)):
        m = PracticeMatch(R, quick_brain, quick_brain.population, mode, seed=6)
        s = m.bet(100.0)
        book = make_style("book", R)
        while s["phase"] in ("insurance", "play"):
            s = m.act("no_insurance" if s["phase"] == "insurance" else book(m.current_view()))
        assert s["phase"] == "round_over" and (s["grades"] is not None) == shown
        if shown:
            assert s["grades"][0]["kind"] == "bet" and {"yours", "best", "fine", "reason"} <= set(s["grades"][0])
    done = play_book(PracticeMatch(R, quick_brain, quick_brain.population, "match", seed=6))
    assert all(r["grades"] for r in done["summary"]["rounds"])

def test_the_down_card_is_never_shown_early(quick_brain):
    m = PracticeMatch(R, quick_brain, [FLAT], "coach", seed=1, shoes=[shoe("10S", "6H", "9D", "KC", "5S")] * 5)
    s = m.bet(100.0)
    assert s["phase"] == "play" and s["dealer"] == ["6H", None] and s["dealer_total"] is None
    assert "KC" not in json.dumps(s)
    s = m.act("stand")
    assert s["dealer"][:2] == ["6H", "KC"] and s["result"]["net"] == -100.0

def test_moves_at_the_wrong_moment_are_refused(quick_brain):
    m = PracticeMatch(R, quick_brain, [FLAT], "coach", seed=1, shoes=[shoe("10S", "6H", "9D", "KC", "5S")] * 5)
    with pytest.raises(TrainerError, match="bet"):
        m.act("hit")
    with pytest.raises(TrainerError):
        m.next_round()
    with pytest.raises(TrainerError):
        m.bet(102.0)
    assert m.state()["phase"] == "bet"
    m.bet(100.0)
    with pytest.raises(TrainerError):
        m.bet(100.0)
    with pytest.raises(TrainerError):
        m.act("split")
    assert m.state()["phase"] == "play" and m.state()["chips"] == 900.0

def test_losing_everything_ends_the_match(quick_brain):
    m = PracticeMatch(R, quick_brain, [FLAT], "match", seed=1, shoes=[shoe("10S", "10H", "6D", "7C")] * 5)
    m.bet(1000.0)
    s = m.act("stand")
    s = s if s["phase"] == "match_over" else m.next_round()
    assert s["phase"] == "match_over" and s["summary"]["you"] == 0.0 and len(s["summary"]["rounds"]) == 1
    assert s["summary"]["outcome"] == "behind"

def test_half_chips_survive(quick_brain):
    m = PracticeMatch(R, quick_brain, [FLAT], "match", seed=1, shoes=[shoe("AS", "9H", "KD", "7C")] * 5)
    s = m.bet(5.0)
    assert s["phase"] == "round_over" and s["chips"] == 1007.5
    m.next_round()
    with pytest.raises(TrainerError):
        m.bet(1007.5)
    assert m.bet(1005.0)["bet"]["main"] == 1005.0

def test_the_clock_forfeits_a_match_but_not_a_coaching_session(quick_brain):
    for mode, outcome in (("match", "forfeit"), ("coach", None)):
        t = [0.0]
        m = PracticeMatch(R, quick_brain, [FLAT], mode, seed=1, now=lambda: t[0], shoes=[shoe("10S", "6H", "9D", "KC", "5S")] * 5)
        assert m.state()["clock"]["left"] == 360
        t[0] = 100.0
        assert m.state()["clock"]["left"] == 260
        t[0] = 361.0
        s = m.state()
        if outcome:
            assert s["phase"] == "match_over" and s["summary"]["outcome"] == "forfeit"
        else:
            assert s["phase"] == "bet" and s["clock"]["left"] == 0

def test_opponent_total_is_a_straight_replay(quick_brain):
    shoes = [shoe("10S", "6H", "9D", "KC", "5S")] * 5          # 19 against 16 drawing 5: lose every round
    assert replay_opponent(R, FLAT, shoes, np.random.default_rng(0)) == 500.0
    s = play_book(PracticeMatch(R, quick_brain, [FLAT], "match", seed=1, shoes=shoes))
    assert s["summary"]["opponent"] == 500.0 and s["summary"]["opponent_name"] == "Flat 100, plays by the book"
    assert s["summary"]["outcome"] == "level"


def test_a_click_that_arrives_twice_is_refused(quick_brain):
    # 9 against 6; each hit brings a small card
    m = PracticeMatch(R, quick_brain, [FLAT], "coach", seed=1, shoes=[shoe("5S", "6H", "4D", "KC", "2S", "3S", "9S")] * 5)
    s = m.state()
    s = m.bet(100.0, step=s["step"])
    after_hit = m.act("hit", step=s["step"])
    assert after_hit["step"] > s["step"] and len(after_hit["hands"][0]["cards"]) == 3
    with pytest.raises(TrainerError, match="twice"):
        m.act("hit", step=s["step"])                      # the same click again
    assert m.state()["hands"][0]["cards"] == after_hit["hands"][0]["cards"]
    over = m.act("stand", step=after_hit["step"])
    assert over["phase"] == "round_over"
    with pytest.raises(TrainerError, match="twice"):
        m.next_round(step=after_hit["step"])
    assert m.next_round(step=over["step"])["phase"] == "bet"
    assert m.bet(100.0)["phase"] == "play"                # no step given: nothing to compare, as before
