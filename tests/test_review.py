# tests/test_review.py
import json, pytest
from trainer.errors import TrainerError
from trainer.review import analyze_match, check_rounds, recommend_bet, recommend_play, save_review
from trainer.rules import Rules

R = Rules()
def rnd(main, player, up, actions, dealer, **kw):
    return dict(main=main, pp=0, fct=0, player=player, upcard=up, insurance=False, actions=actions, dealer=dealer, **kw)
STAND = [{"action": "stand"}]
MATCH = [
    rnd(100, ["5", "6"], "6", [{"action": "double", "card": "K"}], ["10", "8"]),
    rnd(300, ["K", "Q"], "8", STAND, ["9"]),
    rnd(200, ["9", "7"], "10", STAND, ["9"]),
    rnd(100, ["A", "7"], "9", STAND, ["10"]),
    rnd(500, ["K", "Q"], "6", STAND, ["10", "7"]),
]

def test_check_rounds_adds_up_the_chips():
    out = check_rounds(R, MATCH)
    assert [r["chips_after"] for r in out] == [1200.0, 1500.0, 1300.0, 1200.0, 1700.0]
    assert all(r["ok"] and r["complete"] for r in out) and out[2]["chips_before"] == 1500.0
    assert set(out[0]) == {"ok", "error", "complete", "net", "chips_before", "chips_after"}   # results only, never a grade

def test_dealer_cards_must_follow_the_dealer_rules():
    short = check_rounds(R, [rnd(100, ["K", "Q"], "6", STAND, ["9"])])[0]
    assert not short["ok"] and short["error"].startswith("Round 1") and "15" in short["error"]
    extra = check_rounds(R, [rnd(100, ["K", "Q"], "8", STAND, ["9", "4"])])[0]
    assert not extra["ok"] and "17" in extra["error"]

@pytest.mark.parametrize("bad", [
    rnd(100, ["K", "1"], "6", STAND, ["10", "8"]),
    rnd(5000, ["K", "Q"], "6", STAND, ["10", "8"]),
    rnd(100, ["K", "6"], "6", [{"action": "hit", "card": "9"}, {"action": "hit", "card": "2"}], ["10"]),
    rnd(102, ["K", "Q"], "6", STAND, ["10", "8"]),
    rnd(100, ["K", "6"], "6", [{"action": "split", "cards": ["3", "2"]}, {"action": "stand"}, {"action": "stand"}], ["10", "8"]),
    rnd(100, ["K", "K"], "6", STAND, ["10", "8"]) | {"pp": 10},
])
def test_typos_get_a_plain_message_naming_the_round(bad):
    out = check_rounds(R, [bad])[0]
    assert not out["ok"] and out["error"].startswith("Round 1:") and "Traceback" not in out["error"]

def test_an_unfinished_round_is_incomplete_not_wrong():
    out = check_rounds(R, [rnd(100, ["K", "Q"], "6", [], [])])[0]
    assert out["ok"] and not out["complete"] and out["chips_after"] is None

def test_the_down_card_may_be_missing_after_a_bust():
    out = check_rounds(R, [rnd(100, ["K", "6"], "6", [{"action": "hit", "card": "9"}], [])])[0]
    assert out["ok"] and out["complete"] and out["chips_after"] == 900.0

def test_split_cards_go_in_table_order():
    r = rnd(100, ["8", "8"], "6", [{"action": "split", "cards": ["3", "2"]}, {"action": "hit", "card": "K"},
                                   {"action": "hit", "card": "9"}, {"action": "stand"}], ["10", "5"])
    out = check_rounds(R, [r])[0]
    assert out["ok"] and out["complete"] and out["chips_after"] == 900.0

def test_perfect_pairs_uses_the_suits():
    r = rnd(100, ["K", "K"], "6", STAND, ["10", "8"], suits=["S", "C"]) | {"pp": 10}
    assert check_rounds(R, [r])[0]["chips_after"] == 1000.0 + 100.0 + 120.0

def test_analyze_needs_the_whole_match(quick_brain):
    with pytest.raises(TrainerError, match="Round 4"):
        analyze_match(R, quick_brain, MATCH[:3])
    with pytest.raises(TrainerError, match="Round 2"):
        analyze_match(R, quick_brain, [MATCH[0], rnd(300, ["K", "Q"], "8", [], [])] + MATCH[2:])

def test_best_bet_available_before_match_ends(quick_brain):
    for previous in ([], MATCH[:2]):
        advice = recommend_bet(R, quick_brain, previous)
        assert advice["best"] >= R.min_bet
        assert 0 <= advice["best_chance"] <= 1
    with pytest.raises(TrainerError, match="Round 1"):
        recommend_bet(R, quick_brain, [rnd(100, ["K", "Q"], "8", [], [])])

def test_suggested_play_uses_current_hand_before_match_ends(quick_brain):
    current = rnd(100, ["5", "6"], "6", [], [])
    advice = recommend_play(R, quick_brain, [], current)
    assert advice["best"] in advice["chances"]
    assert {"hit", "stand", "double"} <= set(advice["chances"])
    assert all(0 <= chance <= 1 for chance in advice["chances"].values())
    current["actions"] = [{"action": "hit", "card": "2"}]
    next_advice = recommend_play(R, quick_brain, [], current)
    assert "double" not in next_advice["chances"]

def test_a_bust_out_ends_the_match_early(quick_brain):
    report = analyze_match(R, quick_brain, [rnd(1000, ["9", "7"], "10", STAND, ["9"])])
    assert report["final_chips"] == 0.0 and len(report["rounds"]) == 1

def test_report_shape(quick_brain, tmp_path):
    report = analyze_match(R, quick_brain, MATCH)
    assert report["final_chips"] == 1700.0 and len(report["rounds"]) == 5
    r3 = report["rounds"][2]
    assert r3["bet"]["yours"] == 200.0 and {"best", "your_chance", "best_chance", "fine"} <= set(r3["bet"])
    assert r3["plays"][0]["yours"] == "stand" and r3["plays"][0]["spot"] == "16 against 10"
    assert report["rounds"][0]["plays"][0]["yours"] == "double"
    assert report["off_bets"] == sum(not r["bet"]["fine"] for r in report["rounds"])
    assert report["off_plays"] == sum(not p["fine"] for r in report["rounds"] for p in r["plays"])
    if report["off_bets"] + report["off_plays"]:
        assert report["biggest_miss"]["text"].startswith("Round ")
    path = save_review(tmp_path, {"rounds": MATCH, "opponent_final": 1450, "won": True}, report)
    saved = json.loads(path.read_text())
    assert saved["payload"]["opponent_final"] == 1450 and saved["report"]["final_chips"] == 1700.0
