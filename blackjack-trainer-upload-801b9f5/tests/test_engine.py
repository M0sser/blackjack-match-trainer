import pytest

from trainer.cards import parse_card
from trainer.engine import Round, play_round, validate_bet
from trainer.errors import TrainerError
from trainer.rules import Rules

R = Rules()
shoe = lambda *cards: [parse_card(c) for c in cards]


def run(cards, actions, chips=1000.0, main=100.0, pp=0.0, fct=0.0):
    r = Round(R, shoe(*cards), chips, main, pp, fct)
    for a in actions:
        r.act(a)
    assert r.phase == "done"
    return r.result


def test_stand_and_dealer_busts():
    res = run(["10S", "6H", "9D", "10C", "KS"], ["stand"])
    assert (res.main_net, res.chips_after, res.cards_used, res.x) == (100.0, 1100.0, 5, 1.0)


def test_blackjack_pays_three_to_two_exactly():
    assert run(["AS", "9H", "KD", "7C"], []).main_net == 150.0
    assert run(["AS", "9H", "KD", "7C"], [], main=15.0).chips_after == 1022.5


def test_blackjack_against_dealer_blackjack_pushes():
    assert run(["AS", "AH", "KD", "QC"], ["no_insurance"]).main_net == 0.0


def test_insurance_wins_when_dealer_has_blackjack():
    res = run(["10S", "AH", "9D", "KC"], ["insurance"])
    assert (res.insurance_net, res.main_net, res.chips_after) == (100.0, -100.0, 1000.0)


def test_insurance_loses_then_the_hand_plays_on():
    res = run(["10S", "AH", "9D", "5C", "2S"], ["insurance", "stand"])
    assert (res.insurance_net, res.main_net, res.chips_after) == (-50.0, 100.0, 1050.0)


def test_double_takes_one_card_and_doubles_the_stake():
    res = run(["5S", "6H", "6D", "10C", "KS", "8D"], ["double"])
    assert (res.main_net, res.chips_after, res.x) == (200.0, 1200.0, 2.0)


@pytest.mark.parametrize("ace", ["AS", "AH", "AD", "AC"])
def test_cannot_double_when_the_first_player_card_is_an_ace(ace):
    r = Round(R, shoe(ace, "6H", "7D", "10C", "3S", "8D"), 1000.0, 100.0)
    assert r.legal_actions() == ["hit", "stand"] and not r.view().can_double
    with pytest.raises(TrainerError, match="first card is an ace"):
        r.act("double")
    assert r.pos == 4 and r.chips == 900 and r.actions == []


def test_can_still_double_when_the_ace_is_second():
    r = Round(R, shoe("7D", "6H", "AS", "10C", "3S", "8D"), 1000.0, 100.0)
    assert "double" in r.legal_actions() and r.view().can_double
    r.act("double")
    assert r.phase == "done" and r.hands[0].doubled and r.hands[0].stake == 200


def test_dealer_ace_does_not_block_doubling_after_insurance():
    r = Round(R, shoe("5S", "AH", "6D", "5C", "KS", "8D"), 1000.0, 100.0)
    r.act("no_insurance")
    assert "double" in r.legal_actions()


def test_double_and_split_need_the_chips():
    assert "double" not in Round(R, shoe("5S", "6H", "6D", "10C", "KS"), 150.0, 100.0).legal_actions()
    assert "split" not in Round(R, shoe("8S", "6H", "8D", "10C", "3S"), 150.0, 100.0).legal_actions()
    assert "split" in Round(R, shoe("8S", "6H", "8D", "10C", "3S"), 200.0, 100.0).legal_actions()


def test_split_plays_the_first_hand_then_the_second():
    res = run(["8S", "6H", "8D", "10C", "3S", "KH", "2C", "9S", "5H"], ["split", "hit", "hit", "stand"])
    assert (res.main_net, res.chips_after) == (-100.0, 900.0)
    assert [h.outcome for h in res.hands] == ["push", "lose"]


def test_split_aces_get_one_card_each_and_21_is_not_blackjack():
    assert run(["AS", "6H", "AD", "10C", "KS", "5H", "10D"], ["split"]).main_net == 200.0


def test_ten_value_cards_split_by_value_but_not_by_rank():
    assert "split" in Round(R, shoe("KS", "6H", "QD", "10C"), 1000.0, 100.0).legal_actions()
    by_rank = Rules(split_by_value=False)
    assert "split" not in Round(by_rank, shoe("KS", "6H", "QD", "10C"), 1000.0, 100.0).legal_actions()


def test_bust_loses_and_dealer_does_not_draw():
    res = run(["10S", "10H", "6D", "7C", "9S"], ["hit"])
    assert (res.main_net, res.cards_used) == (-100.0, 5)


def test_dealer_stands_on_soft_17():
    res = run(["10S", "6H", "8D", "AC", "5S"], ["stand"])
    assert (res.main_net, res.cards_used) == (100.0, 4)


def test_hitting_to_21_ends_the_hand():
    r = Round(R, shoe("5S", "6H", "6D", "10C", "KS", "8D"), 1000.0, 100.0)
    r.act("hit")
    assert r.phase == "done" and r.result.main_net == 100.0


@pytest.mark.parametrize("second, net", [("KS", 250.0), ("KC", 120.0), ("KH", 60.0), ("QS", -10.0)])
def test_perfect_pairs(second, net):
    assert run(["KS", "5H", second, "7C", "10D"], ["stand"], pp=10.0).pp_net == net


@pytest.mark.parametrize("up, hole, net", [("10H", "AC", 100.0), ("9H", "9C", 80.0), ("9H", "7C", 20.0), ("9H", "6C", -10.0)])
def test_four_card_total(up, hole, net):
    r = Round(R, shoe("KS", up, "QD", hole, "10D", "10C"), 1000.0, 100.0, 0.0, 10.0)
    if r.phase != "done":
        r.act("stand")
    assert r.result.fct_net == net


def test_bet_rules():
    validate_bet(R, 1000.0, 1000.0)
    validate_bet(R, 1007.5, 1005.0)
    for chips, main, pp, fct in [(1000.0, 3.0, 0, 0), (1000.0, 102.0, 0, 0), (1007.5, 1007.5, 0, 0),
                                 (100.0, 100.0, 5.0, 0), (1000.0, 0.0, 0, 0), (1000.0, 100.0, 7.0, 0)]:
        with pytest.raises(TrainerError):
            validate_bet(R, chips, main, pp, fct)


def test_wrong_action_is_refused():
    r = Round(R, shoe("10S", "6H", "9D", "10C", "KS"), 1000.0, 100.0)
    for bad in ("split", "insurance", "dance"):
        with pytest.raises(TrainerError):
            r.act(bad)
    assert r.phase == "play" and r.actions == []


def test_dealer_cards_stay_hidden_until_the_round_ends():
    r = Round(R, shoe("10S", "6H", "9D", "10C", "KS"), 1000.0, 100.0)
    assert r.visible_dealer() == [parse_card("6H"), None]
    r.act("stand")
    assert r.visible_dealer() == shoe("6H", "10C", "KS")


def test_play_round_with_a_style():
    res = play_round(R, shoe("10S", "6H", "9D", "10C", "KS"), 1000.0, 100.0, lambda view: "stand")
    assert res.main_net == 100.0 and res.x == 1.0
