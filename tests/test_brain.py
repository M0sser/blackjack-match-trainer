from trainer.brain import Brain, BrainConfig, PlaySpot
from trainer.cards import parse_card
from trainer.opponents import OpponentType
from trainer.rules import Rules

C = parse_card
WON4 = [{4: 1.0}] * 4          # four rounds that each paid +1: a turtle is on 1,040
LOST4 = [{2: 1.0}] * 4         # four rounds that each lost 1: a turtle is on 960


def spot(**kw):
    base = dict(round_no=5, chips_start=1000.0, main_bet=40.0, pp_bet=0.0, fct_bet=0.0, insurance_bet=0.0,
                player=[C("5S"), C("6D")], upcard=C("6H"), drawn=[], script=[],
                legal=["hit", "stand", "double"], stage="play")
    base.update(kw)
    return PlaySpot(**base)


def test_a_minimum_bet_cannot_catch_a_turtle(turtle_brain):
    g = turtle_brain.grade_bet(5, 1000.0, WON4, 10.0)
    assert g.your_chance == 0.0 and not g.fine and g.best[0] >= 60.0 and g.best_chance > 0.38


def test_sitting_tight_when_ahead_of_a_turtle(turtle_brain):
    assert turtle_brain.grade_bet(5, 1000.0, LOST4, 10.0).your_chance >= 0.999
    assert turtle_brain.grade_bet(5, 1000.0, LOST4, 20.0).fine
    assert not turtle_brain.grade_bet(5, 1000.0, LOST4, 200.0).fine


def test_own_side_bets_are_scored(turtle_brain):
    plain = turtle_brain.grade_bet(5, 1000.0, LOST4, 10.0)
    sided = turtle_brain.grade_bet(5, 1000.0, LOST4, 10.0, 100.0, 100.0)
    assert sided.yours == (10.0, 100.0, 100.0) and sided.your_chance < plain.your_chance and not sided.fine


def test_bet_grade_early_in_the_match(quick_brain):
    g = quick_brain.grade_bet(1, 1000.0, [], 100.0)
    assert 0.0 < g.your_chance <= g.best_chance <= 1.0 and g.best[1:] == (0.0, 0.0)
    assert abs(g.best_chance - quick_brain.start_chance()) < 0.03


def test_the_double_that_is_needed(turtle_brain):
    g = turtle_brain.grade_play(spot(), WON4, "hit")
    assert g.best == "double" and g.chances["double"] > 0.55 and g.chances["hit"] < 0.02 and not g.fine


def test_book_play_is_fine_in_an_ordinary_spot(quick_brain):
    s = spot(round_no=1, main_bet=100.0, player=[C("10S"), C("6D")], upcard=C("KH"))
    g = quick_brain.grade_play(s, [], "hit")
    assert g.fine and g.best in ("hit", "stand") and set(g.chances) == {"hit", "stand", "double"}


def test_insurance_scores_both_choices(turtle_brain):
    s = spot(main_bet=100.0, player=[C("10S"), C("9D")], upcard=C("AH"), legal=["insurance", "no_insurance"], stage="insurance")
    g = turtle_brain.grade_play(s, LOST4, "no_insurance")
    assert set(g.chances) == {"insurance", "no_insurance"} and all(0.0 <= v <= 1.0 for v in g.chances.values())


def test_same_spot_same_grade(quick_brain):
    s = spot(round_no=2, main_bet=100.0)
    assert quick_brain.grade_play(s, [{2: 1.0}], "double").chances == quick_brain.grade_play(s, [{2: 1.0}], "double").chances


def test_round_class_from_a_full_shoe(turtle_brain):
    shoe = [C(c) for c in ("5S", "6H", "6D", "10C", "KS", "8D")] + [C("2C")] * 34
    assert turtle_brain.round_class(shoe) == 6


def test_round_class_when_a_card_was_never_seen(turtle_brain):
    dist = turtle_brain.round_class_dist({0: C("10S"), 1: C("10H"), 2: C("6D"), 3: C("7C")})
    assert abs(sum(dist.values()) - 1.0) < 1e-9 and set(dist) <= {2, 3, 4} and dist[2] > 0.5


def test_round_class_is_certain_when_every_card_needed_was_seen(turtle_brain):
    known = {0: C("5S"), 1: C("6H"), 2: C("6D"), 3: C("10C"), 4: C("KS"), 5: C("8D")}
    assert turtle_brain.round_class_dist(known) == {6: 1.0}


def test_uncertain_history_is_averaged(turtle_brain):
    # on 970 with the same bet as the turtle: ahead for sure if it is on 960, behind for sure if it is on 980
    assert turtle_brain.grade_bet(5, 970.0, LOST4, 10.0).your_chance >= 0.999
    mixed = turtle_brain.grade_bet(5, 970.0, LOST4[:3] + [{2: 0.5, 4: 0.5}], 10.0).your_chance
    assert abs(mixed - 0.5) < 0.005


def test_table_bet_is_a_legal_bet(quick_brain):
    assert quick_brain.table_bet(1, 1000.0, []) >= 10.0
    assert quick_brain.table_bet(3, 12.5, [2, 2]) == 10.0
    assert quick_brain.table_bet(3, 7.5, [2, 2]) == 0.0


def test_prepare_reuses_its_cache(tmp_path):
    pop = [OpponentType("Turtle", "book", "turtle", 0.0, 1.0)]
    cfg = BrainConfig(table_rounds=50_000, samples=2_000, coarse=True)
    first = []
    a = Brain.prepare(Rules(), pop, tmp_path, cfg, progress=lambda f, m: first.append(m))
    calls = []
    b = Brain.prepare(Rules(), pop, tmp_path, cfg, progress=lambda f, m: calls.append(m))
    assert a.start_chance() == b.start_chance() and len(list(tmp_path.glob("*.npz"))) == 2
    assert any("solv" in m.lower() for m in first)
    assert not any("solv" in m.lower() for m in calls)
