# tests/test_opponents.py
from trainer.opponents import OpponentType, bets_for, build_population, initial_memory, legal_bet, next_memory
from trainer.rules import Rules

R = Rules()
opp = lambda rule, param=0.0, style="book": OpponentType("t", style, rule, param, 1.0)
def only(options):
    assert len(options) == 1 and options[0][1] == 1.0
    return options[0][0]

def test_legal_bet():
    assert legal_bet(R, 1000.0, 103.75) == 100.0 and legal_bet(R, 60.0, 100.0) == 60.0
    assert legal_bet(R, 1007.5, 5000.0) == 1005.0 and legal_bet(R, 1000.0, 1.0) == 10.0
    assert legal_bet(R, 2.5, 100.0) == 0.0

def test_simple_rules():
    assert only(bets_for(opp("turtle"), R, 1, 1000.0, 0.0)) == 10.0
    assert only(bets_for(opp("flat", 100), R, 3, 60.0, 0.0)) == 60.0
    assert only(bets_for(opp("share", 0.25), R, 1, 1000.0, 0.0)) == 250.0
    assert only(bets_for(opp("share", 0.10), R, 2, 1037.5, 0.0)) == 100.0
    assert only(bets_for(opp("all_in"), R, 2, 1007.5, 0.0)) == 1005.0

def test_double_up_and_sit():
    o = opp("double_up_sit")
    assert only(bets_for(o, R, 1, 1000.0, 0.0)) == 1000.0
    assert only(bets_for(o, R, 2, 1000.0, 0.0)) == 1000.0
    assert only(bets_for(o, R, 2, 2000.0, 0.0)) == 10.0

def test_late_lunge():
    o = opp("late_lunge")
    assert [only(bets_for(o, R, r, 990.0, 0.0)) for r in (1, 4, 5)] == [10.0, 10.0, 990.0]

def test_chaser_memory():
    o = opp("chaser", 50)
    m = initial_memory(o)
    assert m == 50.0 and only(bets_for(o, R, 1, 1000.0, m)) == 50.0
    m = next_memory(o, m, 50.0, -1.0)
    assert only(bets_for(o, R, 2, 950.0, m)) == 100.0
    m = next_memory(o, m, 100.0, 0.0)
    assert m == 100.0
    assert next_memory(o, m, 100.0, 1.5) == 50.0

def test_random_is_five_equal_picks_merged():
    assert bets_for(opp("random"), R, 1, 1000.0, 0.0) == [(10.0, 0.2), (100.0, 0.2), (250.0, 0.2), (500.0, 0.2), (1000.0, 0.2)]
    short = bets_for(opp("random"), R, 1, 20.0, 0.0)
    assert [b for b, _ in short] == [10.0, 20.0] and abs(short[0][1] - 0.8) < 1e-12 and abs(short[1][1] - 0.2) < 1e-12

def test_mixed_is_five_equal_shares():
    assert bets_for(opp("mixed"), R, 1, 1000.0, 0.0) == [(140.0, 0.2), (250.0, 0.2), (315.0, 0.2), (525.0, 0.2), (840.0, 0.2)]
    assert [b for b, _ in bets_for(opp("mixed"), R, 2, 40.0, 0.0)] == [10.0, 20.0, 30.0]

def test_out_of_chips_bets_nothing():
    assert bets_for(opp("flat", 100), R, 3, 2.5, 0.0) == [(0.0, 1.0)]

def test_default_population():
    pop = build_population()
    assert len(pop) == 54 and abs(sum(o.weight for o in pop) - 1.0) < 1e-12
    o = next(o for o in pop if o.label == "Flat 100" and o.style == "book")
    assert abs(o.weight - (5.75 / 98) * 0.50) < 1e-12 and o.name == "Flat 100, plays by the book"
    m = next(o for o in pop if o.label == "Mixed" and o.style == "book")
    assert abs(m.weight - (44.30 / 98) * 0.50) < 1e-12
    big = next(o for o in pop if o.label == "Flat 500" and o.style == "book")
    assert abs(big.weight - (7.03 / 98) * 0.50) < 1e-12
    assert len(build_population(style_weights={"book": 1, "cautious": 0, "dealer_copy": 0})) == 18
    assert len(build_population(rule_weights={"Turtle": 1, "Flat 100": 3})) == 6
