from trainer.rules import Rules


def test_defaults_match_the_spec():
    r = Rules()
    assert (r.starting_chips, r.rounds, r.decks, r.blackjack_pays) == (1000.0, 5, 6, 1.5)
    assert (r.min_bet, r.bet_step, r.tie_credit, r.clock_seconds) == (10.0, 5.0, 0.5, 360)
    assert r.pp_pays == (25, 12, 6) and r.fct_pays == (2, 8, 10)
    assert r.insurance_offered and r.double_after_split and r.split_by_value


def test_round_trip_and_fingerprint():
    r = Rules(min_bet=20.0)
    assert Rules.from_dict({**r.to_dict(), "unknown": 1}) == r
    assert len(r.fingerprint()) == 16 and r.fingerprint() != Rules().fingerprint()
    assert Rules().fingerprint() == Rules().fingerprint()


def test_floor_bet():
    assert Rules().floor_bet(103.75) == 100.0 and Rules().floor_bet(1007.5) == 1005.0
