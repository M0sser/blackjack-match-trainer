# Match Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local practice table and match review for five-round head-to-head blackjack, with a brain that grades every bet and play by the chance of finishing ahead of a hidden opponent.

**Architecture:** A Python package `trainer` holds a reference blackjack engine, a NumPy engine that must match it exactly, simulated outcome tables, and a backward-induction solver whose value tables grade decisions. A small `http.server` app exposes the practice match and the review as JSON; a plain HTML/JS page is the screen.

**Tech Stack:** Python 3.13, NumPy, pytest, stdlib `http.server`, vanilla HTML/CSS/JS (no build step).

**Spec:** `docs/superpowers/specs/2026-11-14-match-trainer-design.md`

## Global Constraints

- Python is started with `py -3.13`; the project uses a virtual environment at `.venv`. Tests run with `.venv/Scripts/python.exe -m pytest`.
- Dependencies: `numpy` and `pytest` only. No other packages, no CDN assets, no network requests at runtime.
- The server binds `127.0.0.1` only.
- Chip amounts are floats that are always multiples of 2.5; arithmetic on them must stay exact (bets are multiples of 5; blackjack pays 1.5).
- Rule defaults (spec section 5): 1,000 starting chips, 5 rounds, 6 decks shuffled fresh per round, dealer stands on every 17, blackjack 3:2, insurance at half the main bet paying 2:1, no surrender, double any first two cards including after a split, one split per round on equal value, split aces one card each, minimum bet 5 in steps of 5, equal final chips count 0.5, clock 360 seconds.
- Side-bet payouts: Perfect Pairs 25 / 12 / 6 to 1; 4 Card Total 2 / 8 / 10 to 1 for 36–37 / 38–39 / 40+ with ace = 11.
- "Fine" means within 0.005 of the best chance, or within twice the standard error of a sampled difference.
- On screen the tool is called "Match Trainer". No DraftKings name, logo or artwork anywhere in the UI.
- The review reports only on a complete finished match. No endpoint or screen returns a grade for cards typed in hand by hand.
- Every user-facing error is a `TrainerError` with a plain-words message; the UI shows the message as is.
- Speed targets: first solve under 5 minutes; bet grade under 0.5 s; play grade under 1 s; review report under 15 s.

## Review Focus

1. **Brain still solving.** Starting a match or a review before the first solve finishes must return a clear "still working out the best bets" message, not a crash or a hang. Test in Task 12.
2. **Chips ending in .5.** After a blackjack on an odd multiple of 5 the balance is e.g. 1,007.5: all-in is 1,005, the total is shown as 1,007.5, and nothing rounds it. Tests in Tasks 2 and 10.
3. **Running out of chips.** Losing an all-in ends the match early: remaining rounds are skipped, the result still shows, and a review of such a match is accepted. Tests in Tasks 10 and 11.
4. **Typos in a review.** Dealer cards that break the dealer's rules, an unknown card, a bet above the chips, a hit after a bust: each gets a message naming the round, never a stack trace. Tests in Task 11.
5. **Clicks at the wrong moment.** A double-click or an action sent in the wrong phase (hit while betting, bet while playing) is refused with a message and leaves the match unchanged. Tests in Tasks 10 and 12.

## File Structure

```text
requirements.txt            numpy, pytest
pytest.ini                  registers the "slow" marker; default run skips slow tests
Start Trainer.bat           first-run setup, then starts the server and opens the browser
README.md                   what it is, how to start, what "best" assumes
trainer/
  __init__.py
  __main__.py               `python -m trainer` -> server.main()
  errors.py                 TrainerError
  rules.py                  Rules dataclass (settings with defaults), fingerprint
  cards.py                  card codes, parsing, totals, shuffled shoe
  sidebets.py               Perfect Pairs and 4 Card Total multipliers (scalar and array)
  engine.py                 reference engine: Round, RoundResult, HandView, play_round, validate_bet
  styles.py                 Book, Cautious, Dealer-copy as functions and as lookup tables
  fast.py                   NumPy engine: sample_sequences, sample_completions, play_rounds
  opponents.py              OpponentType, betting rules, default population
  tables.py                 result classes and simulated outcome tables
  solver.py                 opponent chip distributions and backward induction
  brain.py                  Brain: grade_bet, grade_play, round classes, cache
  benchmark.py              full-match simulation report (also the acceptance test)
  practice.py               PracticeMatch: the practice table's state machine
  review.py                 check_rounds, analyze_match, save_review
  settings.py               Settings load/save, population from settings
  server.py                 App (routing, brain status), HTTP server, main()
  web/index.html, style.css, app.js, practice.js, review.js, settings.js
scripts/smoke_ui.py         drives the real page in headless Edge
tests/                      one test file per module, conftest.py with shared brains
```

---

### Task 1: Project setup, rules and cards

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `trainer/__init__.py`, `trainer/errors.py`, `trainer/rules.py`, `trainer/cards.py`, `tests/__init__.py`
- Test: `tests/test_rules.py`, `tests/test_cards.py`

**Interfaces:**
- Produces:
  - `errors.TrainerError(Exception)`
  - `rules.Rules` (frozen dataclass): `starting_chips=1000.0, rounds=5, decks=6, blackjack_pays=1.5, insurance_offered=True, double_after_split=True, split_by_value=True, min_bet=5.0, bet_step=5.0, tie_credit=0.5, clock_seconds=360, pp_pays=(25, 12, 6), fct_pays=(2, 8, 10)`; methods `to_dict() -> dict`, `from_dict(d) -> Rules` (classmethod, ignores unknown keys), `fingerprint() -> str` (16 hex chars of a SHA-256 over the sorted JSON of `to_dict()`), `floor_bet(amount: float) -> float` (round down to a multiple of `bet_step`)
  - `cards.RANKS = ("A","2",...,"10","J","Q","K")`, `cards.SUITS = ("S","H","D","C")`; a card is `rank_index * 4 + suit_index` (0..51)
  - `cards.parse_card(text) -> int` (`"10S"`; rank alone means spades; raises `TrainerError` on anything else), `cards.card_str(card) -> str`
  - `cards.rank_of(card) -> int`, `suit_of(card) -> int`, `value_of(card) -> int` (ace 1, tens 10), `is_red(card) -> bool` (hearts, diamonds)
  - `cards.VALUES: np.ndarray` shape (52,), int8, `VALUES[card] == value_of(card)`
  - `cards.hand_total(cards) -> tuple[int, bool]` (best total, soft flag)
  - `cards.new_shoe(rng: np.random.Generator, decks: int = 6) -> np.ndarray` shuffled int8

- [ ] **Step 1: Create the environment**

Run: `py -3.13 -m venv .venv && .venv/Scripts/python.exe -m pip install --quiet -r requirements.txt` after writing `requirements.txt` (`numpy`, `pytest`) and `pytest.ini`:

```ini
[pytest]
markers =
    slow: needs the full solve (minutes); run with -m slow
addopts = -m "not slow" -q
testpaths = tests
pythonpath = .
```

Expected: `.venv/Scripts/python.exe -c "import numpy, pytest; print(numpy.__version__)"` prints a version.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_rules.py
from trainer.rules import Rules

def test_defaults_match_the_spec():
    r = Rules()
    assert (r.starting_chips, r.rounds, r.decks, r.blackjack_pays) == (1000.0, 5, 6, 1.5)
    assert (r.min_bet, r.bet_step, r.tie_credit, r.clock_seconds) == (5.0, 5.0, 0.5, 360)
    assert r.pp_pays == (25, 12, 6) and r.fct_pays == (2, 8, 10)
    assert r.insurance_offered and r.double_after_split and r.split_by_value

def test_round_trip_and_fingerprint():
    r = Rules(min_bet=10.0)
    assert Rules.from_dict({**r.to_dict(), "unknown": 1}) == r
    assert len(r.fingerprint()) == 16 and r.fingerprint() != Rules().fingerprint()
    assert Rules().fingerprint() == Rules().fingerprint()

def test_floor_bet():
    assert Rules().floor_bet(103.75) == 100.0 and Rules().floor_bet(1007.5) == 1005.0
```

```python
# tests/test_cards.py
import numpy as np, pytest
from trainer.cards import parse_card, card_str, value_of, is_red, hand_total, new_shoe, VALUES
from trainer.errors import TrainerError

def test_parse_and_format():
    for text in ("AS", "10H", "KD", "2C"):
        assert card_str(parse_card(text)) == text
    assert card_str(parse_card("10")) == "10S"
    for bad in ("1", "11", "XS", "", "10Z"):
        with pytest.raises(TrainerError):
            parse_card(bad)

def test_values_and_colours():
    assert [value_of(parse_card(t)) for t in ("AS", "5H", "10D", "JC", "QS", "KH")] == [1, 5, 10, 10, 10, 10]
    assert all(VALUES[c] == value_of(c) for c in range(52))
    assert is_red(parse_card("KH")) and is_red(parse_card("KD"))
    assert not is_red(parse_card("KS")) and not is_red(parse_card("KC"))

def test_hand_total():
    T = lambda *t: hand_total([parse_card(x) for x in t])
    assert T("AS", "6H") == (17, True)
    assert T("AS", "6H", "10D") == (17, False)
    assert T("AS", "AH") == (12, True)
    assert T("10S", "9H", "5D") == (24, False)

def test_new_shoe_is_six_decks_and_seeded():
    a, b = new_shoe(np.random.default_rng(7)), new_shoe(np.random.default_rng(7))
    assert len(a) == 312 and (a == b).all() and (np.bincount(a, minlength=52) == 6).all()
```

- [ ] **Step 3: Run them to see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rules.py tests/test_cards.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer'`.

- [ ] **Step 4: Implement `errors.py`, `rules.py`, `cards.py`** to the interfaces above.

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rules.py tests/test_cards.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Add project setup, rules and cards"
```

---

### Task 2: Reference engine and side bets

**Files:**
- Create: `trainer/sidebets.py`, `trainer/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `Rules`, `TrainerError`, `cards.*`
- Produces:
  - `sidebets.pp_multiplier(c1, c2, pays=(25, 12, 6)) -> int`: 25 same rank and suit, 12 same rank and colour, 6 same rank, else -1
  - `sidebets.fct_multiplier(p1, p2, up, hole, pays=(2, 8, 10)) -> int`: sum with ace = 11; 40+ gives 10, 38–39 gives 8, 36–37 gives 2, else -1
  - `sidebets.pp_multipliers(seq, pays) -> np.ndarray` and `fct_multipliers(seq, pays) -> np.ndarray` for `(N, W)` card arrays (player cards in columns 0 and 2, dealer in 1 and 3)
  - `engine.HandView` (frozen): `total: int, soft: bool, pair_value: int, upcard: int, can_double: bool, can_split: bool`. `pair_value` is the card value (ace 1) when the hand is the unsplit first two cards and they pair under the split rule, else 0. `upcard` is the dealer's up card value (ace 1).
  - `engine.Hand`: `cards: list[int], stake: float, doubled: bool, from_split: bool, done: bool, outcome: str | None` (`"win" | "lose" | "push" | "blackjack" | "bust"`)
  - `engine.RoundResult`: `main_net, pp_net, fct_net, insurance_net, net, chips_after, x` (`main_net / main_bet`), `cards_used: int, dealer_cards: list[int], hands: list[Hand]`
  - `engine.Round(rules, shoe, chips, main_bet, pp_bet=0.0, fct_bet=0.0)`: attributes `phase` (`"insurance" | "play" | "done"`), `hands`, `active` (index of the hand to act), `upcard`, `pos`, `chips` (unstaked), `main_bet`, `pp_bet`, `fct_bet`, `insurance_bet`, `actions` (play actions so far, as names), `result`; methods `legal_actions() -> list[str]`, `act(action: str) -> None`, `view() -> HandView`, `visible_dealer() -> list[int | None]` (`[upcard, None]` until done, then every dealer card). `shoe` may be a list that grows while the round is live; reading past its end raises `IndexError`.
  - `engine.validate_bet(rules, chips, main_bet, pp_bet=0.0, fct_bet=0.0) -> None` (raises `TrainerError`)
  - `engine.play_round(rules, shoe, chips, main_bet, style, pp_bet=0.0, fct_bet=0.0) -> RoundResult` where `style(view) -> str`; it always declines insurance

**Round semantics** (the fast engine in Task 4 must reproduce these exactly):

1. Stakes come off `chips` first. `validate_bet`: main bet a multiple of `bet_step`, at least `min_bet`; side bets zero or multiples of `bet_step`; the total at most `chips`.
2. Deal order: `shoe[0]` player, `shoe[1]` dealer up, `shoe[2]` player, `shoe[3]` dealer down; `pos = 4`.
3. Up card an ace and `insurance_offered`: phase `"insurance"`, actions `"insurance"` (costs `main_bet / 2`, legal only with the chips) or `"no_insurance"`.
4. With an ace or ten-value up card the dealer checks the down card. Dealer blackjack ends the round at once: insurance pays 2:1, the main bet pushes against a player blackjack and otherwise loses. Without dealer blackjack, insurance is lost.
5. A player blackjack (first two cards, never after a split) pays `blackjack_pays` and ends the round; the dealer draws nothing.
6. Play. A hand that reaches 21 or more is finished without a decision. `hit` draws one card. `double` (two cards, chips cover the stake, not a split-ace hand, after a split only if `double_after_split`) adds one stake, draws one card and finishes the hand. `split` (only as the first decision of the round: one hand, two cards equal by value, or by rank when `split_by_value` is false, chips cover the stake) adds one stake; hand 1 keeps the first card and draws at once; hand 2 keeps the second card and draws its second card only when hand 1 is finished. Split aces: each hand is finished after its one card.
7. Dealer: draws nothing if every player hand is bust; otherwise draws while the total is under 17 (an ace counts 11 when that does not bust, so soft 17 stands).
8. Settle each hand: bust loses the stake; otherwise dealer bust or a higher total wins 1:1; equal totals push.
9. Side bets settle from the four dealt cards; a lost side bet nets minus its stake.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_engine.py
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
    assert run(["AS", "9H", "KD", "7C"], [], main=5.0).chips_after == 1007.5

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
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_engine.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.engine'`.

- [ ] **Step 3: Implement `sidebets.py` and `engine.py`** to the interfaces and the nine semantics above.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_engine.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add reference blackjack engine and side bets"
```

---

### Task 3: Play styles

**Files:**
- Create: `trainer/styles.py`
- Test: `tests/test_styles.py`

**Interfaces:**
- Consumes: `engine.HandView`, `Rules`
- Produces:
  - `styles.STYLE_NAMES = ("book", "cautious", "dealer_copy")`, `styles.STYLE_INDEX = {"book": 0, "cautious": 1, "dealer_copy": 2}`
  - `styles.ACTIONS = ("stand", "hit", "double", "split")` (the index is the action code)
  - `styles.make_style(name: str, rules: Rules) -> Callable[[HandView], str]`
  - `styles.style_table(name: str, rules: Rules) -> np.ndarray`: int8, shape `(2, 2, 11, 2, 22, 11)`, indexed `[can_double][can_split][pair_value][soft][total][upcard]`, built by calling the style function, so the two forms cannot disagree

**The charts** (up card: 1 is an ace):

- **Book**, pairs (only when `can_split`): aces and 8s always; 10s never; 9s against 2–6, 8, 9; 7s against 2–7; 6s against 2–6; 5s never; 4s against 5–6; 2s and 3s against 2–7. With `double_after_split` false: 6s against 3–6, 4s never, 2s and 3s against 4–7.
- **Book**, soft totals: 13–14 double against 5–6 else hit; 15–16 double against 4–6 else hit; 17 double against 3–6 else hit; 18 double against 3–6, stand against 2, 7, 8, hit against 9, 10, ace; 19+ stand; soft 12 hits. When `can_double` is false, a would-be double becomes hit, except soft 18 which stands.
- **Book**, hard totals: 17+ stand; 13–16 stand against 2–6 else hit; 12 stand against 4–6 else hit; 11 double against 2–10 else hit; 10 double against 2–9 else hit; 9 double against 3–6 else hit; 8 or less hit. A would-be double without `can_double` hits.
- **Cautious**: splits only aces and 8s; doubles only hard 10 and 11 against 2–9; stands on hard 12+; hits hard 11 or less; hits soft 17 or less; stands on soft 18+.
- **Dealer-copy**: hits under 17, stands on 17+ (soft 17 included); never doubles or splits.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_styles.py
import itertools, pytest
from trainer.engine import HandView
from trainer.rules import Rules
from trainer.styles import ACTIONS, STYLE_NAMES, make_style, style_table

R = Rules()
def V(total, up, soft=False, pair=0, dbl=True, spl=False):
    return HandView(total=total, soft=soft, pair_value=pair, upcard=up, can_double=dbl, can_split=spl)

BOOK = [
    (V(16, 10), "hit"), (V(16, 6), "stand"), (V(12, 3), "hit"), (V(12, 4), "stand"),
    (V(11, 10), "double"), (V(11, 10, dbl=False), "hit"), (V(11, 1), "hit"),
    (V(10, 10), "hit"), (V(10, 9), "double"), (V(9, 3), "double"), (V(9, 2), "hit"), (V(8, 6), "hit"),
    (V(17, 1), "stand"),
    (V(18, 9, soft=True), "hit"), (V(18, 2, soft=True), "stand"), (V(18, 5, soft=True), "double"),
    (V(18, 5, soft=True, dbl=False), "stand"), (V(17, 3, soft=True), "double"),
    (V(17, 3, soft=True, dbl=False), "hit"), (V(13, 5, soft=True), "double"), (V(13, 4, soft=True), "hit"),
    (V(19, 6, soft=True), "stand"),
    (V(16, 10, pair=8, spl=True), "split"), (V(16, 10, pair=8), "hit"),
    (V(20, 6, pair=10, spl=True), "stand"), (V(18, 7, pair=9, spl=True), "stand"),
    (V(18, 8, pair=9, spl=True), "split"), (V(10, 9, pair=5, spl=True), "double"),
    (V(8, 5, pair=4, spl=True), "split"), (V(8, 4, pair=4, spl=True), "hit"),
    (V(12, 1, soft=True, pair=1, spl=True), "split"), (V(12, 1, soft=True, pair=1), "hit"),
    (V(4, 7, pair=2, spl=True), "split"), (V(4, 8, pair=2, spl=True), "hit"),
]

@pytest.mark.parametrize("view, action", BOOK)
def test_book(view, action):
    assert make_style("book", R)(view) == action

def test_book_without_double_after_split():
    book = make_style("book", Rules(double_after_split=False))
    assert book(V(8, 5, pair=4, spl=True)) == "hit"
    assert book(V(4, 3, pair=2, spl=True)) == "hit" and book(V(4, 4, pair=2, spl=True)) == "split"
    assert book(V(12, 2, pair=6, spl=True)) == "hit" and book(V(12, 3, pair=6, spl=True)) == "split"

def test_cautious():
    c = make_style("cautious", R)
    assert c(V(12, 10)) == "stand" and c(V(11, 5)) == "double" and c(V(11, 10)) == "hit"
    assert c(V(9, 5)) == "hit" and c(V(17, 6, soft=True)) == "hit" and c(V(18, 9, soft=True)) == "stand"
    assert c(V(16, 10, pair=8, spl=True)) == "split" and c(V(18, 6, pair=9, spl=True)) == "stand"

def test_dealer_copy():
    d = make_style("dealer_copy", R)
    assert d(V(16, 6)) == "hit" and d(V(17, 10)) == "stand" and d(V(17, 10, soft=True)) == "stand"
    assert d(V(11, 6)) == "hit" and d(V(16, 10, pair=8, spl=True)) == "hit"

@pytest.mark.parametrize("name", STYLE_NAMES)
def test_table_equals_function(name):
    style, table = make_style(name, R), style_table(name, R)
    assert table.shape == (2, 2, 11, 2, 22, 11)
    for dbl, spl, pair, soft, total, up in itertools.product((0, 1), (0, 1), range(11), (0, 1), range(4, 21), range(1, 11)):
        view = HandView(total=total, soft=bool(soft), pair_value=pair, upcard=up, can_double=bool(dbl), can_split=bool(spl))
        assert ACTIONS[table[dbl, spl, pair, soft, total, up]] == style(view)
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_styles.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.styles'`.

- [ ] **Step 3: Implement `styles.py`** from the charts above. A style never returns an action its flags forbid (`double` needs `can_double`, `split` needs `can_split`).

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_styles.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add Book, Cautious and Dealer-copy play styles"
```

---

### Task 4: Fast engine

**Files:**
- Create: `trainer/fast.py`
- Test: `tests/test_fast.py`

**Interfaces:**
- Consumes: `Rules`, `cards.VALUES`, `styles.style_table`, `styles.ACTIONS`, `engine.Round`/`play_round` (tests only)
- Produces:
  - `fast.WIDTH = 40`
  - `fast.sample_sequences(rng, n: int, decks: int = 6, width: int = WIDTH) -> np.ndarray`: `(n, width)` int8, each row the first `width` cards of an independent shuffled shoe
  - `fast.sample_completions(rng, n: int, known: dict[int, int], decks: int = 6, width: int = WIDTH) -> np.ndarray`: rows whose positions in `known` hold those cards and whose other positions are drawn without replacement from the rest of the shoe
  - `fast.play_rounds(rules, seq, table, extra_bets, script=()) -> np.ndarray`: float64 `(N,)`, the main result per chip bet. `extra_bets` (int or `(N,)` array, 0–3) is how many extra stakes the player can afford this round. `script` is a sequence of action codes forced on the first `len(script)` player decisions of every row; later decisions come from `table`.

**Algorithm.** A masked state machine over all rows, following Task 2's semantics 2 and 4–8. Per row: next card position; for up to two hands the hard sum (aces as 1), an ace flag, the card count, the stake multiplier, done and bust flags; the current hand; the count of extra stakes used; the count of decisions made. Each pass picks an action for every row whose current hand is live (the script while the decision count is below its length, else a table lookup with `can_double = two cards and used < extra_bets and (unsplit or double_after_split) and not a split-ace hand`, `can_split = unsplit two-card pair and used < extra_bets`), applies it with masks, and advances finished rows to their second hand or to the dealer. A hand at 21 or more is finished without consuming a decision. Rows resolved by a blackjack skip play. Card positions are clamped to `width - 1`.

Sampling without replacement uses a partial Fisher–Yates shuffle over the columns of an `(n, 52 * decks)` array, in chunks of at most 100,000 rows.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fast.py
import numpy as np, pytest
from trainer.cards import VALUES, parse_card
from trainer.engine import Round, play_round
from trainer.fast import WIDTH, play_rounds, sample_completions, sample_sequences
from trainer.rules import Rules
from trainer.sidebets import pp_multipliers
from trainer.styles import ACTIONS, STYLE_NAMES, make_style, style_table

R = Rules()

def test_sequences_are_real_shoe_prefixes():
    seq = sample_sequences(np.random.default_rng(1), 2000)
    assert seq.shape == (2000, WIDTH) and seq.dtype == np.int8
    assert max(np.bincount(row, minlength=52).max() for row in seq) <= 6
    assert not (seq[0] == seq[1]).all()

def test_completions_keep_known_cards_and_respect_the_shoe():
    known = {0: parse_card("KS"), 1: parse_card("KS"), 2: parse_card("KS"), 4: parse_card("KS")}
    seq = sample_completions(np.random.default_rng(2), 3000, known)
    for pos, card in known.items():
        assert (seq[:, pos] == card).all()
    assert max((row == parse_card("KS")).sum() for row in seq) <= 6

@pytest.mark.parametrize("style", STYLE_NAMES)
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_fast_engine_equals_reference(style, k):
    seq = sample_sequences(np.random.default_rng(100 + k), 3000)
    fast = play_rounds(R, seq, style_table(style, R), k)
    fn = make_style(style, R)
    slow = np.array([play_round(R, [int(c) for c in row], 100.0 * (1 + k), 100.0, fn).x for row in seq])
    assert np.array_equal(fast, slow)

def test_fast_engine_equals_reference_without_double_after_split():
    rules = Rules(double_after_split=False, split_by_value=False)
    seq = sample_sequences(np.random.default_rng(9), 3000)
    fn = make_style("book", rules)
    slow = np.array([play_round(rules, [int(c) for c in row], 400.0, 100.0, fn).x for row in seq])
    assert np.array_equal(play_rounds(rules, seq, style_table("book", rules), 3), slow)

@pytest.mark.parametrize("first", ["hit", "stand", "double"])
def test_script_forces_the_first_action_then_follows_the_style(first):
    known = {0: parse_card("9S"), 1: parse_card("10H"), 2: parse_card("7D")}
    seq = sample_completions(np.random.default_rng(5), 3000, known)
    seq = seq[VALUES[seq[:, 3]] != 1]                      # the dealer checked: no blackjack
    book = make_style("book", R)
    def slow_one(row):
        r = Round(R, [int(c) for c in row], 400.0, 100.0)
        r.act(first)
        while r.phase != "done":
            r.act(book(r.view()))
        return r.result.x
    fast = play_rounds(R, seq, style_table("book", R), 3, script=[ACTIONS.index(first)])
    assert np.array_equal(fast, np.array([slow_one(row) for row in seq]))

def test_book_loses_a_little_per_chip():
    total, n = 0.0, 0
    for seed in range(5):
        x = play_rounds(R, sample_sequences(np.random.default_rng(seed), 400_000), style_table("book", R), 3)
        total, n = total + x.sum(), n + len(x)
    assert -0.009 < total / n < -0.002

def test_perfect_pairs_come_up_at_the_six_deck_rates():
    m = pp_multipliers(sample_sequences(np.random.default_rng(3), 400_000), R.pp_pays)
    for mult, chances in ((25, 5), (12, 6), (6, 12)):
        assert abs((m == mult).mean() - chances / 311) < 0.001
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fast.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.fast'`.

- [ ] **Step 3: Implement `fast.py`** as described under Algorithm.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fast.py`
Expected: PASS. If the equality tests fail, print the first differing row with both engines' card-by-card trace and fix the fast engine; the reference engine is the definition.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add NumPy engine that matches the reference engine"
```

---

### Task 5: Stand-in opponents

**Files:**
- Create: `trainer/opponents.py`
- Test: `tests/test_opponents.py`

**Interfaces:**
- Consumes: `Rules`
- Produces:
  - `opponents.OpponentType` (frozen dataclass): `label: str, style: str, rule: str, param: float, weight: float`; property `name` = `label` + `", plays by the book" | ", plays cautiously" | ", copies the dealer"`
  - `opponents.RULE_VARIANTS`: ordered `(label, rule, param)` list: `("Turtle","turtle",0)`, `("Flat 25","flat",25)`, `("Flat 50","flat",50)`, `("Flat 100","flat",100)`, `("Flat 200","flat",200)`, `("Flat 500","flat",500)`, `("Share 10%","share",0.10)`, `("Share 25%","share",0.25)`, `("Share 50%","share",0.50)`, `("All-in","all_in",0)`, `("Double up and sit","double_up_sit",0)`, `("Late lunge","late_lunge",0)`, `("Chaser 50","chaser",50)`, `("Chaser 100","chaser",100)`, `("Random","random",0)`
  - `opponents.DEFAULT_RULE_WEIGHTS`: label to weight, in the order above: 5, 7, 7, 15, 11, 8, 5, 8, 5, 8, 6, 5, 3, 3, 4
  - `opponents.DEFAULT_STYLE_WEIGHTS = {"book": 50, "cautious": 40, "dealer_copy": 10}`
  - `opponents.build_population(rule_weights=DEFAULT_RULE_WEIGHTS, style_weights=DEFAULT_STYLE_WEIGHTS) -> list[OpponentType]`: every rule crossed with every style, weight = product, labels or styles missing from the dicts count as 0, zero-weight entries dropped, weights normalised to sum 1
  - `opponents.legal_bet(rules, chips, wanted) -> float`: `wanted` rounded down to a bet step, raised to `min_bet`, capped at `rules.floor_bet(chips)`; 0.0 when `rules.floor_bet(chips) < min_bet`
  - `opponents.bets_for(opp, rules, round_no, chips, memory) -> list[tuple[float, float]]`: `(bet, probability)` pairs, equal bets merged, sorted by bet; `[(0.0, 1.0)]` when no bet is possible
  - `opponents.initial_memory(opp) -> float`, `opponents.next_memory(opp, memory, bet, x) -> float`

**Rules:** turtle bets `min_bet`. flat bets `param`. share bets `param * chips`. all_in bets everything. double_up_sit bets everything while `chips <= starting_chips`, else `min_bet`. late_lunge bets `min_bet` in rounds before the last and everything in the last. chaser bets `memory`, which starts at `param`, doubles after `x < 0`, resets to `param` after `x > 0`, and is unchanged after a push. random picks with probability 0.2 each from `min_bet`, 10%, 25%, 50% of chips, and everything. Every wanted amount goes through `legal_bet`. Only chaser uses memory; the others keep 0.0.

- [ ] **Step 1: Write the failing tests**

```python
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
    assert legal_bet(R, 1007.5, 5000.0) == 1005.0 and legal_bet(R, 1000.0, 1.0) == 5.0
    assert legal_bet(R, 2.5, 100.0) == 0.0

def test_simple_rules():
    assert only(bets_for(opp("turtle"), R, 1, 1000.0, 0.0)) == 5.0
    assert only(bets_for(opp("flat", 100), R, 3, 60.0, 0.0)) == 60.0
    assert only(bets_for(opp("share", 0.25), R, 1, 1000.0, 0.0)) == 250.0
    assert only(bets_for(opp("share", 0.10), R, 2, 1037.5, 0.0)) == 100.0
    assert only(bets_for(opp("all_in"), R, 2, 1007.5, 0.0)) == 1005.0

def test_double_up_and_sit():
    o = opp("double_up_sit")
    assert only(bets_for(o, R, 1, 1000.0, 0.0)) == 1000.0
    assert only(bets_for(o, R, 2, 1000.0, 0.0)) == 1000.0
    assert only(bets_for(o, R, 2, 2000.0, 0.0)) == 5.0

def test_late_lunge():
    o = opp("late_lunge")
    assert [only(bets_for(o, R, r, 990.0, 0.0)) for r in (1, 4, 5)] == [5.0, 5.0, 990.0]

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
    assert bets_for(opp("random"), R, 1, 1000.0, 0.0) == [(5.0, 0.2), (100.0, 0.2), (250.0, 0.2), (500.0, 0.2), (1000.0, 0.2)]
    short = bets_for(opp("random"), R, 1, 10.0, 0.0)
    assert [b for b, _ in short] == [5.0, 10.0] and abs(short[0][1] - 0.8) < 1e-12 and abs(short[1][1] - 0.2) < 1e-12

def test_out_of_chips_bets_nothing():
    assert bets_for(opp("flat", 100), R, 3, 2.5, 0.0) == [(0.0, 1.0)]

def test_default_population():
    pop = build_population()
    assert len(pop) == 45 and abs(sum(o.weight for o in pop) - 1.0) < 1e-12
    o = next(o for o in pop if o.label == "Flat 100" and o.style == "book")
    assert abs(o.weight - 0.15 * 0.50) < 1e-12 and o.name == "Flat 100, plays by the book"
    assert len(build_population(style_weights={"book": 1, "cautious": 0, "dealer_copy": 0})) == 15
    assert len(build_population(rule_weights={"Turtle": 1, "Flat 100": 3})) == 6
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_opponents.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.opponents'`.

- [ ] **Step 3: Implement `opponents.py`** from the Rules paragraph.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_opponents.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add stand-in opponents and their betting rules"
```

---

### Task 6: Outcome tables

**Files:**
- Create: `trainer/tables.py`
- Test: `tests/test_tables.py`

**Interfaces:**
- Consumes: `fast.sample_sequences`, `fast.play_rounds`, `styles.style_table`, `styles.STYLE_INDEX`, `sidebets.pp_multipliers`, `sidebets.fct_multipliers`
- Produces:
  - `tables.CLASS_VALUES = (-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0, 3.0)`, `tables.N_CLASSES = 8`
  - `tables.class_of(x) -> int | np.ndarray`: x ≤ -3 gives 0; -2, -1, 0, 1, 1.5, 2 give 1–6; x ≥ 3 gives 7
  - `tables.Tables` dataclass:
    - `p_class`: `(8,)` chance of each class
    - `vals`, `probs`: `(3, 4, 8, J)`: for style s, spare-chip level k (0–3 extra stakes) and class g, the distinct results per chip and their chances given the class; unused slots have chance 0
    - `side_vals`: `(4, 8, Js, 3)` and `side_probs`: `(4, 8, Js)`: for Book at level k and class g, the distinct triples (result per chip, Perfect Pairs multiplier, 4 Card Total multiplier) and their chances
  - `tables.build_tables(rules, n_rounds: int = 1_000_000, seed: int = 20261114) -> Tables`
  - `tables.save_tables(tables, path)`, `tables.load_tables(path) -> Tables` (`.npz`)

**How.** In chunks of at most 100,000 rounds: sample sequences, play each of the 3 styles at each of the 4 levels, take the class from Book at level 3, and count. The class of a round is always defined by Book at level 3, whatever style or level is being tabulated.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tables.py
import numpy as np, pytest
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

def test_same_seed_same_tables_and_round_trip(tmp_path, tables):
    again = build_tables(Rules(), n_rounds=200_000, seed=3)
    assert np.array_equal(again.probs, tables.probs)
    save_tables(tables, tmp_path / "t.npz")
    loaded = load_tables(tmp_path / "t.npz")
    assert np.array_equal(loaded.vals, tables.vals) and np.array_equal(loaded.side_probs, tables.side_probs)
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tables.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.tables'`.

- [ ] **Step 3: Implement `tables.py`.**

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tables.py`
Expected: PASS. Note the time `build_tables` takes for 200,000 rounds; a million rounds must stay under 90 seconds.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add simulated outcome tables"
```

---

### Task 7: Solver

**Files:**
- Create: `trainer/solver.py`
- Test: `tests/test_solver.py`

**Interfaces:**
- Consumes: `Rules`, `OpponentType`, `bets_for`, `initial_memory`, `next_memory`, `Tables`, `N_CLASSES`, `STYLE_INDEX`
- Produces:
  - `solver.chip_grid(coarse: bool = False) -> np.ndarray`. Fine: 0–2000 step 5, then to 5000 step 25, to 10000 step 100, to 40000 step 1000, plus 100000. Coarse: 0–900 step 25, 900–1100 step 5, 1100–3000 step 25, to 10000 step 100, to 40000 step 1000, plus 100000.
  - `solver.bet_menu(coarse: bool = False) -> np.ndarray`. Fine: 5–100 step 5, 125–1000 step 25, 1100–5000 step 100, 5500–20000 step 500, 22500–100000 step 2500. Coarse: 5, 10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 50000. All-in (`rules.floor_bet(chips)`) is always an extra candidate.
  - `solver.history_index(classes: Sequence[int]) -> int`: base-8 number, first round most significant
  - `solver.propagate(rules, opp, tables, history: Sequence[int]) -> dict[tuple[float, float], float]`: the opponent's `(chips, memory) -> probability` after the history
  - `solver.final_distribution(rules, population, tables, history4: Sequence[int], g5: int) -> tuple[np.ndarray, np.ndarray]`: the distinct final chip totals of the whole population in ascending order, and the weight on each (weights sum to 1)
  - `solver.credit(final_chips, final_weights, my_chips: np.ndarray, tie_credit: float) -> np.ndarray`: weight strictly below, plus `tie_credit` times the weight exactly equal
  - `solver.Solution` dataclass: `grid: np.ndarray`, `value: list[np.ndarray]`, `bet: list[np.ndarray]`; `value[r - 1]` has shape `(8 ** (r - 1), len(grid))`: the chance of finishing ahead at the start of round r for each history and chip total; `bet[r - 1]` is the best main bet (0.0 where no bet is possible)
  - `solver.solve(rules, population, tables, coarse: bool = False, progress=None) -> Solution`; `progress(fraction: float, message: str)`
  - `solver.save_solution(solution, path)`, `solver.load_solution(path) -> Solution`

**Algorithm.**

*Opponents forward.* An opponent's state is a dict `(chips, memory) -> probability`, starting at `(starting_chips, initial_memory)`. One round with class g: for each state and each `(bet, p)` from `bets_for`: a bet of 0 keeps the state; otherwise `k = min(3, int((chips - bet) // bet))`, and for each result v with chance q in `tables.vals/probs[style, k, g]` add `p * q` to `(chips + bet * v, next_memory(...))`. Drop states under 1e-7 and renormalise.

*The player backward.* For round r from 5 down to 1, for each history h and each grid chip total c: candidates are the menu bets not above `rules.floor_bet(c)`, plus all-in; with no legal bet the round is skipped (the value is the class-weighted next value at c). For a bet b: `k = min(3, int((c - b) // b))`, and

`Q(b) = sum over g of p_class[g] * sum over Book results (v, q) at level k, class g of q * Next(h, g, c + b * v)`

where `Next` is `value[r]` for history `h * 8 + g`, linearly interpolated on the grid (totals above the grid use the last point); in round 5 it is `credit` against `final_distribution` for `(h, g)`. `value[r - 1][h, c] = max Q`, and `bet` is the smallest bet within 1e-9 of the maximum.

Round 5 dominates the run time. Build each opponent's round-4 states for all 4,096 histories once, flatten them to arrays with their round-5 bet options and levels, and for each history compute the eight final distributions and all candidate bets with array operations (one `searchsorted` per class and Book result). Report progress per history.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_solver.py
import numpy as np, pytest
from trainer.opponents import OpponentType
from trainer.rules import Rules
from trainer.solver import chip_grid, credit, final_distribution, history_index, load_solution, propagate, save_solution, solve
from trainer.tables import build_tables

R = Rules()
one = lambda rule, param=0.0: [OpponentType("x", "book", rule, param, 1.0)]
start = lambda sol: float(np.interp(1000.0, sol.grid, sol.value[0][0]))

@pytest.fixture(scope="module")
def tables():
    return build_tables(R, n_rounds=200_000, seed=3)

@pytest.fixture(scope="module")
def vs_turtle(tables):
    return solve(R, one("turtle"), tables, coarse=True)

def test_history_index_and_grid():
    assert history_index([]) == 0 and history_index([2]) == 2 and history_index([1, 0, 7]) == 71
    for coarse in (False, True):
        g = chip_grid(coarse)
        assert g[0] == 0.0 and (np.diff(g) > 0).all() and 1000.0 in g and 1005.0 in g

def test_propagate_follows_the_betting_rule(tables):
    assert propagate(R, one("turtle")[0], tables, [2, 2, 4]) == {(995.0, 0.0): 1.0}
    assert propagate(R, one("chaser", 50)[0], tables, [2, 2]) == {(850.0, 200.0): 1.0}

def test_final_distribution_and_credit(tables):
    chips, weight = final_distribution(R, one("turtle"), tables, [2, 2, 2, 2], 2)
    assert chips.tolist() == [975.0] and abs(weight.sum() - 1.0) < 1e-12
    assert credit(chips, weight, np.array([970.0, 975.0, 980.0]), 0.5).tolist() == [0.0, 0.5, 1.0]

def test_beats_an_opponent_who_only_bets_the_minimum(vs_turtle):
    assert start(vs_turtle) >= 0.88

def test_beats_an_opponent_who_always_goes_all_in(tables):
    assert start(solve(R, one("all_in"), tables, coarse=True)) >= 0.90

def test_sitting_on_a_lead_is_certain(vs_turtle):
    h = history_index([2, 2, 2, 2])                       # the turtle is on 980
    assert np.interp(1000.0, vs_turtle.grid, vs_turtle.value[4][h]) >= 0.999
    assert np.interp(1000.0, vs_turtle.grid, vs_turtle.bet[4][h]) <= 10.0

def test_values_are_chances_and_a_huge_stack_cannot_be_caught(tables):
    sol = solve(R, one("flat", 100), tables, coarse=True)
    for v in sol.value:
        assert v.min() >= 0.0 and v.max() <= 1.0 + 1e-9
    assert sol.value[4][:, int(np.searchsorted(sol.grid, 20000.0))].min() >= 0.99
    assert [v.shape[0] for v in sol.value] == [1, 8, 64, 512, 4096]

def test_solution_round_trip(tmp_path, vs_turtle):
    save_solution(vs_turtle, tmp_path / "s.npz")
    again = load_solution(tmp_path / "s.npz")
    assert all(np.array_equal(a, b) for a, b in zip(again.value, vs_turtle.value))
    assert np.array_equal(again.grid, vs_turtle.grid)
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_solver.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.solver'`.

- [ ] **Step 3: Implement `solver.py`** as described under Algorithm.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_solver.py`
Expected: PASS, each coarse solve in well under a minute.

- [ ] **Step 5: Time the full solve**

Run: `.venv/Scripts/python.exe -c "import time; from trainer.rules import Rules; from trainer.opponents import build_population; from trainer.tables import build_tables; from trainer.solver import solve; import numpy as np; R=Rules(); t=time.time(); T=build_tables(R); print('tables', round(time.time()-t)); t=time.time(); s=solve(R, build_population(), T); print('solve', round(time.time()-t), float(np.interp(1000, s.grid, s.value[0][0])))"`
Expected: tables plus solve under 300 seconds in total, and a start value between 0.5 and 0.8. If the solve is over 240 seconds, split the round-5 histories across a `multiprocessing.Pool` before going on.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Add backward-induction solver"
```

---

### Task 8: Brain

**Files:**
- Create: `trainer/brain.py`, `tests/conftest.py`
- Test: `tests/test_brain.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `brain.BrainConfig` (frozen): `table_rounds: int = 1_000_000, samples: int = 40_000, coarse: bool = False, seed: int = 20261114`
  - `brain.History = list[dict[int, float]]`: one dict per past round, class to probability (a certain round is `{g: 1.0}`)
  - `brain.BetGrade`: `yours: tuple[float, float, float]`, `best: tuple[float, float, float]` (main, Perfect Pairs, 4 Card Total), `your_chance: float`, `best_chance: float`, `fine: bool`
  - `brain.PlaySpot`: `round_no: int, chips_start: float, main_bet: float, pp_bet: float, fct_bet: float, insurance_bet: float, player: list[int]` (the two dealt cards), `upcard: int, drawn: list[int]` (cards drawn since the deal, in table order), `script: list[str]` (play actions already taken this round), `legal: list[str], stage: str` (`"insurance" | "play"`)
  - `brain.PlayGrade`: `yours: str, best: str, chances: dict[str, float], your_chance: float, best_chance: float, fine: bool`
  - `brain.Brain(rules, population, tables, solution, config)`, keeping those five as attributes of the same names, with:
    - `Brain.prepare(rules, population, cache_dir, config=BrainConfig(), progress=None) -> Brain` (classmethod): loads `tables-<fp>.npz` and `solution-<fp>.npz` from `cache_dir`, building and saving whatever is missing; `fp` is 16 hex chars of a SHA-256 over the rules, the population and `table_rounds`, `coarse`, `seed`
    - `start_chance() -> float`
    - `round_class(shoe: Sequence[int]) -> int`: the class of a fully known round (Book, unlimited chips, reference engine)
    - `round_class_dist(known: dict[int, int]) -> dict[int, float]`: the class distribution of a round whose sequence is only partly known (positions to cards), by sampling 2,000 completions
    - `grade_bet(round_no, chips, history, main_bet, pp_bet=0.0, fct_bet=0.0) -> BetGrade`
    - `grade_play(spot: PlaySpot, history, chosen: str) -> PlayGrade`
    - `table_bet(round_no, chips, classes: Sequence[int]) -> float`: the solved table's bet at the nearest grid total not above `chips`, capped at `rules.floor_bet(chips)` (used by the benchmark)

**Scoring.** `Next(round_no, history, g, chips_array)` is the mean, over history combinations (products of the per-round class dicts, dropping combinations under 0.001 and renormalising), of the solver's `Next` from Task 7.

- `grade_bet`: for every main bet from `min_bet` to `rules.floor_bet(chips)` in steps of `bet_step`, with no side bets: `k = min(3, int((chips - b) // b))` and `Q(b)` as in Task 7. Best main bet: the smallest within 1e-9 of the maximum. The player's own choice is scored with `tables.side_vals/side_probs`: totals `chips + main * x + pp * m_pp + fct * m_fct`, level from the chips left after all three stakes. In round 5 only, also score side bets from the menu {0, 5, 10, 25, 50, 100, 200, 500, 1000} for each side, paired with main bets from {multiples of 25 up to 1000, then of 100, plus the largest affordable}, and let `best` include side bets when that beats the best main-only bet by more than 0.005. `fine = your_chance >= best_chance - 0.005`.
- `grade_play`: build `known = {0: player[0], 1: upcard, 2: player[1], 4 + i: drawn[i]}` and take `config.samples` completions. At stage `"play"` with an ace or ten-value up card, keep only rows whose down card does not make a dealer blackjack. For each legal choice: at stage `"play"`, `x = play_rounds(rules, seq, book_table, extra, script=codes(spot.script + [choice]))` with `extra = min(3, int((chips_start - main - pp - fct - insurance) // main))`; at stage `"insurance"`, the stake is `main / 2` for `"insurance"` and 0 otherwise, the insurance result per row is `+main` when the down card is ten-value and `-main / 2` otherwise (0 without insurance), and `x = play_rounds(...)` with an empty script and the level recomputed after the stake. Each row's class is `class_of(play_rounds(rules, seq, book_table, 3))`. Each row's final total is `chips_start + main * x + pp result + fct result + insurance result`, and the choice's chance is the mean of `Next` over rows, grouped by class. `fine` if the chosen action is within 0.005 of the best, or within twice the standard error of the paired difference. The sampler is seeded from `config.seed` and the spot, so the same spot always gives the same grade.

- [ ] **Step 1: Write the shared fixtures**

```python
# tests/conftest.py
from pathlib import Path
import pytest
from trainer.brain import Brain, BrainConfig
from trainer.opponents import OpponentType, build_population
from trainer.rules import Rules

QUICK = BrainConfig(table_rounds=200_000, samples=8_000, coarse=True)
QUICK_RULE_WEIGHTS = {"Turtle": 1, "Flat 100": 3, "Flat 500": 1, "Share 25%": 2, "All-in": 1}
QUICK_STYLE_WEIGHTS = {"book": 2, "cautious": 1, "dealer_copy": 0}

@pytest.fixture(scope="session")
def quick_brain(tmp_path_factory):
    pop = build_population(rule_weights=QUICK_RULE_WEIGHTS, style_weights=QUICK_STYLE_WEIGHTS)
    return Brain.prepare(Rules(), pop, tmp_path_factory.mktemp("quick"), QUICK)

@pytest.fixture(scope="session")
def turtle_brain(tmp_path_factory):
    pop = [OpponentType("Turtle", "book", "turtle", 0.0, 1.0)]
    return Brain.prepare(Rules(), pop, tmp_path_factory.mktemp("turtle"), QUICK)

@pytest.fixture(scope="session")
def full_brain():
    cache = Path(__file__).resolve().parent.parent / "data" / "cache"
    return Brain.prepare(Rules(), build_population(), cache, BrainConfig())
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_brain.py
from trainer.brain import Brain, BrainConfig, PlaySpot
from trainer.cards import parse_card
from trainer.rules import Rules

C = parse_card
WON4 = [{4: 1.0}] * 4          # four rounds that each paid +1: a turtle is on 1,020
LOST4 = [{2: 1.0}] * 4         # four rounds that each lost 1: a turtle is on 980

def spot(**kw):
    base = dict(round_no=5, chips_start=1000.0, main_bet=20.0, pp_bet=0.0, fct_bet=0.0, insurance_bet=0.0,
                player=[C("5S"), C("6D")], upcard=C("6H"), drawn=[], script=[],
                legal=["hit", "stand", "double"], stage="play")
    base.update(kw)
    return PlaySpot(**base)

def test_a_minimum_bet_cannot_catch_a_turtle(turtle_brain):
    g = turtle_brain.grade_bet(5, 1000.0, WON4, 5.0)
    assert g.your_chance == 0.0 and not g.fine and g.best[0] >= 30.0 and g.best_chance > 0.38

def test_sitting_tight_when_ahead_of_a_turtle(turtle_brain):
    assert turtle_brain.grade_bet(5, 1000.0, LOST4, 5.0).your_chance >= 0.999
    assert turtle_brain.grade_bet(5, 1000.0, LOST4, 10.0).fine
    assert not turtle_brain.grade_bet(5, 1000.0, LOST4, 200.0).fine

def test_own_side_bets_are_scored(turtle_brain):
    plain = turtle_brain.grade_bet(5, 1000.0, LOST4, 5.0)
    sided = turtle_brain.grade_bet(5, 1000.0, LOST4, 5.0, 100.0, 100.0)
    assert sided.yours == (5.0, 100.0, 100.0) and sided.your_chance < plain.your_chance and not sided.fine

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

def test_uncertain_history_is_averaged(turtle_brain):
    # on 985 with the same bet as the turtle: ahead for sure if it is on 980, behind for sure if it is on 990
    assert turtle_brain.grade_bet(5, 985.0, LOST4, 5.0).your_chance >= 0.999
    mixed = turtle_brain.grade_bet(5, 985.0, LOST4[:3] + [{2: 0.5, 4: 0.5}], 5.0).your_chance
    assert abs(mixed - 0.5) < 0.005

def test_prepare_reuses_its_cache(tmp_path):
    from trainer.opponents import OpponentType
    pop = [OpponentType("Turtle", "book", "turtle", 0.0, 1.0)]
    cfg = BrainConfig(table_rounds=50_000, samples=2_000, coarse=True)
    a = Brain.prepare(Rules(), pop, tmp_path, cfg)
    calls = []
    b = Brain.prepare(Rules(), pop, tmp_path, cfg, progress=lambda f, m: calls.append(m))
    assert a.start_chance() == b.start_chance() and len(list(tmp_path.glob("*.npz"))) == 2
    assert not any("solv" in m.lower() for m in calls)
```

- [ ] **Step 3: Run to see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.brain'`.

- [ ] **Step 4: Implement `brain.py`** as described under Scoring.

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Add the brain that grades bets and plays"
```

---

### Task 9: Benchmark and acceptance

**Files:**
- Create: `trainer/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `Brain`, `fast.sample_sequences`, `fast.play_rounds`, `opponents.*`, `tables.class_of`
- Produces:
  - `benchmark.PLAYERS = ("Brain", "Minimum only", "Flat 100", "Flat 500", "25% of chips", "All-in once, then minimum")`
  - `benchmark.run_benchmark(brain, matches: int = 20_000, seed: int = 1) -> list[dict]`: one row per player and opponent, plus one per player with `opponent == "All opponents"`; keys `player, opponent, ahead, level, credit, low, high` (`credit = ahead + tie_credit * level`; `low`/`high` a 95% range on `credit`)
  - `benchmark.main()`: `python -m trainer.benchmark [--matches N]` prints the "All opponents" table and writes every row to `data/benchmark.csv`

**How.** Every player uses Book play. Per round, sample `matches` sequences and play all 3 styles at all 4 levels once; each player and opponent then takes the result at their own level `min(3, int((chips - bet) // bet))`. Opponents bet by their rule (a Random opponent samples one option per match). The Brain player bets `brain.table_bet(round, chips, classes so far)`. The "All opponents" credit weights each opponent by its population weight.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_benchmark.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.benchmark'`.

- [ ] **Step 3: Implement `benchmark.py`.**

- [ ] **Step 4: Run the quick test, then the slow acceptance tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_benchmark.py` then `.venv/Scripts/python.exe -m pytest tests/test_benchmark.py -m slow`
Expected: PASS. The slow run builds and caches the full brain in `data/cache/`, which the app reuses. If the claim test fails by more than noise, the fault is in the solver's shortcuts (grid, menu, conditional tables): find which with the per-opponent rows before changing any threshold.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add benchmark report and brain acceptance tests"
```

---

### Task 10: Practice match

**Files:**
- Create: `trainer/practice.py`
- Test: `tests/test_practice.py`

**Interfaces:**
- Consumes: `Round`, `play_round`, `make_style`, `Brain`, `PlaySpot`, `opponents.*`, `cards.card_str`, `cards.new_shoe`
- Produces:
  - `practice.replay_opponent(rules, opp, shoes, rng) -> float`: the opponent's final chips over the given shoes (reference engine, its own style and betting rule; `rng` only for a Random opponent)
  - `practice.PracticeMatch(rules, brain, population, mode: str, seed: int, now=time.monotonic, shoes=None, opponent=None)`: `mode` is `"coach"` or `"match"`; `shoes` and `opponent` override the seeded choices (tests)
    - `state() -> dict`, `bet(main, pp=0.0, fct=0.0) -> dict`, `act(action: str) -> dict`, `next_round() -> dict`, `current_view() -> HandView`
    - every method returns the new state; a call that does not fit the phase raises `TrainerError` and changes nothing

**State** (JSON-ready; cards as `card_str`, a face-down card as `None`):

```text
mode, phase ("bet" | "insurance" | "play" | "round_over" | "match_over"), round, rounds,
chips, clock {left, limit, running}, bet {main, pp, fct, insurance} | None,
last_bet {main, pp, fct} | None, dealer [cards], dealer_total | None,
hands [{cards, total, soft, stake, active, outcome}], legal [actions],
result {main, pp, fct, insurance, net} | None, grades [grade] | None, summary | None

grade:   {kind ("bet" | "insurance" | "play"), hand, spot, yours, best, your_chance, best_chance, fine, reason}
summary: {you, opponent, opponent_name, outcome ("ahead" | "level" | "behind" | "forfeit"),
          start_chance, rounds [{round, bet, net, chips_after, grades}]}
```

**Behaviour.**

- The match holds five seeded shoes, one hidden opponent drawn by weight, and the classes of finished rounds (`brain.round_class`).
- Grades are computed from recorded spots when they are revealed: in Coach at `round_over`, in Match only at `match_over`. `grades` is `None` whenever they are not being shown.
- Bet grade text: `yours`/`best` like `"200"` or `"200 + 25 on Perfect Pairs"`; `spot` like `"Round 3, 1,150 chips"`. Play grade text: `yours`/`best` capitalised action names; `spot` like `"16 against 10"` or `"soft 18 against 9"`.
- `reason`: one plain sentence. Fine: "Good bet." / "Good play." Otherwise for a bet: "A smaller bet finishes ahead more often from here." or "A bigger bet finishes ahead more often from here."; for a play: "<Best> finishes ahead more often than <yours> here."
- Every round ends in `round_over`. `next_round()` then moves to the next round's `bet`, or to `match_over` after the last round or when the chips are below `min_bet`. The opponent's total comes from `replay_opponent`. Outcome: `ahead`, `level` or `behind` by final chips.
- Clock: starts at the first `state()` after construction. Coach stops it during `round_over`. In Match, any call made after `clock_seconds` of running time ends the match with outcome `forfeit`. In Coach the clock stops at 0 and nothing else happens.
- The down card and undealt cards never appear in any state before the round is over.

- [ ] **Step 1: Write the failing tests**

```python
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
```

`Brain` must expose `rules` and `population` attributes for these tests.

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_practice.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.practice'`.

- [ ] **Step 3: Implement `practice.py`** to the State and Behaviour sections.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_practice.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add the practice match"
```

---

### Task 11: Match review

**Files:**
- Create: `trainer/review.py`
- Test: `tests/test_review.py`

**Interfaces:**
- Consumes: `Round`, `Brain`, `PlaySpot`, `cards.parse_card`, `TrainerError`
- Produces:
  - `review.check_rounds(rules, rounds: list[dict]) -> list[dict]`: one entry per round given: `{"ok": bool, "error": str | None, "complete": bool, "net": float | None, "chips_before": float, "chips_after": float | None}`; never raises on bad input
  - `review.analyze_match(rules, brain, rounds) -> dict`: raises `TrainerError` naming the first round that is missing, unfinished or wrong
  - `review.save_review(data_dir, payload: dict, report: dict) -> Path`: writes `data_dir/reviews/<YYYYMMDD-HHMMSS>.json`

**A round as entered:**

```text
{"main": 200, "pp": 0, "fct": 0,
 "player": ["9", "7"], "suits": ["H", "C"] | absent,
 "upcard": "10", "insurance": false,
 "actions": [{"action": "hit", "card": "5"}, {"action": "stand"},
             {"action": "double", "card": "K"}, {"action": "split", "cards": ["3", "2"]}],
 "dealer": ["9", "4"]}          # the down card first, then each card the dealer drew
```

A split carries the second card of hand 1 and of hand 2; the following actions are hand 1's, then hand 2's.

**How.**

- Replay each round on the reference engine with a growing shoe: start with `[player1, upcard, player2, down card] + dealer draws`; before each action insert its card (for a split, both cards) at the engine's current position; then `act`. Suits default to spades; `suits` matter only for Perfect Pairs.
- A round is **complete** when the engine reaches `"done"` and every entered card was used. With no actions and no dealer cards yet it is incomplete but not wrong, unless the dealt cards already settle it.
- Errors, each starting `"Round N:"`: unknown card; bet not allowed; an action the engine refuses; the dealer needing a card that was not entered ("the dealer had 15 and must draw again"); dealer cards left over ("the dealer had 17 and would not draw"); a Perfect Pairs stake on a pair without suits; a missing down card when it is needed. The down card may be left out only when every player hand busted and there is no 4 Card Total stake; the replay then uses any card that does not make a dealer blackjack.
- The match must have `rules.rounds` rounds, or end at the round where the chips fall below `min_bet`.
- Analysis. History for round r: `brain.round_class_dist(known)` of each earlier round, where `known` maps every sequence position that was seen. For each round: `brain.grade_bet(...)`; then for the insurance choice (if offered) and every play decision a `PlaySpot` holding only what was visible at that moment, graded with `brain.grade_play`.

**Report:**

```text
{"rounds": [{"round", "chips_before", "chips_after", "net",
             "bet": {"yours", "best", "pp", "fct", "best_pp", "best_fct", "your_chance", "best_chance", "fine"},
             "plays": [{"hand", "spot", "yours", "best", "your_chance", "best_chance", "fine"}]}],
 "final_chips", "off_bets", "off_plays",
 "biggest_miss": {"round", "kind", "text", "cost"} | None}
```

`bet.yours`/`bet.best` are main bets; `plays[].yours`/`best` are lower-case action names; `spot` as in Task 10. `biggest_miss` is the decision with the largest `best_chance - your_chance` among those not fine; its `text` reads like "Round 4: you bet 100 where 400 was best." or "Round 3: you stood on 16 against 10 where hitting was best."

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_review.py
import json, pytest
from trainer.errors import TrainerError
from trainer.review import analyze_match, check_rounds, save_review
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
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_review.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.review'`.

- [ ] **Step 3: Implement `review.py`** as described under How and Report.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_review.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add match review"
```

---

### Task 12: Settings and server

**Files:**
- Create: `trainer/settings.py`, `trainer/server.py`, `trainer/__main__.py`, `trainer/web/index.html` (a placeholder page whose title and heading are "Match Trainer"; Task 13 replaces it)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `Rules`, `build_population`, `DEFAULT_RULE_WEIGHTS`, `DEFAULT_STYLE_WEIGHTS`, `Brain`, `BrainConfig`, `PracticeMatch`, `review.*`
- Produces:
  - `settings.Settings` dataclass: `rules: Rules, rule_weights: dict[str, float], style_weights: dict[str, float], default_mode: str = "coach"`; `to_dict()`, `from_dict(d)` (classmethod; unknown keys ignored; raises `TrainerError` unless `min_bet` is a positive multiple of `bet_step`, `tie_credit` is between 0 and 1, `clock_seconds` is between 30 and 3600, every weight is a number of 0 or more, at least one rule weight and one style weight are above 0, and `default_mode` is `"coach"` or `"match"`)
  - `settings.load_settings(data_dir) -> Settings` (defaults when `settings.json` is missing or unreadable), `settings.save_settings(data_dir, settings)`
  - `settings.population_from(settings) -> list[OpponentType]`
  - `server.App(data_dir: Path, config: BrainConfig = BrainConfig())`: `start_brain()` (background thread calling `Brain.prepare` into `data_dir/"cache"`), `set_brain(brain)` (tests), `handle(method: str, path: str, body: dict | None) -> tuple[int, dict]`
  - `server.make_server(app, port: int = 8721) -> ThreadingHTTPServer` on `127.0.0.1`
  - `server.main()`: loads settings from `<project>/data`, starts the brain, serves on 8721 (trying the next five ports if taken), opens the browser unless `TRAINER_NO_BROWSER` is set, and prints the address

**Routes** (JSON in and out; errors are `{"error": message}`):

| Method and path | Body | Answer |
| --- | --- | --- |
| GET `/api/status` | | `{"brain": {"ready", "progress", "message"}, "settings": {...}, "default_rule_weights", "default_style_weights"}` |
| POST `/api/practice/start` | `{"mode", "seed"?}` | practice state |
| GET `/api/practice/state` | | practice state, or 404 when no match is running |
| POST `/api/practice/bet` | `{"main", "pp"?, "fct"?}` | practice state |
| POST `/api/practice/action` | `{"action"}` | practice state |
| POST `/api/practice/next` | | practice state |
| POST `/api/review/check` | `{"rounds"}` | `{"rounds": check_rounds(...)}` |
| POST `/api/review/analyze` | `{"rounds", "opponent_final"?, "won"?}` | the report; the review is saved |
| POST `/api/settings` | settings dict | `{"settings", "resolving": bool}`; saves, and when rules or weights changed drops the brain and starts a new solve |

Status codes: 200; 400 for a `TrainerError` or a malformed body; 404 for an unknown route; 503 with "Still working out the best bets. This takes a few minutes the first time." for practice and analyze routes while the brain is not ready; 500 with "Something went wrong inside the trainer." for anything unexpected (the traceback goes to the console, never to the page). `/api/review/check` works without the brain. Static files: `/` serves `web/index.html`; `/static/<name>` serves files that exist directly inside `trainer/web/`. A single lock guards all state.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_server.py
import json, threading, urllib.request
import pytest
from trainer.server import App, make_server
from tests.test_review import MATCH

@pytest.fixture
def app(tmp_path, quick_brain):
    a = App(tmp_path)
    a.set_brain(quick_brain)
    return a

def test_status(app):
    code, body = app.handle("GET", "/api/status", None)
    assert code == 200 and body["brain"]["ready"] is True and body["settings"]["default_mode"] == "coach"
    assert body["settings"]["rules"]["min_bet"] == 5.0 and "Flat 100" in body["default_rule_weights"]

def test_practice_flow(app):
    assert app.handle("GET", "/api/practice/state", None)[0] == 404
    code, s = app.handle("POST", "/api/practice/start", {"mode": "match", "seed": 5})
    assert code == 200 and s["phase"] == "bet" and s["chips"] == 1000.0
    code, s = app.handle("POST", "/api/practice/bet", {"main": 100})
    assert code == 200 and s["phase"] in ("insurance", "play", "round_over")
    assert app.handle("GET", "/api/practice/state", None)[1]["round"] == 1

def test_a_wrong_move_is_a_400_with_a_plain_message(app):
    app.handle("POST", "/api/practice/start", {"mode": "coach", "seed": 5})
    code, body = app.handle("POST", "/api/practice/action", {"action": "hit"})
    assert code == 400 and "bet" in body["error"].lower()
    assert app.handle("POST", "/api/practice/bet", {"main": "lots"})[0] == 400
    assert app.handle("POST", "/api/practice/start", {"mode": "turbo"})[0] == 400
    assert app.handle("GET", "/api/practice/state", None)[1]["phase"] == "bet"

def test_a_brain_that_is_still_solving_says_so(tmp_path):
    a = App(tmp_path)
    for path, body in (("/api/practice/start", {"mode": "coach"}), ("/api/review/analyze", {"rounds": MATCH})):
        code, answer = a.handle("POST", path, body)
        assert code == 503 and "working out" in answer["error"].lower()
    assert a.handle("GET", "/api/status", None)[1]["brain"]["ready"] is False
    assert a.handle("POST", "/api/review/check", {"rounds": MATCH})[0] == 200

def test_review_check_and_analyze(app, tmp_path):
    code, body = app.handle("POST", "/api/review/check", {"rounds": MATCH})
    assert code == 200 and body["rounds"][4]["chips_after"] == 1700.0
    code, report = app.handle("POST", "/api/review/analyze", {"rounds": MATCH, "opponent_final": 1450, "won": True})
    assert code == 200 and report["final_chips"] == 1700.0 and len(list((tmp_path / "reviews").glob("*.json"))) == 1
    code, body = app.handle("POST", "/api/review/analyze", {"rounds": MATCH[:2]})
    assert code == 400 and "Round 3" in body["error"]

def test_settings_are_saved_and_a_rule_change_resolves(app, tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(app, "start_brain", lambda: started.append(True))
    current = app.handle("GET", "/api/status", None)[1]["settings"]
    code, body = app.handle("POST", "/api/settings", {**current, "default_mode": "match"})
    assert code == 200 and body["resolving"] is False and not started
    changed = {**current, "rules": {**current["rules"], "min_bet": 10.0}}
    code, body = app.handle("POST", "/api/settings", changed)
    assert code == 200 and body["resolving"] is True and started == [True]
    assert json.loads((tmp_path / "settings.json").read_text())["rules"]["min_bet"] == 10.0
    assert app.handle("POST", "/api/settings", {**current, "rules": {**current["rules"], "min_bet": -5}})[0] == 400

def test_unknown_routes_and_crashes(app, monkeypatch):
    assert app.handle("GET", "/api/nope", None)[0] == 404
    monkeypatch.setattr("trainer.server.check_rounds", lambda *a: 1 / 0)
    code, body = app.handle("POST", "/api/review/check", {"rounds": []})
    assert code == 500 and "Traceback" not in body["error"]

def test_pages_over_a_real_socket(app):
    server = make_server(app, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"Match Trainer" in urllib.request.urlopen(base + "/").read()
        assert json.loads(urllib.request.urlopen(base + "/api/status").read())["brain"]["ready"] is True
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(base + "/static/../server.py")
        assert err.value.code == 404
    finally:
        server.shutdown()
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_server.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.server'`.

- [ ] **Step 3: Implement `settings.py`, `server.py`, `__main__.py`** and the placeholder page.

- [ ] **Step 4: Run the tests, then the whole quick suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_server.py` then `.venv/Scripts/python.exe -m pytest`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add settings and the local server"
```

---

### Task 13: Practice screen

**Files:**
- Create: `trainer/web/style.css`, `trainer/web/app.js`, `trainer/web/practice.js`, `scripts/smoke_ui.py`
- Modify: `trainer/web/index.html` (replace the placeholder)

**Interfaces:**
- Consumes: the routes and practice state from Tasks 10 and 12
- Produces: `window.Trainer = {api(method, path, body), showError(message), fmtChips(n), fmtChance(p), cardHtml(card), onTab(name, fn)}` in `app.js`; `scripts/smoke_ui.py [practice|review|settings]`

**The page.**

- Header "Match Trainer" with three tabs: Practice, Review, Settings. `index.html` loads `/static/style.css`, `/static/app.js`, `/static/practice.js`, `/static/review.js`, `/static/settings.js`.
- While the brain is not ready, a banner over the Practice and Review tabs shows the status message and a progress bar, polling `/api/status` every second.
- Practice layout follows `docs/mockups/practice-round.html`: a status bar (round, clock, chips); a table area with dealer cards and player hands; under it the controls for the current phase.
  - Before a match: Coach / Match choice (preselected from settings) and Start.
  - `bet`: chip buttons 5, 10, 25, 100, 500, 1,000 that add to the selected target (main bet by default; clicking a side-bet box selects it); Clear, All in, Same bet (when `last_bet` exists and is affordable); Deal. The running main and side stakes are shown.
  - `insurance`: Insurance and No insurance buttons.
  - `play`: Hit, Stand, Double, Split; buttons not in `legal` are disabled. The active hand is marked.
  - `round_over`: result line, and in Coach the grades as rows (yours, best, both chances as percentages, the reason; a tick when fine); Next round.
  - `match_over`: both totals, the opponent's description, the outcome in words, a per-round list with every grade, and Play again.
- Keyboard: H hit, S stand, D double, P split, I insurance, N no insurance, Enter for Deal / Next round / Start, A all-in, C clear, R same bet, 1–6 for the chips 5, 10, 25, 100, 500, 1,000.
- The clock ticks down locally from the last state's `left` while `running`, and re-syncs on every answer. Under 60 seconds it turns red.
- Errors from the server show in a strip that fades after four seconds; the screen keeps its last good state.
- Chips print with thousands separators and one decimal only when needed ("1,007.5"). Chances print as whole percentages.
- Look: plain and its own, light and dark by system setting, felt-green table, white cards with red or black ranks, no logos. Usable at 400 px wide.
- Under any grades, one small line: "Best means finishing ahead most often against the stand-in opponents in Settings."
- Test hooks: opening `/#autoplay` starts a Match and plays it to the end using only the keyboard shortcuts (dispatched as key events): a bet of 100 built from the chip keys, then stand on 17 or more and hit otherwise, declining insurance. It then sets `document.title` to `"autoplay done"`. `/#autoplay-coach` does the same in Coach, pressing Enter for Next round each time. For the bet, the keys 1–6 add the six chip values.

`scripts/smoke_ui.py` starts an `App` on port 8799 with a temporary data folder and the quick brain settings from `tests/conftest.py`, runs Microsoft Edge headless (`--headless=new --disable-gpu --virtual-time-budget=60000 --dump-dom` and again with `--screenshot`), and checks the dumped page. It prints `SMOKE OK` or the reason, and where the screenshot is. If Edge is not at either standard install path it prints `SMOKE SKIPPED` and exits 0.

- [ ] **Step 1: Write `scripts/smoke_ui.py`** with the `practice` check: the dump for `/#autoplay` contains "autoplay done", "finished", both chip totals, and no element with class `error-strip visible`.

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/Scripts/python.exe scripts/smoke_ui.py practice`
Expected: a failure naming the missing text (the placeholder page has no practice screen).

- [ ] **Step 3: Build the page**: `index.html`, `style.css`, `app.js`, `practice.js`.

- [ ] **Step 4: Check the scripts parse and the smoke test passes**

Run: `node --check trainer/web/app.js && node --check trainer/web/practice.js && .venv/Scripts/python.exe scripts/smoke_ui.py practice`
Expected: `SMOKE OK` and a screenshot path. Open the screenshot and confirm the end-of-match screen is readable and nothing overlaps.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add the practice table screen"
```

---

### Task 14: Review and settings screens

**Files:**
- Create: `trainer/web/review.js`, `trainer/web/settings.js`
- Modify: `trainer/web/index.html`, `trainer/web/style.css`, `scripts/smoke_ui.py`

**Interfaces:**
- Consumes: `window.Trainer`, `/api/review/check`, `/api/review/analyze`, `/api/status`, `/api/settings`

**Review screen** (layout from `docs/mockups/match-review.html`):

- Five round tabs; a tab shows a tick when its round is complete and a warning mark when it has an error. Tabs after a bust-out are disabled.
- Step 1, start of the hand: main bet and two side-bet number fields (step 5); two player card slots; one dealer up-card slot. When there is a Perfect Pairs stake and the two cards share a rank, two suit pickers appear.
- Step 2, how it ended: an Insurance yes/no choice when the up card is an ace; action buttons that append to the action list (Hit, Stand, Double, Split), each asking for its card (Split asks for two); the list can be undone from the end; dealer slots for the down card and drawn cards.
- A rank picker (A, 2–10, J, Q, K) fills the selected slot and moves to the next one. Number keys, A, J, Q, K and T (for 10) work too.
- After every change the form posts all rounds to `/api/review/check` (debounced 300 ms) and shows each round's result, running chips and any error under the round.
- A field for the other player's final chips (optional) and a won / lost / not sure choice.
- **Analyze match** is disabled until every round the match needs is complete. It posts to `/api/review/analyze` and shows the report as a table (round, your bet, best bet, each play against the best play, both chances), a summary line with the counts, the biggest miss, and the same "Best means…" line as the practice screen. A "Start a new review" button clears the form.
- The form is kept in `localStorage` so a refresh does not lose it.
- Test hook: `/#review-demo` fills the five rounds from the mockup example (the `MATCH` list in `tests/test_review.py`), presses Analyze, and sets `document.title` to `"review done"` when the report is on screen.

**Settings screen:**

- Rules: minimum bet, how equal chips count (half a win / a loss / a win, stored as 0.5 / 0 / 1), double after split, split by value or by rank, insurance offered, clock length in seconds.
- Opponents: a number field for each of the 15 betting rules and each of the 3 styles, showing the defaults, with a Reset to defaults button. One sentence above them: "These are guesses about how real opponents bet. Every best bet depends on them."
- Default mode: Coach or Match.
- Save posts to `/api/settings`. When the answer says `resolving`, the solving banner appears.
- Test hook: `/#settings-demo` opens the tab and sets `document.title` to `"settings done"` once the fields are filled from `/api/status`.

- [ ] **Step 1: Extend `scripts/smoke_ui.py`** with `review` (the dump for `/#review-demo` contains "review done", "1,700" and "Biggest miss" or "No misses") and `settings` (contains "settings done" and "Flat 100").

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python.exe scripts/smoke_ui.py review`
Expected: a failure naming the missing text.

- [ ] **Step 3: Build `review.js` and `settings.js`** and their markup and styles.

- [ ] **Step 4: Check everything**

Run: `node --check trainer/web/review.js && node --check trainer/web/settings.js && .venv/Scripts/python.exe scripts/smoke_ui.py practice && .venv/Scripts/python.exe scripts/smoke_ui.py review && .venv/Scripts/python.exe scripts/smoke_ui.py settings`
Expected: three `SMOKE OK` lines. Open each screenshot and confirm the report table and the settings form are readable.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add the match review and settings screens"
```

---

### Task 15: Launcher, README and final check

**Files:**
- Create: `Start Trainer.bat`, `README.md`

**Interfaces:**
- Consumes: `python -m trainer`

`Start Trainer.bat`: changes to its own folder; if `.venv\Scripts\python.exe` is missing, prints "Setting up for the first time…", creates the environment with `py -3.13 -m venv .venv` and installs `requirements.txt`; on failure prints one plain line saying what to check (Python 3.13 installed, internet for the first setup) and pauses; then runs `.venv\Scripts\python.exe -m trainer`. The window stays open while the trainer runs; closing it stops the trainer.

`README.md`, in plain words and under 60 lines: what it is; how to start it (double-click `Start Trainer.bat`; the first start works out the best bets for a few minutes); the two screens; what "best" means and that it depends on the stand-in opponents in Settings; that it is for practice and for going back over finished matches, not for use during a live match; which rules are assumptions (point to Settings and to the design doc); how to run the benchmark (`.venv\Scripts\python.exe -m trainer.benchmark`).

- [ ] **Step 1: Write both files.**

- [ ] **Step 2: Run the launcher without a browser**

Run (PowerShell): `$env:TRAINER_NO_BROWSER = '1'; $p = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', '"Start Trainer.bat"' -PassThru -WindowStyle Hidden; Start-Sleep 8; (Invoke-WebRequest http://127.0.0.1:8721/api/status -UseBasicParsing).Content; Stop-Process -Id $p.Id -Force`
Expected: a JSON status with `"ready": true` (the full brain is already cached from Task 9), then the server stops. Make sure no `python.exe` from this project is left running.

- [ ] **Step 3: Run every test**

Run: `.venv/Scripts/python.exe -m pytest` then `.venv/Scripts/python.exe -m pytest -m slow`
Expected: all pass.

- [ ] **Step 4: Run the benchmark and the three smoke checks against the full brain**

Run: `.venv/Scripts/python.exe -m trainer.benchmark --matches 20000` and the three `scripts/smoke_ui.py` commands.
Expected: the Brain row has the highest "All opponents" credit, or is within one point of the highest; three `SMOKE OK` lines.

- [ ] **Step 5: Check the speed targets**

Run a short script that times `full_brain.grade_bet` in rounds 1 and 5, `grade_play` on a round-3 spot, and `analyze_match` on the example match.
Expected: under 0.5 s, 1 s and 15 s.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Add launcher and README"
```
