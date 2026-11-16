# Blackjack Match Trainer: design

Date: 2026-11-14

Status: the user saw both screen mockups, then said "go" and "make". They have not
read this document. Everything marked **(my call)** was decided by Claude and
is open to correction.

## 1. What this is and why

A practice and review tool for five-round, head-to-head blackjack matches.
Both players
start with 1,000 chips and play five rounds against a dealer. Whoever has
more chips after round five wins. The other player's bets, plays and chips
are never visible.

The user plays this format. They want to know the best bet and the best play in
every spot, and to get fast enough to play a real match from memory inside
its six-minute clock. Their words: "optimized enough that it can do all rounds
as fast as possible", "close to the real experience", "put in the cards for
each match … and from there calculate the best play", "the simulator should
be calculating optimal bet size as well".

One fact shapes everything. The program gives both players the same shuffle,
the same dealer cards (hole card included), and the same player cards for as
long as they play alike. So two players mostly win and lose the same rounds,
and the match is decided by how much each had riding on them. "Best" in this
tool therefore always means "finishes ahead of the other player most often".
It never means "wins the most chips".

## 2. Goals, non-goals, anti-goals

### Goals

1. **Practice table.** The tool deals five-round matches by the real rules,
   on a six-minute clock, against a hidden stand-in opponent. Each round has
   two inputs: the bet, then the play. The user approved this flow (mockup:
   `docs/mockups/practice-round.html`).
2. **Match review.** After a real match is over, the player enters what happened in
   all five rounds and gets a report: best bet and best play for every
   round, next to what they did (mockup: `docs/mockups/match-review.html`).
3. **One brain** behind both, grading bets and plays by the chance of
   finishing ahead.
4. **Fast and familiar.** Same chip values, same order of steps, same clock
   pressure as the real game. Nothing slow between rounds.
5. **Honest numbers.** Every "best" is best against assumed opponents, and
   the tool says so.

### Non-goals for this version

- Tuning the stand-in opponents from the user's logged matches. The review saves
  the data, so this can be added later.
- The research dashboard, Streamlit interface and open-ended policy search
  from the original plan
  (`21_rivals_tournament_optimizer_plan.md`, in the Downloads folder). A small
  benchmark report replaces them (section 6.9).
- Re-splitting, surrender, and a one-page betting plan to memorise.

### Anti-goals

- **False certainty.** No bet is presented as simply correct. It is the best
  against the stand-in opponents described in section 6.4.

## 3. Constraints

- Windows 11. Python 3.13 is installed and starts with `py -3.13`. Plain
  `python` is not on the PATH.
- Works offline once set up. The only add-on is NumPy (pytest for tests).
- The user is not a programmer. They start the tool by double-clicking a file, and
  anything that goes wrong is explained in plain words.
- Several rules of the real game are unpublished. Each is a setting with a
  default (section 5).

## 4. Approaches considered

### 4.1 How the brain finds the best bet and play

**A. Search among simple rules** (the original plan's Phase 4). Simulate
fixed, proportional and round-aware betting rules and keep the winner.
Simple to build and gives a rule a person can remember. But the best bet it
can name is only the best among the rules tried, and it cannot grade an
arbitrary spot.

**B. Play out each option on demand.** At each decision, try every option and
simulate the rest of the match thousands of times. Flexible, and nothing to
precompute. But the answer depends on what it assumes you do in later
rounds, and in this game that matters enormously. Example: against an
opponent who always bets the minimum, looking one round ahead says "copy
them and tie". The real best plan raises the stake after each miss until one
round wins, and finishes ahead roughly nine times in ten.

**C. Solve backwards from round five, then look up.** Work out the best
round-five bet for every situation, then the best round-four bet knowing
round five will be played well, and so on back to round one. This captures
when to chase and when to sit tight, the answers are instant once solved,
and the result can be checked against plain simulation.

**Chosen: C (my call).** The user talks about outcomes, not methods, so the
method was left to Claude. C is the only one that makes "best bet" mean what
they asked for. Its cost is a one-off solve of a few minutes whenever the
rules or opponents change.

### 4.2 How it runs and looks

**A. Streamlit** (suggested in the original plan). Quick for charts, awkward
for a ticking clock and a table that reacts instantly.

**B. A local web page with a small built-in server.** Python does the
thinking; a plain web page in the browser is the table. Needs no extra
installs beyond NumPy.

**C. A desktop window (tkinter).** No browser, but slow to make look and
feel right.

**Chosen: B (my call).** The user has already used a browser tab for the mockups,
and a web page handles the clock and quick clicks best.

## 5. Rules

"Published" means stated in The program' launch post (section 9). "User" means
the user confirmed it. Everything else is an assumption **(my call)**. In this
version Settings can change: the minimum bet, how equal chips count, double
after split, splitting by value or by rank, whether insurance is offered,
and the clock length. The other assumptions are fixed in the code and
listed here so they are easy to find and change later.

| Rule | Default | Source |
| --- | --- | --- |
| Starting chips | 1,000 | User |
| Rounds | 5 | Published |
| Shoe | Six decks, shuffled fresh for every round | Published ("the card shuffle for each round") |
| Shared cards | Both players get the same card order each round | Published |
| Deal order | Player, dealer up card, player, dealer down card | Assumption |
| Order after the deal | One sequence: the player's extra cards come first, then the dealer's | Assumption |
| Dealer | Stands on every 17, soft 17 included | Published |
| Dealer checks for blackjack | Yes, with an ace or ten showing, before the player acts | Assumption |
| Blackjack | Pays 3:2; a tie with dealer blackjack is a push | Published |
| Insurance | Offered when the dealer shows an ace; costs half the main bet; pays 2:1 | Payout published, the rest assumed |
| Surrender | Not offered | Not mentioned anywhere |
| Double | First two cards, including after a split, unless the first card is an ace | First-ace restriction confirmed by the user |
| Split | Two cards of equal value, once per round; split aces get one card each; 21 after a split is not blackjack | Assumption |
| Main bet | Multiples of 5, minimum 10, maximum all chips | All-in from the user; minimum assumed |
| Betting every round | Required while you have chips | Assumption |
| Perfect Pairs | Same rank and suit 25:1, same rank and colour 12:1, same rank 6:1 | Published |
| 4 Card Total | Your two cards plus both dealer cards, ace = 11: 36–37 pays 2:1, 38–39 pays 8:1, 40+ pays 10:1 | Published |
| Side bet stakes | Multiples of 5; main bet plus side bets cannot exceed your chips | Assumption |
| Doubling and splitting | Each costs one more main bet and needs the chips to cover it | Assumption |
| Zero chips | You are out; remaining rounds are skipped | Assumption |
| Winner | More chips after round five | Published |
| Equal chips | Counts as half a win | Assumption |
| Clock | Six minutes for all five rounds; running out forfeits | Published |

Chips can end in .5 (blackjack on a bet of 5 pays 7.5). All chip amounts are
multiples of 2.5 and must be handled exactly.

## 6. Design

### 6.1 The parts

1. **Rules engine**: cards, shoe, hands, dealer, payouts, side bets.
2. **Play styles**: fixed ways of playing a hand (section 6.3).
3. **Stand-in opponents**: a play style plus a betting rule (section 6.4).
4. **The brain**: solves the match backwards and grades decisions
   (section 6.5).
5. **Practice table** and **match review**: the two screens (6.6, 6.7).
6. **Settings**, a **benchmark report**, and a **launcher** (6.8 to 6.10).

### 6.2 Two engines that must agree

- A **reference engine**: clear, step-by-step code that plays one round from
  a given card order. The practice table and the review use it. It enforces
  chips for doubles and splits.
- A **fast engine**: plays hundreds of thousands of rounds at once with
  NumPy, for fixed play styles. The brain uses it.

The fast engine must give exactly the same result as the reference engine
for the same cards, style and available chips. A test checks this on
thousands of random rounds. This is the main guard against a wrong brain.

### 6.3 Play styles

- **Book**: standard basic strategy for six decks, dealer standing on soft
  17, double after split allowed, no surrender. Never takes insurance. When a
  double or split is not affordable or not allowed, it falls back to the
  best hit-or-stand choice (soft 18 stands, other would-be doubles hit, a
  pair is played as its total).
- **Cautious**: a common casual player. Never risks busting (stands on any
  hard 12 or more), doubles only on 10 or 11, splits only aces and eights.
- **Dealer-copy**: hits until 17, never doubles or splits.

### 6.4 Stand-in opponents

Each opponent is a play style plus a betting rule. Betting rules:

| Rule | What it does | Default weight |
| --- | --- | --- |
| Turtle | Minimum bet every round | 5 |
| Flat 25 / 50 / 100 / 200 / 500 | The same bet every round, or all chips if fewer | 7 / 7 / 15 / 11 / 8 |
| Share 10% / 25% / 50% | That share of current chips, rounded down to 5 | 5 / 8 / 5 |
| All-in | Everything, every round | 8 |
| Double up and sit | All-in until ahead of 1,000, then minimum bets | 6 |
| Late lunge | Minimum bets, then all-in on round five | 5 |
| Chaser 50 / 100 | Doubles the bet after a loss, resets after a win | 3 / 3 |
| Random | A random pick each round from minimum, 10%, 25%, 50%, all-in | 4 |
| Mixed | A random pick each round from 10%, 18%, 22.5%, 37.5%, 60% of chips | (added 2026-11-15) |

**Update 2026-11-15:** the default rule weights are no longer the numbers above. They are counts from
28 real finished matches (match-stats screenshots), plus the numbers above counted as 10 matches.
Flat 250, Flat 300 and Mixed were added because many real opponents bet that way. The
counts and the sum are in `trainer/opponents.py`. Play styles cannot be seen in match stats,
so the style weights are unchanged. The minimum bet is 10, not 5.

Style weights: Book 50, Cautious 40, Dealer-copy 10. The default population
is every betting rule crossed with every style, weighted by the product.
These weights are Claude's guess **(my call)**. Nobody has data on how real
opponents bet, so the weights are adjustable in Settings and the benchmark
report shows results against each type separately. Stand-in opponents do
not place side bets in this version.

### 6.5 The brain

**Result class.** For any round's cards, take the result per chip bet that
Book play gets with unlimited chips, and sort it into one of eight classes:
−3 or worse, −2, −1, 0, +1, +1.5, +2, +3 or better. A match history is the
list of classes so far. The history is what both players have lived
through, so it tells the brain how each kind of opponent is likely doing.

**Tables from simulation.** The fast engine plays a large number of random
rounds once and records:

- how often each class comes up;
- for each play style and each level of spare chips (enough for 0, 1, 2, or
  3+ extra bets), the spread of results per chip within each class;
- the same, jointly with the two side-bet payouts, for Book play.

**Opponents forward.** For every history and every opponent type, work out
the spread of that opponent's chips, round by round, from its betting rule
and the tables. After five rounds this gives, for every full history, how
likely the other player is to hold fewer chips than any given total, or
exactly that total.

**The player's bets backward.** For round five, for every history and every chip
total on a grid, try a menu of bets and keep the one with the best chance of
finishing ahead (equal chips count half). Then do round four using the
round-five answers, and so on back to round one. The player is assumed to play
Book in later rounds. The result is a table: best chance and best bet for
every round, history and chip total. It is saved to disk and reused until
the rules or opponents change.

**Grading a bet.** At a bet decision the tool knows the round, the player's chips
and the cards of earlier rounds. It scores every legal bet in steps of 5
with one exact step forward into the solved table, then reports the best
bet, the player's bet, and each one's chance of finishing ahead. Side bets are
scored the same way. The tool recommends a side bet only in round five,
where the scoring is exact; in earlier rounds it shows what the player's side bet
cost or gained.

**Grading a play.** At a hit, stand, double, split or insurance decision the
tool deals out the unseen cards many thousands of times. For each deal it
finds the player's result under each legal choice (followed by Book play) and the
round's result class, then reads the chance of finishing ahead from the
solved table. It reports the best choice and the player's choice.

**Close calls.** If the player's choice is within half a percentage point of the
best, or within the noise of the estimate, it counts as fine. The tool does
not nag over coin flips.

**Unseen cards in a review.** When Book play would have used a card the player
never saw, the tool averages over what that card could have been.

**Speed targets.** First solve: under five minutes on the user's PC, with
progress shown. After that: a bet grade in under half a second, a play grade
in under a second, a full review report in under fifteen seconds.

### 6.6 Practice table

- **Start**: pick Coach or Match and press Start. The tool shuffles five
  shoes and secretly picks one stand-in opponent by weight.
- **Bet**: chips 5, 10, 25, 100, 500, 1,000 add to the bet. Clear, All in and
  Same bet buttons. Two side-bet boxes. Deal.
- **Play**: the player's cards, the dealer's up card and face-down card. Hit,
  Stand, Double, Split, with unavailable buttons greyed out. Insurance is
  offered when the dealer shows an ace. Keyboard shortcuts for every button.
- **After the hand**: the dealer's cards, the result, the new chip total.
- **Coach** shows the grade for the bet and for each play right after the
  hand, with one line of reason, and stops the clock while it is on screen.
  **Match** shows nothing until the end, like a real match.
- **Clock**: counts down from 6:00 across all five rounds. In Match, hitting
  0:00 forfeits.
- **End**: the player's total, the opponent's total and what kind of opponent it
  was, who finished ahead, and every grade.

### 6.7 Match review

- Five round tabs. Each round is entered in two steps.
  - **Start of the hand**: main bet, side bets, the player's two cards, the
    dealer's up card. Suits are asked for only when they matter (a Perfect
    Pairs stake on a pair).
  - **How it ended**: what the player did in order, the card each hit, double or
    split brought, whether they took insurance, the dealer's face-down card
    and any cards the dealer drew.
- Cards are entered by tapping a rank. The form works out each round's
  result and the running chip total, so a typo shows up as a total that
  does not match what the player remembers.
- Rounds after a bust-out are skipped.
- An optional box records the other player's final chips, if the game shows
  them, and whether the player won.
- **Analyze match** 
  The report then lists, per round, the player's bet against the best bet and each
  play against the best play, with the chance of finishing ahead for both,
  followed by a short summary and the biggest miss.
- Each analyzed match is saved under `data/reviews/`.

### 6.8 Settings

The adjustable rules named in section 5, the opponent weights from section
6.4, and the Coach/Match default. Changing rules or opponents re-solves the
brain in the background and says so.

### 6.9 Benchmark report

A command that plays a large number of full matches, card by card, and
prints and saves a table: how often the brain's betting finishes ahead,
compared with simple rules (minimum only, flat 100, flat 500, 25% of chips,
all-in once then minimum), against each opponent type and against the whole
mix, with 95% ranges. This is what remains of the original plan's research
phases, and it doubles as the brain's acceptance test.

### 6.10 Storage and starting it

- `data/` holds settings, the solved tables and saved reviews.
- `Start Trainer.bat` starts the server on this PC only (127.0.0.1) and
  opens the browser. Setup (creating the Python environment and installing
  NumPy) is done once as part of the build.
- The tool makes no network requests.

## 7. How we know it is right

- Every payout in section 5 is tested against a hand-worked example,
  including blackjack, doubles, splits, insurance and both side bets.
- The same seed always gives the same shuffle and the same results.
- The fast engine equals the reference engine on thousands of random rounds,
  for every style and chip level.
- Book play loses between 0.2% and 0.9% per chip bet over a large flat-bet
  run, as these rules should.
- Perfect Pairs comes up at the exact rates for six decks: 5, 6 and 12
  chances in 311.
- Against Turtle-only opponents the brain finishes ahead at least 88% of the
  time. Against All-in-only opponents, at least 90%.
- Played out card by card, the brain's betting finishes ahead about as often
  as its own table predicts (within two percentage points), and at least as
  often as every simple rule in the benchmark, allowing for noise.
- The practice table can be played through a full match by keyboard alone,
  and a match entered in the review produces the report in the mockup's
  shape.

## 8. Left to the builder

File layout, the server's messages, grid sizes, bet menus and sample sizes
(set by the speed targets and the checks in section 7), the exact wording
of grade reasons, and the visual design within "plain and its own".

## 9. Sources
- The user's original plan:
  `21_rivals_tournament_optimizer_plan.md`, in the Downloads folder
- The two mockups the user saw: `docs/mockups/practice-round.html` and
  `docs/mockups/match-review.html` (page fragments from the brainstorm
  browser tab; they need its frame styles to render).
