# Match Trainer

A practice table and match review for five-round, head-to-head blackjack: both
players start with 1,000 chips, get the same cards each round, and whoever has
more chips after round five wins. You never see the other player.

It runs on this computer only, in a browser tab. It does not connect to any
game or website.

## Starting it

On Mac, double-click **Start Trainer.command** in Finder. On Windows,
double-click **Start Trainer.bat**. A browser tab opens. Leave the terminal
window open while you use the trainer. Close it to stop.

The first start works out the best bets, which takes a minute or two. After
that it starts straight away, unless you change the rules or the opponents in
Settings.

## The two screens

- **Practice.** The trainer deals a five-round match on a six-minute clock
  against a hidden stand-in opponent. Each round you bet, then play. *Coach*
  grades your bet and your plays after every round. *Match* shows nothing until
  the end, like the real thing.
- **Review.** After a real match is over, enter what happened in all five
  rounds. The trainer checks that the chips add up, then shows the best bet and
  the best play for every round next to what you did.

In Review, click any card box to open a dropdown of card ranks. Suggested bets
are filled in automatically. Tick **Change amount** beside a bet to enter what
you actually bet, or untick it to use the suggestion again. Perfect Pairs and
4 Card Total have their own side panel, with **Recommended** or **Skip** labels.
Review fills in the suggested play as you enter cards. Use **Undo last move** if
you played differently. A completed hand advances to the next round.
You can also type a card's rank; **-** (next to **0**) enters a 10.
When you have blackjack, the dealer's second card stays hidden. If needed, record
a blackjack push or the 4 Card Total payout using the result fields instead.

Every match is saved automatically on this computer. Reviews are saved under
`data/reviews/` as you enter them, including unfinished reviews; analyzing updates
the same match file. Practice matches are saved under `data/matches/` after every
bet and action, including their cards, results, and final report. Starting a new
match keeps the previous file. These local records are excluded from GitHub.

The review is for going over a match. It takes the round
and then reports


## What "best" means

Best is never "wins the most chips". It is the bet or play that most often
leaves you **ahead of the other player** at the end.

That depends on how the other player bets, and nobody can see that. The trainer
assumes a mix of stand-in opponents: cautious ones, steady ones, big bettors,
all-in types, players who make common mistakes. The mix is a guess. You can
change it in **Settings**, and every best bet changes with it.

## Rules it assumes

The published rules are built in: six decks shuffled each round, dealer stands
on every 17, blackjack pays 3:2, insurance pays 2:1, Perfect Pairs and 4 Card
Total side bets, a six-minute clock.

Some rules are not published, so the trainer assumes them: minimum bet 10,
one split per round, double on the first two cards unless the first is an ace,
equal chips count as half a win, zero chips means you are out. Settings can change a few of these. The
full list is in `docs/superpowers/specs/2026-11-14-match-trainer-design.md`,
section 5.

## Checking the brain

To see how the trainer's betting does against each kind of stand-in opponent,
compared with simple rules like "always bet 100":

```
.venv\Scripts\python.exe -m trainer.benchmark
```

It prints a table and saves every row to `data\benchmark.csv`.

## For developers

Tests: `.venv\Scripts\python.exe -m pytest` (quick) and
`.venv\Scripts\python.exe -m pytest -m slow` (needs the full solve).
The design and the build plan are under `docs/superpowers/`.

## Comparing opponent models

Run `.venv/bin/python -m trainer.calibration` on Mac to compare the frozen models
against newer completed reviews. The per-match recommendations, hypothetical final
stacks and results are written to `data/calibration/comparison.json`. The comparison
uses the recorded plays and the cards available before each bet. A bet requiring
missing information or making a recorded play illegal is marked unscorable.

The local model file freezes the original settings and the revised opponent weights.
For initial setup on a fresh copy, use `--fit-through REVIEW_ID` with the ID of the
last review in the training cohort. Later runs reuse those weights. Training gives
each eligible match one fractional vote based on how well each betting style predicts
the opponent's final chips given the shared hands. The current comparison uses
positive opponent totals and outcomes consistent with the chip counts. A positive
forecast log gain means the revised model assigned more weight near the observed
opponent score. The live trainer continues to use its Settings.
