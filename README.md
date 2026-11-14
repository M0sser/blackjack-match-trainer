# Match Trainer

A practice table and match review for five-round, head-to-head blackjack: both
players start with 1,000 chips, get the same cards each round, and whoever has
more chips after round five wins. You never see the other player.

It runs on this computer only, in a browser tab. It does not connect to any
game or website.

## Starting it

Double-click **Start Trainer.bat**. A browser tab opens. Leave the black window
open while you use the trainer; close it to stop.

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

The review is for going back over a finished match. It takes the whole match
and then reports; it is not built to give answers hand by hand while a match
is being played, and using it that way in a paid contest would be unfair to
the person you are playing.

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

Some rules are not published, so the trainer assumes them: minimum bet 5,
one split per round, double on any first two cards, equal chips count as half
a win, zero chips means you are out. Settings can change a few of these. The
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
