"""The reference engine: one round of blackjack, played step by step from a given card order.

This file is the definition of the rules. The fast engine in fast.py must give the
same result for the same cards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from trainer.cards import hand_total, rank_of, value_of
from trainer.errors import TrainerError
from trainer.rules import Rules
from trainer.sidebets import fct_multiplier, pp_multiplier

PLAY_ACTIONS = ("hit", "stand", "double", "split")


@dataclass(frozen=True)
class HandView:
    """What a play style needs to know to choose an action."""
    total: int
    soft: bool
    pair_value: int      # card value (ace 1) when the hand is an unsplit pair, else 0
    upcard: int          # dealer's up card value (ace 1)
    can_double: bool
    can_split: bool


@dataclass
class Hand:
    cards: list[int]
    stake: float
    doubled: bool = False
    from_split: bool = False
    done: bool = False
    outcome: str | None = None   # "win", "lose", "push", "blackjack" or "bust"


@dataclass
class RoundResult:
    main_net: float
    pp_net: float
    fct_net: float
    insurance_net: float
    net: float
    chips_after: float
    x: float                     # main_net per chip of the main bet
    cards_used: int
    dealer_cards: list[int]
    hands: list[Hand]


def _amount(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TrainerError("A bet has to be a number.") from None
    if not math.isfinite(number):
        raise TrainerError("A bet has to be a number.")
    return number


def _chips_text(amount: float) -> str:
    return f"{amount:,.1f}".rstrip("0").rstrip(".")


def _in_steps(amount: float, step: float) -> bool:
    return abs(amount / step - round(amount / step)) < 1e-9


def validate_bet(rules: Rules, chips: float, main_bet, pp_bet=0.0, fct_bet=0.0) -> None:
    """Raise TrainerError unless the bets are allowed with this many chips."""
    main_bet, pp_bet, fct_bet = _amount(main_bet), _amount(pp_bet), _amount(fct_bet)
    step = _chips_text(rules.bet_step)
    if main_bet < rules.min_bet:
        raise TrainerError(f"The smallest bet is {_chips_text(rules.min_bet)}.")
    if not _in_steps(main_bet, rules.bet_step):
        raise TrainerError(f"Bets go in steps of {step}.")
    for name, amount in (("Perfect Pairs", pp_bet), ("4 Card Total", fct_bet)):
        if amount < 0 or not _in_steps(amount, rules.bet_step):
            raise TrainerError(f"The {name} bet has to be 0 or a multiple of {step}.")
    if main_bet + pp_bet + fct_bet > chips + 1e-9:
        raise TrainerError(f"That is more than your {_chips_text(chips)} chips.")


class Round:
    """One round. `shoe` is read one card at a time and may be a list that grows while
    the round is live; reading past its end raises IndexError."""

    def __init__(self, rules: Rules, shoe: Sequence[int], chips: float, main_bet: float,
                 pp_bet: float = 0.0, fct_bet: float = 0.0):
        validate_bet(rules, chips, main_bet, pp_bet, fct_bet)
        self.rules = rules
        self.shoe = shoe
        self.main_bet = float(main_bet)
        self.pp_bet = float(pp_bet)
        self.fct_bet = float(fct_bet)
        self.insurance_bet = 0.0
        self.actions: list[str] = []
        self.result: RoundResult | None = None
        self._start_chips = float(chips)
        self.chips = self._start_chips - self.main_bet - self.pp_bet - self.fct_bet

        first, up, second, hole = (int(shoe[i]) for i in range(4))
        self.pos = 4
        self.upcard = up
        self._dealt = (first, second)
        self._dealer = [up, hole]
        self._insurance_net = 0.0
        self._split_aces = False
        self.hands = [Hand(cards=[first, second], stake=self.main_bet)]
        self.active = 0

        if value_of(up) == 1 and rules.insurance_offered:
            self.phase = "insurance"
        else:
            self.phase = "play"
            self._check_for_blackjacks()

    # ---- what the player may do -------------------------------------------------

    def legal_actions(self) -> list[str]:
        if self.phase == "insurance":
            if self.chips >= self.main_bet / 2:
                return ["insurance", "no_insurance"]
            return ["no_insurance"]
        if self.phase != "play":
            return []
        hand = self.hands[self.active]
        actions = ["hit", "stand"]
        if self._can_double(hand):
            actions.append("double")
        if self._can_split(hand):
            actions.append("split")
        return actions

    def view(self) -> HandView:
        hand = self.hands[self.active]
        total, soft = hand_total(hand.cards)
        unsplit_pair = len(self.hands) == 1 and len(hand.cards) == 2 and self._is_pair(hand.cards)
        return HandView(
            total=total,
            soft=soft,
            pair_value=value_of(hand.cards[0]) if unsplit_pair else 0,
            upcard=value_of(self.upcard),
            can_double=self._can_double(hand),
            can_split=self._can_split(hand),
        )

    def visible_dealer(self) -> list[int | None]:
        """The dealer's cards as the player sees them: the down card is hidden until the round ends."""
        if self.phase != "done":
            return [self.upcard, None]
        return list(self._dealer)

    def act(self, action: str) -> None:
        if action not in self.legal_actions():
            raise TrainerError(self._refusal(action))

        if self.phase == "insurance":
            if action == "insurance":
                self.insurance_bet = self.main_bet / 2
                self.chips -= self.insurance_bet
            self.phase = "play"
            self._check_for_blackjacks()
            return

        hand = self.hands[self.active]
        if action == "stand":
            hand.done = True
        elif action == "hit":
            hand.cards.append(self._draw())
        elif action == "double":
            card = self._draw()
            self.chips -= hand.stake
            hand.stake *= 2
            hand.doubled = True
            hand.cards.append(card)
            hand.done = True
        else:  # split
            first, second = hand.cards
            card = self._draw()
            self.chips -= self.main_bet
            self._split_aces = value_of(first) == 1
            self.hands = [
                Hand(cards=[first, card], stake=self.main_bet, from_split=True, done=self._split_aces),
                Hand(cards=[second], stake=self.main_bet, from_split=True),
            ]
        self.actions.append(action)
        self._advance()

    # ---- internals --------------------------------------------------------------

    def _is_pair(self, cards: list[int]) -> bool:
        if self.rules.split_by_value:
            return value_of(cards[0]) == value_of(cards[1])
        return rank_of(cards[0]) == rank_of(cards[1])

    def _can_double(self, hand: Hand) -> bool:
        if len(hand.cards) != 2 or self.chips < hand.stake:
            return False
        if hand.from_split and (self._split_aces or not self.rules.double_after_split):
            return False
        return True

    def _can_split(self, hand: Hand) -> bool:
        return (len(self.hands) == 1 and len(hand.cards) == 2 and self._is_pair(hand.cards)
                and self.chips >= self.main_bet)

    def _refusal(self, action) -> str:
        if self.phase == "done":
            return "This hand is already over."
        if self.phase == "insurance":
            if action == "insurance":
                return "You don't have the chips for insurance."
            return "First choose: insurance or no insurance."
        if action == "double":
            return "You can only double on your first two cards, with the chips to cover it."
        if action == "split":
            return "You can only split two matching cards, once, with the chips to cover it."
        if action in ("insurance", "no_insurance"):
            return "Insurance is only offered when the dealer shows an ace."
        return f'"{action}" is not a move. Use hit, stand, double or split.'

    def _draw(self) -> int:
        card = int(self.shoe[self.pos])
        self.pos += 1
        return card

    def _check_for_blackjacks(self) -> None:
        """The dealer checks the down card; a blackjack on either side ends the round here."""
        hand = self.hands[0]
        dealer_blackjack = hand_total(self._dealer)[0] == 21
        player_blackjack = hand_total(hand.cards)[0] == 21
        if dealer_blackjack:
            self._insurance_net = 2 * self.insurance_bet
            hand.outcome = "push" if player_blackjack else "lose"
        else:
            self._insurance_net = -self.insurance_bet
            if player_blackjack:
                hand.outcome = "blackjack"
        if hand.outcome:
            hand.done = True
            self._finish(dealer_plays=False)

    def _advance(self) -> None:
        """Finish hands that need no decision, move to the next hand, or end the round."""
        while True:
            hand = self.hands[self.active]
            if not hand.done and hand_total(hand.cards)[0] >= 21:
                hand.done = True
            if not hand.done:
                return
            if self.active + 1 == len(self.hands):
                self._finish(dealer_plays=True)
                return
            self.active += 1
            following = self.hands[self.active]
            following.cards.append(self._draw())
            if self._split_aces:
                following.done = True

    def _finish(self, dealer_plays: bool) -> None:
        if dealer_plays and any(hand_total(h.cards)[0] <= 21 for h in self.hands):
            while hand_total(self._dealer)[0] < 17:
                self._dealer.append(self._draw())
        dealer_total = hand_total(self._dealer)[0]

        main_net = 0.0
        for hand in self.hands:
            if hand.outcome is None:
                total = hand_total(hand.cards)[0]
                if total > 21:
                    hand.outcome = "bust"
                elif dealer_total > 21 or total > dealer_total:
                    hand.outcome = "win"
                elif total < dealer_total:
                    hand.outcome = "lose"
                else:
                    hand.outcome = "push"
            if hand.outcome == "blackjack":
                main_net += hand.stake * self.rules.blackjack_pays
            elif hand.outcome == "win":
                main_net += hand.stake
            elif hand.outcome in ("lose", "bust"):
                main_net -= hand.stake

        first, second = self._dealt
        up, hole = self._dealer[0], self._dealer[1]
        pp_net = self.pp_bet * pp_multiplier(first, second, self.rules.pp_pays) if self.pp_bet else 0.0
        fct_net = self.fct_bet * fct_multiplier(first, second, up, hole, self.rules.fct_pays) if self.fct_bet else 0.0
        net = main_net + pp_net + fct_net + self._insurance_net
        self.chips = self._start_chips + net
        self.result = RoundResult(
            main_net=main_net,
            pp_net=pp_net,
            fct_net=fct_net,
            insurance_net=self._insurance_net,
            net=net,
            chips_after=self.chips,
            x=main_net / self.main_bet,
            cards_used=self.pos,
            dealer_cards=list(self._dealer),
            hands=self.hands,
        )
        self.phase = "done"


def play_round(rules: Rules, shoe: Sequence[int], chips: float, main_bet: float,
               style: Callable[[HandView], str], pp_bet: float = 0.0, fct_bet: float = 0.0) -> RoundResult:
    """Play one round to the end with a play style. Styles never take insurance."""
    round_ = Round(rules, shoe, chips, main_bet, pp_bet, fct_bet)
    if round_.phase == "insurance":
        round_.act("no_insurance")
    while round_.phase == "play":
        round_.act(style(round_.view()))
    return round_.result
