"""Opponents whose behaviour depends on something the observation does not say.

Agent A's input is one encoded state per decision: cards, chips, per-street
raise counts, street and position.  In heads-up limit hold'em that summary
implies most of what happened, which is why the three style opponents in
``training.py`` are effectively memoryless to exploit: a policy that sees only
the current state does as well against them as one that remembers.

``ContinuationBettor`` was the first attempt at changing that.  It raises
pre-flop on a coin flip and then plays the whole hand according to whether *it*
was the pre-flop aggressor.  Its mode is real, but the probe in
``scripts/history_probe.py`` found that it is also visible: a raise changes the
chips and the raise count, so a memoryless policy can read the mode straight off
the state.  On 3,000 deals, 0 of 4,302 post-flop decision states had a card-free
betting summary that appeared in both modes, and features-only classification
was 100 % on every street.  It is kept as a negative control.

``FirstRaiserBettor`` hides something the observation cannot encode: **who made
the first pre-flop raise**.  It raises on a coin flip at its first pre-flop
decision, then bets or raises post-flop only when the *agent* opened the
betting.  With the bot on the small blind and the agent on the big blind, both
of these lines reach the agent's first flop decision with the same board, 4
chips each and ``raise_nums = [1, 0, 0, 0]``:

* bot calls, agent raises, bot calls -- the agent opened, so the bot bets;
* bot raises, agent calls, bot checks -- the bot opened, so the bot keeps
  calling.

The action records differ, which is the only place the distinction lives.  The
probe confirms that this is not a restatement of the pot size: 29.3 % of the
agent's post-flop decisions sit in card-free mixed groups, the observation alone
is at chance there (52.7 %), and the history recovers the mode (95.6 %).

One caveat is inherent to the collision.  To leave both lines with the same
chips and raise count the agent has to act differently in them, so the hidden
fact is partly the agent's own pre-flop action.  A memoryless policy can
approximate it with a deterministic card-based pre-flop policy; the GRU should
help most when that policy is genuinely mixed.  The nine-run experiment in the
README measures which effect wins.

``CoinOpenerBettor`` removes the "agent caused it" caveat.  A per-hand coin,
drawn from the opponent's own RNG, selects which street the bot is aggressive
on: heads raises pre-flop and plays passive post-flop, tails calls pre-flop and
betting post-flop.  The coin is independent of the agent's cards and actions.
When the agent uses the complementary pre-flop response (call a bot raise,
raise a bot call), both coin values reach the flop with the same public
features, so only the action record reveals the coin.  The identifiability
probe in ``scripts/history_probe.py`` tests the collision before training.

``tests/test_opponents.py`` asserts both collisions on real environment states
rather than trusting this comment.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

#: Probability that the bot raises at its first preflop decision.
PREFLOP_RAISE_PROBABILITY = 0.5


def notify_hand_start(opponent: Any, state: Mapping[str, Any]) -> None:
    """Tell a stateful opponent that a new hand has begun.

    Most scripted opponents are pure functions of the current state and ignore
    this.  One that keeps per-hand state (``ContinuationBettor``) exposes
    ``start_hand``; calling it here keeps that detail out of the four play loops.
    """

    hook = getattr(opponent, "start_hand", None)
    if hook is not None:
        hook(state)


class ContinuationBettor:
    """Raise preflop on a coin flip, then play only as the preflop aggressor.

    It is intentionally simple and deterministic *given* the coin flip: every
    postflop street, an aggressor bets or raises when it can, and a non-aggressor
    calls or checks.  That makes the hidden fact -- who raised preflop -- the
    only thing that predicts its line, so it is a test of memory rather than of
    hand reading.

    Args:
        seed: Seeds the per-hand coin flip.  A fresh instance built with the same
            seed makes the same decision, which is what paired evaluation needs.
        preflop_raise_probability: Chance of raising at the first preflop
            decision.  The default splits hands evenly between the two modes.
    """

    def __init__(
        self,
        seed: int = 0,
        preflop_raise_probability: float = PREFLOP_RAISE_PROBABILITY,
    ) -> None:
        if not 0.0 <= float(preflop_raise_probability) <= 1.0:
            raise ValueError(
                "preflop_raise_probability must be in [0, 1], got "
                f"{preflop_raise_probability!r}"
            )
        self._rng = np.random.default_rng(seed)
        self.preflop_raise_probability = float(preflop_raise_probability)
        self._aggressor = False
        self._preflop_decided = False

    @property
    def was_preflop_aggressor(self) -> bool:
        """Whether the bot, not its opponent, raised preflop this hand."""

        return self._aggressor

    def start_hand(self, state: Mapping[str, Any]) -> None:
        """Clear the per-hand state; called by every play loop."""

        self._aggressor = False
        self._preflop_decided = False

    def __call__(self, state: Mapping[str, Any], player_id: int) -> int:
        legal = state["legal_actions"]

        if _is_preflop(state):
            if not self._preflop_decided:
                self._preflop_decided = True
                if 1 in legal and self._rng.random() < self.preflop_raise_probability:
                    self._aggressor = True
                    return 1
            return _call_check_fold(legal)

        if self._aggressor and 1 in legal:
            return 1
        return _call_check_fold(legal)


class CoinOpenerBettor:
    """A per-hand private coin selects which street the bot is aggressive on.

    The coin is drawn at the start of the hand from the opponent's own RNG, so it
    is independent of the agent's cards and of the agent's actions.  It controls
    the pre-flop opening: heads raises, tails calls/checks.  Post-flop the line is
    the *opposite*: the bot bets or raises when the coin was tails, and calls or
    checks when the coin was heads.  That inversion is what makes the two coin
    values collide at the agent's first flop decision.

    The collision needs the agent's complementary pre-flop response: call when
    the bot raises, raise when the bot calls.  Then both lines reach the big
    blind's first flop decision with the same board, chips and raise counts, and
    the bot's first flop action happens *after* that decision (a check when it
    opened, or the agent acts first when the agent opened).  Only the action
    record reveals the coin.  Without the complementary response the coin leaks
    through the chips, which is exactly what the identifiability probe checks.

    Args:
        seed: Seeds the per-hand coin.  A fresh instance with the same seed makes
            the same sequence of coin flips, which paired evaluation needs.
        raise_probability: Chance that the coin is heads (bot raises pre-flop and
            plays passive post-flop).
    """

    def __init__(
        self,
        seed: int = 0,
        raise_probability: float = PREFLOP_RAISE_PROBABILITY,
    ) -> None:
        if not 0.0 <= float(raise_probability) <= 1.0:
            raise ValueError(
                f"raise_probability must be in [0, 1], got {raise_probability!r}"
            )
        self._rng = np.random.default_rng(seed)
        self.raise_probability = float(raise_probability)
        self._coin: bool | None = None
        self._preflop_decisions = 0

    @property
    def coin(self) -> bool | None:
        """The current hand's coin, or ``None`` before ``start_hand``.

        ``True`` means the bot raised pre-flop (and will play passive post-flop);
        ``False`` means it called pre-flop (and will bet post-flop).
        """

        return self._coin

    def start_hand(self, state: Mapping[str, Any]) -> None:
        """Draw this hand's coin and clear the pre-flop decision count."""

        self._coin = bool(self._rng.random() < self.raise_probability)
        self._preflop_decisions = 0

    def __call__(self, state: Mapping[str, Any], player_id: int) -> int:
        legal = state["legal_actions"]

        if _is_preflop(state):
            first_decision = self._preflop_decisions == 0
            self._preflop_decisions += 1
            if first_decision and self._coin and 1 in legal:
                return 1
            return _call_check_fold(legal)

        if not self._coin and 1 in legal:
            return 1
        return _call_check_fold(legal)


class FirstRaiserBettor:
    """Bet post-flop only when the *opponent* made the first pre-flop raise.

    ``ContinuationBettor`` hides "did I raise?", which the observation leaks:
    a raise changes the chips and the raise count, so a memoryless policy can
    read the mode straight off the state.  This bot hides something the
    observation cannot encode instead -- **who raised first**.

    Pre-flop it raises on a coin flip at its first decision, then only calls or
    checks.  Post-flop it bets or raises when the *other* player was the first
    raiser, and calls or checks otherwise.  With the bot on the small blind and
    the agent on the big blind, both of these lines reach the agent's first
    flop decision with identical features (same board, 4 chips each, one raise):

    * bot calls, agent raises, bot calls -- agent raised first, bot bets;
    * bot raises, agent calls, bot checks -- bot raised first, bot keeps calling.

    In the first line the agent acts first and the bot's aggression is still
    hidden; in the second the bot has already checked, which is invisible in the
    observation.  Only the action order distinguishes them, so this is a real
    memory test rather than a restatement of the pot size.

    Args:
        seed: Seeds the per-hand pre-flop coin flip.  A fresh instance with the
            same seed makes the same decision, which paired evaluation needs.
        preflop_raise_probability: Chance of raising at the first pre-flop
            decision.  The default splits lines evenly.
    """

    def __init__(
        self,
        seed: int = 0,
        preflop_raise_probability: float = PREFLOP_RAISE_PROBABILITY,
    ) -> None:
        if not 0.0 <= float(preflop_raise_probability) <= 1.0:
            raise ValueError(
                "preflop_raise_probability must be in [0, 1], got "
                f"{preflop_raise_probability!r}"
            )
        self._rng = np.random.default_rng(seed)
        self.preflop_raise_probability = float(preflop_raise_probability)
        self._preflop_decided = False
        self._bot_raised = False
        self._agent_raised_first: bool | None = None

    @property
    def agent_raised_first(self) -> bool | None:
        """Whether the opponent opened the pre-flop betting, once known."""

        return self._agent_raised_first

    def start_hand(self, state: Mapping[str, Any]) -> None:
        """Clear the per-hand state; called by every play loop."""

        self._preflop_decided = False
        self._bot_raised = False
        self._agent_raised_first = None

    def __call__(self, state: Mapping[str, Any], player_id: int) -> int:
        legal = state["legal_actions"]

        if _is_preflop(state):
            if not self._preflop_decided:
                self._preflop_decided = True
                if 1 in legal and self._rng.random() < self.preflop_raise_probability:
                    self._bot_raised = True
                    return 1
            return _call_check_fold(legal)

        if self._agent_raised_first is None:
            # Only two players, so a pre-flop raise the bot did not make must
            # have come from the agent.  The count is final once the flop is out.
            raises = int((state.get("raw_obs") or {}).get("raise_nums", [0])[0])
            self._agent_raised_first = raises >= 1 and not self._bot_raised

        if self._agent_raised_first and 1 in legal:
            return 1
        return _call_check_fold(legal)


def _is_preflop(state: Mapping[str, Any]) -> bool:
    raw = state.get("raw_obs") or {}
    return len(raw.get("public_cards") or []) == 0


def _call_check_fold(legal: Any) -> int:
    """Check, else call, else fold.  Never raises.

    Check is preferred over call because the history tracker records the action
    id it is given.  RLCard silently turns an illegal ``call`` into ``check``
    when no bet is pending, so returning ``call`` there would make the recorded
    history disagree with what actually happened.
    """

    for action in (3, 0, 2):
        if action in legal:
            return action
    return 1  # unreachable: an action is always call/raise/fold/check
