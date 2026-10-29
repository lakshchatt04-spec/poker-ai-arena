"""Fixed-size encoding of the betting history that Agent B's GRU will read.

Agent A sees only the 72-dimensional observation, which says nothing about who
acted or in what order.  Agent B needs that order, so every action becomes one
16-number token, and a whole hand becomes a ``(32, 16)`` table plus a mask.

Layout of one token (all values in ``[0, 1]``)::

    index  width  feature
    0-1      2    actor one-hot: P0, P1
    2        1    is_self   (actor is the player making this decision)
    3-6      4    action one-hot: call, raise, fold, check
    7-10     4    street one-hot: preflop, flop, turn, river
    11       1    raises on this street / 4
    12       1    pot in big blinds / 48
    13       1    chips to call in big blinds / 2
    14       1    actor is the small blind this hand
    15       1    actor is not first to act on this street (in position)

Why the last two flags are not redundant: RLCard randomises the blinds every
hand (``np_random.randint`` in ``LimitHoldemGame.init_game``) and starts each
new street with whoever follows the player who closed the previous one
(``Round.proceed_round`` + ``start_new_round``).  Neither is derivable from the
actor id or the street number, so both are supplied explicitly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .environment import ACTION_NAMES


STREET_NAMES = ("preflop", "flop", "turn", "river")
NUM_STREETS = len(STREET_NAMES)

#: Hard ceiling on stored tokens per hand.  A max-raise policy completes a hand
#: in exactly 20 actions, so 32 leaves comfortable headroom.
MAX_HISTORY_TOKENS = 32

TOKEN_DIM = 16

#: RLCard's ``LimitHoldemGame.allowed_raise_num`` is 4.
RAISE_LIMIT = 4

#: Measured maxima with the default blinds and 4-raise cap.  Values are clipped
#: rather than allowed to exceed 1.0, so an unusual configuration degrades
#: gracefully instead of feeding the network out-of-range inputs.
POT_CEILING_BB = 48.0
CALL_CEILING_BB = 2.0

ACTOR_SLICE = slice(0, 2)
IS_SELF_INDEX = 2
ACTION_SLICE = slice(3, 7)
STREET_SLICE = slice(7, 11)
RAISES_INDEX = 11
POT_INDEX = 12
TO_CALL_INDEX = 13
SMALL_BLIND_INDEX = 14
IN_POSITION_INDEX = 15

#: ``len(raw_obs["public_cards"])`` -> street index.  RLCard deals 3 cards at
#: once on the flop and 1 on each later street.
_STREET_BY_BOARD_SIZE = {0: 0, 3: 1, 4: 2, 5: 3}


@dataclass(frozen=True)
class ActionToken:
    """One completed action, recorded together with its decision-time context.

    ``pot_bb`` and ``to_call_bb`` are snapshots taken *before* the action, because
    RLCard's ``action_record`` drops that information.
    """

    actor: int
    action: str
    street: int
    raises_this_street: int
    pot_bb: float
    to_call_bb: float
    is_small_blind: bool
    is_in_position: bool

    @property
    def street_name(self) -> str:
        return STREET_NAMES[self.street]


@dataclass(frozen=True)
class HistorySequence:
    """A right-padded table of tokens plus the mask that marks real rows."""

    tokens: np.ndarray  # (max_tokens, TOKEN_DIM) float32
    mask: np.ndarray  # (max_tokens,) bool, True only where a token exists
    length: int  # number of real tokens, 0 .. max_tokens

    def real_tokens(self) -> np.ndarray:
        """Return only the rows that carry an action, shape ``(length, 16)``."""

        return self.tokens[: self.length]


def street_from_board_size(board_size: int) -> int:
    """Map the number of public cards to a street index."""

    try:
        return _STREET_BY_BOARD_SIZE[int(board_size)]
    except KeyError:
        raise ValueError(
            f"unexpected number of public cards: {board_size!r}"
        ) from None


def normalize(value: float, ceiling: float) -> float:
    """Scale ``value`` into ``[0, 1]`` and clip, so features stay bounded."""

    if ceiling <= 0:
        raise ValueError(f"ceiling must be positive, got {ceiling!r}")
    return float(min(max(float(value) / float(ceiling), 0.0), 1.0))


def encode_token(token: ActionToken, viewer: int) -> np.ndarray:
    """Encode one action token from ``viewer``'s point of view.

    ``viewer`` is the player about to act, so ``is_self`` flips between P0 and
    P1 for the same action stream.
    """

    if viewer not in (0, 1):
        raise ValueError(f"viewer must be 0 or 1, got {viewer!r}")
    if token.actor not in (0, 1):
        raise ValueError(f"actor must be 0 or 1, got {token.actor!r}")
    if token.action not in ACTION_NAMES:
        raise ValueError(f"unknown action {token.action!r}")
    if not 0 <= token.street < NUM_STREETS:
        raise ValueError(f"street out of range: {token.street!r}")
    if not 0 <= token.raises_this_street <= RAISE_LIMIT:
        raise ValueError(
            f"raises_this_street out of range: {token.raises_this_street!r}"
        )

    vector = np.zeros(TOKEN_DIM, dtype=np.float32)
    vector[ACTOR_SLICE.start + token.actor] = 1.0
    vector[IS_SELF_INDEX] = 1.0 if token.actor == viewer else 0.0
    vector[ACTION_SLICE.start + ACTION_NAMES.index(token.action)] = 1.0
    vector[STREET_SLICE.start + token.street] = 1.0
    vector[RAISES_INDEX] = normalize(token.raises_this_street, RAISE_LIMIT)
    vector[POT_INDEX] = normalize(token.pot_bb, POT_CEILING_BB)
    vector[TO_CALL_INDEX] = normalize(token.to_call_bb, CALL_CEILING_BB)
    vector[SMALL_BLIND_INDEX] = 1.0 if token.is_small_blind else 0.0
    vector[IN_POSITION_INDEX] = 1.0 if token.is_in_position else 0.0
    return vector


class HistoryTracker:
    """Collects one ``ActionToken`` per action, capped at ``max_tokens``.

    Call order for one hand::

        state, player_id = env.reset()
        tracker.start_hand(state)

        while not env.is_over():
            sequence = tracker.tokens(viewer=player_id)   # the agent's input
            action_name = env.actions[chosen_action_id]
            tracker.record(state, player_id, action_name)  # snapshot, then append
            state, player_id = env.step(chosen_action_id)

    ``record`` must be called *before* ``step`` so pot, call amount and street
    describe the state the decision was made in.  When more than ``max_tokens``
    actions arrive, the oldest tokens are dropped and the newest kept.
    """

    def __init__(self, max_tokens: int = MAX_HISTORY_TOKENS) -> None:
        self.max_tokens = int(max_tokens)
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {max_tokens!r}")

        self._tokens: deque[ActionToken] = deque(maxlen=self.max_tokens)
        self._street_first_actor: dict[int, int] = {}
        self._dropped = 0
        self._sb_player: int | None = None
        self._big_blind: int | None = None

    # -- hand lifecycle ---------------------------------------------------
    def start_hand(self, state: dict[str, Any]) -> None:
        """Reset the tracker and learn this hand's blinds from the first state."""

        raw_obs = state["raw_obs"]
        if state.get("action_record"):
            raise ValueError("start_hand() expects a freshly reset state")
        if raw_obs["public_cards"]:
            raise ValueError("start_hand() expects no public cards yet")

        chips = [int(chips) for chips in raw_obs["all_chips"]]
        if len(chips) != 2:
            raise ValueError(
                f"expected 2 blinds for heads-up play, got {len(chips)}"
            )
        small, big = sorted(chips)
        if small <= 0 or big != 2 * small:
            raise ValueError(f"unexpected blind posting: {chips!r}")

        self._tokens.clear()
        self._street_first_actor.clear()
        self._dropped = 0
        self._sb_player = chips.index(small)
        self._big_blind = big

    # -- recording --------------------------------------------------------
    def record(
        self, state: dict[str, Any], player_id: int, action: str
    ) -> ActionToken:
        """Snapshot the decision context and append the action that followed."""

        token = self.build_token(state, player_id, action)
        self.append(token)
        return token

    def build_token(
        self, state: dict[str, Any], player_id: int, action: str
    ) -> ActionToken:
        """Create the token for ``action`` without storing it yet."""

        self._require_started()
        if player_id not in (0, 1):
            raise ValueError(f"player_id must be 0 or 1, got {player_id!r}")
        if action not in ACTION_NAMES:
            raise ValueError(f"unknown action {action!r}")

        raw_obs = state["raw_obs"]
        street = street_from_board_size(len(raw_obs["public_cards"]))
        chips = [int(chips) for chips in raw_obs["all_chips"]]
        big_blind = self._big_blind
        assert big_blind is not None  # guaranteed by _require_started()

        raises = int(raw_obs["raise_nums"][street])
        if not 0 <= raises <= RAISE_LIMIT:
            raise ValueError(f"unexpected raise count {raises!r} on street {street}")

        first_actor = self._street_first_actor.setdefault(street, player_id)

        return ActionToken(
            actor=player_id,
            action=action,
            street=street,
            raises_this_street=raises,
            pot_bb=sum(chips) / big_blind,
            to_call_bb=(max(chips) - int(raw_obs["my_chips"])) / big_blind,
            is_small_blind=player_id == self._sb_player,
            is_in_position=player_id != first_actor,
        )

    def append(self, token: ActionToken) -> None:
        """Append a pre-built token, dropping the oldest if the buffer is full."""

        if len(self._tokens) == self.max_tokens:
            self._dropped += 1
        self._tokens.append(token)

    # -- reading ----------------------------------------------------------
    def tokens(self, viewer: int) -> HistorySequence:
        """Return the encoded history as seen by ``viewer``."""

        self._require_started()
        length = len(self._tokens)
        table = np.zeros((self.max_tokens, TOKEN_DIM), dtype=np.float32)
        for index, token in enumerate(self._tokens):
            table[index] = encode_token(token, viewer)

        mask = np.zeros(self.max_tokens, dtype=bool)
        mask[:length] = True
        return HistorySequence(tokens=table, mask=mask, length=length)

    @property
    def length(self) -> int:
        """Number of tokens currently stored (never above ``max_tokens``)."""

        return len(self._tokens)

    @property
    def dropped(self) -> int:
        """How many oldest tokens were discarded because the buffer was full."""

        return self._dropped

    @property
    def sb_player(self) -> int:
        self._require_started()
        assert self._sb_player is not None
        return self._sb_player

    @property
    def big_blind(self) -> int:
        self._require_started()
        assert self._big_blind is not None
        return self._big_blind

    @property
    def actions(self) -> tuple[ActionToken, ...]:
        """The stored tokens, oldest first."""

        return tuple(self._tokens)

    def _require_started(self) -> None:
        if self._big_blind is None:
            raise RuntimeError("call start_hand(state) before using the tracker")
