"""Agent A's fixed-size encoder for the 72-dimensional observation.

RLCard's observation carries cards plus per-street raise counts and nothing
else.  Pot, chips to call and position never appear, so no amount of training can
recover them, and the first layer needs one consistent scale.  This module turns
``state["raw_obs"]`` into the single vector the policy/value network reads.

Layout, all values in ``[0, 1]``, dtype ``float32``::

    index      width  feature
    0-51        52    own hole cards, multi-hot over the deck
    52-103      52    public cards, multi-hot over the deck
    104-123     20    4 streets x 5 raise-count slots, row = street
    124-127      4    street one-hot: preflop, flop, turn, river
    128          1    pot in big blinds / 48
    129          1    chips to call in big blinds / 2
    130          1    this player posted the small blind

Why hole and board are split into two blocks: RLCard builds ``obs[0:52]`` from
``public_cards + hand`` as one multi-hot (``envs/limitholdem.py:59-62``), so
holding an ace on a king-high board and holding a king on an ace-high board
produce identical input.  Two blocks keep the distinction.

Why cards go through a mapping: ``raw_obs["hand"]`` and ``raw_obs["public_cards"]``
hold card *strings* such as ``"SQ"``, not slot numbers.  ``env.card2index`` is the
only authority on slot order, so the encoder captures it once instead of
re-deriving an order of its own.

Why position is passed in instead of derived: the small blind is drawn at random
every hand (``games/limitholdem/game.py:75``) and each street opens with whoever
follows the player who closed the previous one, so position is neither constant
nor recoverable from ``raw_obs``.  Capture it once per hand with
:func:`small_blind_player` and pass it to every :meth:`ObservationEncoder.encode`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .environment import (
    ACTION_NAMES,
    CARD_SLOT_COUNT,
    NUM_BETTING_ROUNDS,
    NUM_PLAYERS,
    RAISE_BLOCK_SIZE,
)
from .history import (
    CALL_CEILING_BB,
    POT_CEILING_BB,
    RAISE_LIMIT,
    normalize,
    street_from_board_size,
)

NUM_ACTIONS = len(ACTION_NAMES)

#: RLCard's default blinds, in chips.  ``big_blind = 2 * small_blind``.
SMALL_BLIND_CHIPS = 1
BIG_BLIND_CHIPS = 2

#: RLCard's constant raise size for the preflop and flop betting rounds.  It is
#: doubled to 4 chips on the turn and river (``game.py:148-149``).
RAISE_AMOUNT_CHIPS = 2

HOLE_SLICE = slice(0, CARD_SLOT_COUNT)
BOARD_SLICE = slice(CARD_SLOT_COUNT, 2 * CARD_SLOT_COUNT)
RAISE_SLICE = slice(
    2 * CARD_SLOT_COUNT,
    2 * CARD_SLOT_COUNT + NUM_BETTING_ROUNDS * RAISE_BLOCK_SIZE,
)
STREET_SLICE = slice(RAISE_SLICE.stop, RAISE_SLICE.stop + NUM_BETTING_ROUNDS)
POT_INDEX = STREET_SLICE.stop
TO_CALL_INDEX = POT_INDEX + 1
SMALL_BLIND_INDEX = TO_CALL_INDEX + 1

#: Width of :attr:`EncodedState.features`.
FEATURE_DIM = SMALL_BLIND_INDEX + 1


@dataclass(frozen=True)
class EncodedState:
    """One decision state, ready to be fed to the network.

    Attributes:
        features: ``(FEATURE_DIM,)`` float32 vector, every value in ``[0, 1]``.
        legal_mask: ``(4,)`` bool mask copied from ``state["legal_actions"]``.
            The encoder copies legality, it never recomputes it.
        street: ``0..3`` for preflop, flop, turn, river.  Kept for debugging and
            for indexing the raise block in tests.
    """

    features: np.ndarray
    legal_mask: np.ndarray
    street: int


class ObservationEncoder:
    """Turns decision states into fixed-size feature vectors.

    The card mapping is configuration, so it is captured once::

        encoder = ObservationEncoder.from_environment(env)

        state, player_id = env.reset()
        small_blind = small_blind_player(state)

        while not env.is_over():
            encoded = encoder.encode(
                state, is_small_blind=(player_id == small_blind)
            )
            ...
    """

    def __init__(self, card2index: Mapping[str, int]) -> None:
        mapping = {str(card): int(index) for card, index in card2index.items()}
        if len(mapping) != CARD_SLOT_COUNT:
            raise ValueError(
                f"expected {CARD_SLOT_COUNT} cards, got {len(mapping)}"
            )
        if sorted(mapping.values()) != list(range(CARD_SLOT_COUNT)):
            raise ValueError("card indices must be exactly 0..51")

        self._card2index: dict[str, int] = mapping

    @classmethod
    def from_environment(cls, env: Any) -> "ObservationEncoder":
        """Build an encoder for ``env`` and check its betting configuration."""

        validate_environment_contract(env)
        return cls(env.card2index)

    def encode(
        self, state: Mapping[str, Any], *, is_small_blind: bool
    ) -> EncodedState:
        """Encode one decision state for the player whose observation it is.

        Args:
            state: A state dict from ``env.reset()``, ``env.step()`` or
                ``env.get_state(player_id)``.
            is_small_blind: Whether the observing player posted the small blind
                this hand.  See :func:`small_blind_player`.

        Returns:
            The encoded features, the legal-action mask and the street index.

        Raises:
            ValueError: If the state violates the contract the layout assumes.
        """

        raw_obs = state["raw_obs"]
        hand = self._card_indices(raw_obs["hand"], "hand")
        board = self._card_indices(raw_obs["public_cards"], "public_cards")

        duplicates = set(hand) & set(board)
        if duplicates:
            raise ValueError(f"cards appear on both hand and board: {duplicates}")

        street = street_from_board_size(len(board))
        raises = _raise_counts(raw_obs["raise_nums"])
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        features[HOLE_SLICE.start + np.asarray(hand, dtype=np.intp)] = 1.0
        features[BOARD_SLICE.start + np.asarray(board, dtype=np.intp)] = 1.0
        for index, count in enumerate(raises):
            features[RAISE_SLICE.start + index * RAISE_BLOCK_SIZE + count] = 1.0
        features[STREET_SLICE.start + street] = 1.0

        chips = [int(value) for value in raw_obs["all_chips"]]
        if len(chips) != NUM_PLAYERS:
            raise ValueError(f"expected {NUM_PLAYERS} chip totals, got {chips!r}")

        my_chips = int(raw_obs["my_chips"])
        if my_chips not in chips:
            raise ValueError(f"my_chips={my_chips} is not among all_chips={chips!r}")

        pot_bb = sum(chips) / BIG_BLIND_CHIPS
        to_call_bb = (max(chips) - my_chips) / BIG_BLIND_CHIPS
        features[POT_INDEX] = normalize(pot_bb, POT_CEILING_BB)
        features[TO_CALL_INDEX] = normalize(to_call_bb, CALL_CEILING_BB)
        features[SMALL_BLIND_INDEX] = 1.0 if is_small_blind else 0.0

        return EncodedState(
            features=features,
            legal_mask=_legal_mask(state),
            street=street,
        )

    def encode_batch(
        self,
        states: Sequence[Mapping[str, Any]],
        is_small_blind: Sequence[bool],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode many decision states at once.

        Returns:
            ``(features, masks)`` shaped ``(n, FEATURE_DIM)`` float32 and
            ``(n, 4)`` bool.  Both come back together because a rollout buffer
            needs them aligned row for row.

        Raises:
            ValueError: If the inputs have different lengths.
        """

        if len(states) != len(is_small_blind):
            raise ValueError(
                f"got {len(states)} states but {len(is_small_blind)} position flags"
            )
        if not states:
            return (
                np.zeros((0, FEATURE_DIM), dtype=np.float32),
                np.zeros((0, NUM_ACTIONS), dtype=bool),
            )

        encoded = [
            self.encode(state, is_small_blind=bool(flag))
            for state, flag in zip(states, is_small_blind, strict=True)
        ]
        return (
            np.stack([item.features for item in encoded]).astype(np.float32),
            np.stack([item.legal_mask for item in encoded]),
        )

    def _card_indices(self, cards: Iterable[str], label: str) -> list[int]:
        """Map RLCard card strings to slots, rejecting unknown cards."""

        indices: list[int] = []
        for card in cards:
            try:
                indices.append(self._card2index[str(card)])
            except KeyError:
                raise ValueError(f"unknown card {card!r} in {label}") from None
        return indices


def small_blind_player(state: Mapping[str, Any]) -> int:
    """Return the player who posted the small blind in this hand.

    Must be called on the state returned by ``env.reset()``, before any action:
    the blinds are the only moment the contributions identify the seats.  The
    result is fixed for the whole hand, so capture it once and reuse it.

    Raises:
        ValueError: If the state is not a freshly posted hand.
    """

    chips = [int(value) for value in state["raw_obs"]["all_chips"]]
    if len(chips) != NUM_PLAYERS:
        raise ValueError(
            f"expected {NUM_PLAYERS} blind postings, got {chips!r}"
        )

    small, big = sorted(chips)
    if small <= 0 or big != 2 * small:
        raise ValueError(
            f"expected a fresh 1:2 blind posting, got {chips!r}; "
            "call small_blind_player() on the state from reset()"
        )
    return chips.index(small)


def validate_environment_contract(env: Any) -> None:
    """Check the game still matches the ceilings this encoder was built for.

    ``pot / 48`` and ``to_call / 2`` are the only facts about the simulator baked
    into the layout, and they are only true for RLCard's default blinds and
    4-raise cap.  Validate once, while building the encoder, instead of
    re-deriving the bound on every step: a changed configuration should fail at
    startup, not silently saturate a feature in the middle of a run.

    Raises:
        RuntimeError: If any betting parameter differs from the measured default.
    """

    game = env.game
    observed = {
        "small_blind": getattr(game, "small_blind", None),
        "big_blind": getattr(game, "big_blind", None),
        "raise_amount": getattr(game, "raise_amount", None),
        "allowed_raise_num": getattr(game, "allowed_raise_num", None),
    }
    expected = {
        "small_blind": SMALL_BLIND_CHIPS,
        "big_blind": BIG_BLIND_CHIPS,
        "raise_amount": RAISE_AMOUNT_CHIPS,
        "allowed_raise_num": RAISE_LIMIT,
    }

    mismatched = {
        name: (expected[name], value)
        for name, value in observed.items()
        if value != expected[name]
    }
    if mismatched:
        details = ", ".join(
            f"{name}: expected {want!r}, got {got!r}"
            for name, (want, got) in mismatched.items()
        )
        raise RuntimeError(
            "the encoder's pot/to_call ceilings "
            f"({POT_CEILING_BB:g} bb / {CALL_CEILING_BB:g} bb) were measured for "
            f"RLCard's default betting: {details}. Update the ceilings and "
            "retrain, or the scalars will clip."
        )


def _raise_counts(raises: Iterable[int]) -> list[int]:
    """Copy and validate the per-street raise counts."""

    counts = [int(value) for value in raises]
    if len(counts) != NUM_BETTING_ROUNDS:
        raise ValueError(
            f"expected {NUM_BETTING_ROUNDS} raise counts, got {counts!r}"
        )
    for street, count in enumerate(counts):
        if not 0 <= count <= RAISE_LIMIT:
            raise ValueError(
                f"raise count {count} on street {street} is outside "
                f"0..{RAISE_LIMIT}"
            )
    return counts


def _legal_mask(state: Mapping[str, Any]) -> np.ndarray:
    """Copy ``state["legal_actions"]`` into a boolean mask.

    Deliberately a copy, not a derivation: RLCard is the only authority on
    legality, and an independently computed mask is a second source of truth
    that can drift from it.
    """

    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for action_id in state["legal_actions"]:
        action_id = int(action_id)
        if not 0 <= action_id < NUM_ACTIONS:
            raise ValueError(f"legal action id out of range: {action_id}")
        mask[action_id] = True
    return mask
