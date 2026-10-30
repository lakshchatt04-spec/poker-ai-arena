"""Poker AI Arena: a heads-up Limit Texas Hold'em research testbed.

Stage 1 exposes simulator-inspection helpers plus the fixed-size betting-history
encoding Agent B's GRU will read.  Stage 2 adds Agent A's 131-number observation
encoder.  Learning agents, PPO and the networks themselves are not implemented
yet.
"""

from .encoding import (
    FEATURE_DIM,
    EncodedState,
    ObservationEncoder,
    small_blind_player,
    validate_environment_contract,
)
from .environment import (
    ACTION_NAMES,
    CARD_SLOT_COUNT,
    ENVIRONMENT_ID,
    NUM_BETTING_ROUNDS,
    NUM_PLAYERS,
    OBSERVATION_SIZE,
    RAISE_BLOCK_SIZE,
    HandResult,
    ObservationView,
    StepRecord,
    action_mask,
    decode_observation,
    make_environment,
    play_random_hand,
    verify_observation_is_legal,
)
from .history import (
    MAX_HISTORY_TOKENS,
    RAISE_LIMIT,
    STREET_NAMES,
    TOKEN_DIM,
    ActionToken,
    HistorySequence,
    HistoryTracker,
    encode_token,
    normalize,
    street_from_board_size,
)

__all__ = [
    "ACTION_NAMES",
    "CARD_SLOT_COUNT",
    "ENVIRONMENT_ID",
    "FEATURE_DIM",
    "MAX_HISTORY_TOKENS",
    "NUM_BETTING_ROUNDS",
    "NUM_PLAYERS",
    "OBSERVATION_SIZE",
    "RAISE_BLOCK_SIZE",
    "RAISE_LIMIT",
    "STREET_NAMES",
    "TOKEN_DIM",
    "ActionToken",
    "EncodedState",
    "HandResult",
    "HistorySequence",
    "HistoryTracker",
    "ObservationEncoder",
    "ObservationView",
    "StepRecord",
    "action_mask",
    "decode_observation",
    "encode_token",
    "make_environment",
    "normalize",
    "play_random_hand",
    "small_blind_player",
    "street_from_board_size",
    "validate_environment_contract",
    "verify_observation_is_legal",
]

__version__ = "0.1.0"
