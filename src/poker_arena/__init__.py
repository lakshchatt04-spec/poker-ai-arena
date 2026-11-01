"""Poker AI Arena: a heads-up Limit Texas Hold'em research testbed.

Stage 1 exposes simulator-inspection helpers plus the fixed-size betting-history
encoding Agent B's GRU will read.  Stage 2 adds Agent A's 131-number observation
encoder, its policy/value network, the rollout buffer that turns played hands into
GAE advantages, the clipped PPO update, and a training loop with a fixed-seed
evaluation harness.
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
from .networks import (
    HIDDEN_WIDTH,
    MASKED_LOGIT,
    PolicyValueNet,
)
from .ppo import PPOConfig, PPOUpdater
from .rollout import (
    GAE_LAMBDA,
    GAMMA,
    RolloutBatch,
    RolloutBuffer,
    Trajectory,
    Transition,
    collect_trajectories,
    compute_gae,
    normalize_advantages,
)
from .training import (
    Z_95,
    EvaluationRecord,
    EvaluationResult,
    TrainingConfig,
    TrainingResult,
    evaluate,
    load_checkpoint,
    random_opponent,
    save_checkpoint,
    scripted_opponent,
    train,
)

__all__ = [
    "ACTION_NAMES",
    "CARD_SLOT_COUNT",
    "ENVIRONMENT_ID",
    "FEATURE_DIM",
    "GAE_LAMBDA",
    "GAMMA",
    "HIDDEN_WIDTH",
    "MASKED_LOGIT",
    "MAX_HISTORY_TOKENS",
    "NUM_BETTING_ROUNDS",
    "NUM_PLAYERS",
    "OBSERVATION_SIZE",
    "RAISE_BLOCK_SIZE",
    "RAISE_LIMIT",
    "STREET_NAMES",
    "TOKEN_DIM",
    "Z_95",
    "ActionToken",
    "EncodedState",
    "EvaluationRecord",
    "EvaluationResult",
    "HandResult",
    "HistorySequence",
    "HistoryTracker",
    "ObservationEncoder",
    "ObservationView",
    "PPOConfig",
    "PPOUpdater",
    "PolicyValueNet",
    "RolloutBatch",
    "RolloutBuffer",
    "StepRecord",
    "Trajectory",
    "TrainingConfig",
    "TrainingResult",
    "Transition",
    "action_mask",
    "collect_trajectories",
    "compute_gae",
    "decode_observation",
    "encode_token",
    "evaluate",
    "load_checkpoint",
    "make_environment",
    "normalize",
    "normalize_advantages",
    "play_random_hand",
    "random_opponent",
    "save_checkpoint",
    "scripted_opponent",
    "small_blind_player",
    "street_from_board_size",
    "train",
    "validate_environment_contract",
    "verify_observation_is_legal",
]

__version__ = "0.1.0"
