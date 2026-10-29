"""Inspection helpers for RLCard's two-player limit Texas Hold'em.

The goal of Stage 1 is to pin down the simulator contract before any neural
network is written.  Nothing in this module trains or evaluates a policy; the
only agent behaviour is uniform random action selection for smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import rlcard

ENVIRONMENT_ID = "limit-holdem"
NUM_PLAYERS = 2
OBSERVATION_SIZE = 72
CARD_SLOT_COUNT = 52
RAISE_BLOCK_SIZE = 5
NUM_BETTING_ROUNDS = 4
ACTION_NAMES = ("call", "raise", "fold", "check")


@dataclass(frozen=True)
class ObservationView:
    """Human-readable interpretation of RLCard's 72-dimensional observation."""

    cards: tuple[str, ...]
    raises_per_round: tuple[int, ...]


@dataclass(frozen=True)
class StepRecord:
    """Snapshot of one decision, taken *before* the action is applied.

    ``history`` is copied rather than referencing RLCard's live
    ``action_record`` list, so records remain valid after later steps.
    """

    player_id: int
    action_id: int
    action_name: str
    legal_action_ids: tuple[int, ...]
    legal_action_names: tuple[str, ...]
    observed_cards: tuple[str, ...]
    raises_per_round: tuple[int, ...]
    pot: int
    board: tuple[str, ...]
    history: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class HandResult:
    """Outcome of one complete hand."""

    payoffs: np.ndarray
    steps: tuple[StepRecord, ...]


def make_environment(seed: int = 42, allow_step_back: bool = False) -> Any:
    """Create a two-player limit Hold'em environment and validate its contract.

    RLCard 1.2.0 supports this setup.  If a future installation changes the
    player count, action vocabulary or observation size, fail loudly instead of
    silently training on a different game.

    Args:
        seed: Seed for RLCard's dealer/card RNG.
        allow_step_back: Kept off for PPO-style rollouts (RLCard default).

    Returns:
        The ready-to-use ``rlcard.Env``.

    Raises:
        RuntimeError: If the installed RLCard API does not match the expected
            heads-up limit Hold'em contract.
    """

    env = rlcard.make(
        ENVIRONMENT_ID,
        config={"seed": seed, "allow_step_back": allow_step_back},
    )

    problems: list[str] = []
    if env.num_players != NUM_PLAYERS:
        problems.append(f"expected {NUM_PLAYERS} players, got {env.num_players}")
    if tuple(env.actions) != ACTION_NAMES:
        problems.append(
            f"expected actions {ACTION_NAMES}, got {tuple(env.actions)}"
        )

    state_sizes = [int(np.prod(shape)) for shape in env.state_shape]
    if state_sizes != [OBSERVATION_SIZE] * NUM_PLAYERS:
        problems.append(
            f"expected observations of size {OBSERVATION_SIZE}, got {state_sizes}"
        )

    if not hasattr(env, "card2index") or len(env.card2index) != CARD_SLOT_COUNT:
        problems.append("expected a 52-card `env.card2index` mapping")

    if problems:
        raise RuntimeError(
            f"RLCard environment {ENVIRONMENT_ID!r} does not match the expected "
            "two-player limit Hold'em contract: " + "; ".join(problems)
        )

    return env


def decode_observation(env: Any, obs: np.ndarray) -> ObservationView:
    """Turn a 72-dimensional observation into readable cards and raise counts.

    Layout verified against RLCard 1.2.0:

    * ``obs[0:52]``: one-hot over the deck for the acting player's hole cards
      plus public cards.
    * ``obs[52:72]``: four betting rounds x five slots.  Slot ``n`` is set when
      that round has seen ``n`` raises, so every block is one-hot (future
      rounds show zero raises).
    """

    obs = np.asarray(obs)
    if obs.shape != (OBSERVATION_SIZE,):
        raise ValueError(
            f"expected observation shape {(OBSERVATION_SIZE,)}, got {obs.shape}"
        )

    index_to_card = {index: card for card, index in env.card2index.items()}
    cards = tuple(
        index_to_card[int(index)]
        for index in np.flatnonzero(obs[:CARD_SLOT_COUNT] != 0)
    )

    raise_blocks = obs[CARD_SLOT_COUNT:].reshape(
        NUM_BETTING_ROUNDS, RAISE_BLOCK_SIZE
    )
    if not np.all(raise_blocks.sum(axis=1) == 1):
        raise ValueError("expected exactly one raise-count slot per betting round")

    raises_per_round = tuple(
        int(np.argmax(block)) for block in raise_blocks
    )
    return ObservationView(cards=cards, raises_per_round=raises_per_round)


def action_mask(env: Any, state: dict[str, Any]) -> np.ndarray:
    """Return a 0/1 mask over ``env.num_actions`` for the current state.

    RLCard silently substitutes an illegal action, so future policies must mask
    their action distribution rather than relying on the environment to reject
    bad choices.
    """

    mask = np.zeros(env.num_actions, dtype=np.int8)
    for action_id in state["legal_actions"]:
        action_id = int(action_id)
        if not 0 <= action_id < env.num_actions:
            raise ValueError(f"legal action id out of range: {action_id}")
        mask[action_id] = 1
    return mask


def play_random_hand(
    env: Any, rng: np.random.Generator | None = None
) -> HandResult:
    """Play one complete hand, choosing uniformly among legal actions.

    ``reset()`` is called first, so the same environment can be reused for
    consecutive hands.  Returns the final payoffs and a snapshot of every
    decision for inspection.
    """

    if rng is None:
        rng = np.random.default_rng()

    state, player_id = env.reset()
    steps: list[StepRecord] = []

    while not env.is_over():
        legal_ids = tuple(int(action) for action in state["legal_actions"])
        legal_names = tuple(str(name) for name in state["raw_legal_actions"])

        if not legal_ids:
            raise RuntimeError("environment returned no legal actions")
        if len(legal_ids) != len(legal_names):
            raise RuntimeError("legal action ids and names are not parallel")
        if legal_names != tuple(env.actions[action] for action in legal_ids):
            raise RuntimeError("legal action ids and names disagree")

        action_id = int(rng.choice(legal_ids))
        view = decode_observation(env, state["obs"])
        raw_obs = state["raw_obs"]

        steps.append(
            StepRecord(
                player_id=int(player_id),
                action_id=action_id,
                action_name=str(env.actions[action_id]),
                legal_action_ids=legal_ids,
                legal_action_names=legal_names,
                observed_cards=view.cards,
                raises_per_round=view.raises_per_round,
                pot=int(sum(raw_obs["all_chips"])),
                board=tuple(str(card) for card in raw_obs["public_cards"]),
                history=tuple(
                    (int(history_player), str(action))
                    for history_player, action in state["action_record"]
                ),
            )
        )

        state, player_id = env.step(action_id)

    payoffs = np.asarray(env.get_payoffs(), dtype=float)
    if payoffs.shape != (NUM_PLAYERS,):
        raise RuntimeError(
            f"expected {NUM_PLAYERS} payoffs, got shape {payoffs.shape}"
        )

    return HandResult(payoffs=payoffs, steps=tuple(steps))


def verify_observation_is_legal(
    env: Any,
    player_id: int,
    state: dict[str, Any] | None = None,
) -> None:
    """Assert that a player's observation contains no private opponent cards.

    The observation must be exactly the player's own hole cards plus the public
    cards.  ``env.get_perfect_information()`` is used here for verification
    only; it exposes both hands and must never become an agent input.

    Raises:
        AssertionError: If a card is leaked, missing, or unexpected.
    """

    if state is None:
        state = env.get_state(player_id)

    perfect = env.get_perfect_information()
    observed = set(decode_observation(env, state["obs"]).cards)
    own_cards = set(perfect["hand_cards"][player_id])
    public_cards = set(perfect["public_card"] or [])
    opponent_private = set(perfect["hand_cards"][1 - player_id]) - public_cards

    allowed = own_cards | public_cards
    leaked = observed & opponent_private
    unexpected = observed - allowed
    missing = allowed - observed

    if leaked or unexpected or missing:
        raise AssertionError(
            f"observation leakage for P{player_id}: "
            f"leaked={sorted(leaked)}, unexpected={sorted(unexpected)}, "
            f"missing={sorted(missing)}"
        )
