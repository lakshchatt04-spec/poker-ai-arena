"""Rollout collection for Agent B: Agent A's data plus the history tokens.

The maths is identical to :mod:`poker_arena.rollout`: terminal-only rewards, GAE
with ``gamma = 1``, a stored sampling-time ``log_prob`` and ``value``.  What
changes is that every decision also carries the ``(32, 16)`` token table and its
mask, because Agent B's policy reads them.  The GAE functions are imported from
:mod:`poker_arena.rollout` rather than reimplemented -- two copies of the credit
assignment would eventually disagree.

``collect_history_trajectories`` mirrors ``collect_trajectories`` with one extra
step: a single :class:`~poker_arena.history.HistoryTracker` per hand is advanced
for **every** action, including the opponent's.  The tracker stores actor-relative
tokens, and ``tokens(viewer=player_id)`` encodes them for whoever is deciding, so
the same tracker serves both seats of a self-play hand.  Recording before
``env.step`` is required: pot, chips to call and street must describe the decision
that was made, not the state after it.

Why the history is not "the previous observation": action order is exactly the
part of the game RLCard's 72-number observation throws away.  If B were fed a
window of past observations its input would still not say who acted, only what
the counts were.  The token stream is the smallest input that contains the order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from .encoding import FEATURE_DIM, ObservationEncoder, small_blind_player
from .environment import NUM_PLAYERS, reset_hand
from .history import MAX_HISTORY_TOKENS, TOKEN_DIM, HistorySequence, HistoryTracker
from .networks import NUM_ACTIONS
from .networks_b import HistoryPolicyValueNet
from .opponents import notify_hand_start
from .rollout import GAE_LAMBDA, GAMMA, compute_gae

Opponent = Callable[[Mapping[str, Any], int], int]


@dataclass(frozen=True)
class HistoryTransition:
    """One decision with the history that preceded it.

    ``log_prob`` and ``value`` come from the forward pass that chose ``action``.
    Recomputation after an update would make the PPO ratio meaningless.
    """

    features: np.ndarray  # (FEATURE_DIM,) float32
    tokens: np.ndarray  # (MAX_HISTORY_TOKENS, TOKEN_DIM) float32
    history_mask: np.ndarray  # (MAX_HISTORY_TOKENS,) bool
    legal_mask: np.ndarray  # (NUM_ACTIONS,) bool
    action: int
    log_prob: float
    value: float


@dataclass(frozen=True)
class HistoryTrajectory:
    """One player's decisions in one hand, plus what the hand paid them."""

    transitions: tuple[HistoryTransition, ...]
    final_payoff: float

    def __len__(self) -> int:
        return len(self.transitions)

    def rewards(self) -> np.ndarray:
        rewards = np.zeros(len(self.transitions), dtype=np.float64)
        if rewards.size:
            rewards[-1] = float(self.final_payoff)
        return rewards

    def values(self) -> np.ndarray:
        return np.asarray(
            [transition.value for transition in self.transitions], dtype=np.float64
        )


@dataclass(frozen=True)
class HistoryRolloutBatch:
    """Flat decisions ready for Agent B's PPO update, one row per decision."""

    features: np.ndarray  # (N, FEATURE_DIM) float32
    tokens: np.ndarray  # (N, MAX_HISTORY_TOKENS, TOKEN_DIM) float32
    history_masks: np.ndarray  # (N, MAX_HISTORY_TOKENS) bool
    legal_masks: np.ndarray  # (N, NUM_ACTIONS) bool
    actions: np.ndarray  # (N,) int64
    log_probs: np.ndarray  # (N,) float32
    values: np.ndarray  # (N,) float32
    advantages: np.ndarray  # (N,) float32, raw (not normalised)
    returns: np.ndarray  # (N,) float32

    def __len__(self) -> int:
        return int(self.actions.shape[0])


def _is_prefix_mask(mask: np.ndarray) -> bool:
    """True when every ``True`` precedes every ``False`` on each row."""

    lengths = mask.sum(axis=1).astype(np.int64)
    expected = np.arange(mask.shape[1])[None, :] < lengths[:, None]
    return bool(np.array_equal(mask, expected))


class HistoryRolloutBuffer:
    """Collects history-aware trajectories and flattens them into one batch."""

    def __init__(self, gamma: float = GAMMA, lam: float = GAE_LAMBDA) -> None:
        if not 0.0 <= float(gamma) <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma!r}")
        if not 0.0 <= float(lam) <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam!r}")

        self.gamma = float(gamma)
        self.lam = float(lam)
        self._trajectories: list[HistoryTrajectory] = []

    def add_trajectory(self, trajectory: HistoryTrajectory) -> None:
        """Validate and store one trajectory.

        The checks mirror :meth:`poker_arena.rollout.RolloutBuffer.add_trajectory`
        and add the history shapes and the prefix-mask rule.  A malformed mask
        would silently reorder the actions inside the GRU, so it fails here
        rather than in a training curve three hours later.
        """

        if len(trajectory) == 0:
            raise ValueError(
                "cannot store a trajectory with no decisions; a hand where the "
                "small blind folds preflop leaves the big blind with none"
            )
        if not np.isfinite(float(trajectory.final_payoff)):
            raise ValueError(
                f"final_payoff must be finite, got {trajectory.final_payoff!r}"
            )

        for index, transition in enumerate(trajectory.transitions):
            features = np.asarray(transition.features)
            tokens = np.asarray(transition.tokens)
            history_mask = np.asarray(transition.history_mask)
            mask = np.asarray(transition.legal_mask)

            if features.shape != (FEATURE_DIM,):
                raise ValueError(
                    f"transition {index}: features must be ({FEATURE_DIM},), "
                    f"got {features.shape}"
                )
            if tokens.shape != (MAX_HISTORY_TOKENS, TOKEN_DIM):
                raise ValueError(
                    f"transition {index}: tokens must be "
                    f"({MAX_HISTORY_TOKENS}, {TOKEN_DIM}), got {tokens.shape}"
                )
            if tokens.dtype != np.float32:
                raise ValueError(
                    f"transition {index}: tokens must be float32, got {tokens.dtype}"
                )
            if history_mask.shape != (MAX_HISTORY_TOKENS,):
                raise ValueError(
                    f"transition {index}: history_mask must be "
                    f"({MAX_HISTORY_TOKENS},), got {history_mask.shape}"
                )
            if history_mask.dtype != np.bool_:
                raise ValueError(
                    f"transition {index}: history_mask must be bool, got "
                    f"{history_mask.dtype}"
                )
            if not _is_prefix_mask(history_mask.reshape(1, -1)):
                raise ValueError(
                    f"transition {index}: history_mask must mark a prefix of "
                    f"real tokens"
                )
            if mask.shape != (NUM_ACTIONS,):
                raise ValueError(
                    f"transition {index}: legal_mask must be ({NUM_ACTIONS},), "
                    f"got {mask.shape}"
                )
            if mask.dtype != np.bool_:
                raise ValueError(
                    f"transition {index}: legal_mask must be bool, got {mask.dtype}"
                )
            if not 0 <= int(transition.action) < NUM_ACTIONS:
                raise ValueError(
                    f"transition {index}: action {transition.action!r} is not a "
                    f"valid action id"
                )
            if not bool(mask[int(transition.action)]):
                raise ValueError(
                    f"transition {index}: action {transition.action!r} is illegal "
                    f"under its own mask {mask.tolist()}"
                )
            for name in ("log_prob", "value"):
                if not np.isfinite(float(getattr(transition, name))):
                    raise ValueError(
                        f"transition {index}: {name} is not finite "
                        f"({getattr(transition, name)!r})"
                    )

        self._trajectories.append(trajectory)

    def batch(self) -> HistoryRolloutBatch:
        """Flatten every stored trajectory, keeping advantages raw."""

        if not self._trajectories:
            raise ValueError("no trajectories stored; call add_trajectory() first")

        features = np.concatenate(
            [np.stack([np.asarray(t.features) for t in traj.transitions])
             for traj in self._trajectories]
        ).astype(np.float32)
        tokens = np.concatenate(
            [np.stack([np.asarray(t.tokens) for t in traj.transitions])
             for traj in self._trajectories]
        ).astype(np.float32)
        history_masks = np.concatenate(
            [np.stack([np.asarray(t.history_mask) for t in traj.transitions])
             for traj in self._trajectories]
        )
        masks = np.concatenate(
            [np.stack([np.asarray(t.legal_mask) for t in traj.transitions])
             for traj in self._trajectories]
        )
        actions = np.concatenate(
            [np.asarray([int(t.action) for t in traj.transitions], dtype=np.int64)
             for traj in self._trajectories]
        )
        log_probs = np.concatenate(
            [np.asarray([float(t.log_prob) for t in traj.transitions], dtype=np.float32)
             for traj in self._trajectories]
        )
        values = np.concatenate(
            [traj.values() for traj in self._trajectories]
        ).astype(np.float32)

        advantage_parts: list[np.ndarray] = []
        return_parts: list[np.ndarray] = []
        for trajectory in self._trajectories:
            advantages, returns = compute_gae(
                trajectory.rewards(),
                trajectory.values(),
                gamma=self.gamma,
                lam=self.lam,
            )
            advantage_parts.append(advantages)
            return_parts.append(returns)

        return HistoryRolloutBatch(
            features=features,
            tokens=tokens,
            history_masks=history_masks,
            legal_masks=masks,
            actions=actions,
            log_probs=log_probs,
            values=values,
            advantages=np.concatenate(advantage_parts),
            returns=np.concatenate(return_parts),
        )

    def clear(self) -> None:
        self._trajectories.clear()

    def __len__(self) -> int:
        return sum(len(trajectory) for trajectory in self._trajectories)


def collect_history_trajectories(
    env: Any,
    encoder: ObservationEncoder,
    net: HistoryPolicyValueNet,
    *,
    opponent: Opponent | None = None,
    hands: int = 1,
    rng: np.random.Generator | None = None,
    use_history: bool = True,
) -> list[HistoryTrajectory]:
    """Play ``hands`` hands and return history-aware trajectories.

    ``opponent=None`` is self-play with both seats recorded; a callable plays
    player 1 and only the learner's seat is recorded.  ``rng`` controls action
    sampling, while the cards come from the environment's seed.

    ``use_history`` is threaded through to the network for the ablation run; it
    does not change what is collected, only which forward pass chooses actions.
    """

    if int(hands) < 1:
        raise ValueError(f"hands must be positive, got {hands!r}")

    rng = np.random.default_rng() if rng is None else rng
    recorded = tuple(range(NUM_PLAYERS)) if opponent is None else (0,)

    trajectories: list[HistoryTrajectory] = []
    was_training = net.training
    net.eval()

    try:
        for _ in range(int(hands)):
            decisions: dict[int, list[HistoryTransition]] = {
                player: [] for player in recorded
            }

            state, player_id = reset_hand(env)
            small_blind = small_blind_player(state)
            tracker = HistoryTracker()
            tracker.start_hand(state)
            if opponent is not None:
                notify_hand_start(opponent, state)

            while not env.is_over():
                sequence = tracker.tokens(viewer=player_id)
                encoded = encoder.encode(
                    state, is_small_blind=(player_id == small_blind)
                )

                if player_id in decisions:
                    action = _act(
                        net,
                        encoded,
                        sequence,
                        decisions[player_id],
                        rng,
                        use_history=use_history,
                    )
                else:
                    action = int(opponent(state, player_id))  # type: ignore[misc]
                    if action not in state["legal_actions"]:
                        raise ValueError(
                            f"opponent returned illegal action {action!r} for "
                            f"player {player_id}; legal: "
                            f"{sorted(state['legal_actions'])}"
                        )

                # Snapshot the context that produced this action, then advance.
                # Doing it after ``env.step`` would record the next street's pot.
                tracker.record(state, player_id, env.actions[int(action)])
                state, player_id = env.step(int(action))

            payoffs = env.get_payoffs()
            for player, transitions in decisions.items():
                if transitions:
                    trajectories.append(
                        HistoryTrajectory(
                            transitions=tuple(transitions),
                            final_payoff=float(payoffs[player]),
                        )
                    )
    finally:
        net.train(was_training)

    return trajectories


def _act(
    net: HistoryPolicyValueNet,
    encoded: Any,
    sequence: HistorySequence,
    sink: list[HistoryTransition],
    rng: np.random.Generator,
    *,
    use_history: bool,
) -> int:
    """Sample one legal action, record the decision, and return its id."""

    features = torch.from_numpy(encoded.features).unsqueeze(0)
    tokens = torch.from_numpy(sequence.tokens).unsqueeze(0)
    history_mask = torch.from_numpy(sequence.mask).unsqueeze(0)
    mask = torch.from_numpy(encoded.legal_mask).unsqueeze(0)

    with torch.no_grad():
        logits, value = net(
            features, tokens, history_mask, mask, use_history=use_history
        )
        distribution = Categorical(logits=logits)
        probabilities = torch.softmax(logits, dim=-1).reshape(-1).numpy()
        probabilities = probabilities.astype(np.float64)
        probabilities /= probabilities.sum()
        action = int(rng.choice(NUM_ACTIONS, p=probabilities))
        log_prob = float(distribution.log_prob(torch.tensor(action)))

    sink.append(
        HistoryTransition(
            features=np.array(encoded.features, dtype=np.float32, copy=True),
            tokens=np.array(sequence.tokens, dtype=np.float32, copy=True),
            history_mask=np.array(sequence.mask, dtype=bool, copy=True),
            legal_mask=np.array(encoded.legal_mask, dtype=bool, copy=True),
            action=action,
            log_prob=log_prob,
            value=float(value.reshape(-1)[0]),
        )
    )
    return action
