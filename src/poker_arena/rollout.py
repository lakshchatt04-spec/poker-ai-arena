"""Rollout collection and generalised advantage estimation for Agent A.

A poker hand pays out once, at the end.  Every decision inside the hand sees a
reward of ``0`` and the hand's payoff arrives in the final slot.  Handing that
same number to every decision and subtracting the baseline is wrong in a way
that matters: on a three-decision hand that loses 3 bb, the naive
``R - V(s_t)`` assigns ``-3.0 / -4.0 / -4.5``, punishing the two decisions that
*raised* the value estimate.

So credit is assigned with temporal-difference residuals instead::

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t     = delta_t + (gamma * lam) * A_{t+1}

``delta_t`` asks "did the state get better or worse than the value head
expected?", which is a question each decision can answer on its own.  The same
example then reads ``+1.0 / +0.5 / -4.5`` at ``lam = 0``: the call was good, the
check was good, the fold gave away 4.5 bb.

``lam = 1`` reproduces the naive ``R - V(s_t)`` exactly, which is the check that
the two are one family rather than two ideas; :func:`compute_gae` is tested
against that identity.  ``lam`` is the dial between "blame only the decision that
moved the value" and "blame the whole hand equally".

There is no free fix for the remaining problem.  Between two of your decisions
the opponent acts and a card falls, so their effect is attributed to your
previous action.  Over many hands that noise averages out, but it never
disappears per hand, and no choice of ``lam`` separates luck from skill.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from .encoding import FEATURE_DIM, ObservationEncoder, small_blind_player
from .environment import ACTION_NAMES, reset_hand
from .networks import PolicyValueNet
from .opponents import notify_hand_start

NUM_ACTIONS = len(ACTION_NAMES)

#: Chips do not decay, and the objective is the expected final payoff, so the
#: episode is undiscounted.
GAMMA = 1.0

#: Bias/variance dial for GAE.  ``1.0`` is the naive terminal-total version,
#: ``0.0`` blames only the transition that moved the value.
GAE_LAMBDA = 0.95


@dataclass(frozen=True)
class Transition:
    """One decision, captured at sampling time.

    ``log_prob`` and ``value`` must come from the same forward pass that chose
    ``action``.  Recomputing them later with updated weights makes the PPO ratio
    meaningless, which is why :func:`collect_trajectories` stores them inline.
    """

    features: np.ndarray  # (FEATURE_DIM,) float32
    legal_mask: np.ndarray  # (NUM_ACTIONS,) bool
    action: int
    log_prob: float
    value: float


@dataclass(frozen=True)
class Trajectory:
    """One player's decisions in one hand, plus what the hand paid them."""

    transitions: tuple[Transition, ...]
    final_payoff: float

    def __len__(self) -> int:
        return len(self.transitions)

    def rewards(self) -> np.ndarray:
        """Per-decision rewards: zero everywhere, the payoff after the last one."""

        rewards = np.zeros(len(self.transitions), dtype=np.float64)
        if rewards.size:
            rewards[-1] = float(self.final_payoff)
        return rewards

    def values(self) -> np.ndarray:
        return np.asarray(
            [transition.value for transition in self.transitions], dtype=np.float64
        )


@dataclass(frozen=True)
class RolloutBatch:
    """Flat transitions ready for a PPO update, one row per decision."""

    features: np.ndarray  # (N, FEATURE_DIM) float32
    legal_masks: np.ndarray  # (N, NUM_ACTIONS) bool
    actions: np.ndarray  # (N,) int64
    log_probs: np.ndarray  # (N,) float32
    values: np.ndarray  # (N,) float32
    advantages: np.ndarray  # (N,) float32, raw (not normalised)
    returns: np.ndarray  # (N,) float32

    def __len__(self) -> int:
        return int(self.actions.shape[0])


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    *,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(advantages, returns)`` for one trajectory.

    Args:
        rewards: ``(T,)`` reward received *after* each action.  Poker puts the
            whole payoff in the last slot and zeros everywhere else.
        values: ``(T,)`` ``V(s_t)`` from the sampling forward pass.
        gamma: discount factor, ``1.0`` for this game.
        lam: GAE lambda.  ``1.0`` recovers ``final_return - V(s_t)`` exactly.

    Returns:
        ``advantages`` and ``returns = advantages + values``, both ``(T,)``
        float32.  The value after the last decision is ``0``: the hand is over,
        so there is nothing to bootstrap from.

    Raises:
        ValueError: If the inputs differ in length, are not 1-D, or are empty.
    """

    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if rewards.ndim != 1 or values.ndim != 1:
        raise ValueError(
            f"rewards and values must be 1-D, got {rewards.shape} and {values.shape}"
        )
    if rewards.shape != values.shape:
        raise ValueError(
            f"rewards and values must have the same length, got "
            f"{rewards.shape} and {values.shape}"
        )
    if rewards.size == 0:
        raise ValueError("cannot compute advantages for an empty trajectory")

    advantages = np.zeros_like(values)
    running = 0.0
    for index in range(values.size - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < values.size else 0.0
        delta = rewards[index] + gamma * next_value - values[index]
        running = delta + gamma * lam * running
        advantages[index] = running

    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def normalize_advantages(advantages: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Centre and scale advantages, guarding the no-spread case.

    A batch whose advantages are all equal carries no preference signal; scaling
    it would divide by ~0 and turn numerical dust into a large update, so it
    comes back as zeros instead.
    """

    advantages = np.asarray(advantages, dtype=np.float32)
    if advantages.size == 0:
        return advantages.astype(np.float32)

    spread = float(advantages.std())
    if not np.isfinite(spread) or spread < eps:
        return np.zeros_like(advantages)

    return ((advantages - advantages.mean()) / (spread + eps)).astype(np.float32)


class RolloutBuffer:
    """Collects trajectories and flattens them into one :class:`RolloutBatch`."""

    def __init__(self, gamma: float = GAMMA, lam: float = GAE_LAMBDA) -> None:
        _check_unit_interval("gamma", gamma)
        _check_unit_interval("lam", lam)

        self.gamma = float(gamma)
        self.lam = float(lam)
        self._trajectories: list[Trajectory] = []

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Validate and store one trajectory.

        Raises:
            ValueError: If the trajectory is empty, its arrays disagree, a
                feature vector or mask has the wrong width, the stored action is
                illegal under its own mask, or a scalar is not finite.
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
            mask = np.asarray(transition.legal_mask)
            if features.shape != (FEATURE_DIM,):
                raise ValueError(
                    f"transition {index}: features must be ({FEATURE_DIM},), "
                    f"got {features.shape}"
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

    def batch(self) -> RolloutBatch:
        """Flatten every stored trajectory into one batch.

        Advantages are returned **raw**.  Normalise them inside the PPO step,
        after computing ``returns = advantages + values`` — normalising here
        would silently break that identity.

        Raises:
            ValueError: If nothing has been stored.
        """

        if not self._trajectories:
            raise ValueError("no trajectories stored; call add_trajectory() first")

        features = np.concatenate(
            [np.stack([np.asarray(t.features) for t in traj.transitions])
             for traj in self._trajectories]
        ).astype(np.float32)
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

        return RolloutBatch(
            features=features,
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


def collect_trajectories(
    env: Any,
    encoder: ObservationEncoder,
    net: PolicyValueNet,
    *,
    opponent: Callable[[Mapping[str, Any], int], int] | None = None,
    hands: int = 1,
    rng: np.random.Generator | None = None,
) -> list[Trajectory]:
    """Play ``hands`` hands and return the recorded trajectories.

    Args:
        env: A two-player limit Hold'em environment.
        encoder: The observation encoder built for ``env``.
        net: The policy/value network.  Used in eval mode here, and its previous
            mode is restored before returning.
        opponent: ``None`` means self-play: both seats are the network, and both
            seats' decisions are recorded as separate trajectories.  A callable
            ``(state, player_id) -> action_id`` instead plays player 1, and only
            player 0 (the learner) is recorded.
        hands: Number of hands to play.
        rng: Generator used to draw the sampled actions.  Pass a seeded one to
            make a run reproducible; the cards come from the environment's own
            seed.  ``None`` draws a fresh OS-seeded generator.

    Returns:
        One :class:`Trajectory` per recorded player per hand, in play order.
        Hands that leave a recorded player with no decisions are skipped.

    Raises:
        ValueError: If ``hands`` is not positive, or the opponent returns an
            action that is not legal in that state.
    """

    if int(hands) < 1:
        raise ValueError(f"hands must be positive, got {hands!r}")

    rng = np.random.default_rng() if rng is None else rng
    recorded = (0, 1) if opponent is None else (0,)

    trajectories: list[Trajectory] = []
    was_training = net.training
    net.eval()

    try:
        for _ in range(int(hands)):
            decisions: dict[int, list[Transition]] = {player: [] for player in recorded}

            state, player_id = reset_hand(env)
            small_blind = small_blind_player(state)
            if opponent is not None:
                notify_hand_start(opponent, state)

            while not env.is_over():
                encoded = encoder.encode(
                    state, is_small_blind=(player_id == small_blind)
                )

                if player_id in decisions:
                    action = _act(net, encoded, decisions[player_id], rng)
                else:
                    action = opponent(state, player_id)  # type: ignore[misc]
                    if int(action) not in state["legal_actions"]:
                        raise ValueError(
                            f"opponent returned illegal action {action!r} for "
                            f"player {player_id}; legal: "
                            f"{sorted(state['legal_actions'])}"
                        )

                state, player_id = env.step(int(action))

            payoffs = env.get_payoffs()
            for player, transitions in decisions.items():
                if transitions:
                    trajectories.append(
                        Trajectory(
                            transitions=tuple(transitions),
                            final_payoff=float(payoffs[player]),
                        )
                    )
    finally:
        net.train(was_training)

    return trajectories


def _act(
    net: PolicyValueNet,
    encoded: Any,
    sink: list[Transition],
    rng: np.random.Generator,
) -> int:
    """Sample one legal action, record the decision, and return the action id.

    Sampling goes through ``rng`` rather than torch's global generator so that a
    seeded callable reproduces a rollout exactly.  The distribution is unchanged:
    the probabilities are the same softmax the PPO update will use, and
    ``log_prob`` is read from that distribution for the drawn action.
    """

    features = torch.from_numpy(encoded.features).unsqueeze(0)
    mask = torch.from_numpy(encoded.legal_mask).unsqueeze(0)

    with torch.no_grad():
        logits, value = net(features, mask)
        distribution = Categorical(logits=logits)
        probabilities = torch.softmax(logits, dim=-1).reshape(-1).numpy()
        probabilities = probabilities.astype(np.float64)
        probabilities /= probabilities.sum()
        action = int(rng.choice(NUM_ACTIONS, p=probabilities))
        log_prob = float(distribution.log_prob(torch.tensor(action)))

    sink.append(
        Transition(
            features=np.array(encoded.features, dtype=np.float32, copy=True),
            legal_mask=np.array(encoded.legal_mask, dtype=bool, copy=True),
            action=action,
            log_prob=log_prob,
            value=float(value.reshape(-1)[0]),
        )
    )
    return action


def _check_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
