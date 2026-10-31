"""Rollout and GAE report.

Run from the project root:

    python scripts\\test_rollout.py

Prints the worked credit-assignment example, one real hand decision by decision,
and a summary of a small self-play batch.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch

from poker_arena.encoding import FEATURE_DIM, ObservationEncoder
from poker_arena.environment import ACTION_NAMES, make_environment
from poker_arena.networks import NUM_ACTIONS, PolicyValueNet
from poker_arena.rollout import (
    GAE_LAMBDA,
    GAMMA,
    RolloutBuffer,
    collect_trajectories,
    compute_gae,
)

DEMO_SEED = 11
BATCH_HANDS = 25


def _header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _deltas(rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
    next_values = np.concatenate([values[1:], [0.0]])
    return rewards + GAMMA * next_values - values


def main() -> int:
    print("Poker AI Arena - rollout collection and GAE report")
    print(f"Python  : {sys.version.split()[0]}")
    print(f"Torch   : {torch.__version__}")
    print(f"gamma   : {GAMMA:g}")
    print(f"lambda  : {GAE_LAMBDA:g}")

    # ------------------------------------------------------------------
    _header("[1] The worked example: three decisions, then a 3 bb loss")
    values = np.array([0.0, 1.0, 1.5])
    payoff = -3.0
    rewards = np.zeros(3)
    rewards[-1] = payoff

    delta = _deltas(rewards, values)
    at_zero, _ = compute_gae(rewards, values, lam=0.0)
    at_default, _ = compute_gae(rewards, values, lam=GAE_LAMBDA)
    at_one, returns = compute_gae(rewards, values, lam=1.0)

    values_desc = [
        "preflop call (value rose: good)",
        "flop check  (value rose: good)",
        "river fold  (value collapsed)",
    ]
    print(f"terminal payoff: {payoff:+.2f} bb\n")
    print(f"{'t':>2}  {'decision':<34} {'V(s_t)':>8} {'delta':>8} "
          f"{'lam=0':>8} {'lam=.95':>8} {'lam=1':>8}")
    for t in range(3):
        print(f"{t:>2}  {values_desc[t]:<34} {values[t]:>8.2f} {delta[t]:>8.2f} "
              f"{at_zero[t]:>8.2f} {at_default[t]:>8.2f} {at_one[t]:>8.2f}")
    print()
    print(f"naive R - V(s_t)          : "
          f"{np.array2string(payoff - values, precision=2)}")
    print(f"lam=1 reproduces it       : "
          f"{np.allclose(at_one, payoff - values, atol=1e-5)}")
    print(f"returns == advantages + V : "
          f"{np.allclose(returns, at_one + values, atol=1e-5)}")

    assert np.allclose(at_zero, [1.0, 0.5, -4.5], atol=1e-5)
    assert np.allclose(at_one, payoff - values, atol=1e-5)
    assert np.allclose(returns, at_one + values, atol=1e-5)

    # ------------------------------------------------------------------
    _header("[2] One real self-play hand, decision by decision")
    torch.manual_seed(0)
    net = PolicyValueNet()

    hand_env = None
    hand_encoder = None
    trajectories = []
    for offset in range(40):
        candidate = make_environment(seed=400 + offset)
        candidate_encoder = ObservationEncoder.from_environment(candidate)
        candidate_trajectories = collect_trajectories(
            candidate,
            candidate_encoder,
            net,
            hands=1,
            rng=np.random.default_rng(offset),
        )
        if len(candidate_trajectories) == 2:
            hand_env = candidate
            hand_encoder = candidate_encoder
            trajectories = candidate_trajectories
            break

    assert hand_env is not None and hand_encoder is not None
    assert len(trajectories) == 2

    print(f"payoffs by seat: "
          f"{[round(t.final_payoff, 2) for t in trajectories]} bb "
          f"(sum {sum(t.final_payoff for t in trajectories):+.2f})")
    for seat, trajectory in enumerate(trajectories):
        hand_rewards = trajectory.rewards()
        hand_values = trajectory.values()
        hand_delta = _deltas(hand_rewards, hand_values)
        hand_advantages, hand_returns = compute_gae(
            hand_rewards, hand_values, lam=GAE_LAMBDA
        )
        print(f"\nseat {seat}: {len(trajectory)} decisions, "
              f"payoff {trajectory.final_payoff:+.2f} bb")
        print(f"{'t':>2}  {'action':<6} {'V(s_t)':>8} {'reward':>8} "
              f"{'delta':>8} {'advantage':>10} {'return':>8}")
        for t, transition in enumerate(trajectory.transitions):
            print(f"{t:>2}  {ACTION_NAMES[transition.action]:<6} "
                  f"{hand_values[t]:>8.3f} {hand_rewards[t]:>8.3f} "
                  f"{hand_delta[t]:>8.3f} {hand_advantages[t]:>10.3f} "
                  f"{hand_returns[t]:>8.3f}")
        assert np.allclose(
            hand_returns, hand_advantages + hand_values, atol=1e-5
        )

    # ------------------------------------------------------------------
    _header(f"[3] A self-play batch from {BATCH_HANDS} hands")
    batch_env = make_environment(seed=DEMO_SEED)
    batch_encoder = ObservationEncoder.from_environment(batch_env)
    batch_trajectories = collect_trajectories(
        batch_env, batch_encoder, net, hands=BATCH_HANDS,
        rng=np.random.default_rng(7),
    )

    buffer = RolloutBuffer()
    for trajectory in batch_trajectories:
        buffer.add_trajectory(trajectory)
    batch = buffer.batch()

    illegal = sum(
        int(not batch.legal_masks[row, batch.actions[row]])
        for row in range(len(batch))
    )
    payoffs = np.array([t.final_payoff for t in batch_trajectories])

    print(f"trajectories        : {len(batch_trajectories)}")
    print(f"decisions           : {len(batch)}")
    print(f"features            : {batch.features.shape} "
          f"{batch.features.dtype}")
    print(f"legal masks         : {batch.legal_masks.shape} "
          f"{batch.legal_masks.dtype}")
    print(f"actions             : {batch.actions.shape} "
          f"{batch.actions.dtype}")
    print(f"actions per id      : "
          f"{ {ACTION_NAMES[i]: int((batch.actions == i).sum()) for i in range(NUM_ACTIONS)} }")
    print(f"stored illegal      : {illegal}")
    print(f"feature width       : {batch.features.shape[1]} "
          f"(expected {FEATURE_DIM})")
    print(f"payoff range        : {payoffs.min():+.1f} .. {payoffs.max():+.1f} bb")
    print(f"advantages mean/std : {batch.advantages.mean():+.4f} / "
          f"{batch.advantages.std():.4f}   (raw, not normalised)")
    print(f"returns identity    : "
          f"{np.allclose(batch.returns, batch.advantages + batch.values, atol=1e-5)}")
    print(f"finite              : "
          f"{bool(np.isfinite(batch.advantages).all() and np.isfinite(batch.returns).all())}")

    assert len(batch) == len(buffer) == sum(len(t) for t in batch_trajectories)
    assert batch.features.shape[1] == FEATURE_DIM
    assert illegal == 0
    assert np.allclose(batch.returns, batch.advantages + batch.values, atol=1e-5)
    assert np.isfinite(batch.advantages).all() and np.isfinite(batch.returns).all()

    # ------------------------------------------------------------------
    _header("[4] Reproducibility")
    def run_once():
        run_env = make_environment(seed=DEMO_SEED)
        run_encoder = ObservationEncoder.from_environment(run_env)
        torch.manual_seed(0)
        run_net = PolicyValueNet()
        return collect_trajectories(
            run_env, run_encoder, run_net, hands=5,
            rng=np.random.default_rng(3),
        )

    first = run_once()
    second = run_once()
    identical = len(first) == len(second) and all(
        len(a) == len(b)
        and [t.action for t in a.transitions] == [t.action for t in b.transitions]
        and np.allclose(a.values(), b.values())
        and a.final_payoff == b.final_payoff
        for a, b in zip(first, second)
    )
    print(f"same env seed, net seed and action seed -> identical: {identical}")
    print(f"trajectories per run: {len(first)}")
    assert identical

    # ------------------------------------------------------------------
    _header("[5] API summary")
    print("trajectories = collect_trajectories(env, encoder, net, hands=...)")
    print("  opponent=None  -> self-play, both seats recorded")
    print("  opponent=f     -> f plays player 1, only player 0 is recorded")
    print("buffer.add_trajectory(trajectory); batch = buffer.batch()")
    print("advantages, returns = compute_gae(rewards, values, lam=0.95)")
    print("normalised = normalize_advantages(batch.advantages)  # inside the PPO step")
    print("returns are raw: returns == advantages + values")

    print("\nAll rollout checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
