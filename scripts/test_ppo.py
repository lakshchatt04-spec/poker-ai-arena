"""PPO update report.

Run from the project root:

    python scripts\\test_ppo.py

Collects a real rollout, runs one update, and prints what changed: the losses,
the trust-region diagnostics, the entropy, and the value error.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from torch.distributions import Categorical

from poker_arena.encoding import FEATURE_DIM, ObservationEncoder
from poker_arena.environment import ACTION_NAMES, make_environment
from poker_arena.networks import NUM_ACTIONS, PolicyValueNet
from poker_arena.ppo import PPOConfig, PPOUpdater
from poker_arena.rollout import RolloutBatch, RolloutBuffer, collect_trajectories

DEMO_SEED = 11
#: Enough hands to exceed four 256-row minibatches per epoch.
BATCH_HANDS = 320


def _header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _new_net(seed: int = 0) -> PolicyValueNet:
    torch.manual_seed(seed)
    return PolicyValueNet()


def _clone(net: PolicyValueNet) -> PolicyValueNet:
    clone = PolicyValueNet()
    clone.load_state_dict(copy.deepcopy(net.state_dict()))
    return clone


def _batch_tensors(batch: RolloutBatch):
    return (
        torch.as_tensor(batch.features),
        torch.as_tensor(batch.legal_masks),
        torch.as_tensor(batch.actions),
        torch.as_tensor(batch.returns),
    )


def _measure(net: PolicyValueNet, batch: RolloutBatch) -> dict[str, float]:
    features, masks, actions, returns = _batch_tensors(batch)
    with torch.no_grad():
        logits, values = net(features, masks)
        distribution = Categorical(logits=logits)
        probabilities = torch.softmax(logits, dim=-1)
        return {
            "entropy": float(distribution.entropy().mean()),
            "value_loss": float((values - returns).pow(2).mean()),
            "mean_value": float(values.mean()),
            "max_illegal_probability": float(probabilities[~masks].abs().max()),
            "mean_log_prob": float(distribution.log_prob(actions).mean()),
        }


def main() -> int:
    print("Poker AI Arena - PPO update report")
    print(f"Python  : {sys.version.split()[0]}")
    print(f"Torch   : {torch.__version__}")

    config = PPOConfig()
    # ------------------------------------------------------------------
    _header("[1] Configuration")
    print(f"clip epsilon        : {config.clip_epsilon}")
    print(f"entropy coefficient : {config.entropy_coef}")
    print(f"value coefficient   : {config.value_coef}")
    print(f"epochs              : {config.epochs}")
    print(f"minibatch size      : {config.minibatch_size}")
    print(f"max gradient norm   : {config.max_grad_norm}")
    print(f"learning rate       : {config.learning_rate}")
    print(f"normalise advantages: {config.normalize_advantages}")

    assert config.clip_epsilon == 0.2
    assert config.value_coef == 0.5

    # ------------------------------------------------------------------
    _header(f"[2] A real self-play rollout: {BATCH_HANDS} hands")
    net = _new_net(0)
    env = make_environment(seed=DEMO_SEED)
    encoder = ObservationEncoder.from_environment(env)
    trajectories = collect_trajectories(
        env, encoder, net, hands=BATCH_HANDS, rng=np.random.default_rng(7)
    )

    buffer = RolloutBuffer()
    for trajectory in trajectories:
        buffer.add_trajectory(trajectory)
    batch = buffer.batch()

    payoffs = np.array([t.final_payoff for t in trajectories])
    print(f"trajectories        : {len(trajectories)}")
    print(f"decisions           : {len(batch)}")
    print(f"decisions per hand  : {len(batch) / BATCH_HANDS:.2f}")
    print(f"payoff range        : {payoffs.min():+.1f} .. {payoffs.max():+.1f} bb")
    print(f"raw advantage std   : {batch.advantages.std():.3f} bb")
    print(f"illegal stored      : "
          f"{int(sum(not batch.legal_masks[i, batch.actions[i]] for i in range(len(batch))))}")

    before = _measure(net, batch)
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in net.named_parameters()
    }
    print(f"\nentropy before      : {before['entropy']:.4f}")
    print(f"value loss before   : {before['value_loss']:.4f}")
    print(f"mean value before   : {before['mean_value']:+.4f} bb")

    # ------------------------------------------------------------------
    _header("[3] One update: 4 epochs over the same batch")
    stats = PPOUpdater(net, config, seed=0).step(batch)
    after = _measure(net, batch)

    print(f"minibatches         : {stats['minibatches']} "
          f"({stats['epochs']} epochs x {config.minibatch_size} rows)")
    print()
    print(f"{'metric':<24} {'value':>10}")
    for key in (
        "policy_loss",
        "value_loss",
        "entropy",
        "total_loss",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
    ):
        print(f"{key:<24} {stats[key]:>10.4f}")

    print()
    print(f"{'metric':<24} {'before':>10} {'after':>10} {'change':>10}")
    for key in ("entropy", "value_loss", "mean_value", "mean_log_prob"):
        print(f"{key:<24} {before[key]:>10.4f} {after[key]:>10.4f} "
              f"{after[key] - before[key]:>+10.4f}")

    parameter_delta = sum(
        float((parameter.detach() - parameters_before[name]).pow(2).sum())
        for name, parameter in net.named_parameters()
    ) ** 0.5
    print(f"\nparameter delta L2  : {parameter_delta:.4e}")
    print(f"illegal probability : {after['max_illegal_probability']:.3e} "
          "(must be exactly 0)")
    print(f"finite diagnostics  : "
          f"{all(np.isfinite(float(v)) for v in stats.values())}")

    assert parameter_delta > 0.0
    assert after["max_illegal_probability"] == 0.0
    assert all(np.isfinite(float(v)) for v in stats.values())
    assert stats["clip_fraction"] >= 0.0
    assert stats["approx_kl"] >= -1e-6

    # ------------------------------------------------------------------
    _header("[4] The trust region: fresh vs stale log-probabilities")
    fresh_net = _clone(net)
    stale_net = _clone(net)
    fresh_stats = PPOUpdater(
        fresh_net, PPOConfig(epochs=1, minibatch_size=len(batch)), seed=0
    ).step(batch)

    stale_batch = RolloutBatch(
        features=batch.features,
        legal_masks=batch.legal_masks,
        actions=batch.actions,
        log_probs=(batch.log_probs + 5.0).astype(np.float32),
        values=batch.values,
        advantages=batch.advantages,
        returns=batch.returns,
    )
    stale_stats = PPOUpdater(
        stale_net, PPOConfig(epochs=1, minibatch_size=len(batch)), seed=0
    ).step(stale_batch)

    print(f"fresh log-probs     : clip fraction "
          f"{fresh_stats['clip_fraction']:.4f}, "
          f"approx KL {fresh_stats['approx_kl']:.2e}")
    print(f"stale log-probs     : clip fraction "
          f"{stale_stats['clip_fraction']:.4f}, "
          f"approx KL {stale_stats['approx_kl']:.4f}")
    print("old log-probs are constants: on the first pass the ratio is exactly 1,")
    print("so the clip cannot bind; a stale store pushes every sample outside it.")

    assert fresh_stats["clip_fraction"] == 0.0
    assert stale_stats["clip_fraction"] == 1.0

    # ------------------------------------------------------------------
    _header("[5] Reproducibility")
    first_net = _clone(net)
    second_net = _clone(net)
    PPOUpdater(first_net, PPOConfig(epochs=2), seed=3).step(batch)
    PPOUpdater(second_net, PPOConfig(epochs=2), seed=3).step(batch)
    identical = all(
        torch.equal(left, right)
        for left, right in zip(
            first_net.parameters(), second_net.parameters()
        )
    )
    print(f"same weights, batch and minibatch seed -> identical: {identical}")
    assert identical

    # ------------------------------------------------------------------
    _header("[6] API summary")
    print("updater = PPOUpdater(net, PPOConfig(), seed=0)")
    print("stats = updater.step(buffer.batch())")
    print("stats keys: policy_loss, value_loss, entropy, total_loss,")
    print("            approx_kl, clip_fraction, grad_norm, epochs,")
    print("            minibatches, samples")
    print("advantages are normalised once per rollout; returns stay raw")
    print("log_probs from the buffer are treated as constants")

    print("\nAll PPO checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
