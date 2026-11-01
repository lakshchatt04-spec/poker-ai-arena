"""The PPO update for Agent A.

One rollout batch is reused for several epochs.  On the first epoch the policy
that samples and the policy being scored are the same network, so the
probability ratio

    ratio = exp(log_pi_new(a) - log_pi_old(a))

is exactly ``1`` and the clip cannot bind.  Clipping exists for the later
epochs: after the first gradient step the two policies differ, and the clip
stops a single batch from being over-exploited.  That only means anything if
``log_pi_old`` is a *fixed constant* read from the buffer.  If it is recomputed
from the current weights, ``ratio`` is always ``1``, the clip never binds, and
PPO silently becomes a plain policy gradient that can walk arbitrarily far.

The objective, clipped:

    L_policy = -mean( min(ratio * A, clamp(ratio, 1-eps, 1+eps) * A) )
    L_value  =  mean( (V(s) - return)^2 )
    L        =  L_policy - entropy_coef * H[pi] + value_coef * L_value

Advantages are normalised **once** per rollout, before the epochs start.  Doing
it per minibatch would change the objective between epochs and let the same
sample mean something different on every pass.  Returns stay raw: they are the
value targets, and rescaling them would change the value loss's scale.

The value loss is deliberately not clipped.  Clipping the value is a second
mechanism with its own bugs; it is worth adding only if the value loss is
observed to explode.  Global gradient-norm clipping is here from the start,
because one bad state can otherwise move all 100k weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .encoding import FEATURE_DIM
from .networks import NUM_ACTIONS, PolicyValueNet
from .rollout import RolloutBatch, normalize_advantages


@dataclass(frozen=True)
class PPOConfig:
    """Hyper-parameters for one PPO update.

    Defaults are the usual PPO values, chosen here for a 131-input, 100k-parameter
    network on short poker episodes.
    """

    #: Bound on how far ``ratio`` may leave 1 before the objective stops paying.
    clip_epsilon: float = 0.2
    #: Weight on the entropy bonus.  Small: enough to delay collapse, not enough
    #: to fight the policy gradient.
    entropy_coef: float = 0.01
    #: Weight on the value loss, kept near 0.5 so critic and actor learn together.
    value_coef: float = 0.5
    #: Passes over the same rollout batch.
    epochs: int = 4
    #: Rows per gradient step.  The last minibatch may be smaller.
    minibatch_size: int = 256
    #: Global gradient-norm limit applied before every optimiser step.
    max_grad_norm: float = 0.5
    #: Adam learning rate.
    learning_rate: float = 3e-4
    #: Normalise advantages once per rollout, before the epochs.
    normalize_advantages: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < float(self.clip_epsilon) < 1.0:
            raise ValueError(
                f"clip_epsilon must be in (0, 1), got {self.clip_epsilon!r}"
            )
        if float(self.entropy_coef) < 0.0:
            raise ValueError(
                f"entropy_coef must be non-negative, got {self.entropy_coef!r}"
            )
        if float(self.value_coef) < 0.0:
            raise ValueError(
                f"value_coef must be non-negative, got {self.value_coef!r}"
            )
        if int(self.epochs) < 1:
            raise ValueError(f"epochs must be positive, got {self.epochs!r}")
        if int(self.minibatch_size) < 1:
            raise ValueError(
                f"minibatch_size must be positive, got {self.minibatch_size!r}"
            )
        if float(self.max_grad_norm) <= 0.0:
            raise ValueError(
                f"max_grad_norm must be positive, got {self.max_grad_norm!r}"
            )
        if float(self.learning_rate) <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate!r}"
            )


class PPOUpdater:
    """Optimises a policy/value network on one :class:`RolloutBatch` at a time.

    Example::

        updater = PPOUpdater(net)
        buffer = RolloutBuffer()
        for trajectory in collect_trajectories(env, encoder, net, hands=200):
            buffer.add_trajectory(trajectory)
        stats = updater.step(buffer.batch())
        print(stats["approx_kl"], stats["clip_fraction"])
    """

    def __init__(
        self,
        net: PolicyValueNet,
        config: PPOConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.net = net
        self.config = PPOConfig() if config is None else config
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=float(self.config.learning_rate)
        )
        # Only used to shuffle minibatches.  A private generator keeps the update
        # reproducible without touching torch's global RNG.
        self._generator = (
            torch.Generator().manual_seed(int(seed)) if seed is not None else None
        )

    def step(self, batch: RolloutBatch) -> dict[str, float | int]:
        """Run ``config.epochs`` passes over ``batch`` and return diagnostics.

        The network is left in training mode.  ``log_probs``, ``advantages`` and
        ``returns`` are read once and never recomputed, so every epoch scores the
        same fixed data.

        Raises:
            ValueError: If the batch arrays disagree in length, are empty, contain
                non-finite values, or a feature/mask row has the wrong width.
        """

        _validate_batch(batch)

        advantages = (
            normalize_advantages(batch.advantages)
            if self.config.normalize_advantages
            else np.asarray(batch.advantages, dtype=np.float32)
        )

        features = torch.as_tensor(np.ascontiguousarray(batch.features))
        legal_masks = torch.as_tensor(np.ascontiguousarray(batch.legal_masks))
        actions = torch.as_tensor(batch.actions, dtype=torch.long)
        old_log_probs = torch.as_tensor(
            np.ascontiguousarray(batch.log_probs)
        )
        returns = torch.as_tensor(np.ascontiguousarray(batch.returns))
        advantages_tensor = torch.as_tensor(np.ascontiguousarray(advantages))

        count = int(batch.features.shape[0])
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
        }
        visited = 0
        minibatches = 0

        self.net.train()

        for _ in range(int(self.config.epochs)):
            order = torch.randperm(count, generator=self._generator).tolist()
            for start in range(0, count, int(self.config.minibatch_size)):
                rows = order[start : start + int(self.config.minibatch_size)]
                index = torch.as_tensor(rows, dtype=torch.long)
                weight = len(rows)

                logits, values = self.net(
                    features[index], legal_masks[index]
                )
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(actions[index])
                entropy = distribution.entropy().mean()

                # old_log_probs is a stored constant: this subtraction is what
                # makes the ratio a ratio rather than an identity.
                ratio = torch.exp(new_log_probs - old_log_probs[index])
                unclipped = ratio * advantages_tensor[index]
                clipped = (
                    ratio.clamp(
                        1.0 - float(self.config.clip_epsilon),
                        1.0 + float(self.config.clip_epsilon),
                    )
                    * advantages_tensor[index]
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (values - returns[index]).pow(2).mean()

                loss = (
                    policy_loss
                    - float(self.config.entropy_coef) * entropy
                    + float(self.config.value_coef) * value_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), float(self.config.max_grad_norm)
                )
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_probs - old_log_probs[index]
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > float(self.config.clip_epsilon)
                    ).float().mean()

                totals["policy_loss"] += float(policy_loss.detach()) * weight
                totals["value_loss"] += float(value_loss.detach()) * weight
                totals["entropy"] += float(entropy.detach()) * weight
                totals["total_loss"] += float(loss.detach()) * weight
                totals["approx_kl"] += float(approx_kl) * weight
                totals["clip_fraction"] += float(clip_fraction) * weight
                totals["grad_norm"] += float(grad_norm) * weight
                visited += weight
                minibatches += 1

        stats: dict[str, float | int] = {
            name: total / visited for name, total in totals.items()
        }
        stats["epochs"] = int(self.config.epochs)
        stats["minibatches"] = minibatches
        stats["samples"] = count
        return stats


def _validate_batch(batch: RolloutBatch) -> None:
    """Reject a batch that would misalign samples or poison the gradients."""

    count = int(batch.actions.shape[0])
    if count == 0:
        raise ValueError("cannot update on an empty batch")

    lengths = {
        "features": int(batch.features.shape[0]),
        "legal_masks": int(batch.legal_masks.shape[0]),
        "actions": count,
        "log_probs": int(batch.log_probs.shape[0]),
        "values": int(batch.values.shape[0]),
        "advantages": int(batch.advantages.shape[0]),
        "returns": int(batch.returns.shape[0]),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"batch arrays disagree in length: {lengths}")

    if batch.features.shape[1:] != (FEATURE_DIM,):
        raise ValueError(
            f"features must be (N, {FEATURE_DIM}), got {batch.features.shape}"
        )
    if batch.legal_masks.shape[1:] != (NUM_ACTIONS,):
        raise ValueError(
            f"legal_masks must be (N, {NUM_ACTIONS}), got "
            f"{batch.legal_masks.shape}"
        )
    if batch.legal_masks.dtype != np.bool_:
        raise ValueError(
            f"legal_masks must be bool, got {batch.legal_masks.dtype}"
        )

    for name in ("log_probs", "values", "advantages", "returns"):
        values = getattr(batch, name)
        if not np.isfinite(values).all():
            raise ValueError(f"batch.{name} contains a non-finite value")
