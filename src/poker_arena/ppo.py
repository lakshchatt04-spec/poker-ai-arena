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
    L_value  =  mean( ((V(s) - return) / std(return))^2 )
    L        =  L_policy - entropy_coef * H[pi] + value_coef * L_value

When the optional auxiliary coin head is used (Agent B only), the update adds
``aux_coef * BCE(aux_logit, coin)`` over the rows that carry a coin label.
Self-play rows have no opponent coin and are excluded.

Advantages are normalised **once** per rollout, before the epochs start.  Doing
it per minibatch would change the objective between epochs and let the same
sample mean something different on every pass.

The value targets are divided by their own rollout standard deviation, so
``L_value`` is the *fraction of return variance left unexplained*: ``0`` is a
perfect critic, ``1`` is a critic that predicts the mean.  ``explained_variance``
is the same number from the other end, ``1 - L_value``.

The first reason is that a raw MSE on returns is not comparable between
iterations: the returns' spread moves with the policy, so the same critic can
show a rising loss while it is getting *better*.  The first Agent A run rose
from ``7.95`` to ``128.07`` and back to ``60.57`` while the return standard
deviation went ``9.7 -> 7.9``; almost all of that shape was the target moving.

The second reason is the shared gradient.  Measured on checkpoint 50, the raw
value term was ``99.8 %`` of the pre-clip gradient norm (``5.07``) and the
policy term ``1.7 %`` (``0.087``), so the global clip at ``0.5`` was not an
emergency brake but the permanent operating point, rescaling every minibatch by
a factor set by the critic's error.  Adam is almost invariant to a *constant*
rescale -- it divides by a running RMS of each parameter's gradient -- but a
factor that jumps between minibatches perturbs the momentum and second-moment
estimates in a way that has nothing to do with the policy.  After this change
that batch reports a norm of ``0.48``, so the clip acts occasionally instead of
always.  Measured directly, clipping at ``0.5`` versus not clipping changes the
summed parameter update by about ``4 %``, which is the honest size of this
effect: it is a reason not to let the clip be permanently saturated, not a 10x
speed-up.

The value loss is deliberately not clipped.  Clipping the value is a second
mechanism with its own bugs; it is worth adding only if the value loss is
observed to explode.  Global gradient-norm clipping is here from the start,
because one bad state can otherwise move all 100k weights.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .encoding import FEATURE_DIM, preflop_rows
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
    #: Extra entropy weight applied to pre-flop decisions only.  Zero leaves the
    #: update identical to :attr:`entropy_coef`.  Raising it keeps the opening
    #: action stochastic, so the agent's own cards cannot reveal which action it
    #: took -- the point of the pre-flop-proxy experiment.
    entropy_coef_preflop: float = 0.0
    #: Weight on the value loss, kept near 0.5 so critic and actor learn together.
    value_coef: float = 0.5
    #: Divide the value targets by the rollout's return standard deviation, so
    #: the loss is a fraction of variance rather than a squared number of big
    #: blinds.  See the module docstring for why this also matters to the clip.
    normalize_value_loss: bool = True
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
    #: Weight on the auxiliary coin-prediction loss.  Zero is a no-op, and a
    #: caller that does not supply ``aux_labels`` ignores it.  Agent B's aux
    #: experiment sets this positive; Agent A never uses the term.
    aux_coef: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < float(self.clip_epsilon) < 1.0:
            raise ValueError(
                f"clip_epsilon must be in (0, 1), got {self.clip_epsilon!r}"
            )
        if float(self.entropy_coef) < 0.0:
            raise ValueError(
                f"entropy_coef must be non-negative, got {self.entropy_coef!r}"
            )
        if float(self.entropy_coef_preflop) < 0.0:
            raise ValueError(
                "entropy_coef_preflop must be non-negative, got "
                f"{self.entropy_coef_preflop!r}"
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
        if float(self.aux_coef) < 0.0:
            raise ValueError(
                f"aux_coef must be non-negative, got {self.aux_coef!r}"
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

        features = torch.as_tensor(np.ascontiguousarray(batch.features))
        legal_masks = torch.as_tensor(np.ascontiguousarray(batch.legal_masks))

        entropy_weights = preflop_rows(batch.features).astype(np.float64)

        def forward(index: Tensor) -> tuple[Tensor, Tensor]:
            # Agent A's network reads only the observation and the legal mask.
            return self.net(features[index], legal_masks[index])

        return run_ppo_update(
            net=self.net,
            config=self.config,
            optimizer=self.optimizer,
            generator=self._generator,
            count=int(batch.features.shape[0]),
            forward=forward,
            actions=batch.actions,
            log_probs=batch.log_probs,
            values=batch.values,
            advantages=batch.advantages,
            returns=batch.returns,
            entropy_weights=entropy_weights,
        )


def entropy_terms(
    entropy_by_row: Tensor, weights: Tensor | None = None
) -> tuple[Tensor, Tensor | None]:
    """Return the mean entropy and, when weighted, the marked rows' mean.

    ``weights`` marks the rows that receive the extra coefficient.  The weighted
    mean is normalised by the total weight, so the bonus does not shrink merely
    because a batch contains few marked rows.
    """

    mean = entropy_by_row.mean()
    if weights is None:
        return mean, None
    total = weights.sum()
    if float(total) <= 0.0:
        return mean, None
    return mean, (entropy_by_row * weights).sum() / total


def run_ppo_update(
    *,
    net: nn.Module,
    config: PPOConfig,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator | None,
    count: int,
    forward: Callable[
        [Tensor], tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]
    ],
    actions: np.ndarray,
    log_probs: np.ndarray,
    values: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
    entropy_weights: np.ndarray | None = None,
    aux_labels: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Run the clipped PPO update over one fixed batch of decisions.

    This is the optimiser core shared by Agent A and Agent B.  It is deliberately
    network-agnostic: the caller supplies ``forward(index)``, which maps a tensor
    of row indices to ``(logits, values)``.  Agent A closes over its observation
    batch; Agent B closes over observations *and* history tensors.  Everything
    that defines PPO -- the ratio, the clipped surrogate, the entropy bonus, the
    variance-scaled value loss, the diagnostics and the gradient clip -- lives
    here once, so an A/B experiment isolates the architecture rather than
    comparing two subtly different optimisers.

    Args:
        net: The module being optimised.  Switched to training mode.
        config: Hyper-parameters.
        optimizer: Owns the parameters of ``net``; its moment estimates persist
            across calls.
        generator: Optional private generator for minibatch shuffling.
        count: Number of rows in the batch.
        forward: Row-index -> ``(logits, values)``.
        actions: ``(count,)`` sampled action ids.
        log_probs: ``(count,)`` log-probabilities from the sampling forward pass.
        values: ``(count,)`` critic estimates from the sampling forward pass.
        advantages: ``(count,)`` raw advantages; normalised here if configured.
        returns: ``(count,)`` value targets.
        entropy_weights: Optional ``(count,)`` non-negative weights.  When given,
            ``entropy_coef_preflop`` multiplies the weighted mean entropy over
            the marked rows (the caller marks pre-flop decisions), instead of
            spreading the bonus over every decision.
        aux_labels: Optional ``(count,)`` integer coin labels in ``{0, 1}``.  A
            ``-1`` marks a decision with no label -- self-play hands have no
            opponent coin -- and those rows are excluded from the auxiliary
            loss.  Supplying this requires ``forward`` to return a third tensor,
            the auxiliary logits.

    Returns:
        Batch-mean diagnostics plus the critic's explained variance and the
        rollout's return standard deviation.  When ``aux_labels`` is supplied,
        ``aux_loss`` and ``aux_accuracy`` are included as well.
    """

    normalised_advantages = (
        normalize_advantages(advantages)
        if config.normalize_advantages
        else np.asarray(advantages, dtype=np.float32)
    )

    entropy_weights_tensor: Tensor | None = None
    if entropy_weights is not None:
        weights = np.asarray(entropy_weights, dtype=np.float64)
        if weights.shape != (int(count),):
            raise ValueError(
                f"entropy_weights must be ({int(count)},), got {weights.shape}"
            )
        if not np.isfinite(weights).all() or (weights < 0.0).any():
            raise ValueError(
                "entropy_weights must be finite and non-negative"
            )
        entropy_weights_tensor = torch.as_tensor(weights)

    aux_labels_tensor: Tensor | None = None
    labelled_rows = 0
    if aux_labels is not None:
        labels_array = np.asarray(aux_labels)
        if labels_array.shape != (int(count),):
            raise ValueError(
                f"aux_labels must be ({int(count)},), got {labels_array.shape}"
            )
        if not np.issubdtype(labels_array.dtype, np.integer):
            raise ValueError(
                f"aux_labels must be integers, got {labels_array.dtype}"
            )
        if not np.isin(labels_array, (-1, 0, 1)).all():
            raise ValueError("aux_labels must contain only -1, 0 or 1")
        labelled_rows = int((labels_array >= 0).sum())
        aux_labels_tensor = torch.as_tensor(labels_array.astype(np.int64))

    # Diagnostics about the critic that produced this rollout.  Measured
    # before any gradient step, so it describes the sampling policy's own
    # values rather than the ones being fitted right now.
    sampled_returns = np.asarray(returns, dtype=np.float64)
    sampled_values = np.asarray(values, dtype=np.float64)
    return_variance = float(sampled_returns.var())
    return_std = float(np.sqrt(return_variance))
    if return_variance > 0.0:
        explained_variance = 1.0 - float(
            np.mean((sampled_returns - sampled_values) ** 2)
        ) / return_variance
    else:
        # Every return is identical: there is nothing to explain, and a mean
        # predictor is already perfect.
        explained_variance = 0.0

    # A degenerate batch (all returns equal) has nothing to scale by.
    return_scale = (
        return_std
        if config.normalize_value_loss and return_std > 1e-8
        else 1.0
    )

    actions_tensor = torch.as_tensor(actions, dtype=torch.long)
    old_log_probs = torch.as_tensor(np.ascontiguousarray(log_probs))
    returns_tensor = torch.as_tensor(np.ascontiguousarray(returns))
    advantages_tensor = torch.as_tensor(
        np.ascontiguousarray(normalised_advantages)
    )

    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "entropy_preflop": 0.0,
        "total_loss": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "grad_norm": 0.0,
    }
    if aux_labels_tensor is not None:
        totals["aux_loss"] = 0.0
        totals["aux_accuracy"] = 0.0
    visited = 0
    minibatches = 0
    preflop_visited = 0
    aux_visited = 0

    net.train()

    for _ in range(int(config.epochs)):
        order = torch.randperm(int(count), generator=generator).tolist()
        for start in range(0, int(count), int(config.minibatch_size)):
            rows = order[start : start + int(config.minibatch_size)]
            index = torch.as_tensor(rows, dtype=torch.long)
            weight = len(rows)

            outputs = forward(index)
            if len(outputs) == 3:
                logits, batch_values, aux_logits = outputs
            else:
                logits, batch_values = outputs
                aux_logits = None
            if aux_labels_tensor is not None and aux_logits is None:
                raise ValueError(
                    "aux_labels were supplied but forward returned only "
                    "(logits, values); pass the auxiliary head's logits as a "
                    "third tensor"
                )
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions_tensor[index])
            entropy, entropy_preflop = entropy_terms(
                distribution.entropy(),
                None
                if entropy_weights_tensor is None
                else entropy_weights_tensor[index],
            )

            # old_log_probs is a stored constant: this subtraction is what
            # makes the ratio a ratio rather than an identity.
            ratio = torch.exp(new_log_probs - old_log_probs[index])
            unclipped = ratio * advantages_tensor[index]
            clipped = (
                ratio.clamp(
                    1.0 - float(config.clip_epsilon),
                    1.0 + float(config.clip_epsilon),
                )
                * advantages_tensor[index]
            )
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = (
                (batch_values - returns_tensor[index]) / return_scale
            ).pow(2).mean()

            entropy_bonus = float(config.entropy_coef) * entropy
            if entropy_preflop is not None:
                entropy_bonus = (
                    entropy_bonus
                    + float(config.entropy_coef_preflop) * entropy_preflop
                )

            loss = (
                policy_loss
                - entropy_bonus
                + float(config.value_coef) * value_loss
            )

            aux_loss = None
            aux_accuracy = 0.0
            aux_rows = 0
            if aux_labels_tensor is not None:
                row_labels = aux_labels_tensor[index]
                keep = row_labels >= 0
                aux_rows = int(keep.sum())
                if aux_rows:
                    targets = row_labels[keep].to(torch.float32)
                    scores = aux_logits[keep]  # type: ignore[index]
                    aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        scores, targets
                    )
                    with torch.no_grad():
                        aux_accuracy = float(
                            ((scores > 0.0) == targets.bool()).float().mean()
                        )
                    loss = loss + float(config.aux_coef) * aux_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                net.parameters(), float(config.max_grad_norm)
            )
            optimizer.step()

            with torch.no_grad():
                log_ratio = new_log_probs - old_log_probs[index]
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > float(config.clip_epsilon)
                ).float().mean()

            totals["policy_loss"] += float(policy_loss.detach()) * weight
            totals["value_loss"] += float(value_loss.detach()) * weight
            totals["entropy"] += float(entropy.detach()) * weight
            if entropy_preflop is not None:
                totals["entropy_preflop"] += (
                    float(entropy_preflop.detach()) * weight
                )
                preflop_visited += weight
            totals["total_loss"] += float(loss.detach()) * weight
            totals["approx_kl"] += float(approx_kl) * weight
            totals["clip_fraction"] += float(clip_fraction) * weight
            totals["grad_norm"] += float(grad_norm) * weight
            if aux_loss is not None:
                totals["aux_loss"] += float(aux_loss.detach()) * aux_rows
                totals["aux_accuracy"] += aux_accuracy * aux_rows
                aux_visited += aux_rows
            visited += weight
            minibatches += 1

    stats: dict[str, float | int] = {
        name: total / visited
        for name, total in totals.items()
        if name not in ("entropy_preflop", "aux_loss", "aux_accuracy")
    }
    if aux_labels_tensor is not None:
        stats["aux_loss"] = (
            totals["aux_loss"] / aux_visited if aux_visited else 0.0
        )
        stats["aux_accuracy"] = (
            totals["aux_accuracy"] / aux_visited if aux_visited else 0.0
        )
    if entropy_weights_tensor is not None:
        stats["entropy_preflop"] = (
            totals["entropy_preflop"] / preflop_visited
            if preflop_visited
            else 0.0
        )
        stats["preflop_fraction"] = float(entropy_weights_tensor.mean())
    stats["explained_variance"] = explained_variance
    stats["return_std"] = return_std
    stats["epochs"] = int(config.epochs)
    stats["minibatches"] = minibatches
    stats["samples"] = int(count)
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
