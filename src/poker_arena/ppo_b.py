"""Agent B's PPO update: the same clipped objective, one extra forward argument.

The only difference from Agent A's updater is what the network reads.  Everything
that defines PPO -- the ratio, the clipped surrogate, the entropy bonus, the
variance-scaled value loss, the diagnostics, the minibatch order and the global
gradient-norm clip -- lives in :func:`poker_arena.ppo.run_ppo_update`.  This class
supplies a forward callback that includes the history tensors.

Two PPO implementations would eventually disagree about something subtle, and an
A/B experiment with two optimisers would no longer isolate the architecture.  So
the duplication is avoided at the one place that matters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .encoding import FEATURE_DIM, preflop_rows
from .history import MAX_HISTORY_TOKENS, TOKEN_DIM
from .networks import NUM_ACTIONS
from .networks_b import HistoryPolicyValueNet, validate_history
from .ppo import PPOConfig, run_ppo_update
from .rollout_b import HistoryRolloutBatch


class HistoryPPOUpdater:
    """Optimises Agent B on one :class:`HistoryRolloutBatch` at a time.

    Example::

        updater = HistoryPPOUpdater(net)
        buffer = HistoryRolloutBuffer()
        for trajectory in collect_history_trajectories(env, encoder, net, hands=200):
            buffer.add_trajectory(trajectory)
        stats = updater.step(buffer.batch())
    """

    def __init__(
        self,
        net: HistoryPolicyValueNet,
        config: PPOConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.net = net
        self.config = PPOConfig() if config is None else config
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=float(self.config.learning_rate)
        )
        # Private generator for minibatch shuffling, so a seeded update is
        # reproducible without touching torch's global RNG.
        self._generator = (
            torch.Generator().manual_seed(int(seed)) if seed is not None else None
        )

    def step(self, batch: HistoryRolloutBatch) -> dict[str, float | int]:
        """Run ``config.epochs`` passes over ``batch`` and return diagnostics.

        Raises:
            ValueError: If the arrays disagree in length, the batch is empty,
                contains a non-finite value, or a history mask is not a prefix.
        """

        _validate_history_batch(batch)

        features = torch.as_tensor(np.ascontiguousarray(batch.features))
        tokens = torch.as_tensor(np.ascontiguousarray(batch.tokens))
        history_masks = torch.as_tensor(
            np.ascontiguousarray(batch.history_masks)
        )
        legal_masks = torch.as_tensor(np.ascontiguousarray(batch.legal_masks))

        # The buffer already checked shapes; this re-check is the one the GRU
        # actually depends on, after the numpy -> torch conversion.
        validate_history(tokens, history_masks)

        entropy_weights = preflop_rows(batch.features).astype(np.float64)

        def forward(index: Tensor) -> tuple[Tensor, Tensor]:
            return self.net(
                features[index],
                tokens[index],
                history_masks[index],
                legal_masks[index],
            )

        return run_ppo_update(
            net=self.net,
            config=self.config,
            optimizer=self.optimizer,
            generator=self._generator,
            count=int(batch.actions.shape[0]),
            forward=forward,
            actions=batch.actions,
            log_probs=batch.log_probs,
            values=batch.values,
            advantages=batch.advantages,
            returns=batch.returns,
            entropy_weights=entropy_weights,
        )


def _validate_history_batch(batch: HistoryRolloutBatch) -> None:
    """Reject a batch that would misalign decisions or reorder a history."""

    count = int(batch.actions.shape[0])
    if count == 0:
        raise ValueError("cannot update on an empty batch")

    lengths = {
        "features": int(batch.features.shape[0]),
        "tokens": int(batch.tokens.shape[0]),
        "history_masks": int(batch.history_masks.shape[0]),
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
    if batch.tokens.shape[1:] != (MAX_HISTORY_TOKENS, TOKEN_DIM):
        raise ValueError(
            f"tokens must be (N, {MAX_HISTORY_TOKENS}, {TOKEN_DIM}), got "
            f"{batch.tokens.shape}"
        )
    if batch.tokens.dtype != np.float32:
        raise ValueError(f"tokens must be float32, got {batch.tokens.dtype}")
    if batch.history_masks.shape[1:] != (MAX_HISTORY_TOKENS,):
        raise ValueError(
            f"history_masks must be (N, {MAX_HISTORY_TOKENS}), got "
            f"{batch.history_masks.shape}"
        )
    if batch.history_masks.dtype != np.bool_:
        raise ValueError(
            f"history_masks must be bool, got {batch.history_masks.dtype}"
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
