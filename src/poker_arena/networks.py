"""Agent A's policy/value network: 131 numbers in, four scores and one value out.

One shared trunk reads the encoded state; two small heads read the trunk.  The
policy head scores each of RLCard's four actions, the value head estimates how
good the state is for the player to act.  Sharing the trunk is deliberate: the
things that make a hand strong and the things that make an action good are the
same features, and 100k parameters is small enough that a second trunk would
mostly add noise.

Layout of one forward pass::

    features (B, 131) float32
        |
        +-- Linear(131, 256) -> ReLU -> Linear(256, 256) -> ReLU   shared trunk
                |                                    |
        Linear(256, 4)                        Linear(256, 1)
                |                                    |
        masked_fill(~legal, -1e9)                  squeeze
                |                                    |
           logits (B, 4)                          value (B,)

Why the mask is applied here instead of by the caller: a rollout touches the mask
three times (sampling, the PPO ratio, evaluation) and every forgotten one fails
silently.  RLCard substitutes an illegal action rather than raising
(``envs/limitholdem.py:90-96``: play ``check`` if possible, otherwise **fold**),
so a policy that samples ``raise`` where raising is capped out actually folds the
hand while the buffer records a raise.  Measured on an untrained net: 25% of
cap-state samples landed on an illegal action.  There is no error message for
any of it, so the mask is not something to remember.

Why ``-1e9`` and not ``-inf``: both give an illegal action probability exactly
0 and both leave the chosen action's log-probability untouched, but ``-inf``
makes the entropy ``0 * -inf = NaN``, and one NaN poisons every gradient in the
batch.  ``-1e9`` underflows to the same zero probability while keeping every
value finite.

Why logits and not probabilities: PPO needs ``log_prob`` and ``entropy``, and
``log_softmax`` is more stable than ``log(probs)``.  The network deliberately
does not sample, does not compute a loss and does no bookkeeping -- that is the
rollout's job.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .encoding import FEATURE_DIM
from .environment import ACTION_NAMES

NUM_ACTIONS = len(ACTION_NAMES)

#: Replaces every illegal logit.  Large enough that ``exp()`` underflows to 0,
#: finite so that entropy and gradients stay well defined.
MASKED_LOGIT = -1e9

#: Width of both hidden layers.
HIDDEN_WIDTH = 256

#: Strategy standard deviations for the orthogonal initialisation.  The small
#: policy-head gain keeps the first policy near-uniform instead of committing to
#: one action on random weights; the value head starts at unit scale.
TRUNK_GAIN = math.sqrt(2.0)
POLICY_GAIN = 0.01
VALUE_GAIN = 1.0


class PolicyValueNet(nn.Module):
    """Shared-trunk policy/value network for Agent A.

    Example::

        net = PolicyValueNet()
        logits, value = net(features, legal_mask)   # (B, 4), (B,)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

    Args:
        feature_dim: Width of the encoded state vector.  Must match the encoder.
        hidden: Width of both hidden layers.
        num_actions: Must equal the number of RLCard actions, because action ids
            are used directly as indices into the policy head's output.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden: int = HIDDEN_WIDTH,
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()

        if int(feature_dim) < 1:
            raise ValueError(f"feature_dim must be positive, got {feature_dim!r}")
        if int(hidden) < 1:
            raise ValueError(f"hidden must be positive, got {hidden!r}")
        if int(num_actions) != NUM_ACTIONS:
            raise ValueError(
                f"num_actions must match RLCard's {NUM_ACTIONS} actions "
                f"{ACTION_NAMES}, got {num_actions!r}"
            )

        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.num_actions = int(num_actions)

        self.trunk = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(self.hidden, self.num_actions)
        self.value_head = nn.Linear(self.hidden, 1)

        self._orthogonal_init(self.trunk[0], TRUNK_GAIN)
        self._orthogonal_init(self.trunk[2], TRUNK_GAIN)
        self._orthogonal_init(self.policy_head, POLICY_GAIN)
        self._orthogonal_init(self.value_head, VALUE_GAIN)

    @staticmethod
    def _orthogonal_init(layer: nn.Linear, gain: float) -> None:
        """Orthogonal weights and zero bias, the standard PPO initialisation."""

        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    def forward(self, features: Tensor, legal_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Score the states in ``features`` and estimate their values.

        Args:
            features: ``(B, feature_dim)`` float tensor from the encoder.
            legal_mask: ``(B, num_actions)`` bool tensor, ``True`` where the
                action is legal.  Required -- there is no unmasked path.

        Returns:
            ``(logits, value)`` where ``logits`` is ``(B, num_actions)`` with
            illegal entries replaced by :data:`MASKED_LOGIT`, and ``value`` is
            ``(B,)``.

        Raises:
            ValueError: If the shapes disagree or a row has no legal action.
        """

        if features.ndim != 2:
            raise ValueError(
                f"features must be (batch, feature_dim), got shape "
                f"{tuple(features.shape)}"
            )
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected {self.feature_dim} features, got {features.shape[1]}"
            )
        if legal_mask.dtype != torch.bool:
            raise ValueError(f"legal_mask must be bool, got {legal_mask.dtype}")
        if tuple(legal_mask.shape) != (features.shape[0], self.num_actions):
            raise ValueError(
                f"legal_mask must be {(features.shape[0], self.num_actions)}, "
                f"got {tuple(legal_mask.shape)}"
            )
        if not bool(legal_mask.any(dim=1).all()):
            raise ValueError(
                "every row needs at least one legal action; an all-False row "
                "would silently produce a uniform distribution"
            )

        hidden = self.trunk(features)
        logits = self.policy_head(hidden).masked_fill(~legal_mask, MASKED_LOGIT)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value

    def count_parameters(self) -> int:
        """Number of trainable parameters, for reports and architecture tests."""

        return sum(parameter.numel() for parameter in self.parameters())
