"""Agent B's policy/value network: Agent A's trunk plus a GRU over the history.

Agent A reads 131 numbers that describe *what is true now*: cards, pot, chips to
call, raise counts.  Those numbers cannot say who bet, in what order, or whether
the opponent has check-raised twice this street.  ``history.py`` turns each
completed action into a 16-number token, and this module reads the resulting
sequence with a GRU.

Structure of one forward pass::

    features (B, 131)                      tokens (B, 32, 16), mask (B, 32)
        |                                        |
    Linear(131, 256) -> ReLU                GRU(16, 64), packed by length
        |                                        |
    Linear(256, 256) -> ReLU                last hidden state (B, 64)
        |                                        |
        +---------------- concat ----------------+
                            |
                  (B, 320) -> Linear(320, 4) -> masked_fill(~legal, -1e9)
                            |
                  (B, 320) -> Linear(320, 1) -> value

Why the trunk is *identical* to Agent A's: the experiment is "does adding memory
of the action sequence help?", not "does a bigger network help?".  Keeping the
131-number path byte-for-byte the same means the only architectural difference is
the 64-number side channel.  The policy and value heads grow from 256 to 320
inputs, so B has 116,933 parameters against A's 100,869 -- ``+16 %``, all of it
attached to the memory channel.  That asymmetry is reported, not hidden: a gain
for B is evidence that memory is useful, but it is not proof that memory is the
*only* cause, because B also has slightly more head capacity.

Why a GRU and not a transformer or an MLP over a flat window: 32 tokens is an
episode, not a corpus, and the useful signal is "what changed recently".  A GRU
is causal by construction, has 15,744 parameters, and the longest measured hand
is 20 actions, so it never needs long-range tricks.  A transformer would add
complexity without a question it answers better here.

Why packed sequences: a padded row after ``length`` is a real zero vector, and a
GRU fed a zero vector still updates its hidden state through its biases.  Running
over all 32 rows would therefore make the last-token state depend on how much
padding was appended -- the same hand encoded with 32 rows and with 8 rows would
produce different memory.  Packing stops the recurrence at each sample's real
length.  An empty history (the first decision of a hand) bypasses the GRU
entirely and returns the zero hidden state.

Why the mask is applied here: identical reasoning to Agent A.  RLCard silently
substitutes an illegal action (check if possible, otherwise fold), so sampling an
illegal id records an action that never happened.  The masked logits are the only
place that can be prevented.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from .encoding import FEATURE_DIM
from .environment import ACTION_NAMES
from .history import MAX_HISTORY_TOKENS, TOKEN_DIM
from .networks import HIDDEN_WIDTH, MASKED_LOGIT, NUM_ACTIONS

#: Recurrent width.  64 is two 16-number tokens' worth of room per state update:
#: enough to carry "who raised how often" without letting the side channel
#: dominate the 256-number observation trunk.
HISTORY_HIDDEN = 64

#: Gain for the orthogonal recurrent initialisation.  ``1.0`` is the standard
#: choice for GRUs; the observation trunk keeps Agent A's ``sqrt(2)``.
HISTORY_GAIN = 1.0
POLICY_GAIN = 0.01
VALUE_GAIN = 1.0


def validate_history(tokens: Tensor, mask: Tensor) -> None:
    """Reject a history that the packed GRU would silently misread.

    The mask must be a **prefix**: some number of ``True`` values followed by
    ``False`` values.  ``length = mask.sum(1)`` is only the real sequence length
    under that assumption, and a hole in the middle would be packed as if the
    later actions came earlier.  ``HistoryTracker.tokens`` always produces a
    prefix, so a violation means a hand-rolled caller has corrupted the order.
    """

    if tokens.ndim != 3:
        raise ValueError(
            f"tokens must be (batch, max_tokens, token_dim), got "
            f"{tuple(tokens.shape)}"
        )
    if tokens.shape[2] != TOKEN_DIM:
        raise ValueError(
            f"tokens must have width {TOKEN_DIM}, got {tokens.shape[2]}"
        )
    if tokens.dtype != torch.float32:
        raise ValueError(f"tokens must be float32, got {tokens.dtype}")
    if mask.dtype != torch.bool:
        raise ValueError(f"history mask must be bool, got {mask.dtype}")
    if tuple(mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError(
            f"history mask must be {tuple(tokens.shape[:2])}, got "
            f"{tuple(mask.shape)}"
        )

    length = mask.sum(dim=1)
    expected = (
        torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
        < length.unsqueeze(1)
    )
    if not bool(torch.equal(mask, expected)):
        raise ValueError(
            "history mask must mark a prefix of real tokens (True before "
            "False); padding in the middle would reorder the actions"
        )


class HistoryEncoder(nn.Module):
    """Reads ``(batch, max_tokens, 16)`` tokens into one ``(batch, 64)`` state."""

    def __init__(
        self,
        token_dim: int = TOKEN_DIM,
        hidden: int = HISTORY_HIDDEN,
        max_tokens: int = MAX_HISTORY_TOKENS,
    ) -> None:
        super().__init__()

        if int(token_dim) < 1:
            raise ValueError(f"token_dim must be positive, got {token_dim!r}")
        if int(hidden) < 1:
            raise ValueError(f"hidden must be positive, got {hidden!r}")
        if int(max_tokens) < 1:
            raise ValueError(f"max_tokens must be positive, got {max_tokens!r}")

        self.token_dim = int(token_dim)
        self.hidden = int(hidden)
        self.max_tokens = int(max_tokens)
        self.gru = nn.GRU(self.token_dim, self.hidden, batch_first=True)

        for name, parameter in self.gru.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(parameter, gain=HISTORY_GAIN)
            else:
                nn.init.zeros_(parameter)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        """Return the hidden state after each sample's last real token.

        Rows with an empty history get the zero state, which is also where the
        GRU starts.  The output is independent of how many padding rows follow
        the real tokens.
        """

        validate_history(tokens, mask)
        if tokens.shape[1] > self.max_tokens:
            raise ValueError(
                f"history has {tokens.shape[1]} rows, the encoder was built "
                f"for at most {self.max_tokens}"
            )

        lengths = mask.sum(dim=1)
        output = torch.zeros(
            tokens.shape[0], self.hidden, dtype=tokens.dtype, device=tokens.device
        )

        nonempty = lengths > 0
        if bool(nonempty.any()):
            # ``pack_padded_sequence`` wants CPU lengths in PyTorch's CPU build
            # regardless of where the tensors live.
            packed = pack_padded_sequence(
                tokens[nonempty],
                lengths[nonempty].cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, final = self.gru(packed)
            output[nonempty] = final[-1]

        return output


class HistoryPolicyValueNet(nn.Module):
    """Agent A's trunk with a GRU side channel over the betting history.

    Example::

        net = HistoryPolicyValueNet()
        logits, value = net(features, tokens, history_mask, legal_mask)

    Args:
        feature_dim: Width of the encoded observation.  Must match the encoder.
        hidden: Width of both observation-trunk layers and of the heads' input
            is ``hidden + history_hidden``.
        history_hidden: Width of the GRU state.
        num_actions: Must equal RLCard's action count.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden: int = HIDDEN_WIDTH,
        history_hidden: int = HISTORY_HIDDEN,
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
        self.history_hidden = int(history_hidden)
        self.num_actions = int(num_actions)

        self.trunk = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )
        self.history = HistoryEncoder(hidden=self.history_hidden)

        # The heads read the observation state and the memory state together.
        # There is no extra fusion layer on purpose: adding one would give B a
        # second full-width block of parameters and make the A/B difference
        # mostly "B is bigger" rather than "B can remember".
        head_input = self.hidden + self.history_hidden
        self.policy_head = nn.Linear(head_input, self.num_actions)
        self.value_head = nn.Linear(head_input, 1)

        self._orthogonal_init(self.trunk[0], math.sqrt(2.0))
        self._orthogonal_init(self.trunk[2], math.sqrt(2.0))
        self._orthogonal_init(self.policy_head, POLICY_GAIN)
        self._orthogonal_init(self.value_head, VALUE_GAIN)

    @staticmethod
    def _orthogonal_init(layer: nn.Linear, gain: float) -> None:
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    def forward(
        self,
        features: Tensor,
        tokens: Tensor,
        history_mask: Tensor,
        legal_mask: Tensor,
        *,
        use_history: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Score states and estimate their values.

        Args:
            features: ``(B, feature_dim)`` observation vectors.
            tokens: ``(B, T, 16)`` history tokens.
            history_mask: ``(B, T)`` bool prefix mask.
            legal_mask: ``(B, num_actions)`` bool, ``True`` where legal.
            use_history: When ``False`` the GRU output is replaced by zeros and
                the rest of the network is unchanged.  This is the ablation that
                separates "memory helps" from "these particular weights are
                better": the same trained network is evaluated with and without
                the side channel.

        Returns:
            ``(logits, value)``.  Illegal logits are :data:`MASKED_LOGIT`.
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
        if tokens.shape[0] != features.shape[0]:
            raise ValueError(
                f"history batch {tokens.shape[0]} does not match features batch "
                f"{features.shape[0]}"
            )

        observation = self.trunk(features)
        memory = self.history(tokens, history_mask)
        if not use_history:
            memory = torch.zeros_like(memory)

        hidden = torch.cat((observation, memory), dim=1)
        logits = self.policy_head(hidden).masked_fill(~legal_mask, MASKED_LOGIT)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value

    def count_parameters(self) -> int:
        """Number of trainable parameters, for reports and A/B comparisons."""

        return sum(parameter.numel() for parameter in self.parameters())
