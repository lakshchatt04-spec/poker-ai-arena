"""Agent A policy/value network smoke test.

Run from the project root:

    python scripts\\test_networks.py

Prints the architecture, one real forward pass, the effect of the legal-action
mask, and one PPO-style backward pass.
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
from torch.distributions import Categorical

from poker_arena.encoding import FEATURE_DIM, ObservationEncoder, small_blind_player
from poker_arena.environment import ACTION_NAMES, make_environment
from poker_arena.networks import (
    HIDDEN_WIDTH,
    MASKED_LOGIT,
    NUM_ACTIONS,
    PolicyValueNet,
)

DEMO_SEED = 11
N_SAMPLES = 10_000


def _header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    print("Poker AI Arena - Agent A policy/value network smoke test")
    print(f"Python  : {sys.version.split()[0]}")
    print(f"Torch   : {torch.__version__}")

    torch.manual_seed(0)
    env = make_environment(seed=DEMO_SEED)
    encoder = ObservationEncoder.from_environment(env)

    # ------------------------------------------------------------------
    _header("[1] Architecture")
    net = PolicyValueNet()
    print(f"input               : {FEATURE_DIM} numbers")
    print(f"trunk               : {FEATURE_DIM} -> {HIDDEN_WIDTH} -> {HIDDEN_WIDTH} (ReLU)")
    print(f"policy head         : {HIDDEN_WIDTH} -> {NUM_ACTIONS}   (no softmax)")
    print(f"value head          : {HIDDEN_WIDTH} -> 1")
    print(f"parameters          : {net.count_parameters():,}")
    print(f"  trunk             : {sum(p.numel() for p in net.trunk.parameters()):,}")
    print(f"  policy head       : "
          f"{sum(p.numel() for p in net.policy_head.parameters()):,}")
    print(f"  value head        : "
          f"{sum(p.numel() for p in net.value_head.parameters()):,}")
    print(f"mask value          : {MASKED_LOGIT:g}")
    assert net.count_parameters() == 100_869
    assert NUM_ACTIONS == len(ACTION_NAMES) == 4

    # ------------------------------------------------------------------
    _header("[2] One real forward pass")
    state, player_id = env.reset()
    small_blind = small_blind_player(state)
    encoded = encoder.encode(state, is_small_blind=(player_id == small_blind))

    features = torch.from_numpy(encoded.features).unsqueeze(0)
    masks = torch.from_numpy(encoded.legal_mask).unsqueeze(0)

    with torch.no_grad():
        logits, value = net(features, masks)
        probabilities = torch.softmax(logits, dim=-1)

    raw = state["raw_obs"]
    print(f"acting player       : P{player_id} "
          f"(small blind: {player_id == small_blind})")
    print(f"hand / board        : {raw['hand']} / {raw['public_cards'] or '-'}")
    print(f"chips               : {raw['all_chips']}")
    print()
    print(f"{'action':<8} {'legal':>6} {'logit':>12} {'prob':>8}")
    for index, name in enumerate(ACTION_NAMES):
        print(f"{name:<8} {str(bool(masks[0, index])):>6} "
              f"{logits[0, index].item():>12.4f} {probabilities[0, index].item():>8.4f}")
    print(f"\nvalue               : {value.item():.4f}")
    print(f"probabilities sum   : {probabilities.sum().item():.6f}")

    assert torch.allclose(probabilities.sum(), torch.tensor(1.0))
    assert torch.all(probabilities[~masks] == 0.0)
    assert value.shape == (1,)

    # ------------------------------------------------------------------
    _header("[3] A capped state: masked vs unmasked sampling")
    walk_env = make_environment(seed=DEMO_SEED)
    walk_encoder = ObservationEncoder.from_environment(walk_env)
    cap_state, cap_player = walk_env.reset()
    cap_small = small_blind_player(cap_state)
    while 1 in [int(action) for action in cap_state["legal_actions"]]:
        cap_state, cap_player = walk_env.step(1)

    cap = walk_encoder.encode(
        cap_state, is_small_blind=(cap_player == cap_small)
    )
    cap_features = torch.from_numpy(cap.features).unsqueeze(0)
    cap_masks = torch.from_numpy(cap.legal_mask).unsqueeze(0)

    with torch.no_grad():
        masked_logits, _ = net(cap_features, cap_masks)
        unmasked_logits = net.policy_head(net.trunk(cap_features))
        masked_probs = torch.softmax(masked_logits, dim=-1)
        unmasked_probs = torch.softmax(unmasked_logits, dim=-1)
        unmasked_samples = Categorical(logits=unmasked_logits).sample((N_SAMPLES,))
        masked_samples = Categorical(logits=masked_logits).sample((N_SAMPLES,))

    print(f"legal actions       : "
          f"{[ACTION_NAMES[i] for i in torch.nonzero(cap_masks[0]).flatten().tolist()]}")
    print(f"raise is legal      : {bool(cap_masks[0, 1])}")
    print()
    print(f"{'':<10} {'unmasked':>10} {'masked':>12}")
    for index, name in enumerate(ACTION_NAMES):
        print(f"{name:<10} {unmasked_probs[0, index].item():>10.4f} "
              f"{masked_probs[0, index].item():>12.4f}")
    print()
    print(f"unmasked draws of raise : "
          f"{int((unmasked_samples == 1).sum())} / {N_SAMPLES} "
          "(rlcard would play fold instead)")
    print(f"masked draws of raise   : "
          f"{int((masked_samples == 1).sum())} / {N_SAMPLES}")
    print(f"masked logit            : {masked_logits[0, 1].item():g}")

    assert not bool(cap_masks[0, 1])
    assert int((masked_samples == 1).sum()) == 0
    assert masked_logits[0, 1].item() == MASKED_LOGIT

    # ------------------------------------------------------------------
    _header("[4] One PPO-style backward pass")
    collect_env = make_environment(seed=5)
    collect_encoder = ObservationEncoder.from_environment(collect_env)
    rng = np.random.default_rng(0)

    batch_features, batch_masks = [], []
    while len(batch_features) < 64:
        state, player_id = collect_env.reset()
        small_blind = small_blind_player(state)
        while not collect_env.is_over() and len(batch_features) < 64:
            item = collect_encoder.encode(
                state, is_small_blind=(player_id == small_blind)
            )
            batch_features.append(item.features)
            batch_masks.append(item.legal_mask)
            legal = list(state["legal_actions"])
            state, player_id = collect_env.step(int(rng.choice(legal)))

    batch_features = torch.from_numpy(np.stack(batch_features))
    batch_masks = torch.from_numpy(np.stack(batch_masks))

    net.zero_grad()
    logits, value = net(batch_features, batch_masks)
    distribution = Categorical(logits=logits)
    actions = distribution.sample()
    advantages = torch.randn(len(batch_features))

    log_probs = distribution.log_prob(actions)
    entropy = distribution.entropy().mean()
    returns = torch.randn(len(batch_features))

    loss = (
        -(log_probs * advantages).mean()
        - 0.01 * entropy
        + 0.5 * (value - returns).pow(2).mean()
    )
    loss.backward()

    print(f"batch               : {tuple(batch_features.shape)}")
    print(f"loss                : {loss.item():.4f}")
    print(f"entropy             : {entropy.item():.4f}")
    print(f"log-probs finite    : {bool(torch.isfinite(log_probs).all())}")
    print(f"entropy finite      : {bool(torch.isfinite(entropy))}")
    print()
    print(f"{'layer':<18} {'|grad| mean':>12} {'|grad| max':>12}")
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        print(f"{name:<18} {parameter.grad.abs().mean().item():>12.3e} "
              f"{parameter.grad.abs().max().item():>12.3e}")

    assert torch.isfinite(loss)

    # ------------------------------------------------------------------
    _header("[5] API summary")
    print("net = PolicyValueNet()                     # 100,869 parameters")
    print("logits, value = net(features, legal_mask)  # (B,4) masked, (B,)")
    print("dist = torch.distributions.Categorical(logits=logits)")
    print("action = dist.sample(); log_prob = dist.log_prob(action)")
    print("the mask is applied inside forward(); there is no unmasked path")
    print("illegal logits get -1e9, so their gradient is exactly zero")

    print("\nAll network checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
