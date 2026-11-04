"""Agent B history-aware network smoke test.

Run from the project root:

    python scripts\\test_agent_b.py

Prints the architecture and its size relative to Agent A, one real decision with
the tokens that led to it, the padding/empty-history behaviour, the history
ablation, and one PPO-style backward pass.
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
from poker_arena.history import (
    ACTION_SLICE,
    MAX_HISTORY_TOKENS,
    STREET_NAMES,
    TOKEN_DIM,
    HistoryTracker,
)
from poker_arena.networks import HIDDEN_WIDTH, MASKED_LOGIT, NUM_ACTIONS, PolicyValueNet
from poker_arena.networks_b import HISTORY_HIDDEN, HistoryPolicyValueNet

DEMO_SEED = 11
N_SAMPLES = 10_000


def _header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _describe_token(token: np.ndarray, index: int) -> str:
    actor = "P0" if token[0] else "P1"
    action = ACTION_NAMES[int(np.argmax(token[ACTION_SLICE]))]
    street = STREET_NAMES[int(np.argmax(token[7:11]))]
    return f"{index:>2}. {actor} {action:<5} {street}"


def main() -> int:
    print("Poker AI Arena - Agent B history-aware network smoke test")
    print(f"Python  : {sys.version.split()[0]}")
    print(f"Torch   : {torch.__version__}")

    torch.manual_seed(0)
    env = make_environment(seed=DEMO_SEED)
    encoder = ObservationEncoder.from_environment(env)

    # ------------------------------------------------------------------
    _header("[1] Architecture and size relative to Agent A")
    net = HistoryPolicyValueNet()
    agent_a = PolicyValueNet()
    print(f"observation trunk   : {FEATURE_DIM} -> {HIDDEN_WIDTH} -> {HIDDEN_WIDTH} (ReLU)")
    print(f"history GRU         : {TOKEN_DIM} -> {HISTORY_HIDDEN}, packed by length")
    print(f"heads               : {HIDDEN_WIDTH + HISTORY_HIDDEN} -> {NUM_ACTIONS} / 1")
    print(f"parameters          : {net.count_parameters():,}")
    print(f"  trunk             : {sum(p.numel() for p in net.trunk.parameters()):,}")
    print(f"  history GRU       : "
          f"{sum(p.numel() for p in net.history.parameters()):,}")
    print(f"  policy head       : "
          f"{sum(p.numel() for p in net.policy_head.parameters()):,}")
    print(f"  value head        : "
          f"{sum(p.numel() for p in net.value_head.parameters()):,}")
    print(f"Agent A parameters  : {agent_a.count_parameters():,}")
    print(f"B / A               : "
          f"{net.count_parameters() / agent_a.count_parameters():.2f}x "
          f"(+{net.count_parameters() - agent_a.count_parameters():,} weights)")

    assert net.count_parameters() == 116_933
    assert agent_a.count_parameters() == 100_869
    assert net.hidden == agent_a.hidden

    # ------------------------------------------------------------------
    _header("[2] One real decision with its history")
    # Walk to a live mid-hand state so there is something to remember.  A fold
    # can end the hand early, so retry a different deal until at least three
    # actions have happened and the state is still live.
    for demo_seed in range(DEMO_SEED, DEMO_SEED + 20):
        demo_env = make_environment(seed=demo_seed)
        demo_encoder = ObservationEncoder.from_environment(demo_env)
        state, player_id = demo_env.reset()
        small_blind = small_blind_player(state)
        tracker = HistoryTracker()
        tracker.start_hand(state)
        rng = np.random.default_rng(demo_seed)
        while not demo_env.is_over() and tracker.length < 3:
            action = int(rng.choice(sorted(state["legal_actions"])))
            tracker.record(state, player_id, demo_env.actions[action])
            state, player_id = demo_env.step(action)
        if not demo_env.is_over() and tracker.length >= 3:
            env, encoder = demo_env, demo_encoder
            break
    else:  # pragma: no cover - would mean every deal folded instantly
        raise RuntimeError("could not reach a mid-hand state")

    sequence = tracker.tokens(viewer=player_id)
    encoded = encoder.encode(state, is_small_blind=(player_id == small_blind))

    features = torch.from_numpy(encoded.features).unsqueeze(0)
    tokens = torch.from_numpy(sequence.tokens).unsqueeze(0)
    history_mask = torch.from_numpy(sequence.mask).unsqueeze(0)
    masks = torch.from_numpy(encoded.legal_mask).unsqueeze(0)

    with torch.no_grad():
        logits, value = net(features, tokens, history_mask, masks)
        probabilities = torch.softmax(logits, dim=-1)

    print(f"acting player       : P{player_id} "
          f"(small blind: {player_id == small_blind})")
    print(f"history length      : {sequence.length} of {MAX_HISTORY_TOKENS} tokens")
    for index, token in enumerate(sequence.real_tokens()):
        print(f"  {_describe_token(token, index)}")
    print()
    print(f"{'action':<8} {'legal':>6} {'logit':>12} {'prob':>8}")
    for index, name in enumerate(ACTION_NAMES):
        print(f"{name:<8} {str(bool(masks[0, index])):>6} "
              f"{logits[0, index].item():>12.4f} {probabilities[0, index].item():>8.4f}")
    print(f"\nvalue               : {value.item():.4f}")

    assert torch.allclose(probabilities.sum(), torch.tensor(1.0))
    assert torch.all(probabilities[~masks] == 0.0)
    assert logits[0, ~masks[0]].tolist() == [MASKED_LOGIT] * int((~masks[0]).sum())

    # ------------------------------------------------------------------
    _header("[3] Padding and the empty-history state")
    with torch.no_grad():
        packed = net.history(tokens, history_mask)
        manual, _ = net.history.gru(tokens[0, : sequence.length].unsqueeze(0))
        naive = net.history.gru(tokens)[0][:, -1]

    print(f"packed state |h|    : {packed.norm().item():.6f}")
    print(f"exact-length |h|    : {manual[0, -1].norm().item():.6f}")
    print(f"32-row (padded) |h| : {naive[0].norm().item():.6f}")
    print("packing stops the recurrence at each sample's real length; a GRU fed")
    print("a zero vector still moves its hidden state through its biases.")

    np.testing.assert_allclose(
        packed.numpy(), manual[:, -1].numpy(), rtol=0, atol=1e-6
    )
    assert not torch.allclose(packed[0], naive[0], atol=1e-5)

    empty_tokens = torch.zeros(1, MAX_HISTORY_TOKENS, TOKEN_DIM)
    empty_mask = torch.zeros(1, MAX_HISTORY_TOKENS, dtype=torch.bool)
    with torch.no_grad():
        empty_state = net.history(empty_tokens, empty_mask)
    print(f"empty history |h|   : {empty_state.norm().item():.6f}")
    assert torch.equal(empty_state, torch.zeros_like(empty_state))

    # ------------------------------------------------------------------
    _header("[4] The history ablation (same weights, same state)")
    with torch.no_grad():
        on_logits, _ = net(features, tokens, history_mask, masks)
        off_logits, _ = net(features, tokens, history_mask, masks, use_history=False)
        on_probs = torch.softmax(on_logits, dim=-1)
        off_probs = torch.softmax(off_logits, dim=-1)

    print(f"{'action':<8} {'history on':>12} {'history off':>12} {'shift':>10}")
    for index, name in enumerate(ACTION_NAMES):
        print(f"{name:<8} {on_probs[0, index].item():>12.4f} "
              f"{off_probs[0, index].item():>12.4f} "
              f"{on_probs[0, index].item() - off_probs[0, index].item():>+10.4f}")
    print(f"\nmax |logit shift|   : "
          f"{(on_logits - off_logits).abs().max().item():.6f}")

    assert not torch.allclose(on_logits, off_logits)

    # ------------------------------------------------------------------
    _header("[5] A capped state: masked sampling")
    for cap_seed in range(DEMO_SEED, DEMO_SEED + 20):
        cap_env = make_environment(seed=cap_seed)
        cap_encoder = ObservationEncoder.from_environment(cap_env)
        cap_state, cap_player = cap_env.reset()
        cap_small_blind = small_blind_player(cap_state)
        cap_tracker = HistoryTracker()
        cap_tracker.start_hand(cap_state)
        while not cap_env.is_over() and 1 in [
            int(action) for action in cap_state["legal_actions"]
        ]:
            cap_tracker.record(cap_state, cap_player, "raise")
            cap_state, cap_player = cap_env.step(1)
        if not cap_env.is_over() and 1 not in [
            int(action) for action in cap_state["legal_actions"]
        ]:
            break
    else:  # pragma: no cover - the cap is always reachable
        raise RuntimeError("could not reach a capped state")

    cap_encoded = cap_encoder.encode(
        cap_state, is_small_blind=(cap_player == cap_small_blind)
    )
    cap_sequence = cap_tracker.tokens(viewer=cap_player)
    cap_features = torch.from_numpy(cap_encoded.features).unsqueeze(0)
    cap_tokens = torch.from_numpy(cap_sequence.tokens).unsqueeze(0)
    cap_history = torch.from_numpy(cap_sequence.mask).unsqueeze(0)
    cap_masks = torch.from_numpy(cap_encoded.legal_mask).unsqueeze(0)

    with torch.no_grad():
        masked_logits, _ = net(
            cap_features, cap_tokens, cap_history, cap_masks
        )
        unmasked_logits = net.policy_head(
            torch.cat((net.trunk(cap_features), net.history(cap_tokens, cap_history)), dim=1)
        )
        masked_samples = Categorical(logits=masked_logits).sample((N_SAMPLES,))
        unmasked_samples = Categorical(logits=unmasked_logits).sample((N_SAMPLES,))

    print(f"legal actions       : "
          f"{[ACTION_NAMES[i] for i in torch.nonzero(cap_masks[0]).flatten().tolist()]}")
    print(f"masked draws of raise   : "
          f"{int((masked_samples == 1).sum())} / {N_SAMPLES}")
    print(f"unmasked draws of raise : "
          f"{int((unmasked_samples == 1).sum())} / {N_SAMPLES} "
          "(rlcard would play fold instead)")
    print(f"masked logit            : {masked_logits[0, 1].item():g}")

    if not bool(cap_masks[0, 1]):
        assert int((masked_samples == 1).sum()) == 0
        assert masked_logits[0, 1].item() == MASKED_LOGIT

    # ------------------------------------------------------------------
    _header("[6] One PPO-style backward pass")
    collect_env = make_environment(seed=5)
    collect_encoder = ObservationEncoder.from_environment(collect_env)
    collect_rng = np.random.default_rng(0)

    batch_features, batch_tokens, batch_history, batch_masks = [], [], [], []
    while len(batch_features) < 64:
        state, player_id = collect_env.reset()
        small_blind = small_blind_player(state)
        walk_tracker = HistoryTracker()
        walk_tracker.start_hand(state)
        while not collect_env.is_over() and len(batch_features) < 64:
            sequence = walk_tracker.tokens(viewer=player_id)
            item = collect_encoder.encode(
                state, is_small_blind=(player_id == small_blind)
            )
            batch_features.append(item.features)
            batch_tokens.append(sequence.tokens)
            batch_history.append(sequence.mask)
            batch_masks.append(item.legal_mask)
            action = int(collect_rng.choice(sorted(state["legal_actions"])))
            walk_tracker.record(state, player_id, collect_env.actions[action])
            state, player_id = collect_env.step(action)

    batch_features = torch.from_numpy(np.stack(batch_features))
    batch_tokens = torch.from_numpy(np.stack(batch_tokens))
    batch_history = torch.from_numpy(np.stack(batch_history))
    batch_masks = torch.from_numpy(np.stack(batch_masks))

    net.zero_grad()
    logits, value = net(batch_features, batch_tokens, batch_history, batch_masks)
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
    print()
    print(f"{'layer':<18} {'|grad| mean':>12} {'|grad| max':>12}")
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        print(f"{name:<18} {parameter.grad.abs().mean().item():>12.3e} "
              f"{parameter.grad.abs().max().item():>12.3e}")

    assert torch.isfinite(loss)

    # ------------------------------------------------------------------
    _header("[7] API summary")
    print("net = HistoryPolicyValueNet()                               "
          "# 116,933 parameters")
    print("logits, value = net(features, tokens, history_mask, legal_mask)")
    print("the mask and the history length are validated inside forward()")
    print("history rows after the length are ignored, not merely zero-weighted")
    print("use_history=False zeros the GRU output for the control arm")

    print("\nAll Agent B network checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
