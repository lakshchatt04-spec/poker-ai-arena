"""Measure how much of the value target is predictable at all.

A critic's job is to predict the return ``G`` from the encoded state.  The
interesting question is not "is the network big enough" but "is there anything
to predict": if the conditional mean ``E[G | s]`` barely varies across states,
then no critic -- of any size, trained for any number of steps -- can remove
much variance from the advantage estimate, because

    Var(V(s)) <= Var(E[G | s]) <= Var(G).

This script estimates that ceiling directly, and separates the two ways a
critic can look bad:

1. **Under-trained.**  The architecture can represent the value function but the
   online loop does not give it enough gradient steps.  Then a long supervised
   fit on one fixed batch reaches a much higher explained variance than the
   online number -- and more PPO steps or a bigger value coefficient would help.
2. **Nothing to learn.**  Every model class converges to the same held-out
   explained variance, and that number does not grow when the training set gets
   ten times bigger.  Then the ceiling belongs to the *target*, and critic work
   is wasted effort: the variance has to be attacked somewhere else.

The measurements that separate them:

* a constant predictor, which defines ``0``;
* ridge regression on growing slices of the training set, which answers "is it
  data-bound?";
* ridge on separate feature blocks, which answers "is a feature missing?";
* an MLP with weight decay, which answers "is a linear model too weak?".

Two things this script is careful about.  The split is **by hand**, because
every decision inside a hand shares the same cards and the same payoff, so a
row-level split leaks.  And explained variance is always reported on the held
out hands -- a train figure near ``1.0`` with a negative test figure is
memorisation, not learning.

Run from the project root::

    python scripts\\diagnose_value.py                  # 4,000 hands, ~1 minute
    python scripts\\diagnose_value.py --hands 20000    # tighter ceiling
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from torch import nn

from poker_arena.encoding import ObservationEncoder
from poker_arena.environment import make_environment
from poker_arena.networks import PolicyValueNet
from poker_arena.rollout import RolloutBuffer, collect_trajectories
from poker_arena.training import load_checkpoint

STREETS = ("preflop", "flop", "turn", "river")


def explained_variance(target: np.ndarray, prediction: np.ndarray) -> float:
    """``1 - MSE / Var(target)``; ``0`` is no better than predicting the mean."""

    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    spread = float(target.var())
    if spread <= 0.0:
        return float("nan")
    return 1.0 - float(np.mean((target - prediction) ** 2)) / spread


def collect_dataset(checkpoint: Path | None, hands: int, seed: int):
    """Collect self-play data and split it by hand into train and test."""

    if checkpoint is not None:
        net, _ = load_checkpoint(checkpoint)
    else:
        torch.manual_seed(0)
        net = PolicyValueNet()

    env = make_environment(seed=seed)
    encoder = ObservationEncoder.from_environment(env)
    started = time.perf_counter()
    trajectories = collect_trajectories(
        env, encoder, net, hands=hands, rng=np.random.default_rng(seed + 1)
    )
    seconds = time.perf_counter() - started

    split = max(1, int(round(0.8 * len(trajectories))))
    train_part, test_part = trajectories[:split], trajectories[split:]

    def flatten(part):
        buffer = RolloutBuffer(gamma=1.0, lam=0.95)
        for trajectory in part:
            buffer.add_trajectory(trajectory)
        return buffer.batch()

    hand_ends = np.cumsum([len(trajectory) for trajectory in train_part])
    return flatten(train_part), flatten(test_part), hand_ends, seconds


def ridge_explained_variance(
    train_features, train_returns, test_features, test_returns, penalty
) -> float:
    """Closed-form ridge on standardised-free raw features, plus an intercept."""

    train = np.concatenate(
        [train_features, np.ones((len(train_features), 1))], axis=1
    )
    test = np.concatenate(
        [test_features, np.ones((len(test_features), 1))], axis=1
    )
    gram = train.T @ train
    weights = np.linalg.solve(
        gram + penalty * np.eye(gram.shape[0]), train.T @ train_returns
    )
    return explained_variance(test_returns, test @ weights)


def fit_mlp(
    train_features,
    train_returns,
    test_features,
    test_returns,
    *,
    hidden: int,
    weight_decay: float,
    steps: int,
    learning_rate: float,
    seed: int,
    report_every: int,
):
    """Fit a plain regressor and report the best and final held-out fit."""

    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(train_features.shape[1], hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)

    features = torch.as_tensor(
        np.ascontiguousarray(train_features, dtype=np.float32)
    )
    returns = torch.as_tensor(
        np.ascontiguousarray(train_returns, dtype=np.float32)
    )
    test_features_tensor = torch.as_tensor(
        np.ascontiguousarray(test_features, dtype=np.float32)
    )

    best = (-np.inf, 0, 0.0, 0.0)
    history: list[tuple[int, float, float]] = []
    for step in range(1, steps + 1):
        rows = torch.randint(0, len(features), (512,), generator=generator)
        loss = (model(features[rows]).squeeze(-1) - returns[rows]).pow(2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step % report_every == 0 or step == steps:
            with torch.no_grad():
                train_prediction = model(features).squeeze(-1).numpy()
                test_prediction = model(test_features_tensor).squeeze(-1).numpy()
            train_ev = explained_variance(train_returns, train_prediction)
            test_ev = explained_variance(test_returns, test_prediction)
            history.append((step, train_ev, test_ev))
            if test_ev > best[0]:
                best = (test_ev, step, train_ev, test_ev)

    final_train, final_test = history[-1][1], history[-1][2]
    return best, (final_train, final_test), history


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", default="checkpoints/agent_a_0050.pt",
        help="policy to collect data with (default: checkpoints/agent_a_0050.pt)",
    )
    parser.add_argument(
        "--hands", type=int, default=4_000,
        help="self-play hands to collect (default: 4000)",
    )
    parser.add_argument(
        "--seed", type=int, default=424_242,
        help="collection seed (default: 424242)",
    )
    parser.add_argument(
        "--mlp-steps", type=int, default=5_000,
        help="supervised steps for the MLP comparison (default: 5000)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"warning: {checkpoint} not found, collecting with a fresh network")
        checkpoint = None

    print("Poker AI Arena - how predictable is the value target?")
    print(f"checkpoint : {checkpoint or 'fresh network'}")
    print(f"data       : {args.hands} self-play hands, split by hand 80/20")

    train, test, hand_ends, seconds = collect_dataset(
        checkpoint, args.hands, args.seed
    )
    print(
        f"collected  : {len(train) + len(test)} decisions in {seconds:.1f}s "
        f"-> {len(train)} train, {len(test)} test"
    )

    train_features = train.features.astype(np.float64)
    train_returns = train.returns.astype(np.float64)
    test_features = test.features.astype(np.float64)
    test_returns = test.returns.astype(np.float64)

    spread = float(train_returns.std())
    print(
        f"returns    : std {spread:.3f} bb, mean {train_returns.mean():+.3f} bb, "
        f"range {train_returns.min():+.1f} to {train_returns.max():+.1f}"
    )

    constant = explained_variance(
        test_returns, np.full(len(test_returns), train_returns.mean())
    )
    print()
    print(f"{'predictor':<42}{'held-out explained variance':>28}")
    print(f"{'constant (train mean)':<42}{constant:>28.3f}")

    # -- is it data-bound? -------------------------------------------------
    print()
    print("ridge on all 131 features, fitted on a growing slice of the same data")
    print(f"{'train hands':>12}{'decisions':>11}{'penalty 10':>12}{'100':>8}{'1000':>8}")
    ridge_by_size: list[tuple[int, float]] = []
    for hands in (250, 500, 1_000, 2_000, 4_000, 8_000, len(hand_ends)):
        if hands > len(hand_ends):
            continue
        rows = int(hand_ends[hands - 1])
        values = [
            ridge_explained_variance(
                train_features[:rows], train_returns[:rows],
                test_features, test_returns, penalty,
            )
            for penalty in (10.0, 100.0, 1_000.0)
        ]
        ridge_by_size.append((hands, max(values)))
        print(
            f"{hands:>12}{rows:>11}{values[0]:>12.3f}"
            f"{values[1]:>8.3f}{values[2]:>8.3f}"
        )

    # -- is a feature missing? --------------------------------------------
    print()
    print("ridge on separate feature blocks (penalty 100, all data)")
    blocks = (
        ("own hole cards (0:52)", slice(0, 52)),
        ("public board (52:104)", slice(52, 104)),
        ("raise counts (104:124)", slice(104, 124)),
        ("street one-hot (124:128)", slice(124, 128)),
        ("pot, to call, blind (128:131)", slice(128, 131)),
        ("everything (0:131)", slice(0, 131)),
    )
    for label, block in blocks:
        value = ridge_explained_variance(
            train_features[:, block], train_returns,
            test_features[:, block], test_returns, 100.0,
        )
        print(f"  {label:<32}{value:>8.3f}")

    # -- is a linear model too weak? ---------------------------------------
    print()
    print(
        f"MLP {256}x2 with weight decay, {args.mlp_steps} supervised steps"
    )
    best, final, history = fit_mlp(
        train_features, train_returns, test_features, test_returns,
        hidden=256, weight_decay=1e-2, steps=args.mlp_steps,
        learning_rate=3e-4, seed=0,
        report_every=max(1, args.mlp_steps // 5),
    )
    print(f"{'step':>8}{'train':>9}{'test':>9}")
    for step, train_value, test_value in history:
        print(f"{step:>8}{train_value:>9.3f}{test_value:>9.3f}")
    print(
        f"best test explained variance {best[0]:.3f} at step {best[1]} "
        f"(train {best[2]:.3f}); final test {final[1]:.3f}"
    )

    # -- verdict -----------------------------------------------------------
    candidates = [constant, max(value for _, value in ridge_by_size), best[0]]
    ceiling = max(candidates)
    smallest, largest = ridge_by_size[0], ridge_by_size[-1]
    growth = largest[1] - smallest[1]
    best_curve = max(value for _, value in ridge_by_size)
    curve_at = max(ridge_by_size, key=lambda pair: pair[1])[0]

    print()
    print("verdict")
    print(
        f"  best held-out explained variance found : {ceiling:+.3f} "
        f"({100 * ceiling:.0f}% of the return variance)"
    )
    print(
        f"  ridge, {smallest[0]:,} hands -> {largest[0]:,} hands      : "
        f"{smallest[1]:+.3f} -> {largest[1]:+.3f} "
        f"(peak {best_curve:+.3f} at {curve_at:,} hands)"
    )
    if ceiling > 0.0:
        print(
            f"  that ceiling is a prediction spread of "
            f"{np.sqrt(ceiling) * spread:.2f} bb, against an outcome spread of "
            f"{spread:.2f} bb"
        )
    print()

    if ceiling < 0.15 and growth < 0.05:
        print(
            "This is an information ceiling, not a training problem.  More data,"
        )
        print(
            "a wider network and more PPO steps all converge to the same few"
        )
        print(
            "percent.  The best possible critic is allowed to move its baseline"
        )
        print(
            "by about the prediction spread printed above, so advantages keep"
        )
        print(
            "almost all of the outcome variance.  Gradient noise therefore has"
        )
        print(
            "to be bought with samples per update, not with a better critic."
        )
    elif growth >= 0.05:
        print(
            "The ceiling is still rising with data, so the critic is"
        )
        print(
            "data-starved rather than incapable: collect more hands per"
        )
        print(
            "iteration before changing the architecture."
        )
    else:
        print(
            "The ceiling is well above the constant baseline, so the features"
        )
        print(
            "and the architecture can predict value.  The gap between this and"
        )
        print(
            "the online number is then an optimisation problem inside the loop."
        )

    print("\nValue-head diagnosis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
