"""Training loop and fixed-seed evaluation for Agent A.

One iteration collects a rollout, runs one PPO update, and records the
diagnostics.  Evaluation is deliberately separate: training reward is a noisy
number that the policy is actively overfitting to, while evaluation replays the
same deals against the same scripted opponent every time.  A change in the
evaluation column is therefore a change in the policy, not in the deal.

Evaluation plays greedily (argmax over the masked logits).  The sampling policy
is the thing being trained, but the mode is what a fixed-seed comparison should
measure: sampling would add hand-to-hand noise on top of the deal noise and make
two runs with the same seed diverge for no reason.

"One number per hand" is also why the report prints a confidence interval.  A
hand's payoff has a standard deviation of roughly 2.4 big blinds, so a few
thousand hands cannot resolve a small edge.  The interval says what the sample
actually supports; a point estimate alone invites reading noise as progress.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .encoding import ObservationEncoder, small_blind_player
from .environment import make_environment
from .networks import PolicyValueNet
from .ppo import PPOConfig, PPOUpdater
from .rollout import RolloutBuffer, collect_trajectories

#: Multiplier for a two-sided 95% normal interval.
Z_95 = 1.959963984540054

Opponent = Callable[[Mapping[str, Any], int], int]


def scripted_opponent(state: Mapping[str, Any], player_id: int) -> int:
    """Call when possible, otherwise check, otherwise fold.  Never raises."""

    legal = state["legal_actions"]
    for action in (0, 3, 2):  # call, check, fold
        if action in legal:
            return action
    return 1  # raise; unreachable in practice, kept total for type checkers


def random_opponent(rng: np.random.Generator) -> Opponent:
    """A uniform random legal action, drawn from ``rng`` for reproducibility."""

    def opponent(state: Mapping[str, Any], player_id: int) -> int:
        legal = sorted(state["legal_actions"])
        return int(rng.choice(legal))

    return opponent


@dataclass(frozen=True)
class TrainingConfig:
    """How long to train, how often to evaluate, and where to write results."""

    #: Rollout/update iterations.
    iterations: int = 50
    #: Self-play hands per iteration.
    hands_per_iteration: int = 320
    #: Evaluate every this many iterations, and once on the last one.
    evaluation_every: int = 10
    #: Hands per evaluation.  The 95% interval is roughly +/- 0.105 bb at 2,000.
    evaluation_hands: int = 2_000
    #: Controls the network initialisation, the training environment and the
    #: action sampling.
    seed: int = 0
    #: Separate seed for the evaluation environment, so every evaluation sees
    #: exactly the same deals.
    evaluation_seed: int = 10_000
    #: Name used for the CSV files and checkpoint files.
    run_name: str = "agent_a"
    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    #: PPO hyper-parameters.
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self) -> None:
        if int(self.iterations) < 1:
            raise ValueError(f"iterations must be positive, got {self.iterations!r}")
        if int(self.hands_per_iteration) < 1:
            raise ValueError(
                f"hands_per_iteration must be positive, "
                f"got {self.hands_per_iteration!r}"
            )
        if int(self.evaluation_every) < 1:
            raise ValueError(
                f"evaluation_every must be positive, got {self.evaluation_every!r}"
            )
        if int(self.evaluation_hands) < 2:
            raise ValueError(
                "evaluation_hands must be at least 2 to estimate a spread, "
                f"got {self.evaluation_hands!r}"
            )
        if not str(self.run_name).strip():
            raise ValueError("run_name must not be empty")


@dataclass(frozen=True)
class EvaluationResult:
    """One fixed-seed evaluation of player 0 against one opponent."""

    hands: int
    opponent: str
    mean_bb: float
    stderr_bb: float
    ci95_bb: float
    total_bb: float


@dataclass(frozen=True)
class EvaluationRecord:
    iteration: int
    result: EvaluationResult


@dataclass(frozen=True)
class TrainingResult:
    iterations: tuple[dict[str, float | int], ...]
    evaluations: tuple[EvaluationRecord, ...]
    checkpoints: tuple[str, ...]
    train_log: str
    evaluation_log: str


TRAIN_FIELDS = (
    "iteration",
    "decisions",
    "policy_loss",
    "value_loss",
    "entropy",
    "total_loss",
    "approx_kl",
    "clip_fraction",
    "grad_norm",
    "epochs",
    "minibatches",
    "samples",
)

EVALUATION_FIELDS = (
    "iteration",
    "hands",
    "opponent",
    "mean_bb",
    "stderr_bb",
    "ci95_bb",
    "total_bb",
)


def evaluate(
    net: PolicyValueNet,
    hands: int,
    *,
    seed: int = 10_000,
    opponent: Opponent = scripted_opponent,
    opponent_name: str = "scripted",
) -> EvaluationResult:
    """Play ``hands`` hands as player 0 and summarise the payoff in big blinds.

    Actions are greedy: the highest masked logit.  The environment seed is fixed,
    so two calls with the same weights see the same deals.

    Raises:
        ValueError: If ``hands`` is smaller than 2.
    """

    if int(hands) < 2:
        raise ValueError(f"hands must be at least 2, got {hands!r}")

    env = make_environment(seed=int(seed))
    encoder = ObservationEncoder.from_environment(env)
    payoffs = np.empty(int(hands), dtype=np.float64)

    was_training = net.training
    net.eval()
    try:
        for hand in range(int(hands)):
            state, player_id = env.reset()
            small_blind = small_blind_player(state)

            while not env.is_over():
                if player_id == 0:
                    encoded = encoder.encode(
                        state, is_small_blind=(small_blind == 0)
                    )
                    with torch.no_grad():
                        logits, _ = net(
                            torch.from_numpy(encoded.features).unsqueeze(0),
                            torch.from_numpy(encoded.legal_mask).unsqueeze(0),
                        )
                    action = int(torch.argmax(logits, dim=-1))
                else:
                    action = int(opponent(state, player_id))
                state, player_id = env.step(action)

            payoffs[hand] = float(env.get_payoffs()[0])
    finally:
        net.train(was_training)

    mean = float(payoffs.mean())
    stderr = float(payoffs.std(ddof=1) / np.sqrt(payoffs.size))
    return EvaluationResult(
        hands=int(hands),
        opponent=opponent_name,
        mean_bb=mean,
        stderr_bb=stderr,
        ci95_bb=Z_95 * stderr,
        total_bb=float(payoffs.sum()),
    )


def save_checkpoint(
    net: PolicyValueNet,
    config: TrainingConfig,
    iteration: int,
    path: str | Path,
) -> str:
    """Write weights plus the iteration and config needed to reproduce them."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "iteration": int(iteration),
            "config": asdict(config),
        },
        destination,
    )
    return str(destination)


def load_checkpoint(path: str | Path) -> tuple[PolicyValueNet, dict[str, Any]]:
    """Rebuild a network and its metadata from :func:`save_checkpoint`."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    net = PolicyValueNet()
    net.load_state_dict(payload["state_dict"])
    return net, {
        "iteration": int(payload["iteration"]),
        "config": dict(payload["config"]),
    }


def train(
    config: TrainingConfig,
    *,
    net: PolicyValueNet | None = None,
    progress: Callable[[dict[str, float | int], EvaluationRecord | None], None]
    | None = None,
) -> TrainingResult:
    """Run ``config.iterations`` collect-update steps, evaluating periodically.

    Args:
        config: Length, sizes, seeds and output locations.
        net: Optional starting network.  ``None`` initialises one under
            ``config.seed``.
        progress: Called after every iteration with the training row and the
            evaluation just completed, if any.

    Returns:
        The recorded rows, evaluations and checkpoint paths.
    """

    torch.manual_seed(int(config.seed))
    net = PolicyValueNet() if net is None else net

    env = make_environment(seed=int(config.seed))
    encoder = ObservationEncoder.from_environment(env)
    updater = PPOUpdater(net, config.ppo, seed=int(config.seed))
    rng = np.random.default_rng(int(config.seed))
    buffer = RolloutBuffer(gamma=1.0, lam=0.95)

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    train_log = log_dir / f"{config.run_name}_train.csv"
    evaluation_log = log_dir / f"{config.run_name}_eval.csv"

    rows: list[dict[str, float | int]] = []
    evaluations: list[EvaluationRecord] = []
    checkpoints: list[str] = []

    with train_log.open("w", newline="", encoding="utf-8") as train_handle, \
            evaluation_log.open("w", newline="", encoding="utf-8") as eval_handle:
        train_writer = csv.DictWriter(train_handle, fieldnames=TRAIN_FIELDS)
        evaluation_writer = csv.DictWriter(
            eval_handle, fieldnames=EVALUATION_FIELDS
        )
        train_writer.writeheader()
        evaluation_writer.writeheader()

        for iteration in range(1, int(config.iterations) + 1):
            trajectories = collect_trajectories(
                env,
                encoder,
                net,
                hands=int(config.hands_per_iteration),
                rng=rng,
            )

            buffer.clear()
            for trajectory in trajectories:
                buffer.add_trajectory(trajectory)
            batch = buffer.batch()
            stats = updater.step(batch)

            row: dict[str, float | int] = {
                "iteration": iteration,
                "decisions": len(batch),
                **stats,
            }
            rows.append(row)
            train_writer.writerow(row)
            train_handle.flush()

            record = None
            if (
                iteration % int(config.evaluation_every) == 0
                or iteration == int(config.iterations)
            ):
                result = evaluate(
                    net,
                    int(config.evaluation_hands),
                    seed=int(config.evaluation_seed),
                )
                record = EvaluationRecord(iteration=iteration, result=result)
                evaluations.append(record)
                evaluation_writer.writerow(
                    {
                        "iteration": iteration,
                        "hands": result.hands,
                        "opponent": result.opponent,
                        "mean_bb": result.mean_bb,
                        "stderr_bb": result.stderr_bb,
                        "ci95_bb": result.ci95_bb,
                        "total_bb": result.total_bb,
                    }
                )
                eval_handle.flush()

                path = save_checkpoint(
                    net,
                    config,
                    iteration,
                    Path(config.checkpoint_dir)
                    / f"{config.run_name}_{iteration:04d}.pt",
                )
                checkpoints.append(path)

            if progress is not None:
                progress(row, record)

    return TrainingResult(
        iterations=tuple(rows),
        evaluations=tuple(evaluations),
        checkpoints=tuple(checkpoints),
        train_log=str(train_log),
        evaluation_log=str(evaluation_log),
    )
