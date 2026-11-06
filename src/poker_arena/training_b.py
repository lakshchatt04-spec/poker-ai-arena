"""Training, evaluation and paired comparison for Agent B.

The loop is deliberately the same shape as Agent A's: mixed self-play and scripted
opponent, one PPO update per iteration, greedy fixed-seed evaluation against the
same three opponents, iteration 0 measured, periodic checkpoints.  Keeping the
protocol identical is the point of the experiment -- a difference between the two
agents should be attributable to the history side channel, not to a different
evaluation or a different opponent mix.

Evaluation here is *policy-based* rather than network-based, so three kinds of
comparison share one implementation:

* ``AgentAPolicy`` -- an Agent A checkpoint, which ignores the tracker.
* ``AgentBPolicy`` -- an Agent B checkpoint, reading the tracker.
* the same Agent B checkpoint with ``use_history=False`` -- the ablation.

All three expose ``policy(state, player_id, tracker, small_blind) -> action_id``.
``evaluate_paired_policies`` reseeds the environment before each hand for both
policies, so A and B play identical deals; that is the comparison the project
exists to make.  The interval is computed from the per-hand differences, exactly
as in :func:`poker_arena.training.evaluate_paired`.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .encoding import ObservationEncoder, small_blind_player
from .environment import make_environment, reset_hand
from .history import HistoryTracker
from .networks import PolicyValueNet
from .networks_b import HISTORY_HIDDEN, HistoryPolicyValueNet
from .opponents import notify_hand_start
from .ppo import PPOConfig
from .ppo_b import HistoryPPOUpdater
from .rollout_b import (
    HistoryRolloutBuffer,
    collect_history_trajectories,
)
from .training import (
    EVALUATION_FIELDS,
    OPPONENT_NAMES,
    TRAIN_FIELDS,
    Z_95,
    EvaluationRecord,
    EvaluationResult,
    Opponent,
    PairedEvaluationResult,
    TrainingConfig,
    TrainingResult,
    make_opponent,
    scripted_opponent,
)

#: A decision function shared by A, B and B-without-history.  ``tracker`` holds
#: the actions already taken in this hand and is advanced for both players by the
#: play loop; ``small_blind`` is the player id that posted it this hand.
Policy = Callable[[Any, int, HistoryTracker, int], int]


@dataclass(frozen=True)
class AgentAPolicy:
    """Greedy Agent A action, ignoring the history tracker."""

    net: PolicyValueNet
    encoder: ObservationEncoder

    def __call__(
        self, state: Any, player_id: int, tracker: HistoryTracker, small_blind: int
    ) -> int:
        encoded = self.encoder.encode(
            state, is_small_blind=(small_blind == player_id)
        )
        with torch.no_grad():
            logits, _ = self.net(
                torch.from_numpy(encoded.features).unsqueeze(0),
                torch.from_numpy(encoded.legal_mask).unsqueeze(0),
            )
        return int(torch.argmax(logits, dim=-1))


@dataclass(frozen=True)
class AgentBPolicy:
    """Greedy Agent B action, reading the betting history seen by ``player_id``."""

    net: HistoryPolicyValueNet
    encoder: ObservationEncoder
    use_history: bool = True

    def __call__(
        self, state: Any, player_id: int, tracker: HistoryTracker, small_blind: int
    ) -> int:
        encoded = self.encoder.encode(
            state, is_small_blind=(small_blind == player_id)
        )
        sequence = tracker.tokens(viewer=player_id)
        with torch.no_grad():
            logits, _ = self.net(
                torch.from_numpy(encoded.features).unsqueeze(0),
                torch.from_numpy(sequence.tokens).unsqueeze(0),
                torch.from_numpy(sequence.mask).unsqueeze(0),
                torch.from_numpy(encoded.legal_mask).unsqueeze(0),
                use_history=self.use_history,
            )
        return int(torch.argmax(logits, dim=-1))


@dataclass(frozen=True)
class TrainingConfigB(TrainingConfig):
    """Agent A's training configuration plus B's architecture switches.

    Defaults match the Agent A hardening runs, so an A/B comparison does not have
    to argue about hands per iteration or opponent mix.
    """

    run_name: str = "agent_b"
    #: Width of the GRU state.  Recorded in the checkpoint so loading rebuilds
    #: the same architecture.
    history_hidden: int = HISTORY_HIDDEN
    #: When ``False``, training runs with the history side channel zeroed.  This
    #: is the control arm: same network, same heads, no memory input.
    use_history: bool = True

    def __post_init__(self) -> None:
        TrainingConfig.__post_init__(self)
        if int(self.history_hidden) < 1:
            raise ValueError(
                f"history_hidden must be positive, got {self.history_hidden!r}"
            )


def play_hand(
    env: Any,
    policy: Policy,
    opponent: Opponent,
    tracker: HistoryTracker,
) -> float:
    """Play one hand as player 0 and return its payoff in big blinds.

    The history tracker is advanced for the opponent too: Agent B has to see the
    opponent's actions, not just its own.  It is passed in so evaluation can reuse
    one object for thousands of hands, and it is reset from the first state of the
    hand.  The environment must already be seeded; the caller decides whether the
    deal stream is continuous (``evaluate_policy``) or reseeded per hand
    (``evaluate_paired_policies``).
    """

    state, player_id = reset_hand(env)
    small_blind = small_blind_player(state)
    tracker.start_hand(state)
    notify_hand_start(opponent, state)

    while not env.is_over():
        if player_id == 0:
            action = int(policy(state, player_id, tracker, small_blind))
        else:
            action = int(opponent(state, player_id))
            if action not in state["legal_actions"]:
                raise ValueError(
                    f"opponent returned illegal action {action!r} for player "
                    f"{player_id}; legal: {sorted(state['legal_actions'])}"
                )

        tracker.record(state, player_id, env.actions[action])
        state, player_id = env.step(action)

    return float(env.get_payoffs()[0])


def evaluate_policy(
    policy: Policy,
    hands: int,
    *,
    seed: int = 10_000,
    opponent: Opponent = scripted_opponent,
    opponent_name: str = "passive",
) -> EvaluationResult:
    """Fixed-seed greedy evaluation on a continuous deal stream.

    This mirrors :func:`poker_arena.training.evaluate`, including the shared
    opponent object across hands, so a number stays comparable with the Agent A
    tables.
    """

    if int(hands) < 2:
        raise ValueError(f"hands must be at least 2, got {hands!r}")

    env = make_environment(seed=int(seed))
    tracker = HistoryTracker()
    payoffs = np.empty(int(hands), dtype=np.float64)

    net = getattr(policy, "net", None)
    was_training = None if net is None else net.training
    if net is not None:
        net.eval()
    try:
        for hand in range(int(hands)):
            payoffs[hand] = play_hand(env, policy, opponent, tracker)
    finally:
        if net is not None:
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


def evaluate_paired_policies(
    policy_a: Policy,
    policy_b: Policy,
    hands: int,
    *,
    seed: int = 10_000,
    opponent_name: str = "passive",
    opponent: Opponent | None = None,
) -> PairedEvaluationResult:
    """Play two policies on identical deals; report A minus B.

    Each hand is replayed after ``env.seed(seed + hand)`` and a stochastic
    scripted opponent is rebuilt per play from the same per-hand seed, so both
    policies see the same cards, blinds and opponent draws.  See
    :func:`poker_arena.training.evaluate_paired` for why that makes the interval
    tighter than two independent evaluations.
    """

    if int(hands) < 2:
        raise ValueError(f"hands must be at least 2, got {hands!r}")
    if opponent is None and opponent_name not in OPPONENT_NAMES:
        raise ValueError(
            f"unknown opponent {opponent_name!r}; "
            f"choose one of {', '.join(OPPONENT_NAMES)}"
        )

    env = make_environment(seed=int(seed))
    tracker = HistoryTracker()
    payoffs_a = np.empty(int(hands), dtype=np.float64)
    payoffs_b = np.empty(int(hands), dtype=np.float64)

    net_a = getattr(policy_a, "net", None)
    net_b = getattr(policy_b, "net", None)
    was_training_a = None if net_a is None else net_a.training
    was_training_b = None if net_b is None else net_b.training
    if net_a is not None:
        net_a.eval()
    if net_b is not None:
        net_b.eval()
    try:
        for hand in range(int(hands)):
            deal_seed = int(seed) + hand
            if opponent is None:
                opponent_a = make_opponent(opponent_name, seed=deal_seed)
                opponent_b = make_opponent(opponent_name, seed=deal_seed)
            else:
                opponent_a = opponent_b = opponent

            env.seed(deal_seed)
            payoffs_a[hand] = play_hand(env, policy_a, opponent_a, tracker)

            env.seed(deal_seed)
            payoffs_b[hand] = play_hand(env, policy_b, opponent_b, tracker)
    finally:
        if net_a is not None:
            net_a.train(was_training_a)
        if net_b is not None:
            net_b.train(was_training_b)

    differences = payoffs_a - payoffs_b
    stderr_a = float(payoffs_a.std(ddof=1) / np.sqrt(payoffs_a.size))
    stderr_b = float(payoffs_b.std(ddof=1) / np.sqrt(payoffs_b.size))
    stderr_difference = float(differences.std(ddof=1) / np.sqrt(differences.size))
    return PairedEvaluationResult(
        hands=int(hands),
        opponent=opponent_name,
        mean_a_bb=float(payoffs_a.mean()),
        mean_b_bb=float(payoffs_b.mean()),
        mean_difference_bb=float(differences.mean()),
        stderr_difference_bb=stderr_difference,
        ci95_difference_bb=Z_95 * stderr_difference,
        total_difference_bb=float(differences.sum()),
        unpaired_stderr_bb=float(np.hypot(stderr_a, stderr_b)),
        wins=int(np.count_nonzero(differences > 0.0)),
        losses=int(np.count_nonzero(differences < 0.0)),
        ties=int(np.count_nonzero(differences == 0.0)),
        differences_bb=tuple(float(value) for value in differences),
    )


def save_checkpoint_b(
    net: HistoryPolicyValueNet,
    config: TrainingConfigB,
    iteration: int,
    path: str | Path,
) -> str:
    """Write weights, architecture widths and the iteration."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "iteration": int(iteration),
            "feature_dim": int(net.feature_dim),
            "hidden": int(net.hidden),
            "history_hidden": int(net.history_hidden),
            "config": asdict(config),
        },
        destination,
    )
    return str(destination)


def load_checkpoint_b(
    path: str | Path,
) -> tuple[HistoryPolicyValueNet, dict[str, Any]]:
    """Rebuild an Agent B network and its metadata."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    net = HistoryPolicyValueNet(
        feature_dim=int(payload["feature_dim"]),
        hidden=int(payload["hidden"]),
        history_hidden=int(payload["history_hidden"]),
    )
    net.load_state_dict(payload["state_dict"])
    return net, {
        "iteration": int(payload["iteration"]),
        "config": dict(payload["config"]),
    }


def train_b(
    config: TrainingConfigB,
    *,
    net: HistoryPolicyValueNet | None = None,
    progress: Callable[
        [dict[str, float | int], tuple[EvaluationRecord, ...] | None], None
    ]
    | None = None,
) -> TrainingResult:
    """Run ``config.iterations`` collect-update steps for Agent B.

    The structure is intentionally identical to
    :func:`poker_arena.training.train`; only the network, rollout and updater are
    B's.  ``config.use_history=False`` trains the control arm (history zeroed).
    """

    torch.manual_seed(int(config.seed))
    net = (
        HistoryPolicyValueNet(history_hidden=int(config.history_hidden))
        if net is None
        else net
    )

    env = make_environment(seed=int(config.seed))
    encoder = ObservationEncoder.from_environment(env)
    updater = HistoryPPOUpdater(net, config.ppo, seed=int(config.seed))
    rng = np.random.default_rng(int(config.seed))
    buffer = HistoryRolloutBuffer(gamma=1.0, lam=0.95)

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

        def record_evaluation(iteration: int) -> tuple[EvaluationRecord, ...]:
            completed: list[EvaluationRecord] = []
            for name in config.evaluation_opponents:
                result = evaluate_policy(
                    AgentBPolicy(net, encoder, use_history=config.use_history),
                    int(config.evaluation_hands),
                    seed=int(config.evaluation_seed),
                    opponent=make_opponent(name, seed=int(config.evaluation_seed)),
                    opponent_name=name,
                )
                record = EvaluationRecord(iteration=iteration, result=result)
                evaluations.append(record)
                completed.append(record)
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

            path = save_checkpoint_b(
                net,
                config,
                iteration,
                Path(config.checkpoint_dir)
                / f"{config.run_name}_{iteration:04d}.pt",
            )
            checkpoints.append(path)
            return tuple(completed)

        record_evaluation(0)

        for iteration in range(1, int(config.iterations) + 1):
            total_hands = int(config.hands_per_iteration)
            self_play_hands = int(round(total_hands * config.self_play_fraction))
            opponent_hands = total_hands - self_play_hands

            trajectories = []
            if self_play_hands:
                trajectories.extend(
                    collect_history_trajectories(
                        env,
                        encoder,
                        net,
                        hands=self_play_hands,
                        rng=rng,
                        use_history=config.use_history,
                    )
                )
            if opponent_hands:
                opponent = make_opponent(
                    config.training_opponent,
                    seed=int(config.seed) + 7919 * iteration,
                )
                trajectories.extend(
                    collect_history_trajectories(
                        env,
                        encoder,
                        net,
                        hands=opponent_hands,
                        rng=rng,
                        opponent=opponent,
                        use_history=config.use_history,
                    )
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

            records = None
            if (
                iteration % int(config.evaluation_every) == 0
                or iteration == int(config.iterations)
            ):
                records = record_evaluation(iteration)

            if progress is not None:
                progress(row, records)

    return TrainingResult(
        iterations=tuple(rows),
        evaluations=tuple(evaluations),
        checkpoints=tuple(checkpoints),
        train_log=str(train_log),
        evaluation_log=str(evaluation_log),
    )
