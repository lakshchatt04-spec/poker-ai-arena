"""Training loop and fixed-seed evaluation for Agent A.

One iteration collects a rollout, runs one PPO update, and records the
diagnostics.  The rollout is a **mixture**: ``self_play_fraction`` of the hands
are the network playing both seats against itself, and the rest are the network
playing player 0 against a scripted opponent.  Self-play keeps the policy honest
against a moving copy of itself; the opponent supplies raising decisions that
self-play may never discover.  Both halves feed one update, so the two kinds of
experience share a value scale.

Evaluation is deliberately separate: training reward is a noisy number that the
policy is actively overfitting to, while evaluation replays the same deals
against the same bots every time.  A change in an evaluation column is therefore
a change in the policy, not in the deal.

More than one opponent is evaluated, and they have different jobs.  The passive
caller is the **continuity ruler**: every number in the README was measured
against it, so dropping it would make runs incomparable.  The training opponent
is the **exploitation reference**: beating it can mean pattern-matching it.  At
least one further opponent should be a **held-out generalisation ruler**, never
used for training, so a rising curve there is evidence the skill transferred
rather than that the bot was solved.

Evaluation plays greedily (argmax over the masked logits).  The sampling policy
is the thing being trained, but the mode is what a fixed-seed comparison should
measure: sampling would add hand-to-hand noise on top of the deal noise and make
two runs with the same seed diverge for no reason.

"One number per hand" is also why the report prints a confidence interval.  The
spread is not a property of the game: it follows how many chips the current
policy puts in, and measured `1.6` bb at iteration 10 against `5.1` bb at
iteration 50 of the first run.  At the larger of those, 2,000 hands resolve only
about `0.22` bb/hand.  The interval is computed from the sample actually
observed and says what that sample supports; a point estimate alone invites
reading noise as progress.
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
from .environment import make_environment, reset_hand
from .networks import PolicyValueNet
from .opponents import (
    CoinOpenerBettor,
    ContinuationBettor,
    FirstRaiserBettor,
    notify_hand_start,
)
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


#: Rank order used to score hole cards, worst to best.
CARD_RANKS = "23456789TJQKA"

#: A hole-card score at or above this is raised by the ``raiser_strong`` style.
STRONG_HOLE_SCORE = 0.85

#: Per-decision raise probability of the ``raiser_loose`` style.
LOOSE_RAISE_PROBABILITY = 0.35

#: Every scripted opponent that training and evaluation can ask for.  ``random``
#: is the uniform legal-action bot, kept as a baseline.  ``c_bettor``,
#: ``first_raiser`` and ``coin_opener`` keep per-hand state, so their post-flop
#: play depends on the action sequence rather than only on the current state (see
#: ``poker_arena.opponents``).  ``first_raiser`` and ``coin_opener`` are the two
#: designs the identifiability probe tests for a genuinely hidden mode.
OPPONENT_NAMES = (
    "passive",
    "raiser_strong",
    "raiser_loose",
    "random",
    "c_bettor",
    "first_raiser",
    "coin_opener",
)


def hole_strength(state: Mapping[str, Any]) -> float:
    """A crude, deterministic score for the cards in ``state["raw_obs"]["hand"]``.

    This is not a poker hand evaluator.  It only separates "two big cards, a
    pair, or suited" from trash, which is all a scripted opponent needs.  It
    scores ``0.0`` when the hand is unavailable, so a synthetic state never
    crashes the bot.
    """

    raw = state.get("raw_obs") or {}
    cards = raw.get("hand") or []
    if len(cards) != 2:
        return 0.0

    try:
        ranks = sorted(CARD_RANKS.index(str(card)[1].upper()) for card in cards)
    except (IndexError, ValueError):
        return 0.0

    score = (ranks[0] + ranks[1]) / 24.0
    if ranks[0] == ranks[1]:
        score += 0.6
    if str(cards[0])[0].upper() == str(cards[1])[0].upper():
        score += 0.15
    return score


def raising_opponent(should_raise: Callable[[Mapping[str, Any]], bool]) -> Opponent:
    """Raise when ``should_raise`` says so, otherwise call, check, then fold."""

    def opponent(state: Mapping[str, Any], player_id: int) -> int:
        legal = state["legal_actions"]
        if 1 in legal and should_raise(state):  # raise
            return 1
        for action in (0, 3, 2):  # call, check, fold
            if action in legal:
                return action
        return 1

    return opponent


def make_opponent(name: str, seed: int = 0) -> Opponent:
    """Build a scripted opponent by name.

    ``raiser_strong`` and ``raiser_loose`` differ in *style*, not just in
    aggression: the first raises only hole cards its simple score likes, the
    second raises a fixed fraction of the time regardless of cards.  Training on
    one and evaluating on the other is what turns an evaluation into a
    generalisation test rather than an exploitation test.
    """

    if name == "passive":
        return scripted_opponent
    if name == "raiser_strong":
        return raising_opponent(lambda state: hole_strength(state) >= STRONG_HOLE_SCORE)
    if name == "raiser_loose":
        rng = np.random.default_rng(seed)
        return raising_opponent(
            lambda state: bool(rng.random() < LOOSE_RAISE_PROBABILITY)
        )
    if name == "random":
        return random_opponent(np.random.default_rng(seed))
    if name == "c_bettor":
        return ContinuationBettor(seed=seed)
    if name == "first_raiser":
        return FirstRaiserBettor(seed=seed)
    if name == "coin_opener":
        return CoinOpenerBettor(seed=seed)
    raise ValueError(
        f"unknown opponent {name!r}; choose one of {', '.join(OPPONENT_NAMES)}"
    )


@dataclass(frozen=True)
class TrainingConfig:
    """How long to train, how often to evaluate, and where to write results."""

    #: Rollout/update iterations.
    iterations: int = 50
    #: Hands per iteration, split between self-play and the training opponent.
    hands_per_iteration: int = 320
    #: Fraction of each rollout played as self-play.  Both seats are recorded,
    #: so self-play hands produce two trajectories; opponent hands produce one.
    self_play_fraction: float = 0.5
    #: Scripted opponent for the non-self-play hands.
    training_opponent: str = "raiser_strong"
    #: Opponents evaluated and logged every time, in log order.  Keep at least
    #: one that is never used for training: that column is the generalisation
    #: measure, and the others are the exploitation references.
    evaluation_opponents: tuple[str, ...] = (
        "passive",
        "raiser_strong",
        "raiser_loose",
    )
    #: Evaluate every this many iterations, and once on the last one.
    evaluation_every: int = 10
    #: Hands per evaluation, per opponent.  The 95% interval is roughly
    #: +/- 0.105 bb at 2,000.
    evaluation_hands: int = 2_000
    #: Controls the network initialisation, the training environment and the
    #: action sampling.
    seed: int = 0
    #: Separate seed for the evaluation environment and the scripted opponents,
    #: so every evaluation sees exactly the same deals and the same bot stream.
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
        if not 0.0 <= float(self.self_play_fraction) <= 1.0:
            raise ValueError(
                "self_play_fraction must be between 0 and 1, "
                f"got {self.self_play_fraction!r}"
            )
        if self.training_opponent not in OPPONENT_NAMES:
            raise ValueError(
                f"unknown training_opponent {self.training_opponent!r}; "
                f"choose one of {', '.join(OPPONENT_NAMES)}"
            )
        if not self.evaluation_opponents:
            raise ValueError("evaluation_opponents must not be empty")
        unknown = [
            name
            for name in self.evaluation_opponents
            if name not in OPPONENT_NAMES
        ]
        if unknown:
            raise ValueError(
                f"unknown evaluation opponents {unknown}; "
                f"choose from {', '.join(OPPONENT_NAMES)}"
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
class PairedEvaluationResult:
    """Two networks played on the *same deals*, summarised by their difference.

    ``mean_difference_bb`` is network A's payoff minus network B's, averaged
    over hands.  A positive number means A did better.  ``wins``, ``losses`` and
    ``ties`` count hands by that sign.

    ``unpaired_stderr_bb`` is the interval the same two means would carry if the
    hands were treated as independent samples.  When the networks play similar
    policies, pairing the deal removes most of the payoff variance and
    ``stderr_difference_bb`` is materially smaller.
    """

    hands: int
    opponent: str
    mean_a_bb: float
    mean_b_bb: float
    mean_difference_bb: float
    stderr_difference_bb: float
    ci95_difference_bb: float
    total_difference_bb: float
    unpaired_stderr_bb: float
    wins: int
    losses: int
    ties: int
    #: Payoff of A minus B for each hand, in deal order.  Kept out of ``repr``
    #: because it is as long as the evaluation.
    differences_bb: tuple[float, ...] = field(repr=False)



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
    "explained_variance",
    "return_std",
    "entropy",
    "entropy_preflop",
    "preflop_fraction",
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


def _greedy_action(
    net: PolicyValueNet,
    encoder: ObservationEncoder,
    state: Mapping[str, Any],
    small_blind: int,
) -> int:
    """The highest masked logit for player 0's current state."""

    encoded = encoder.encode(state, is_small_blind=(small_blind == 0))
    with torch.no_grad():
        logits, _ = net(
            torch.from_numpy(encoded.features).unsqueeze(0),
            torch.from_numpy(encoded.legal_mask).unsqueeze(0),
        )
    return int(torch.argmax(logits, dim=-1))


def _play_hand(
    env: Any,
    encoder: ObservationEncoder,
    net: PolicyValueNet,
    opponent: Opponent,
) -> float:
    """Play one hand as player 0 and return its payoff in big blinds.

    The environment must already be seeded.  The caller decides whether to reset
    on a continuous deal stream (``evaluate``) or to reseed before each hand
    (``evaluate_paired``).  :func:`reset_hand` is used so the opening state does
    not carry the previous hand's raise counts.
    """

    state, player_id = reset_hand(env)
    small_blind = small_blind_player(state)
    notify_hand_start(opponent, state)

    while not env.is_over():
        if player_id == 0:
            action = _greedy_action(net, encoder, state, small_blind)
        else:
            action = int(opponent(state, player_id))
        state, player_id = env.step(action)

    return float(env.get_payoffs()[0])


def evaluate(
    net: PolicyValueNet,
    hands: int,
    *,
    seed: int = 10_000,
    opponent: Opponent = scripted_opponent,
    opponent_name: str = "passive",
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
            payoffs[hand] = _play_hand(env, encoder, net, opponent)
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


def evaluate_paired(
    net_a: PolicyValueNet,
    net_b: PolicyValueNet,
    hands: int,
    *,
    seed: int = 10_000,
    opponent_name: str = "passive",
    opponent: Opponent | None = None,
) -> PairedEvaluationResult:
    """Play two networks on the same deals and summarise A's payoff minus B's.

    Each hand is played twice, both times after ``env.seed(seed + hand)``.  RLCard
    rebuilds its dealer, players and judger from that seed inside ``reset()``, so
    both networks see the *same cards and the same blinds*.  :func:`reset_hand`
    also clears the previous hand's raise counts, which RLCard otherwise leaves
    attached to the opening state, so the first observation of the hand matches
    too.  Their actions may diverge, but the luck of the deal is held fixed.
    That is what makes this interval tighter than comparing two independent
    :func:`evaluate` calls.

    By default the opponent is rebuilt for each hand from ``opponent_name`` and a
    per-hand seed, so a stochastic scripted opponent (``raiser_loose``,
    ``random``) also makes the same decisions for both networks.  Pass an
    explicit ``opponent`` callable to override that; a callable with internal
    randomness is then only paired on the deal, not on its own draws.

    Raises:
        ValueError: If ``hands`` is smaller than 2, or if no ``opponent`` is
            given and ``opponent_name`` is not one of :data:`OPPONENT_NAMES`.
    """

    if int(hands) < 2:
        raise ValueError(f"hands must be at least 2, got {hands!r}")
    if opponent is None and opponent_name not in OPPONENT_NAMES:
        raise ValueError(
            f"unknown opponent {opponent_name!r}; "
            f"choose one of {', '.join(OPPONENT_NAMES)}"
        )

    env = make_environment(seed=int(seed))
    encoder = ObservationEncoder.from_environment(env)
    payoffs_a = np.empty(int(hands), dtype=np.float64)
    payoffs_b = np.empty(int(hands), dtype=np.float64)

    was_training_a = net_a.training
    was_training_b = net_b.training
    net_a.eval()
    net_b.eval()
    try:
        for hand in range(int(hands)):
            deal_seed = int(seed) + hand
            if opponent is None:
                # A fresh opponent per play, not one shared object: a stochastic
                # style draws from its own RNG, and sharing it would let network
                # A's decisions advance the stream network B sees.
                opponent_a = make_opponent(opponent_name, seed=deal_seed)
                opponent_b = make_opponent(opponent_name, seed=deal_seed)
            else:
                opponent_a = opponent_b = opponent

            env.seed(deal_seed)
            payoffs_a[hand] = _play_hand(env, encoder, net_a, opponent_a)

            env.seed(deal_seed)
            payoffs_b[hand] = _play_hand(env, encoder, net_b, opponent_b)
    finally:
        net_a.train(was_training_a)
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
    progress: Callable[
        [dict[str, float | int], tuple[EvaluationRecord, ...] | None], None
    ]
    | None = None,
) -> TrainingResult:
    """Run ``config.iterations`` collect-update steps, evaluating periodically.

    Args:
        config: Length, sizes, seeds, opponent mix and output locations.
        net: Optional starting network.  ``None`` initialises one under
            ``config.seed``.
        progress: Called after every iteration with the training row and the
            evaluations just completed (one record per opponent), if any.

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

        def record_evaluation(iteration: int) -> tuple[EvaluationRecord, ...]:
            """Evaluate every configured opponent, log each row, save once."""

            completed: list[EvaluationRecord] = []
            for name in config.evaluation_opponents:
                result = evaluate(
                    net,
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

            path = save_checkpoint(
                net,
                config,
                iteration,
                Path(config.checkpoint_dir)
                / f"{config.run_name}_{iteration:04d}.pt",
            )
            checkpoints.append(path)
            return tuple(completed)

        # Iteration 0 is a real measurement of the starting network.  Without
        # it the first reported change needs a reconstructed network and a
        # replayed evaluation, which is work the harness should be doing.
        record_evaluation(0)

        for iteration in range(1, int(config.iterations) + 1):
            # Mix self-play and the scripted opponent inside one rollout, then
            # make one update from the combined batch.  Both halves are needed:
            # self-play keeps the policy honest against itself, the opponent
            # supplies the raising decisions self-play may never discover.  A
            # single update also keeps the two experiences on the same value
            # scale instead of alternating between two objectives.
            total_hands = int(config.hands_per_iteration)
            self_play_hands = int(round(total_hands * config.self_play_fraction))
            opponent_hands = total_hands - self_play_hands

            trajectories = []
            if self_play_hands:
                trajectories.extend(
                    collect_trajectories(
                        env, encoder, net, hands=self_play_hands, rng=rng
                    )
                )
            if opponent_hands:
                # A fresh stream each iteration, but derived from the run seed,
                # so the opponent is stochastic *and* the run is reproducible.
                opponent = make_opponent(
                    config.training_opponent,
                    seed=int(config.seed) + 7919 * iteration,
                )
                trajectories.extend(
                    collect_trajectories(
                        env,
                        encoder,
                        net,
                        hands=opponent_hands,
                        rng=rng,
                        opponent=opponent,
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
