# Poker AI Arena

Heads-up Limit Texas Hold'em as a controlled testbed for reinforcement learning.

Two agents will eventually be trained with PPO and compared:

| | Agent A | Agent B |
|---|---|---|
| Network | feed-forward MLP | MLP + GRU |
| Input | current observable state | current observable state **+ betting history** |
| Question | baseline | does memory of opponent actions help? |

**Research question:** *Does recurrent memory of opponent actions improve
reinforcement-learning performance in heads-up poker?*

**Current stage: Stage 1 — the simulator contract is verified, and Agent B's
betting-history encoding is implemented and tested. There is no neural network,
PPO, GRU, self-play or opponent-modelling code yet.**

Stage 1 exists so later results are not built on guesses about RLCard's
observations, actions, payoffs or history handling. The history encoding is data
preparation only: it turns actions into fixed-size arrays and learns nothing.

---

## Repository layout

```
poker-ai-arena/
├── src/
│   └── poker_arena/
│       ├── __init__.py
│       ├── encoding.py             # Agent A's 131-number state vector
│       ├── environment.py          # Stage 1 inspection helpers
│       ├── history.py              # Agent B's betting-history tokens
│       ├── networks.py             # Agent A's policy/value MLP
│       ├── ppo.py                  # clipped PPO update
│       ├── rollout.py              # rollout buffer + GAE for the PPO update
│       └── training.py             # training loop + fixed-seed evaluation
├── scripts/
│   ├── test_encoding.py            # executable encoder report
│   ├── test_environment.py         # executable smoke test / report
│   ├── test_history.py             # executable history-encoding report
│   ├── test_networks.py            # executable network report
│   ├── test_ppo.py                 # executable PPO-update report
│   ├── test_rollout.py             # executable rollout / GAE report
│   ├── plot_training.py            # reads a run's CSVs, writes a PNG
│   └── train_agent_a.py            # trains Agent A and evaluates it
├── tests/
│   ├── conftest.py                 # puts src/ on sys.path for pytest
│   ├── test_encoding.py            # encoder tests
│   ├── test_environment.py         # Stage 1 contract tests
│   ├── test_history.py             # history-encoding tests
│   ├── test_networks.py            # network tests
│   ├── test_ppo.py                 # PPO-update tests
│   ├── test_rollout.py             # rollout and GAE tests
│   └── test_training.py            # training-loop and evaluation tests
├── .venv/                          # local environment (git-ignored)
├── docs/
│   └── agent_a_curve.png           # figure from the first full run
├── .gitignore
├── pytest.ini                      # limits collection to tests/
├── requirements.txt
└── README.md
```

The maintained workflow is plain Python modules, scripts and pytest files — there
are no notebooks. An early exploratory notebook was removed from the repository;
it is still recoverable from commit `f8a4a0a` if ever needed. The empty
pre-existing `tets_poker/` folder is left on disk but is not tracked.

---

## Setup

Windows PowerShell. Adjust the path if the project is moved.

```powershell
cd "C:\Users\laksh\OneDrive\Documents\Default Project\poker-ai-arena"

# Create the environment only if .venv does not already exist.
python -m venv .venv

# Activate it.
.\.venv\Scripts\Activate.ps1

# Install the pinned dependencies.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Verified with Python 3.14.6, `rlcard` 1.2.0, `numpy` 2.5.4, `pytest` 9.2.0,
`torch` 2.15.0 and `matplotlib` 3.11.2. Stage 1 imports only `rlcard`, `numpy`
and `pytest`; Agent A's encoder, network and rollout modules add `torch`.
`matplotlib` is installed for later plots.

---

## Running the smoke test and tests

From the project root, with the virtual environment activated:

```powershell
python scripts\test_environment.py
python scripts\test_history.py
python scripts\test_encoding.py
python scripts\test_networks.py
python scripts\test_rollout.py
python scripts\test_ppo.py
python -m pytest -q

# Train Agent A, then evaluate as it goes.
python scripts\train_agent_a.py

# Plot the run's CSV logs to runs\agent_a_curve.png.
python scripts\plot_training.py
```

Without activation:

```powershell
.\.venv\Scripts\python.exe scripts\test_environment.py
.\.venv\Scripts\python.exe scripts\test_history.py
.\.venv\Scripts\python.exe scripts\test_encoding.py
.\.venv\Scripts\python.exe scripts\test_networks.py
.\.venv\Scripts\python.exe scripts\test_rollout.py
.\.venv\Scripts\python.exe scripts\test_ppo.py
.\.venv\Scripts\python.exe -m pytest -q
```

The training script takes flags if the defaults are too slow for a quick check:

```powershell
# A quick smoke run: about 15 seconds.
python scripts\train_agent_a.py --iterations 3 --hands 64 `
  --eval-hands 200 --eval-every 1 --run-name smoke

# The full run, writing runs\agent_a_*.csv and checkpoints\agent_a_*.pt.
python scripts\train_agent_a.py --iterations 50 --hands 320
```

The environment smoke test ends with `All Stage 1 checks passed.`, the history
smoke test with `All history-encoding checks passed.`, the encoder report with
`All observation-encoder checks passed.`, the network report with
`All network checks passed.`, the rollout report with
`All rollout checks passed.`, the PPO report with `All PPO checks passed.`, and
the test suite currently reports `145 passed`.

`pytest.ini` limits collection to `tests/` because
`scripts/test_environment.py` and `tests/test_environment.py` share a filename
(and likewise for `test_history`). Both scripts and `tests/conftest.py` add
`src/` to `sys.path`, so no package installation is required.

### What the smoke test covers

1. **Environment contract** — two players, four actions, observation shape.
2. **Initial state** — a fresh `reset()`, before any action: two hole cards,
   no public cards, empty action history.
3. **Gameplay states** — one complete hand with uniformly random legal actions,
   printing every decision state and the final payoffs.
4. **Consecutive hands** — 100 hands on one environment, checking termination,
   zero-sum payoffs and that `reset()` clears the action recorder.
5. **Information hiding** — at every decision, both players' observations are
   checked against perfect information. Only the player's own hole cards and the
   public cards may appear.
6. **API summary** — the verified simulator contract.

`scripts/test_history.py` plays a tracked hand, prints the token table it
produced, and then checks 200 random hands: token order matches
`env.action_recorder`, masks are prefixes with no padding leaking through, all
features stay in `[0, 1]`, and nothing exceeds the 32-token capacity.

`scripts/test_encoding.py` prints Agent A's 131-number vector block by block for
one hand, then checks 100 random hands: every value stays in `[0, 1]`, all four
streets are visited, the legal mask matches the environment at every decision,
and the pot feature reaches its ceiling.

---

## Agent A observation encoder

`encoding.py` turns `state["raw_obs"]` into the single 131-number vector the
policy/value MLP reads. Anything the network can ever know about a state is in
this vector, so it is also the boundary of the comparison with Agent B: Agent B
receives these same features plus the action tokens from `history.py`.

| Index | Width | Feature | Why it is there |
|---|---|---|---|
| 0–51 | 52 | own hole cards, multi-hot | card identity without implying an order |
| 52–103 | 52 | public cards, multi-hot | RLCard merges both into `obs[0:52]`, so holding A on a K-high board would look identical to holding K on an A-high board |
| 104–123 | 20 | 4 streets × 5 raise slots | already in the observation; kept one-hot so no ordering is pre-committed |
| 124–127 | 4 | street one-hot | recoverable from the board size, but stating it explicitly is 4 dims |
| 128 | 1 | pot ÷ 48 bb | absent from the observation |
| 129 | 1 | chips to call ÷ 2 bb | absent from the observation |
| 130 | 1 | posted the small blind | RLCard randomises the blinds every hand |

All values are in `[0, 1]` and `float32`. The two ceilings are measured rather
than guessed: `pot` reaches 48 bb and `to_call` reaches 2 bb under the 4-raise
cap. `ObservationEncoder.from_environment(env)` checks those betting parameters
and refuses to build the encoder if they change, because a silently saturated
feature is worse than a failed start.

```python
encoder = ObservationEncoder.from_environment(env)
net = PolicyValueNet()

state, player_id = env.reset()
small_blind = small_blind_player(state)          # once per hand
small = player_id == small_blind

while not env.is_over():
    encoded = encoder.encode(state, is_small_blind=small)
    logits, value = net(
        torch.from_numpy(encoded.features).unsqueeze(0),   # (1, 131)
        torch.from_numpy(encoded.legal_mask).unsqueeze(0),  # (1, 4)
    )                                                   # (1, 4), (1,)
    action = Categorical(logits=logits).sample()
    ...
```

The legal-action mask comes back alongside the features and is **applied to the
logits, never fed in as an input**. Enforcing it at the output is exact; teaching
the network to respect it from a feature is not. The mask is copied from
`state["legal_actions"]`, so the encoder cannot disagree with RLCard about what
is legal, and `PolicyValueNet.forward` applies it before the logits leave the
module.

`small_blind_player()` must be called on the state from `reset()`, because the
blind postings are the only moment the chip contributions identify the seats. It
raises if handed a mid-hand state, where the chips are already level or
re-raised.

---

## Agent A policy/value network

`networks.py` holds one `nn.Module`: a shared trunk that reads the 131 numbers
and two heads that read the trunk.

| Part | Shape | Parameters |
|---|---|---|
| trunk | `131 → 256 → 256`, ReLU | 99,584 |
| policy head | `256 → 4`, no softmax | 1,028 |
| value head | `256 → 1` | 257 |
| **total** | | **100,869** |

Orthogonal initialisation: gain `√2` on the trunk, **0.01** on the policy head,
1.0 on the value head. The small policy gain keeps the first policy close to
uniform instead of committing to one action on random weights.

```python
net = PolicyValueNet()
logits, value = net(features, legal_mask)   # (B, 4) masked, (B,)

dist = torch.distributions.Categorical(logits=logits)
action = dist.sample()
log_prob = dist.log_prob(action)            # what PPO needs
```

The module returns **logits, not probabilities**, because PPO wants `log_prob`
and `entropy`, and `log_softmax` is more numerically stable than `log(probs)`.
It does not sample, compute a loss or keep bookkeeping — that is the rollout's
job.

### The mask lives inside `forward`

`legal_mask` is a required argument. There is no unmasked path, because the
alternative fails silently in three separate places. RLCard **substitutes** an
illegal action instead of rejecting it (`envs/limitholdem.py:90-96`: play
`check` if possible, otherwise `fold`). Measured on an untrained network, at a
state where the raise cap is reached and the only legal actions are call and
fold:

```text
unmasked probabilities : call 0.250  raise 0.250  fold 0.250  check 0.250
unmasked draws of raise: 2508 / 10000
environment executed   : 'fold'      <- not an error, a substitution
masked draws of raise  : 0 / 10000
```

So a policy that thinks it is raising **folds the hand**. And if the update
remembers the mask while sampling forgot it, the stored log-probability belongs
to an action that never happened: the ratio
`exp(logπ_new(raise) − logπ_old(raise))` becomes `exp(−1e9 + 1.3)`, which
underflows to exactly `0`. The sample then contributes nothing — no NaN, no
warning — so exactly the rarest cap states are quietly dropped from training.

Illegal logits become `-1e9`, not `-inf`. Both give probability exactly `0` and
both leave the chosen action's log-probability untouched, but `-inf` makes the
entropy `0 × -inf = NaN`, and one NaN poisons every gradient in the batch.

Consequences the tests pin down: probabilities of illegal actions are exactly
`0.0`, 10,000 samples from a capped state never return `raise`, the gradient at a
masked logit is exactly `0.0` while the shared trunk still receives gradient, and
the value head is unaffected by the mask. A mask row with no legal action raises
instead of turning every logit into `-1e9` and silently producing a uniform
distribution.

`scripts/test_networks.py` prints the architecture, one real forward pass, the
masked-vs-unmasked comparison above, and per-layer gradient magnitudes for one
PPO-style loss.

---

## Rollout collection and credit assignment

A poker hand pays out once. Every decision inside the hand receives reward `0`
and the hand's payoff arrives after the last action. The obvious rollout gives
every decision the same terminal number and subtracts its baseline:

```text
A_t = R - V(s_t)
```

On a three-decision hand that loses 3 bb, with `V` rising because the board
helped:

```text
 t  decision                       V(s_t)    R - V
 0  preflop call (value rose)        0.00    -3.00
 1  flop check  (value rose)         1.00    -4.00
 2  river fold  (value collapsed)    1.50    -4.50
```

Every decision is punished, including the two that **raised** the value
estimate. Trained on this, the agent learns to stop calling and to stop checking
when it improves — the opposite of the truth.

`rollout.py` assigns credit with temporal-difference residuals instead:

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t     = delta_t + (gamma * lambda) * A_{t+1}
```

`delta_t` asks "did the state get better or worse than the value head expected?",
which each decision can answer on its own:

```text
 t  decision                       V(s_t)   delta    lam=0   lam=.95   lam=1
 0  preflop call (value rose)        0.00    1.00     1.00     -2.59   -3.00
 1  flop check  (value rose)         1.00    0.50     0.50     -3.78   -4.00
 2  river fold  (value collapsed)    1.50   -4.50    -4.50     -4.50   -4.50
```

`lambda = 1` reproduces `R - V(s_t)` **exactly** (the test asserts the identity),
and `lambda = 0` blames only the transition that moved the value. `gamma = 1.0`
because chips do not decay and the objective is the undiscounted expected payoff.

The table is also the honest caveat: over a four-decision hand the terminal
shock still reaches the first decision with weight `0.95³ ≈ 0.86`, so `0.95`
sits much closer to the naive version than the intuition suggests. `0.95` is a
reasonable starting point, not a solved choice; the PPO step will report the
advantages it actually sees.

There is no version of this that adds information. Rewarding pot growth teaches
the agent to put chips in and never fold, and the only per-decision shaping that
provably preserves the optimal policy is potential-based; with `phi = V` it
reduces to the TD residual above.

### What is stored

One trajectory **per recorded player per hand**, because `V(s)` is from the
acting player's point of view. In self-play both seats are recorded and their
payoffs are `+R` and `-R`.

```python
@dataclass(frozen=True)
class Transition:
    features: np.ndarray    # (131,) float32, from the encoder
    legal_mask: np.ndarray  # (4,) bool, copied at decision time
    action: int
    log_prob: float         # read from the same forward pass that sampled
    value: float            # same forward pass

@dataclass(frozen=True)
class Trajectory:
    transitions: tuple[Transition, ...]
    final_payoff: float     # big blinds, this player's perspective
```

`log_prob` and `value` are captured at sampling time on purpose. Recomputing
them with updated weights makes the PPO ratio meaningless.

```python
trajectories = collect_trajectories(env, encoder, net, hands=25, rng=seeded)
buffer = RolloutBuffer()
for trajectory in trajectories:
    buffer.add_trajectory(trajectory)
batch = buffer.batch()          # (N, 131), (N, 4), (N,)
normalised = normalize_advantages(batch.advantages)   # inside the PPO step
```

`opponent=None` means self-play and records both seats; passing a callable
`(state, player_id) -> action_id` makes it play player 1, so only player 0 (the
learner) is recorded. A hand where the small blind folds preflop leaves the big
blind with zero decisions; those are dropped rather than stacked, because GAE
over zero steps is `0/0` and one empty row would poison the batch.

`RolloutBuffer.batch()` returns **raw** advantages. Normalising here would
silently break `returns == advantages + values`; the PPO step normalises a copy
after computing both.

### What the tests pin down

`tests/test_rollout.py` (30 tests) asserts the `lambda = 1` identity, the
`lambda = 0` residual, both columns of the worked example, `returns ==
advantages + values` for every `lambda`, that the terminal transition
bootstraps from `0` (a losing last decision cannot inherit a phantom
continuation value), that an over-optimistic value estimate is corrected by the
payoff, normalisation to mean `0` / std `1`, zeros instead of a division by zero
on a flat batch, buffer stacking order and dtypes, rejection of bad widths,
non-finite scalars, illegal stored actions and empty trajectories, that stored
log-probabilities reproduce a fresh forward pass, that collected actions are
legal, that self-play payoffs are zero-sum and equal `env.get_payoffs()`, that a
callable opponent only ever plays player 1, and that the same seeds reproduce
identical rollouts.

`scripts/test_rollout.py` prints the tables above, one real hand decision by
decision, and a 25-hand batch summary. A representative batch: 47 trajectories,
82 decisions, zero illegal stored actions, raw advantage std `4.30` bb — one
argument for normalising before the update.

---

## PPO update

`ppo.py` optimises one rollout batch at a time:

```text
L_policy = -mean( min(ratio * A, clamp(ratio, 1-eps, 1+eps) * A) )
L_value  =  mean( (V(s) - return)^2 )
L        =  L_policy - entropy_coef * H[pi] + value_coef * L_value
ratio    =  exp(log_pi_new(a) - log_pi_old(a))
```

| Setting | Value |
|---|---|
| clip epsilon | 0.2 |
| entropy coefficient | 0.01 |
| value coefficient | 0.5 |
| epochs | 4 |
| minibatch size | 256 |
| max gradient norm | 0.5 |
| optimiser | Adam, `3e-4` |

### The old log-probability must be a constant

On the first pass the network that samples and the network being scored are the
same, so `ratio` is exactly `1` and the clip cannot bind. The clip exists for the
later epochs: once the weights have moved, it stops one batch from being
over-exploited.

That only works because `log_pi_old` is a stored constant. Recompute it from the
current weights and `ratio` becomes `exp(0) = 1` on every pass; the clip never
binds, no error is raised, and PPO quietly degrades into a policy gradient that
can walk arbitrarily far from the sampling policy. The report makes this
visible by shifting the stored log-probabilities by `5`:

```text
fresh log-probs : clip fraction 0.0000, approx KL 5.81e-03
stale log-probs : clip fraction 1.0000, approx KL 4.0087
```

### Two details that are easy to get wrong

* **Advantages are normalised once per rollout**, before the epochs start, not
  per minibatch. Per-minibatch normalisation changes what the same sample means
  on every pass. A test asserts that multiplying every advantage by 1,000 leaves
  the update unchanged.
* **Returns stay raw.** They are the value targets; rescaling them would change
  the value loss's scale. `returns == advantages + values` from the buffer.

The value loss is deliberately **not** clipped. Clipping the value is a second
mechanism with its own failure modes; it is worth adding only if the value loss
is observed to explode. Global gradient-norm clipping is there from the start —
in the report the pre-clip norm is `1.01` against a limit of `0.5`, so the clip
is doing real work.

### What the report measured

A 320-hand self-play rollout (about 1,000 decisions, value loss `10.38`, entropy
`1.096`, zero illegal stored actions), then four epochs:

```text
value loss  10.3786 -> 10.0110      entropy 1.0962 -> 1.0903
clip fraction 0.0000                approx KL 0.0017
grad norm (pre-clip) 1.0059         illegal probability 0.0
```

The value loss falls; the policy barely moves in one update, which is expected
at `3e-4` with an advantage signal this noisy.

### A measured footgun: Adam's step size versus the entropy bonus

An early version of the entropy test used `learning_rate=1e-2`. With a strong
entropy coefficient the entropy **fell** (`1.3805 -> 0.6708`) instead of rising.
The gradient pointed the right way; Adam normalises each parameter's step to
roughly the learning rate regardless of gradient magnitude, so the policy
overshot the near-uniform maximum. The test now uses `1e-3`. The lesson is not
"entropy is broken" but "with Adam, a large learning rate can turn any bonus
into an overshoot", so the reported learning rate is part of the objective's
behaviour, not just a training detail.

### Diagnostics

`PPOUpdater.step()` returns `policy_loss`, `value_loss`, `entropy`,
`total_loss`, `approx_kl`, `clip_fraction`, `grad_norm` (all means over the
minibatches), plus `epochs`, `minibatches` and `samples`. A single loss number
hides which term is misbehaving; `approx_kl` and `clip_fraction` are the two
that say whether the update is still inside its trust region.

### What the tests pin down

`tests/test_ppo.py` (23 tests) pins the defaults and rejects invalid settings,
checks that an update changes parameters and returns finite diagnostics, that
fresh log-probabilities start with `ratio == 1` (clip fraction `0`, KL `0`) and
stale ones clip every sample, that zero advantages leave the policy head
bit-identical while the value head still learns, that the value loss moves the
value towards its targets, that the entropy bonus raises entropy from a sharp
policy, that advantage scaling is invariant, that unnormalised huge advantages
engage gradient clipping, minibatch accounting, seeded reproducibility, that the
network is left in training mode, batch validation, illegal actions staying at
exactly zero probability after updates, and a full rollout → update → rollout
cycle.

---

## Training loop and evaluation

`training.py` connects the pieces without adding a new dependency:

```text
for iteration in 1..N:
    collect 320 self-play hands                 -> ~1,000-4,000 decisions
    RolloutBuffer.batch()                       -> raw advantages, returns
    PPOUpdater.step()                           -> one update, 4 epochs
    write the diagnostics row to CSV
    every 10 iterations, and on the last one:
        evaluate 2,000 fixed-seed hands vs a scripted opponent
        write the evaluation row and a checkpoint
```

The decision count is not constant: it depends on how long the current policy
keeps a hand alive, which changes as it learns.

### Why evaluation is separate from the training reward

The training batch is the data the policy is being fitted to, so its average
payoff is an optimistic, high-variance number. Evaluation replays the *same*
deals, from the same `evaluation_seed`, against the *same* scripted opponent
every time. A change in that column is a change in the policy, not in the deal
or the opponent.

Evaluation plays **greedily** (`argmax` over the masked logits). The sampling
policy is what is trained, but the mode is what a fixed-seed comparison should
measure: sampling would add action noise on top of deal noise and make two runs
with the same seed diverge for no reason.

### How many hands does a measurement need?

The per-hand standard deviation is **not a property of the game** — it depends on
how many chips the current policy puts in. In the first run it was `1.6` bb at
iteration 10 and `5.1` bb at iteration 50. The 95 % interval is
`1.96 × σ / sqrt(n)`:

| Hands | Interval at σ = 1.6 | Interval at σ = 5.1 |
|---|---|---|
| 500 | `±0.140` | `±0.446` |
| 2,000 | `±0.070` | `±0.223` |
| 10,000 | `±0.031` | `±0.100` |
| 40,000 | `±0.016` | `±0.050` |

So against the trained policy, resolving a `0.05` bb edge needs about **40,000**
hands, not the ~9,000 an earlier estimate of σ ≈ 2.4 bb suggested. That estimate
came from the untrained policy, which played short, small pots. The harness
computes `stderr` from the sample it actually observed, so the printed interval
stays correct; only the planning number was wrong.

This is also why the interval is printed next to every evaluation and why a
small movement between checkpoints should not be read as progress. It rules out
the training reward as a progress signal too: it is one noisy sample per
iteration, from a moving opponent and a changing policy.

### Outputs

| File | Contents |
|---|---|
| `runs/<run-name>_train.csv` | one row per iteration: decisions, losses, entropy, KL, clip fraction, gradient norm, minibatches, samples |
| `runs/<run-name>_eval.csv` | one row per evaluation: hands, opponent, mean bb, standard error, 95 % interval, total bb |
| `checkpoints/<run-name>_<iter>.pt` | `state_dict` + iteration + the full training config |

`save_checkpoint` / `load_checkpoint` round-trip the weights exactly; the config
is stored so an evaluation can be reproduced from the file alone. The test suite
asserts that a checkpoint's weights reproduce the evaluation it came from.

### Reproducibility

The network initialisation, the environment, the action sampler and the
minibatch shuffle are all seeded from `config.seed`. Two runs with the same
config produce identical CSVs and identical final weights; the test suite
compares every parameter tensor. The evaluation environment uses a separate
`evaluation_seed`, so changing the training seed does not change the deals the
evaluations are played on.

### What the tests pin down

`tests/test_training.py` (20 tests) pins the defaults, rejects invalid settings,
checks the scripted and random opponents, that evaluation returns finite
statistics with `ci95 == 1.96 × stderr` and a zero-sum-consistent total, that it
is deterministic for the same weights and seed and restores the network's
training mode, the checkpoint round-trip and its metadata, the row, evaluation
and checkpoint counts, CSV contents, evaluation on the final iteration when it
is off schedule, that training changes the weights, the progress callback order,
and full-run reproducibility.

---

## First training run: 50 iterations

```powershell
python scripts\train_agent_a.py      # seed 0, 320 hands/iteration, 2,000-hand evaluation every 10
python scripts\plot_training.py      # -> runs\agent_a_curve.png
```

The full run took a few minutes on CPU and produced 159,035 decisions.

| Iteration | Decisions | Value loss | Entropy | Eval (bb/hand) | 95 % |
|---|---|---|---|---|---|
| 1 | 945 | 7.95 | 1.094 | — | — |
| 10 | 3,399 | 78.33 | 0.733 | −0.040 | ±0.069 |
| 15 | 4,150 | **128.07** (peak) | 0.617 | — | — |
| 20 | 3,567 | 78.50 | 0.671 | **+0.292** | ±0.172 |
| 30 | 3,354 | 56.77 | 0.633 | +0.223 | ±0.225 |
| 40 | 3,200 | 55.18 | 0.583 | +0.161 | ±0.241 |
| 50 | 3,229 | 60.57 | 0.596 | +0.230 | ±0.223 |

![Agent A's first 50-iteration training run](docs/agent_a_curve.png)

The plot is written by `scripts/plot_training.py`; the copy above is the seed-0
run.

### What improved

There is no iteration-0 evaluation — a gap in the harness. Reconstructing the
initial network (same seed, `torch.manual_seed(0)` then `PolicyValueNet()`) and
replaying the same 2,000 deals gives:

```text
initial net   : -0.1050 bb/hand  (+/- 0.1411)
final net     : +0.2295 bb/hand  (+/- 0.2228)
paired change : +0.3345 bb/hand  (+/- 0.1539 paired 95%)
```

The unpaired intervals overlap, but pairing the same deal removes most of the
variance: the paired interval excludes zero, so the improvement is resolved even
though neither individual measurement is precise to 0.33 bb. That is the whole
argument for fixed-seed evaluation, and for pairing rather than comparing two
independent means.

Decisions per hand rose from **3.0 to 10.1**: the agent stopped folding and
started playing hands to the end. Against an opponent that never folds, that is
the correct direction.

### What did not

**It plateaued at iteration 20.** Evaluations 20 → 50 are statistically
indistinguishable, and the best point estimate is the first one after the
fast-learning phase.

**The value head is nearly useless.** Replaying 320 fresh hands at each saved
checkpoint and comparing `V(s)` with the GAE return `G`:

```text
ckpt   return std   MSE     explained variance   correlation
  10       9.73     92.04        0.028             0.189
  20       9.10     83.17       -0.004             0.113
  30       8.69     73.38        0.029             0.191
  40       8.35     68.92        0.010             0.174
  50       7.93     57.58        0.085             0.300
```

Explained variance is the scale-free version of the value loss
(`1 - MSE / Var(G)`); `0` means the head is no better than predicting the mean.
It never exceeds **8.5 %**. So the baseline behind every advantage is almost
constant, and PPO is running close to a plain policy gradient with all of the
variance that implies.

**The value-loss curve is misleading.** It rose from `7.95` to a peak of
`128.07`, then fell to `60.57`. But the return standard deviation rose to `9.7`
during the same phase and then fell to `7.9`, so much of the curve is the target
moving, not the critic getting better or worse. A raw MSE on returns cannot be
compared across iterations; `explained variance` can.

**The trust region was never the problem.** Approx KL stayed below `0.0076`
(mean `0.0036`) and the clip fraction below `0.086`, while the pre-clip gradient
norm reached `15.2` against a limit of `0.5`. The actor update stayed inside its
trust region; the instability is in the critic.

**Entropy halved** (`1.094 → 0.596`) while evaluation stopped improving, which
is early convergence rather than useful exploration.

### The caveat that limits the result

The evaluation opponent calls or checks and never folds or raises. Beating it
means "loses less to a passive caller", not "plays well". The observed gain is
consistent with learning not to fold, which is a degenerate best response. A
raising opponent is needed before any play-quality claim.

---

## Verified simulator contract (RLCard 1.2.0)

These facts were checked by running code, not copied from documentation.

| Item | Verified behaviour |
|---|---|
| Environment | `rlcard.make("limit-holdem", config={"seed": ...})` |
| Players | 2 |
| Actions | 4: `call`, `raise`, `fold`, `check` |
| Observation | `np.ndarray`, shape `(72,)`, `float64`, one-hot |
| `obs[0:52]` | acting player's hole cards + public cards |
| `obs[52:72]` | 4 betting rounds × 5 raise-count slots |
| `reset()` | returns `(state dict, acting player id)` and clears `action_recorder` |
| `step(id)` | returns `(next state dict, next player id)` |
| `run(is_training)` | returns `(trajectories, payoffs)` |
| Payoffs | `np.ndarray`, chips ÷ big blind, zero-sum, multiples of `0.5` |
| `legal_actions` | `OrderedDict` mapping legal action ID → `None` |
| `raw_legal_actions` | action names in the same order as `legal_actions` |
| `raw_obs` | `hand`, `public_cards` (strings such as `"D2"`), `all_chips`, `my_chips`, `legal_actions`, `raise_nums` |
| `action_recorder` | list of `(player_id, action_name)`; `reset()` clears it |
| Blinds | the small blind is drawn at random each hand (`np_random.randint`) |
| Street order | each street opens with the player after whoever closed the previous one |
| Observations during play | the acting player's own cards + public cards only; opponent hole cards never appear |

### Action IDs

Read these from `env.actions`; do not hard-code them in agent code.

| ID | Name |
|---|---|
| 0 | `call` |
| 1 | `raise` |
| 2 | `fold` |
| 3 | `check` |

### Observation layout

* `obs[0:52]` is a one-hot over the 52-card deck. The order is given by
  `env.card2index`; decode it by inverting that mapping.
* `obs[52:72]` is four consecutive blocks of five slots. Slot `n` is set when
  that betting round has seen `n` raises.
* All four blocks are one-hot from the start of the hand. Future rounds simply
  read as `0` raises, so the preflop view is `(0, 0, 0, 0)`.

### Information-hiding check

For every decision in a complete hand, the smoke test verifies:

```
observed cards == player's own hole cards ∪ public cards
observed cards ∩ opponent's private cards == ∅
```

The two players' observations differ only inside the card region (`index < 52`);
public information is identical for both. `env.get_perfect_information()` is used
for this check only — it exposes both hands and must never be an agent input.

---

## Agent B history encoding

`src/poker_arena/history.py` turns a hand's actions into the input Agent B's GRU
will read. One action becomes one 16-number token:

| Index | Width | Feature |
|---|---|---|
| 0–1 | 2 | actor one-hot: `P0`, `P1` |
| 2 | 1 | `is_self` — the actor is the player deciding now |
| 3–6 | 4 | action one-hot: `call`, `raise`, `fold`, `check` |
| 7–10 | 4 | street one-hot: preflop, flop, turn, river |
| 11 | 1 | raises on this street ÷ 4 |
| 12 | 1 | pot in big blinds ÷ 48 |
| 13 | 1 | chips to call in big blinds ÷ 2 |
| 14 | 1 | actor posted the small blind |
| 15 | 1 | actor is not first to act on this street (in position) |

A hand becomes `tokens: float32[32, 16]`, `mask: bool[32]`, `length: int`:

* Rows after `length` are zero padding, and `mask` is `False` there, so the GRU
  can never mistake filler for a real action.
* An empty history — the first decision of a hand — is an all-`False` mask. The
  GRU starts from its zero hidden state; there is no special start token.
* If a hand ever exceeds 32 actions, the oldest tokens are dropped. Measured
  maxima: 20 actions under an always-raise policy, mean 3.2 under random play.
* Tokens are encoded **per viewer**, so the same action stream gives P0 and P1
  different `is_self` flags. Only index 2 differs between the two views.

```python
tracker = HistoryTracker()
state, player_id = env.reset()
tracker.start_hand(state)

while not env.is_over():
    sequence = tracker.tokens(viewer=player_id)  # (32, 16) + mask + length
    action_id = policy(sequence, state)
    tracker.record(state, player_id, env.actions[action_id])  # before step()
    state, player_id = env.step(action_id)
```

`record()` must be called **before** `step()`: pot, call amount and street are
snapshots of the state the decision was made in, and RLCard's own
`action_record` does not keep them.

The `is_small_blind` and `is_in_position` flags are not redundant. RLCard draws
the small blind at random every hand, and a street opens with whoever follows the
player who closed the previous street, so neither fact follows from the actor id
or the street number. Dropping them would hide information the network cannot
otherwise recover at the first decision of a hand.

---

## Known pitfalls

1. **`state["action_record"]` is a live list.** It is the same object as
   `env.action_recorder`. A rollout buffer that stores states without copying the
   history will later see the history of the whole hand. Copy it with
   `list(state["action_record"])`.
2. **Illegal actions are silently substituted — and the fallback is `fold`.**
   `envs/limitholdem.py:90-96` plays `check` if it is legal, otherwise `fold`,
   instead of raising. At the raise cap `check` is not legal, so a policy that
   samples an illegal `raise` *folds the hand* and the buffer records a raise.
   Measured on an untrained network: 25% of cap-state samples hit this.
   `PolicyValueNet` therefore applies the legal-action mask inside `forward`.
3. **The environment seed only controls the dealer.** `RandomAgent` draws from
   NumPy's global RNG, so reproducible random play also needs `np.random.seed()`
   or an explicit `np.random.Generator`.
4. **`env.timestep` accumulates across hands.** `reset()` does not clear it.
5. **`env.get_perfect_information()` leaks both hands.** Debugging only.
6. **The observation does not encode who bet, in what order, or from which
   position.** It records only the current cards, chips and raise counts per
   round. This missing history is exactly what Agent B's GRU is intended to
   model, and `history.py` supplies it.
7. **`raw_obs["raise_nums"]` is a live list**, the same object RLCard keeps as
   `history_raise_nums`. It keeps changing after the decision, so copy it
   (`list(...)`) before storing it. `HistoryTracker` reads it immediately, so
   tokens always carry the count that was true at decision time.
8. **`raw_obs` holds card *strings*; `obs` holds slot *numbers*.**
   `raw_obs["hand"]` is `['SQ', 'C7']` while `obs[0:52]` is a 52-slot array, and
   `env.card2index` is the only mapping between them. Code that assumes
   `raw_obs` already contains slot indices fails with
   `invalid literal for int(): 'SQ'`. `ObservationEncoder` captures
   `env.card2index` once for this reason.

---

## Current limitations

* The first 50-iteration run exists and is reported above, but it is one seed
  against one passive opponent. It improved `-0.105` → `+0.230` bb/hand and then
  plateaued after iteration 20.
* The value head explains only `0–8.5 %` of the return variance. The baseline
  behind every advantage is nearly constant, so the policy gradient is much
  noisier than it needs to be. This is the current bottleneck.
* The training loop has no iteration-0 evaluation, so the starting point is
  reconstructed rather than logged.
* The encoder, the policy/value network, the rollout buffer, the PPO update and
  the training loop all exist; Agent B's history tokens are encoded and unused.
* The evaluation opponent calls or checks, so it never folds a bet and never
  raises. It measures "does the agent beat a passive caller", not "is the agent
  strong". The observed gain is consistent with learning not to fold.
* The `lambda = 0.95` choice is a starting point, not a measurement: on a
  four-decision hand the terminal shock still reaches the first decision with
  weight `0.95³ ≈ 0.86`.
* Credit for chance and opponent actions is attributed to the learner's previous
  action. The estimate is unbiased over many hands but noisy within one.
* The smoke test uses uniformly random agents, not competent baselines.
* Only heads-up **limit** Hold'em is set up; No-Limit is deliberately not used.
* No GUI, online play or real-money connection.

---

## Roadmap and recommended next milestone

| Stage | Work | Status |
|---|---|---|
| 1 | Verify simulator; document observations, actions, payoffs and history; encode Agent B's tokens | ✅ done |
| 2 | **Agent A: observation encoding + MLP policy trained with PPO** | **in progress — trains and improves, but the critic is weak** |
| 3 | Agent B: MLP + GRU over betting history, trained with PPO | planned |
| 4 | Head-to-head evaluation, ablations and results write-up | planned |

**Recommended next milestone: diagnose the value head.**

The first full run improved from `-0.105` to `+0.230` bb/hand and then
plateaued, while the critic explained at most `8.5 %` of the return variance.
Agent B would inherit the same broken baseline, so this comes first.

A sensible remaining Stage 2 scope:

1. ~~Build the observation encoder: 131 numbers, masking, bounds checked.~~ ✅
2. ~~Build the policy/value MLP (shared 256×2 trunk, `256→4` and `256→1`
   heads) with the legal-action mask applied inside `forward`.~~ ✅
3. ~~Collect rollouts: encoded states, sampled actions, sampling-time
   log-probabilities and values, terminal-only rewards, GAE advantages.~~ ✅
4. ~~Implement the PPO update: clipped ratio, entropy bonus, value-loss
   coefficient ≈ 0.5, advantage and return normalisation, global gradient-norm
   clipping.~~ ✅
5. ~~Build the training loop, CSV logging, periodic checkpoints.~~ ✅
6. ~~Add a fixed-seed evaluation harness with confidence intervals.~~ ✅
7. ~~Run the full 50-iteration job and plot the curve.~~ ✅
8. Add an iteration-0 evaluation and log `explained variance` per iteration, so
   the critic's quality is measured on the same axis throughout.
9. Sanity-check the value head offline: on one fixed batch, fit `V` with plain
   supervised regression for many steps. If that cannot reach high explained
   variance, the problem is the features or the architecture, not PPO.
10. Only after the baseline is useful, add the GRU history encoder for Agent B
    and compare the two under the same evaluation protocol.
