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
│       └── networks.py             # Agent A's policy/value MLP
├── scripts/
│   ├── test_encoding.py            # executable encoder report
│   ├── test_environment.py         # executable smoke test / report
│   ├── test_history.py             # executable history-encoding report
│   └── test_networks.py            # executable network report
├── tests/
│   ├── conftest.py                 # puts src/ on sys.path for pytest
│   ├── test_encoding.py            # encoder tests
│   ├── test_environment.py         # Stage 1 contract tests
│   ├── test_history.py             # history-encoding tests
│   └── test_networks.py            # network tests
├── .venv/                          # local environment (git-ignored)
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
`torch` 2.15.0 and `matplotlib` 3.11.2. Only `rlcard`, `numpy` and `pytest` are
imported by Stage 1; `torch` and `matplotlib` are installed for later stages.

---

## Running the smoke test and tests

From the project root, with the virtual environment activated:

```powershell
python scripts\test_environment.py
python scripts\test_history.py
python scripts\test_encoding.py
python scripts\test_networks.py
python -m pytest -q
```

Without activation:

```powershell
.\.venv\Scripts\python.exe scripts\test_environment.py
.\.venv\Scripts\python.exe scripts\test_history.py
.\.venv\Scripts\python.exe scripts\test_encoding.py
.\.venv\Scripts\python.exe scripts\test_networks.py
.\.venv\Scripts\python.exe -m pytest -q
```

The environment smoke test ends with `All Stage 1 checks passed.`, the history
smoke test with `All history-encoding checks passed.`, the encoder report with
`All observation-encoder checks passed.`, the network report with
`All network checks passed.`, and the test suite currently reports `72 passed`.

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

* No learned agents, PPO, GRU, self-play or opponent modelling.
* The encoder and the policy/value network exist, but nothing connects them to a
  training loop yet; Agent B's history tokens are also encoded and unused.
* No rollout collection, PPO update, checkpointing or evaluation harness.
* The smoke test uses uniformly random agents, not competent baselines.
* Only heads-up **limit** Hold'em is set up; No-Limit is deliberately not used.
* No GUI, online play or real-money connection.

---

## Roadmap and recommended next milestone

| Stage | Work | Status |
|---|---|---|
| 1 | Verify simulator; document observations, actions, payoffs and history; encode Agent B's tokens | ✅ done |
| 2 | **Agent A: observation encoding + MLP policy trained with PPO** | **in progress — encoder + network done** |
| 3 | Agent B: MLP + GRU over betting history, trained with PPO | planned |
| 4 | Head-to-head evaluation, ablations and results write-up | planned |

**Recommended next milestone: rollout collection and the PPO update.**

A sensible remaining Stage 2 scope:

1. ~~Build the observation encoder: 131 numbers, masking, bounds checked.~~ ✅
2. ~~Build the policy/value MLP (shared 256×2 trunk, `256→4` and `256→1`
   heads) with the legal-action mask applied inside `forward`.~~ ✅
3. Collect rollouts: play hands against a fixed opponent, storing encoded states,
   sampled actions, their log-probabilities, values and per-decision rewards.
4. Implement the PPO update: clipped ratio, entropy bonus, value-loss
   coefficient ≈ 0.5, advantage and return normalisation, global gradient-norm
   clipping.
5. Add deterministic seeding, checkpointing (the network already round-trips
   through `state_dict`) and a fixed-seed evaluation harness with confidence
   intervals. The observed per-hand standard deviation is roughly 2.3–2.6 big
   blinds, so an edge of 0.05 bb needs thousands of hands before it is
   statistically meaningful.
6. Only after Agent A is stable, add the GRU history encoder for Agent B and
   compare the two under the same evaluation protocol.
