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
│       ├── environment.py          # Stage 1 inspection helpers
│       └── history.py              # Agent B's betting-history tokens
├── scripts/
│   ├── test_environment.py         # executable smoke test / report
│   └── test_history.py             # executable history-encoding report
├── tests/
│   ├── conftest.py                 # puts src/ on sys.path for pytest
│   ├── test_environment.py         # Stage 1 contract tests
│   └── test_history.py             # history-encoding tests
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
python -m pytest -q
```

Without activation:

```powershell
.\.venv\Scripts\python.exe scripts\test_environment.py
.\.venv\Scripts\python.exe scripts\test_history.py
.\.venv\Scripts\python.exe -m pytest -q
```

The environment smoke test ends with `All Stage 1 checks passed.`, the history
smoke test with `All history-encoding checks passed.`, and the test suite
currently reports `35 passed`.

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
2. **Illegal actions are silently substituted.** `env.step()` returns `check`
   when possible, otherwise `fold`, instead of raising. Future policies must
   apply the legal-action mask (`action_mask()` is provided for this).
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

---

## Current limitations

* No learned agents, PPO, GRU, self-play or opponent modelling.
* No observation encoder for the 72-dimensional vector beyond
  `decode_observation()`; Agent B's history tokens exist, but nothing consumes
  them yet.
* No training loop, checkpointing or evaluation harness.
* The smoke test uses uniformly random agents, not competent baselines.
* Only heads-up **limit** Hold'em is set up; No-Limit is deliberately not used.
* No GUI, online play or real-money connection.

---

## Roadmap and recommended next milestone

| Stage | Work | Status |
|---|---|---|
| 1 | Verify simulator; document observations, actions, payoffs and history; encode Agent B's tokens | ✅ done |
| 2 | **Agent A: observation encoding + MLP policy trained with PPO** | **next** |
| 3 | Agent B: MLP + GRU over betting history, trained with PPO | planned |
| 4 | Head-to-head evaluation, ablations and results write-up | planned |

**Recommended next milestone: implement Agent A.**

A sensible Stage 2 scope:

1. Build an observation encoder from the 72-dimensional RLCard vector plus
   scalar features such as pot size, chips committed and street.
2. Build the legal-action mask and validate it against `state["legal_actions"]`.
3. Implement a small PyTorch MLP policy/value network and a PPO training loop
   against a random or simple rule-based opponent.
4. Add deterministic seeding, checkpointing and a fixed-seed evaluation
   harness with confidence intervals. The observed per-hand standard deviation
   is roughly 2.3–2.6 big blinds, so an edge of 0.05 bb needs thousands of hands
   before it is statistically meaningful.
5. Only after Agent A is stable, add the GRU history encoder for Agent B and
   compare the two under the same evaluation protocol.
