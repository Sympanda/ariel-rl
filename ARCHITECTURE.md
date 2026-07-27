# Ariel RL — Architecture

This document describes the code structure, the data flow from raw catalogue to agent observation, and the design decisions behind each layer.

**Implementation status at a glance**

| Layer | Status |
|---|---|
| Config system | ✅ Implemented |
| Data preprocessing | ✅ Implemented |
| Simulator (events, clock, state) | ✅ Implemented |
| Event backend abstraction | ✅ Implemented |
| Gymnasium environment | ✅ Implemented |
| Action masking (feasibility-aware) | ✅ Implemented |
| Observation space (audited, 16+25 features) | ✅ Implemented |
| Baselines (5 schedulers) | ✅ Implemented |
| Evaluation framework + diagnostic plots | ✅ Implemented |
| Rewards (per-step + milestones + terminal) | ✅ Implemented |
| RL agents (MLP + Transformer policies, MaskablePPO) | ✅ Implemented |

---

## High-level data flow

```
Raw MCS CSV
    │
    ▼  data/load_catalogue.py
Target table  (814 rows × ~30 cols, static)
    │
    ├──▶  data/population_bins.py           → population_bin, science_weight
    └──▶  data/observation_requirements.py  → obs_cost_days_t1/t2/t3
    │
    ▼  simulator/event_backend.py  (choose one)
    │
    ├── TableBackend   ← event_generator.py produces ~328k event rows (pre-computed)
    │                    sliding-window binary search: O(log N + K) per step
    │
    └── DynamicBackend ← orbital parameters only; computes next window on demand
                         vectorised numpy modular arithmetic: O(N_targets) per step
                         no pre-computation, infinite mission horizon
    │
    ▼  simulator/mission_state.py
Mission state  (mutable: clock + progress table + current pointing)
    │
    ▼  envs/observation_builder.py
Agent observation
    ├── "events"  float32 (K × n_event_features)   per-candidate features
    └── "global"  float32 (n_global_features,)      mission state summary
    │
    ▼  ArielEnv.step(action)  [Gymnasium]
    │
    ├──▶  envs/action_mask.py               → bool mask (visibility + window + feasibility)
    └──▶  mission_state.execute_observation()
              ├── backend.get_event(event_id) → event row
              ├── idle wait → advance clock to window_start
              ├── slew → advance clock
              ├── missed check (arrived after window_end?)
              ├── update progress table + pointing
              └── return info dict
    │
    ▼  rewards/compute_reward.py
    ├── per-step: tier_bonus + progress_shaping + efficiency_bonus − miss_penalty
    ├── one-shot: check_milestone_reward()  → T1 coverage milestones (25/50/75/90/100%)
    └── terminal: compute_terminal_reward() → t1_terminal_weight × (t1_fraction)^power
    │
    ▼  evaluation/
    ├── metrics.py          → EpisodeStats (tier completion, efficiency, diversity)
    ├── population_coverage.py  → per-bin analysis, Gini coefficient
    └── compare_runs.py     → run_episode(), compare_baselines(), summary_table()
```

---

## Layer 0 — Configuration (`src/ariel_rl/utils/config.py`, `configs/`)

All tuneable constants live in config.  Nothing is hardcoded in the env or
simulator layers; values flow down from an `EnvConfig` at construction time.

### Config hierarchy (frozen dataclasses)

```
EnvConfig
├── MissionConfig          start_bjd, lifetime_days, cost_factor, overhead, max_tier_cap
├── SlewConfig             rate_deg_per_min *, min/max_slew_seconds
├── ActionConfig           type: "topk" | "target"
│   ├── TopKActionConfig   k, sort_by
│   └── TargetActionConfig include_completed
├── ObservationConfig      event_features list, global_features list, normalise, min_bin_targets
└── RewardConfig           per-component weights + milestone/terminal bonus config
```

`MissionConfig.max_tier_cap` (default 3) globally caps the maximum tier any target can reach, regardless of what the MCS catalogue says.  Setting it to 1 restricts the whole mission to T1-only observations; all targets that have already reached `max_tier` are **masked from the action space** so the agent cannot waste time on them.

```yaml
mission:
  max_tier_cap: 3   # set to 1 or 2 to restrict mission scope for ablation studies
```

\* Ariel's true slew performance is not yet published — `rate_deg_per_min = 1.0` is a placeholder; change it here without touching any other code.

### Loading

```python
from ariel_rl.utils.config import load_env_config, default_env_config

cfg = load_env_config("configs/env/simple.yaml")
cfg = default_env_config()   # all defaults, no file needed
```

Unknown YAML keys are silently ignored; missing keys fall back to dataclass defaults.  The config is **frozen** — any attempted mutation raises `TypeError`.

### Provided YAML configs

| File | Action space | Notes |
|---|---|---|
| `configs/env/simple.yaml` | `topk`, K=50 | Default starting point |
| `configs/env/full.yaml` | `target` (all N targets) | Full target-level action space |
| `configs/env/with_visibility.yaml` | `topk`, K=50 | Faster slew for sensitivity testing |

---

## Layer 1 — Data (`src/ariel_rl/data/`)

One-time preprocessing: converts the raw catalogue into a clean DataFrame that every other module reads.

| Module | Purpose |
|---|---|
| `schemas.py` | Single source of truth for column names, dtypes, tier constants (`TIER_1/2/3`), mission constants (`MISSION_START_BJD`, `MISSION_LIFETIME_DAYS`), and population bin boundaries |
| `load_catalogue.py` | `load_mcs()` — reads the MCS CSV, selects and renames ~28 columns via `RAW_COL_MAP`, coerces types, drops any rows with missing `period` or `epoch` |
| `population_bins.py` | `assign_population_bins()` — classifies each target along three axes (planet radius, planet temperature, stellar spectral type) to produce a `population_bin` string (e.g. `super_earth_warm_gf`) and an inverse-frequency `science_weight ∈ [0, 1]` |
| `observation_requirements.py` | `compute_progress(obs_completed, target_row)` — given how many observations a target has received, returns `current_tier`, `progress_in_tier ∈ [0, 1]`, and `obs_remaining_next_tier`.  Also `initialise_progress_table()` to seed a fresh episode. |
| `preprocess_targets.py` | `build_target_table()` — orchestrates load → filter → bin → cost in one call.  `load_or_build()` adds Parquet caching. |

### Tier model

Tier observation counts in the MCS are **cumulative**: a target with `tier1=3, tier2=7, tier3=12` needs 3 total observations for Tier 1, 7 for Tier 2, and 12 for Tier 3.  The `max_tier` column (1, 2, or 3) caps what is achievable.

```
obs_completed:  0  1  2  3  4  5  6  7  8  9  10  11  12
tier:           0  0  0  1  1  1  1  2  2  2   2   2   3
progress_in_t:  0 .3 .7  0 .25 .5 .75  0 .2 .4  .6  .8  1
```

`progress_in_tier` is the "10% away from Tier 3" signal the agent sees — it measures how close the next tier transition is within the current tier.

### Observation cost

```
cost_days = COST_FACTOR (2.5) × T14_seconds / 86400
```

where `T14` is the transit duration (or `E14` for eclipse-preferred targets).  All three tiers share the same **per-observation** cost; higher tiers cost more in total because they require more observations.

### Population bins

Three independent dimension classifications, concatenated into a label:

| Dimension | Categories |
|---|---|
| Planet radius | `sub_earth` (<1.5 R⊕), `super_earth` (1.5–2.5), `mini_neptune` (2.5–4), `neptune` (4–6), `saturn` (6–10), `jupiter` (>10) |
| Planet temperature | `cold` (<400 K), `warm` (400–900), `hot` (900–1400), `very_hot` (1400–2000), `ultra_hot` (>2000) |
| Stellar type | `m` (<3900 K), `k` (3900–5200), `gf` (5200–7500), `af_hot` (>7500) |

Example label: `mini_neptune_hot_gf`.  Science weight is the inverse of bin frequency, normalised to [0, 1].

---

## Layer 2 — Simulator (`src/ariel_rl/simulator/`)

Generates the event stream and tracks evolving mission state.  **Completely independent of Gymnasium** — can be used and tested without the env layer.

| Module | Purpose |
|---|---|
| `ephemeris.py` | `propagate()` — given an epoch + period, generates all transit or eclipse mid-times within a BJD window.  Handles the transit→eclipse time offset (including eccentricity correction).  Propagates timing uncertainty σ(t_n) = √(σ_epoch² + n²·σ_period²). |
| `event_generator.py` | `generate_events(targets)` — calls `propagate()` for every target, assembles the full event DataFrame sorted by `window_mid`.  `Preferred Method` column controls whether transits, eclipses, or both are generated. |
| `event_backend.py` | **Pluggable event backend** — `EventBackend` ABC + two concrete implementations (`TableBackend`, `DynamicBackend`).  See below. |
| `slew.py` | `slew_time_seconds(ra1, dec1, ra2, dec2)` — haversine great-circle distance × `rate_s_per_deg`, clamped to [min_slew, max_slew].  `build_slew_matrix(targets)` pre-computes an N×N lookup. |
| `mission_clock.py` | `MissionClock` dataclass — tracks `current_time` (BJD), splits usage into science/slew/overhead, exposes `remaining_time`, `fraction_elapsed`, `can_fit()`. |
| `mission_state.py` | `MissionState` — owns targets, events, clock, progress table, and current pointing.  `execute_observation(event_id)` is the core step function; delegates event lookup to the active backend. |

### Event backend abstraction

`ArielEnv` and `MissionState` are **fully backend-agnostic**.  Every call that needs event data goes through a three-method interface:

```python
class EventBackend(ABC):
    def candidates(self, t_now: float, k: int) -> pd.DataFrame: ...
    # Return up to k upcoming events (same schema as EVENT_COLUMNS), nearest first.

    def get_event(self, event_id: int) -> pd.Series: ...
    # O(1) retrieval of a single event row by id.

    def reset(self) -> None: ...
    # Clear any per-episode mutable state.
```

#### `TableBackend` (default)

Wraps a pre-computed event DataFrame from `generate_events()`.  Uses a **sliding-window binary search** (`np.searchsorted` on `window_mid`) to avoid scanning the full table on every step.

| Characteristic | Value |
|---|---|
| Pre-computation | Required (`generate_events` once per env init) |
| Event table size | ~328k rows for full 3.5-year mission, ~15k for 60-day window |
| Per-step candidate cost | O(log N + K), ~0.15ms |
| Mission horizon | Fixed by the event table window |
| Ephemeris accuracy | Full `propagate()` with eccentricity correction |

#### `DynamicBackend`

Computes observation windows on-the-fly from orbital parameters via **vectorised numpy modular arithmetic** over all targets simultaneously.  No pre-computed table needed.

```python
# At each step, for all targets in parallel:
phase      = (t_now − epoch) % period        # position in current cycle
in_transit = phase < transit_dur / 2         # bool array (N,)
t_center   = where(in_transit,
                   t_now − phase,             # ongoing transit: centre in past
                   t_now + (period − phase))  # next transit: centre in future
window_end = t_center + transit_dur / 2
```

| Characteristic | Value |
|---|---|
| Pre-computation | None — only numpy arrays of orbital params |
| Memory | O(N_targets × ~10 floats) ≈ 50 KB |
| Per-step candidate cost | O(N_targets) numpy, ~0.15ms + DataFrame construction |
| Mission horizon | Infinite — no time-window constraint |
| Ephemeris accuracy | Simple `epoch + n·period` (no eccentricity correction in MVP) |

**Synthetic event IDs**: `target_index × 2` (transit) or `target_index × 2 + 1` (eclipse).  Valid only within one step; `get_event()` reads from a per-step candidate cache populated by `candidates()`.

> **Note**: The `target` action space type requires `TableBackend`.  Use `topk` with `DynamicBackend`.

#### Selecting a backend

```python
from ariel_rl.simulator.event_backend import DynamicBackend

# Default — TableBackend from pre-generated events (same as always)
env = ArielEnv(config, targets=targets, events=events)

# DynamicBackend — no event table, infinite horizon
env = ArielEnv(config, targets=targets, backend=DynamicBackend(targets))
```

`MissionState` also has a `from_backend()` factory for direct simulator use without `ArielEnv`:

```python
state = MissionState.from_backend(targets, DynamicBackend(targets),
                                  mission_start=T0, mission_end=T0+1278)
```

#### Performance after optimisation

Both backends run at ~1.4–1.7 ms/step on the full 814-target catalogue.  The step time is now dominated by observation-builder and action-mask logic rather than event lookup.

| Bottleneck fixed | Saving |
|---|---|
| `sort_values` on pre-sorted table (redundant) | 8.3 ms/step |
| O(N) boolean filter → sliding-window `searchsorted` | 5 ms/step |
| `max_obs_rem` recomputed 20× per step in event loop | 4 ms/step |
| `population_bin_counts` pandas join every step | 2.4 ms/step |
| `progress.loc[id]` (20×/step) → `_progress_dict` | 0.3 ms/step |
| `progress.loc[id, col] = v` → `.at[]` | 0.25 ms/step |
| `event_id` O(N) scan → O(1) indexed lookup | 0.17 ms/step |
| **Total** | **~20 ms/step → ~1.5 ms/step (13× speedup)** |

### Event table schema

| Column | Type | Description |
|---|---|---|
| `event_id` | int | Unique sequential id |
| `target_id` | str | Foreign key → target table |
| `event_type` | str | `"transit"` or `"eclipse"` |
| `window_start` | float (BJD) | `window_mid − duration/2` |
| `window_mid` | float (BJD) | Predicted mid-time |
| `window_end` | float (BJD) | `window_mid + duration/2` |
| `duration` | float (s) | T14 or E14 |
| `duration_days` | float | Convenience in days |
| `tier_goal` | int | Max tier for this target |
| `base_science_value` | float [0,1] | Static rarity × SNR proxy |
| `visibility_valid` | bool | Within pointing constraints (all True for now) |
| `ephemeris_uncertainty` | float (s) | 1-sigma timing uncertainty |
| `event_index` | int | Transit/eclipse number from epoch |

### `execute_observation` step

```
1. Look up event → get target_id, window_start/end, duration
2. Compute slew time from current pointing
3. If current_time < window_start → skip_to(window_start)      [idle wait]
4. If (current_time + slew_days) > window_end → mark as MISSED
5. Else → advance clock by (slew + obs_duration)
         → update current pointing
         → increment progress table via compute_progress()
6. Return info dict: {tier_before, tier_after, tier_completed, missed, …}
```

### Target progress table (mutable per episode)

| Column | Description |
|---|---|
| `target_id` | Primary key |
| `obs_completed` | Total observations so far |
| `current_tier` | Highest completed tier (0 = none) |
| `tier1_done / tier2_done / tier3_done` | Boolean milestones |
| `progress_in_tier` | 0–1 fraction toward **next** tier |
| `obs_remaining_next_tier` | Observations to reach next tier |
| `max_tier` | Ceiling from target table |

### Mission state summary dict

`state.summary()` returns:

```python
{
    "current_time", "remaining_time", "fraction_elapsed",
    "used_science_time", "used_slew_time",
    "n_observations", "n_missed",
    "current_ra", "current_dec",
    "tier1_completed", "tier2_completed", "tier3_completed",
    "total_targets",
    "population_bin_counts",   # dict[bin_label → n_tier1+_completed]
}
```

---

## Layer 3 — Environment (`src/ariel_rl/envs/`)

Wraps `MissionState` as a Gymnasium environment.  The sim and the env are **deliberately separate** — the sim can be exercised directly for debugging without touching Gymnasium.

| Module | Purpose |
|---|---|
| `ariel_env.py` | `ArielEnv(gym.Env)` — `reset()` builds a fresh `MissionState`; `step(action)` calls `execute_observation`, computes per-step + milestone + terminal rewards, updates candidates and mask, returns `(obs, reward, terminated, truncated, info)` |
| `observation_builder.py` | Pure function `build(state, candidates, cfg)` → `{"events": float32 (K×16), "global": float32 (25,)}`.  No Gymnasium dependency.  See observation space section for full feature list. |
| `action_mask.py` | Pure function `compute_mask(state, candidates, cfg)` → bool array.  Checks: (1) visibility, (2) window not yet expired, (3) fits in remaining mission time, (4) slew feasibility — `max(t_now, window_start) + slew ≤ window_end`.  Check (4) was added after auditing a ≈ 43 % miss rate caused by structurally impossible observations being offered to the agent. |
| `wrappers.py` | Normalisation, frame-stacking, etc. (planned) |

### Action spaces

| Type | `action_space` | How it works |
|---|---|---|
| `topk` | `Discrete(K)` | Agent picks index 0…K-1 into the K upcoming events sorted by `window_mid` |
| `target` | `Discrete(N)` | Agent picks target index 0…N-1; env auto-schedules the next available event for that target |

Selected via `config.action.type`.  Invalid actions are penalised with `reward = -invalid_action_penalty` (default −0.5) and do not advance the clock.

#### Action mask feasibility check

Beyond the basic window-expiry check, the mask also enforces:

```
window_start_approx = window_end − duration   # approximate window start
effective_clock     = max(t_now, window_start_approx)
valid iff: effective_clock + slew_days ≤ window_end
```

This filters events where the slew alone exceeds the transit window — structurally impossible to observe regardless of timing.  Before this fix, ≈ 43 % of attempted observations were missed.  After the fix the miss rate is **0 %**.

#### No-valid-action fallback (`_skip_to_next_feasible`)

When all K candidates fail the feasibility mask (e.g., current pointing is far from all upcoming transits), the env progressively looks at `2K, 3K, …` candidates rather than terminating immediately.  This prevents premature episode end when the agent is temporarily in a "bad" sky region.

### Observation space

```python
Dict({
    "events": Box(shape=(K, 16), dtype=float32),   # K = topk.k (default 50)
    "global": Box(shape=(G,),    dtype=float32),   # G = 8 named + n_large_bins (default 25)
})
```

All values are clipped to `[0, 1]` (or `[-3, 3]` for event features that can go negative, e.g. `stellar_metallicity`) after normalisation.  An `ObservationConfig` in the YAML controls which features to include and whether to normalise.

The observation was audited against 1 500 steps of random valid-action rollouts.  Features that were constant or near-zero throughout were replaced.  See **design notes** column for rationale.

#### Per-event features — 16 features, shape `(K, 16)`

Each of the K candidate events contributes one row.  Slots beyond the number of real events are **zero-padded** and correspond to invalid actions (masked out).

| # | Feature | Source | Normalised by | Design notes |
|---|---|---|---|---|
| 0 | `slew_time_days` | angular distance current→target via `slew.slew_time_days` | 2-hr cap (`0.0833 days`) | Core cost signal; varies 0–1 |
| 1 | `window_urgency_norm` | `(t_now − window_start) / window_duration` | already [0,1] | Fraction of the transit window already elapsed; 0 = just opened, →1 = closing |
| 2 | `duration_days` | event table `duration_days` | 1 day | Transit / eclipse duration (T₁₄) |
| 3 | `total_time_cost_days` | `slew + duration` | 1 day | Combined cost excluding wait (wait varies with scheduling lag, not intrinsic to the event) |
| 4 | `progress_in_tier` | progress table | already [0,1] | Fraction of observations completed within the current tier; 0 for unvisited targets |
| 5 | `obs_remaining_next_tier_norm` | `obs_remaining / target_tier3_required_obs` | per-target max | Fraction of total observation budget still needed; normalised per-target so values are comparable across targets |
| 6 | `base_science_value` | event table (SNR-derived) | already [0,1] | Intrinsic scientific value independent of scheduling |
| 7 | `science_weight` | target table | already [0,1] | Priority weight from the Ariel MCS catalogue |
| 8 | `planet_radius_norm` | target table | 20 R⊕ | Physical feature for population diversity |
| 9 | `planet_temperature_norm` | target table | 3 000 K | Equilibrium temperature |
| 10 | `planet_mass_norm` | target table | 4 000 M⊕ | Planet mass |
| 11 | `stellar_temperature_norm` | target table | 10 000 K | Host star Teff |
| 12 | `stellar_metallicity` | target table | 1.5 dex | [Fe/H]; can be negative, clipped to [−3, 3] before output |
| 13 | `tier_goal_norm` | `event.tier_goal / 3` | already [0,1] | Which tier this observation contributes toward |
| 14 | `event_type_binary` | event table | — | 0 = transit, 1 = eclipse |
| 15 | `days_to_window_end_norm` | `(window_end − t_now)` | 2 days | Absolute urgency: time until the window closes; small = act now |

> **Removed features** (vs original design): `wait_time_days` (83 % zeros — superseded by `window_urgency_norm`), `is_valid` (constant 1.0 after action-mask fix — replaced by `days_to_window_end_norm`).

#### Global features — 8 named + N population-bin features, shape `(G,)`

Mission-level state that is the same for all K candidate slots.

**Named features (indices 0–7):**

| # | Feature | Source | Normalised by | Notes |
|---|---|---|---|---|
| 0 | `fraction_elapsed` | `clock.fraction_elapsed` | already [0,1] | Mission progress |
| 1 | `tier1_fraction` | `tier1_completed / total_targets` | already [0,1] | |
| 2 | `tier2_fraction` | `tier2_completed / total_targets` | already [0,1] | |
| 3 | `tier3_fraction` | `tier3_completed / total_targets` | already [0,1] | |
| 4 | `used_science_fraction` | `used_science_time / mission_length` | already [0,1] | |
| 5 | `used_slew_fraction` | `used_slew_time / mission_length` | already [0,1] | |
| 6 | `n_observations_norm` | raw observation count | 5 000 | |
| 7 | `n_completed_targets_norm` | targets at `max_tier` / total | already [0,1] | Fraction of catalogue "used up" (masked) |

> **Removed feature**: `n_missed_norm` (constant 0 after action-mask feasibility fix; see Layer 3 — Action mask).

**Population-bin features (indices 8 … G−1):**

One feature per population bin with `≥ min_bin_targets` targets (default **10**).  Each value is:

```
bin_fraction[b] = observations_made_in_bin_b / targets_in_bin_b
```

Normalised **per-bin** (not by total catalogue size) so a rare bin at 50 % coverage looks the same magnitude as a common bin at 50 % coverage.  Bins with fewer than `min_bin_targets` targets are excluded because they almost never appear in the k-nearest candidates and would be constant-zero throughout training.

With the default catalogue (814 targets, 56 unique bins), `min_bin_targets = 10` retains **17 bins**, giving **G = 25** global features total.

**Configuring the observation space** (`configs/default.yaml` or `ObservationConfig`):

```yaml
observation:
  normalise: true
  include_population_bin_fractions: true
  min_bin_targets: 10          # exclude tiny bins from obs
  event_features:              # subset or reorder as needed
    - slew_time_days
    - window_urgency_norm
    # ... (all 16 listed above are the default)
  global_features:
    - fraction_elapsed
    # ... (all 8 named features are the default)
```

### `info` dict (returned every step)

```python
{
    "action_mask":     np.ndarray[bool, (n_actions,)],
    "step_count":      int,
    "mission_summary": dict,        # state.summary()
    "invalid_action":  bool,
    "step_result":     dict,        # from execute_observation — tier changes etc.
}
```

`step_result` is always present after a valid action and includes `tier_before`, `tier_after`, `tier_completed`, `missed`, `obs_duration_days`, `slew_days`, `progress_before`, `progress_after`, `science_weight`, `population_bin`.  The last four were added during the reward audit to allow `compute_reward` to calculate progress-shaping and diversity components without re-querying the state.

---

## Layer 4 — Baselines (`src/ariel_rl/baselines/`)  ✅ Implemented

Five scheduling heuristics, all sharing a common `BaselineAgent` interface:

```python
class BaselineAgent(ABC):
    def act(self, obs: dict, info: dict) -> int: ...
    def reset(self) -> None: ...
```

| Class | Strategy | What it maximises |
|---|---|---|
| `RandomValid` | Uniform random over valid actions | — (unbiased lower-bound reference) |
| `GreedyValue` | Highest `base_science_value` first | Raw catalogue science value (static, ignores cost) |
| `GreedyBalanced` | `science_weight × (1 + α × progress_in_tier)` | Rarity × tier urgency; `α` trades off between them |
| `EarliestDeadline` | Smallest `days_to_window_end` (window closing soonest) | Urgency; classic scheduling heuristic |
| `SmartGreedy` | `science_weight × (1 + α × progress_in_tier) / (slew + duration + ε)` | Science return per unit time cost |

All agents accept `obs_cfg: ObservationConfig` at construction to locate feature columns by name rather than hard-coded indices.  All are registered in `ALL_BASELINES` and trivially swappable with RL agents.

### What the baselines do and don't optimise

None of these baselines maximise the **actual reward function**.  They each use a proxy:

| Baseline | Knows about slew cost? | Knows about diversity bonus? | Knows about tier bonuses? |
|---|---|---|---|
| `RandomValid` | ✗ | ✗ | ✗ |
| `GreedyValue` | ✗ | ✗ | ✗ |
| `GreedyBalanced` | ✗ | partial (via `science_weight`) | partial |
| `EarliestDeadline` | ✗ | ✗ | ✗ |
| `SmartGreedy` | ✅ | partial | partial |

The gap between `SmartGreedy` (best heuristic) and the true reward function — which accounts for diversity multipliers, tier-completion bonus ratios, coverage milestones, and long-horizon planning — is the space that RL is designed to exploit.

### `SmartGreedy` scoring formula

```
score = science_weight × (1 + alpha × progress_in_tier)
        ─────────────────────────────────────────────────
        slew_time_norm + duration_norm + ε
```

- **Numerator**: science priority × tier-urgency bonus (higher near a tier boundary)
- **Denominator**: total time cost (slew + observation duration)
- `alpha = 1.0`, `ε = 0.01` by default; all configurable at construction

---

## Layer 5 — Evaluation (`src/ariel_rl/evaluation/`)  ✅ Implemented

| Module | Purpose |
|---|---|
| `metrics.py` | `EpisodeStats` dataclass + `compute_stats(state)` — aggregates tier completion rates, schedule efficiency, population coverage, Gini diversity into a single summary object |
| `population_coverage.py` | `coverage_table(state)` — per-bin completion rates; `coverage_matrix(state, tier)` — radius × temperature heat map; `coverage_gini(state, tier)` — diversity index |
| `compare_runs.py` | `run_episode(env, agent)` — runs one episode, returns `EpisodeStats`; `run_episode_with_log(env, agent)` — same but also returns a per-step `pd.DataFrame` of rewards and cumulative rewards; `compare_baselines()` — runs all agents across N episodes; `summary_table()` — mean ± std per agent |
| `plots.py` | Seven diagnostic plot functions (see below) |

### Diagnostic plots (`plots.py`)

All functions return `(fig, ax)` for further customisation.  Colours use the Paul Tol colourblind-safe palette: blue ramp for tiers (T1=sky, T2=mid, T3=navy), red for slew, grey for idle.

| Function | What it shows |
|---|---|
| `plot_episode_summary(state)` | 4-panel: tier progress bars, time budget pie, population bin heatmap, Gini index over time |
| `plot_schedule_timeline(state)` | Gantt chart of the observation sequence, one row per target |
| `plot_coverage_heatmap(state)` | Radius × temperature grid coloured by T1/T2/T3 completion rate |
| `plot_agent_comparison(df)` | Grouped bar charts comparing agents on n_obs, tier completions, efficiency, diversity |
| `plot_training_curves(logs)` | Episode reward, episode length, optional RL losses over training steps |
| `plot_reward_curve(logs)` | Smoothed per-step reward + cumulative reward over mission days, one line per agent |
| `plot_activity_timeline(state)` | Horizontal bar chart — one row per calendar month; segments coloured by activity (slew, T1/T2/T3 obs, idle/waiting).  Idle gaps are placed in their true chronological position (reconstructed from `window_start ≈ mission_day − duration/2`) rather than lumped at the end. |

### `EpisodeStats` fields

| Field | Description |
|---|---|
| `n_observations` | Total steps taken |
| `n_missed` | Events that fell outside the observation window |
| `miss_rate` | `n_missed / (n_observations + n_missed)` |
| `tier1/2/3_completed` | Targets fully completing each tier |
| `tier1/2/3_rate` | Completion / total eligible targets |
| `science_efficiency` | Science time / total elapsed time |
| `bin_coverage` | Fraction of population bins with ≥ 1 tier-1 completion |
| `coverage_gini_t1/t2` | Gini coefficient of tier completions across bins |
| `bin_counts` | Per-bin completion dict (excluded from `to_dict()`) |

### CLI comparison script

```bash
python -m ariel_rl.scripts.run_baseline \
    --baselines random greedy_value greedy_balanced earliest_deadline \
    --n-episodes 5 \
    --config configs/env/simple.yaml
```

---

## Layer 6 — Rewards (`src/ariel_rl/rewards/`)  ✅ Implemented

All reward logic lives in `rewards/compute_reward.py`.  Three public functions are called by `ArielEnv` at different points in the episode lifecycle.  All weights are held in `RewardConfig` — swap reward profiles without touching the environment code.

### Per-step reward (`compute_reward`)

Called after every valid observation.

```
reward = tier_bonus + progress_shaping + efficiency_bonus − miss_penalty
```

| Component | Type | Formula | Default weight |
|---|---|---|---|
| **Tier completion bonus** | Sparse | `tier_weight × science_weight × diversity_mult` fired when `tier_after > tier_before` | T1=1.0, T2=3.0, T3=10.0 |
| **Progress shaping** | Dense | `progress_weight × Δprogress_in_tier × science_weight × diversity_mult × near_boost` | `progress_weight`=0.3 |
| **Efficiency reward** | Dense | `efficiency_weight × obs_duration / (obs_duration + slew_duration)` | `efficiency_weight`=0.5 |
| **Missed-event penalty** | Sparse | `−miss_penalty` (agent arrives after `window_end`; clock still advances) | `miss_penalty`=0.1 |

> **Invalid-action penalty** (−`invalid_action_penalty`) is applied directly in `ArielEnv.step()` before `_compute_reward`; the clock does not advance.

**Near-completion boost** — when `progress_in_tier ≥ near_completion_threshold` (default 0.7), the progress reward is multiplied by `near_completion_scale` (default 3.0).  This incentivises finishing targets rather than abandoning them just before a tier boundary.

### Diversity multiplier (`_diversity_multiplier`)

Every science-facing component (tier bonus + progress shaping) is scaled by:

```
observed_fraction = bin_tier1_completed / bin_total_targets   ∈ [0, 1]
diversity_mult    = 1 + max(0, 1 − observed_fraction)         ∈ [1, 2]
```

- A bin with **nothing observed** yet → multiplier = **2.0** (maximum boost).
- A bin that is **fully saturated** → multiplier = **1.0** (no penalty).
- Computed live each step from `MissionState.population_bin_counts` and `_bin_totals`.

### Coverage milestone bonuses (`check_milestone_reward`)

Called by `ArielEnv.step()` after any step that could change `tier1_completed`.  Fires **at most once per milestone per episode**.

```
fraction = tier1_completed / total_targets
bonus fired if fraction ≥ milestone_threshold (and not already fired this episode)
```

| Milestone | Bonus | Meaning |
|---|---|---|
| 25 % | 20.0 | First quarter of catalogue at T1 |
| 50 % | 20.0 | Half the catalogue at T1 |
| 75 % | 20.0 | Three-quarters |
| 90 % | 20.0 | Near-complete survey |
| 100 % | 40.0 | Full T1 survey complete (double bonus) |

The milestone set is stored on the `ArielEnv` instance and reset to `set()` at each `reset()` call.

### Terminal episode bonus (`compute_terminal_reward`)

Called once at episode termination (natural or no-valid-actions).

```
terminal_bonus = t1_terminal_weight × (tier1_completed / total_targets)^t1_terminal_power
               = 50.0 × (t1_fraction)^2.0   (defaults)
```

The quadratic exponent makes near-complete T1 coverage disproportionately valuable — going from 90 % → 100 % earns ~10 reward while 0 % → 10 % earns only ~0.5.

### Incentive structure summary

| Design goal | Mechanism |
|---|---|
| Prioritise T3 over T2 over T1 within a target | T3 weight (10) is 3.3× T2 (3), which is 3× T1 (1); progression is cumulative so T3 earns all three |
| Finish targets that are almost done | `near_completion_scale=3` when `progress_in_tier ≥ 0.7` |
| Cover the full catalogue at T1 | Coverage milestone bonuses + quadratic terminal bonus |
| Don't miss good T2/T3 opportunities | `progress_in_tier` in obs + near-completion boost signal to RL agent; `SmartGreedy` uses it heuristically |
| Penalise wasted slew time | Efficiency reward = `obs_fraction` of total step cost |
| Reward diversity dynamically | Diversity multiplier updates every step from live bin counts |
| All weights configurable without code changes | `RewardConfig` dataclass, overrideable in YAML |

### RewardConfig reference

```yaml
reward:
  # Per-step tier bonuses
  tier1_completion: 1.0
  tier2_completion: 3.0
  tier3_completion: 10.0

  # Dense shaping
  progress_weight: 0.3
  efficiency_weight: 0.5
  near_completion_threshold: 0.7   # progress_in_tier above which near_boost applies
  near_completion_scale: 3.0       # multiplier on progress reward in the final stretch

  # Coverage milestone bonuses (one-shot per episode)
  t1_milestone_fractions: [0.25, 0.5, 0.75, 0.90, 1.0]
  t1_milestone_bonus: 20.0         # 100% milestone pays 2× (= 40.0)

  # Terminal bonus
  t1_terminal_weight: 50.0
  t1_terminal_power: 2.0           # quadratic: near-complete coverage is most rewarded

  # Penalties
  miss_penalty: 0.1
  invalid_action_penalty: 0.5
```

---

## Layer 7 — RL Agents  ✅ Implemented

All agents are trained with **MaskablePPO** from `sb3-contrib`, which enforces action-validity at every rollout step without penalising the loss function for masked actions.

| Module | Description |
|---|---|
| `agents/ppo_masked.py` | Masked-env factory: `make_masked_env()`, `make_training_envs()` |
| `agents/rl_agent.py` | `RLAgentWrapper` — adapts a trained SB3 model to `BaselineAgent` interface |
| `agents/policies/event_attention_policy.py` | `ArielTransformerPolicy` — transformer encoder over K candidate events |
| `agents/policies/mlp_scorer.py` | `ArielMlpPolicy` — flat MLP baseline (sanity-check policy) |
| `scripts/train_agent.py` | CLI training script with logging, device auto-detection, post-training plots |

### Environment wrappers (`agents/ppo_masked.py`)

SB3 requires two wrappers around `ArielEnv`:

```python
env = ArielEnv(...)
env = ActionMasker(env, _get_action_mask)   # supplies action_masks() to MaskablePPO
env = Monitor(env)                           # lets SB3 log ep_rew_mean / ep_len_mean
```

`make_training_envs(n_envs, ...)` builds a `SubprocVecEnv` (or `DummyVecEnv`) of wrapped environments for parallel rollout collection.

### Policies

Both policies subclass `MaskableActorCriticPolicy` and share the same interface contract:

```
forward(obs, action_masks) → (actions, values, log_probs)
evaluate_actions(obs, actions, action_masks) → (values, log_probs, entropy)
predict_values(obs) → values
```

#### `ArielMlpPolicy` (`policies/mlp_scorer.py`)

A simple sanity-check policy — flattens the `(K × 16)` event matrix and 25-d global vector, processes them through a shared MLP, and outputs independent logit and value heads.

```
obs["events"]  (K×16) ──┐
                         ├─ flatten → shared_mlp(256→256) → policy_head (K,) + value_head (1,)
obs["global"]  (25,)  ──┘
```

Action logits are clipped to `−∞` for masked positions before the softmax.

#### `ArielTransformerPolicy` (`policies/event_attention_policy.py`)

The primary architecture.  Each of the K candidate events is treated as a token; a transformer encoder with multi-head self-attention processes the full set simultaneously, enabling the policy to reason about relative priorities across all candidates in a single pass.

```
obs["events"]  (K×16) → event_proj(16→d_model)  ─┐
                                                    ├─ [CLS | e_1 | … | e_K]
obs["global"]  (25,)  → global_proj(25→d_model) ──┘  (prepend CLS from global embedding)
                                                   │
                                              TransformerEncoder
                                              (n_layers=2, n_heads=4, d_model=128, Pre-LN)
                                                   │
                                    ┌──────────────┴────────────────┐
                              tokens[1:]                        tokens[0]  (CLS)
                           policy_head(K,)                   value_head(1,)
                         (per-token logits)                 (scalar value)
```

Key design choices:

- **Pre-LN transformer** (layer norm before attention + FFN) for training stability with sparse rewards.
- **Padding mask** derived from the SB3 action mask: invalid event slots are excluded from attention and their logits are forced to `−∞`.
- **CLS token** seeded from the global features serves as the critic's summary of the full episode state.
- **Orthogonal weight init** on all linear layers.

### `RLAgentWrapper` (`agents/rl_agent.py`)

Bridges trained SB3 models and the `BaselineAgent` interface so they can be passed into `compare_baselines`, `run_episode_with_log`, and all evaluation / plotting utilities without modification.

```python
class RLAgentWrapper(BaselineAgent):
    def act(self, obs, info) -> int:
        action_masks = info["action_mask"]
        action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
        return int(action)

    @classmethod
    def load(cls, path: str | Path, name: str = "RLAgent") -> "RLAgentWrapper": ...
```

### Training script (`scripts/train_agent.py`)

```bash
python src/ariel_rl/scripts/train_agent.py \
    --policy transformer \   # or mlp
    --timesteps 500000 \
    --n-envs 4 \
    --run-name my_run \
    --device auto            # auto-selects MPS / CUDA / CPU
```

**Outputs** written to `outputs/<run_name>/`:

| File | Contents |
|---|---|
| `model.zip` | Trained `MaskablePPO` model (SB3 format) |
| `training_log.csv` | Per-rollout: timestep, episode reward/length, value loss, policy loss, entropy, KL |
| `plots/training_curves.png` | Training loss + reward curves |
| `plots/activity_<name>.png` | Monthly schedule breakdown (science / slew / idle) |
| `plots/timeline_<name>.png` | Per-target Gantt chart |
| `plots/schedule_<name>.png` | Classic schedule timeline |
| `plots/reward_curve.png` | Smoothed per-step reward during evaluation episode |
| `plots/coverage.png` | Population-bin T1 coverage heatmap |

**Device auto-detection** (set via `--device auto`, the default):

```
MPS  (Apple Silicon)  →  torch.device("mps")
CUDA                  →  torch.device("cuda")
CPU  (fallback)       →  torch.device("cpu")
```

**`TrainingLoggerCallback`** captures rollout metrics from SB3's `logger` at each `on_rollout_end` call and writes them to `training_log.csv` for offline analysis.

### Comparing a trained RL model against baselines

```bash
python scripts/run_short_episode.py \
    --days 60 \
    --model-path outputs/my_run/model.zip \
    --model-name MyTransformer \
    --out-dir plots/my_run_vs_baselines
```

The RL model is loaded via `RLAgentWrapper.load()` and added to the `agents` dict alongside all five baselines.  All plots are generated identically for every agent in the comparison.

---

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Build processed tables (targets.parquet + events.parquet)
python -m ariel_rl.scripts.build_dataset

# 3. Run tests (167 tests)
pytest
```

### Run one episode with a random agent

```python
import numpy as np
from ariel_rl.envs import ArielEnv
from ariel_rl.utils.config import load_env_config

cfg = load_env_config("configs/env/simple.yaml")
env = ArielEnv(config=cfg)

obs, info = env.reset()
terminated = False
while not terminated:
    mask = info["action_mask"]
    valid = np.where(mask)[0]
    if len(valid) == 0:
        break
    action = np.random.choice(valid)
    obs, reward, terminated, truncated, info = env.step(action)

print(env.state.summary())
```

### Compare baselines over several episodes

```python
from ariel_rl.envs import ArielEnv
from ariel_rl.baselines import RandomValid, GreedyBalanced, EarliestDeadline, SmartGreedy
from ariel_rl.evaluation import compare_baselines, summary_table
from ariel_rl.utils.config import default_env_config

cfg = default_env_config()
env = ArielEnv(config=cfg)
agents = {
    "random":       RandomValid(seed=0),
    "balanced":     GreedyBalanced(obs_cfg=cfg.observation, seed=0),
    "deadline":     EarliestDeadline(obs_cfg=cfg.observation, seed=0),
    "smart_greedy": SmartGreedy(obs_cfg=cfg.observation, seed=0),
}
df  = compare_baselines(env, agents, n_episodes=5)
tbl = summary_table(df)
print(tbl.to_string())
```

### Run an episode with per-step reward log

```python
from ariel_rl.evaluation.compare_runs import run_episode_with_log
from ariel_rl.evaluation.plots import plot_reward_curve, plot_activity_timeline

stats, log_df = run_episode_with_log(env, agents["smart_greedy"])
plot_reward_curve(log_df, agent_name="SmartGreedy")
plot_activity_timeline(log_df, agent_name="SmartGreedy")
```

### Use the simulator directly (no Gymnasium)

```python
from ariel_rl.data import build_target_table
from ariel_rl.simulator import generate_events, MissionState

targets = build_target_table()
events  = generate_events(targets)
state   = MissionState.from_tables(targets, events)

# step manually
event_id = int(state.upcoming_events(n=1)["event_id"].iloc[0])
info = state.execute_observation(event_id)
print(info)          # tier_before, tier_after, slew_days, missed, …
print(state.summary())
```

### Use DynamicBackend (no pre-computed event table)

```python
from ariel_rl.data import build_target_table
from ariel_rl.simulator.event_backend import DynamicBackend
from ariel_rl.envs import ArielEnv
from ariel_rl.utils.config import default_env_config

targets = build_target_table()
cfg     = default_env_config()

# No event table required — windows are computed on demand each step.
# Works for any mission duration; ~1.7 ms/step on the full 814-target catalogue.
env = ArielEnv(config=cfg, targets=targets, backend=DynamicBackend(targets))

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(0)
```

---

## File map

```
src/ariel_rl/
├── data/
│   ├── schemas.py                  ← constants, column names, dtypes
│   ├── load_catalogue.py           ← CSV → canonical DataFrame
│   ├── population_bins.py          ← bin labels + science weights
│   ├── observation_requirements.py ← compute_progress(), tier thresholds
│   └── preprocess_targets.py       ← build_target_table() pipeline
│
├── simulator/
│   ├── ephemeris.py                ← propagate transit/eclipse times
│   ├── event_generator.py          ← generate full event table (TableBackend input)
│   ├── event_backend.py            ← EventBackend ABC, TableBackend, DynamicBackend
│   ├── slew.py                     ← angular separation + slew time model
│   ├── mission_clock.py            ← BJD clock, budget tracking
│   └── mission_state.py            ← full mutable episode state; backend-agnostic
│
├── envs/
│   ├── ariel_env.py                ← Gymnasium env (ArielEnv)
│   ├── observation_builder.py      ← build() → {"events", "global"} arrays
│   ├── action_mask.py              ← compute_mask() → bool array
│   └── wrappers.py                 ← normalisation etc. [planned]
│
├── baselines/                      ← ✅ Implemented (5 agents)
│   ├── base.py                     ← BaselineAgent abstract class
│   ├── random_valid.py             ← uniform random over valid actions
│   ├── greedy_value.py             ← highest base_science_value first
│   ├── greedy_balanced.py          ← science_weight × tier_progress score
│   ├── earliest_deadline.py        ← soonest window_end first
│   └── smart_greedy.py             ← science-return per unit time cost (slew-aware)
│
├── evaluation/                     ← ✅ Implemented
│   ├── metrics.py                  ← EpisodeStats + compute_stats()
│   ├── population_coverage.py      ← coverage_table(), coverage_matrix(), gini
│   ├── compare_runs.py             ← run_episode(), run_episode_with_log(), compare_baselines()
│   └── plots.py                    ← 7 diagnostic plot functions (colourblind-safe palette)
│
├── rewards/                        ← ✅ Implemented
│   └── compute_reward.py           ← compute_reward()          per-step reward
│                                      check_milestone_reward()  one-shot T1 coverage bonuses
│                                      compute_terminal_reward() end-of-episode T1 bonus
│                                      _diversity_multiplier()   bin coverage boost
│
├── agents/                         ← ✅ Implemented
│   ├── __init__.py                 ← exports RLAgentWrapper + policy classes
│   ├── ppo_masked.py               ← make_masked_env(), make_training_envs()
│   ├── rl_agent.py                 ← RLAgentWrapper (BaselineAgent adapter for SB3 models)
│   └── policies/
│       ├── __init__.py
│       ├── event_attention_policy.py  ← ArielTransformerNet + ArielTransformerPolicy
│       └── mlp_scorer.py              ← ArielMlpNet + ArielMlpPolicy
│
├── scripts/
│   ├── build_dataset.py            ← CLI: build + cache Parquet files
│   ├── generate_events.py          ← CLI: event generation only
│   ├── run_baseline.py             ← CLI: compare baselines, print summary table
│   └── train_agent.py              ← CLI: train MaskablePPO agent, save model + plots
│
└── utils/
    └── config.py                   ← EnvConfig dataclass hierarchy + YAML loader

configs/
├── env/
│   ├── simple.yaml                 ← topk K=50, all default features
│   ├── full.yaml                   ← target action space (all N targets)
│   └── with_visibility.yaml        ← topk, faster slew for sensitivity tests
└── agent/                          ← [planned]

scripts/                            ← top-level runnable scripts
├── run_short_episode.py            ← compare baselines (+ optional RL model) and save plots
└── (other convenience scripts)

outputs/                            ← created by train_agent.py
└── <run_name>/
    ├── model.zip                   ← saved MaskablePPO weights
    ├── training_log.csv            ← per-rollout training metrics
    └── plots/                      ← post-training diagnostic plots

tests/
├── test_slew.py            (15 tests) ← angular separation, slew time, matrix
├── test_event_generation.py(19 tests) ← ephemeris, eclipse offset, event table
├── test_env_step.py        (32 tests) ← MissionClock, MissionState, DynamicBackend
├── test_reward.py          (33 tests) ← compute_progress, tier transitions, compute_reward,
│                                        diversity, near_completion_scale, milestones, terminal
├── test_action_mask.py      (6 tests) ← visibility/window/budget/feasibility filtering
├── test_visibility.py       (5 tests) ← placeholder visibility contract
├── test_baselines.py       (30 tests) ← all 5 baselines + evaluation framework
└── test_env_gymnasium.py   (27 tests) ← ArielEnv reset/step, config loading, spaces
                        ─────────────
                        167 tests total
```
