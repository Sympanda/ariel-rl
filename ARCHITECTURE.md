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
| Observation space (audited, 17+26 features) | ✅ Implemented |
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
              ├── slew immediately → t_arrive = t_now + slew_days
              ├── capture check (t_arrive vs block_start/block_end → fraction [0,1])
              ├── idle wait = max(0, block_start − t_arrive)
              ├── observe for block_duration_days (= 2.5 × T14)
              ├── advance clock by (slew + idle + obs + overhead)
              ├── update progress table + pointing
              └── return info dict
    │
    ▼  rewards/compute_reward.py
    ├── per-step: tier_bonus + progress_shaping + efficiency_bonus + coverage_potential + host_diversity − idle_penalty − miss_penalty
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
├── ActionConfig           type: "topk" | "target" | "full_set"
│   ├── TopKActionConfig   k, sort_by
│   ├── TargetActionConfig include_completed
│   └── FullSetActionConfig include_completed, cache_static
├── ObservationConfig      event_features list, global_features list, normalise, min_bin_targets
└── RewardConfig           per-component weights + milestone/terminal + coverage + host diversity
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
| `mission_clock.py` | `MissionClock` dataclass — tracks `current_time` (BJD), splits usage into science/slew/overhead, exposes `remaining_time`, `fraction_elapsed` (= `elapsed / (mission_end − mission_start)`, correct for curriculum episodes), `can_fit()`. |
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

#### `DynamicBackend` (default)

Computes observation windows on-the-fly from orbital parameters via **vectorised numpy modular arithmetic** over all targets simultaneously.  No pre-computed table needed.

```python
# At each step, for all targets in parallel:
cf          = COST_FACTOR                     # = 2.5
half_block  = transit_dur / 2 * cf           # = 1.25 × T14  (block half-width)
phase       = (t_now − epoch) % period        # position in current cycle
in_block    = phase < half_block              # still inside observation block?
t_center    = where(in_block,
                    t_now − phase,            # current occurrence
                    t_now + (period − phase)) # next occurrence
block_end   = t_center + half_block
valid       = block_end > t_now               # keep until block closes, not raw T14
```

An event stays available until `block_end = window_mid + 1.25 × T14`, not the raw transit end (`window_mid + 0.5 × T14`).  This is correct because a late arrival can still capture a partial fraction of the block and contribute fractional progress.  `preferred_method` is respected: `"Transit"` targets produce only transit candidates; `"Eclipse"` targets produce only eclipse candidates; `"Either"` targets produce both.

| Characteristic | Value |
|---|---|
| Pre-computation | None — only numpy arrays of orbital params |
| Memory | O(N_targets × ~10 floats) ≈ 50 KB |
| Per-step candidate cost | O(N_targets) numpy, ~0.15ms + DataFrame construction |
| Mission horizon | Infinite — no time-window constraint |
| Ephemeris accuracy | Simple `epoch + n·period` (no eccentricity correction in MVP) |

**Synthetic event IDs**: `target_index × 2` (transit) or `target_index × 2 + 1` (eclipse).  Valid only within one step; `get_event()` reads from a per-step candidate cache populated by `candidates()`.

`block_duration_days` (= 2.5 × T14) is computed and added by `DynamicBackend` at candidate-generation time so that `execute_observation` and the observation builder can use a single authoritative block cost.

#### `TableBackend` (deprecated)

Wraps a pre-computed event DataFrame from `generate_events()`.  Uses a **sliding-window binary search** (`np.searchsorted` on `window_mid`) to avoid scanning the full table on every step.  Kept for backward-compatibility with existing event tables; `DynamicBackend` is preferred for all new training runs.

| Characteristic | Value |
|---|---|
| Pre-computation | Required (`generate_events` once per env init) |
| Event table size | ~328k rows for full 3.5-year mission, ~15k for 60-day window |
| Per-step candidate cost | O(log N + K), ~0.15ms |
| Mission horizon | Fixed by the event table window |
| Ephemeris accuracy | Full `propagate()` with eccentricity correction |

#### Selecting a backend

```python
from ariel_rl.simulator.event_backend import DynamicBackend

# Default — DynamicBackend, no event table required
env = ArielEnv(config, targets=targets)

# Explicit DynamicBackend (same as default)
env = ArielEnv(config, targets=targets, backend=DynamicBackend(targets))

# TableBackend — pre-computed events (deprecated; backward-compat only)
env = ArielEnv(config, targets=targets, events=events)
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
| `window_start` | float (BJD) | `window_mid − duration_days/2` |
| `window_mid` | float (BJD) | Predicted mid-time |
| `window_end` | float (BJD) | `window_mid + duration_days/2` |
| `duration` | float (s) | T14 or E14 |
| `duration_days` | float | Raw transit/eclipse duration (T14/E14) in days |
| `block_duration_days` | float | Full observation block = `COST_FACTOR × duration_days` (= 2.5 × T14); the authoritative time cost per observation |
| `tier_goal` | int | Max tier for this target |
| `base_science_value` | float [0,1] | Static rarity × SNR proxy |
| `visibility_valid` | bool | Within pointing constraints (all True for now) |
| `ephemeris_uncertainty` | float (s) | 1-sigma timing uncertainty |
| `event_index` | int | Transit/eclipse number from epoch |

### `execute_observation` step

The telescope slews **immediately** after the action is chosen, then idles if it arrives before the block starts.  The observation block extends 0.75 × T₁₄ *before* and *after* the raw transit/eclipse, so **late arrivals still contribute partial science**.

```
1.  Look up event → get target_id, window_mid, block_duration_days
2.  Compute slew_days from current pointing (haversine × slew_rate)
3.  t_arrive = t_now + slew_days
4.  block_start = window_mid − block_duration_days / 2
    block_end   = window_mid + block_duration_days / 2

    Case A  (t_arrive ≤ block_start):   full capture, captured_fraction = 1.0
    Case B  (block_start < t_arrive < block_end):
                                         partial capture,
                                         captured_fraction = (block_end − t_arrive)
                                                             / block_duration_days
    Case C  (t_arrive ≥ block_end):     MISSED — pay slew only, no science, no idle

5.  (Cases A + B only — tier-scoped effective fraction)
      idle_days         = max(0, block_start − t_arrive)
      obs_remaining     = obs_remaining_next_tier   (float, from progress table)
      effective_fraction = min(captured_fraction, obs_remaining)
                         # cap at tier boundary — agent regains control as soon
                         # as this tier's threshold is crossed, not at window end
                         # if target is at max_tier: effective_fraction = 0
      obs_duration       = effective_fraction × block_duration_days
      advance clock by (slew + idle + obs_duration)
      update current pointing → (target.ra, target.dec)
      obs_completed += effective_fraction
      update progress table via compute_progress()

6.  Return info dict: {tier_before, tier_after, tier_completed, missed,
                       captured_fraction,  ← geometric (what the window offered)
                       effective_fraction, ← actual (capped at tier boundary)
                       obs_duration_days, slew_days, idle_days,
                       total_cost_days,    ← slew + idle + obs_duration + overhead
                       …}
```

**Key design points:**
- `obs_completed` is a **float** — partial and tier-capped fractions accumulate until an integer tier threshold is crossed.
- An observation is **tier-scoped**: the clock stops advancing as soon as the current tier threshold is hit. The agent then gets control back and may choose to continue with this target (next tier) or switch to something else.
- If a target has already reached `max_tier`, `effective_fraction = 0` and no science is collected. This is enforced in the simulator regardless of what the RL mask does.
- `captured_fraction` and `effective_fraction` are both reported in the info dict so diagnostics can distinguish "window offered 1.0 but we only needed 0.3 to complete the tier" from "window only offered 0.3 due to late arrival".
- `total_time_cost_days` in the observation uses `effective_fraction` (not `captured_fraction`) so the agent sees the actual expected cost, which drops as a tier nears completion.
- The action mask uses `block_end` (not `window_end`) as the miss cutoff, consistent with this model.

### Target progress table (mutable per episode)

| Column | Type | Description |
|---|---|---|
| `target_id` | str | Primary key |
| `obs_completed` | **float** | Equivalent observations accumulated (fractional with partial obs) |
| `current_tier` | int | Highest completed tier (0 = none) |
| `tier1_done / tier2_done / tier3_done` | bool | Boolean milestones |
| `progress_in_tier` | float | 0–1 fraction toward **next** tier |
| `obs_remaining_next_tier` | **float** | Equivalent obs still needed (fractional) |
| `max_tier` | int | Ceiling from target table |

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
| `observation_builder.py` | Pure function `build(state, candidates, cfg)` → `{"events": float32 (K×18), "global": float32 (26,)}`.  No Gymnasium dependency.  See observation space section for full feature list. |
| `action_mask.py` | Pure function `compute_mask(state, candidates, cfg)` → bool array.  Checks: (1) visibility, (2) window not yet expired, (3) fits in remaining mission time, (4) slew feasibility — `max(t_now, window_start) + slew ≤ window_end`.  Check (4) was added after auditing a ≈ 43 % miss rate caused by structurally impossible observations being offered to the agent. |
| `wrappers.py` | Normalisation, frame-stacking, etc. (planned) |

### Action spaces

| Type | `action_space` | How it works |
|---|---|---|
| `topk` | `Discrete(K)` | Agent picks index 0…K-1 into the K upcoming events sorted by `window_mid`.  Default. |
| `target` | `Discrete(N)` | Agent picks target index 0…N-1; env auto-schedules the next available event for that target |
| `full_set` | `Discrete(N_max)` | **Dynamic active planet set.** Only genuinely active targets (those with `current_tier < max_tier`) participate in ISAB/PMA attention as real tokens.  Completed targets are removed from the set after each observation.  `N_max` is the **fixed action/tensor size** and a hard ceiling set via `action.full_set.n_max` (default = `len(targets)`; recommended ≤ 2000 for the full Ariel catalogue).  A `ValueError` is raised at initialisation if `len(catalogue) > N_max`.  Rows beyond `n_active` are zero-padded sentinels (always masked False).  The mapping `action_index i → _active_target_ids[i]` is maintained explicitly and rebuilt after each removal.  Each active-planet token is associated with its **first reachable upcoming event** — the first event whose block has not yet expired given the telescope's current slew time.  Possible-but-expensive choices remain valid actions; only genuinely impossible or completed targets are removed/masked.  **Runtime insertion of genuinely new targets is deferred** — adding a target mid-episode would require simultaneous updates to the catalogue, MissionState, backend ephemeris, static feature cache, and active mapping; leave as future work for missions with dynamic discovery.  Three policy architectures are available: **FullSetISABPolicy** (ISAB, O(N·m)), **FullSetSelfAttentionPolicy** (full O(N²) ablation), and the pre-existing **ArielTransformerPolicy** (Top-K, unchanged). |

Selected via `config.action.type`.  Invalid actions are penalised with `reward = -invalid_action_penalty` (default −0.5) and do not advance the clock.

#### Action mask feasibility check

The mask uses `block_end = window_mid + block_duration_days / 2` (not `window_end`) as the miss cutoff, consistent with the partial-observation model in `execute_observation`.

```
block_end = window_mid + block_duration_days / 2   # = window_mid + 1.25 × T14

valid iff (ALL modes, including full_set):
  1. visibility_valid == True
  2. block_end > t_now          (block not fully elapsed)
  3. t_arrive < block_end       (telescope arrives before block ends → capture_fraction > 0)
  4. can_fit(slew + idle + captured_duration + overhead)   — tier-capped:
       captured_duration = min(capture_fraction, obs_remaining_next_tier) × block_duration
       actions requiring long idle waits are still allowed; only observations whose
       actual time cost exceeds mission_end are rejected
  5. current_tier < max_tier    (not yet fully complete)
```

This allows the agent to see — and choose — observations where it arrives after the raw transit ends but before the observation block closes, yielding a partial capture rather than a hard miss.

`can_fit` uses the **tier-capped** captured duration rather than the full block duration.  When a target needs only 30 % of an observation to reach the next tier, the actual time cost is 30 % × block_duration, not the full 2.5 × T₁₄.  This makes near-completion observations cheaper in the feasibility check, consistent with how `execute_observation` stops the clock when the tier finishes early.

#### No-valid-action fallback

- **`topk` mode** (`_skip_to_next_feasible_topk`): when all K candidates fail the feasibility mask the env progressively looks at `2K, 3K, …` candidates rather than terminating immediately.  This prevents premature episode end when the agent is temporarily in a "bad" sky region.
- **`target` / `full_set` modes**: candidates are exactly one per target (fixed set); if no target is feasible, the episode terminates.  There is no larger window to look at.

### Observation space

```python
Dict({
    "events": Box(shape=(K, 18), dtype=float32),   # K = topk.k (default 50)
    "global": Box(shape=(G,),    dtype=float32),   # G = 9 named + n_large_bins (default 26)
})
```

All values are clipped to `[0, 1]` (or `[-3, 3]` for event features that can go negative, e.g. `stellar_metallicity`) after normalisation.  An `ObservationConfig` in the YAML controls which features to include and whether to normalise.

The observation was audited against 1 500 steps of random valid-action rollouts.  Features that were constant or near-zero throughout were replaced.  See **design notes** column for rationale.

#### Per-event features — 18 features, shape `(K, 18)`

Each of the K candidate events contributes one row.  Slots beyond the number of real events are **zero-padded** and correspond to invalid actions (masked out).

**Dynamic features** (change each step as the mission evolves):

| # | Feature | Source | Normalised by | Notes |
|---|---|---|---|---|
| 0 | `slew_time_days` | angular distance current→target | 2-hr cap (`0.0833 days`) | Scheduling cost from current pointing |
| 1 | `window_urgency_norm` | `(t_now − window_start) / window_duration` | already [0,1] | 0 = just opened, →1 = closing |
| 2 | `duration_days` | `event.duration_days` | 1 day | Raw transit / eclipse duration T₁₄ |
| 3 | `block_duration_days` | `COST_FACTOR × duration_days` | 1 day | Full observation block (2.5 × T₁₄) |
| 4 | `total_time_cost_days` | `slew + idle + effective_fraction × block_duration` | 3 days | True expected cost, reduced when tier is nearly complete (effective_fraction < 1) |
| 5 | `capture_fraction` | `(block_end − t_arrive) / block_dur` | already [0,1] | **New.** Fraction of block capturable if chosen now; 1.0 = full, <1 = late arrival |
| 6 | `progress_in_tier` | progress table | already [0,1] | Fraction of equivalent obs completed toward the **next** tier boundary |
| 7 | `obs_remaining_next_tier_norm` | `obs_remaining / tier3_required_obs` | per-target max | Equivalent obs still needed (float); comparable across targets |
| 17 | `days_to_block_end_norm` | `block_end − t_now` | 5 days | Time until the observation block closes (scheduling deadline); small = act now |

**Static features** (fixed per target across the mission):

| # | Feature | Source | Normalised by | Notes |
|---|---|---|---|---|
| 8 | `base_science_value` | event table (SNR-derived) | already [0,1] | Intrinsic scientific value independent of scheduling |
| 9 | `science_weight` | target table | already [0,1] | Inverse-bin-frequency rarity weight with `science_weight_floor` applied |
| 10 | `planet_radius_norm` | target table | 20 R⊕ | Population diversity feature |
| 11 | `planet_temperature_norm` | target table | 3 000 K | Equilibrium temperature |
| 12 | `planet_mass_norm` | target table | 4 000 M⊕ | Planet mass |
| 13 | `stellar_temperature_norm` | target table | 10 000 K | Host star T_eff |
| 14 | `stellar_metallicity` | target table | 1.5 dex | [Fe/H]; can be negative, clipped to [−3, 3] |
| 15 | `tier_goal_norm` | `event.tier_goal / 3` | already [0,1] | Which tier this observation contributes toward |
| 16 | `event_type_binary` | event table | — | 0 = transit, 1 = eclipse |

> **Removed features** (vs original design): `wait_time_days` (83 % zeros — superseded by `window_urgency_norm`), `is_valid` (constant 1.0 after action-mask fix — replaced by `days_to_block_end_norm`).
>
> **Added**: `capture_fraction` — tells the policy exactly how much science it would receive if it chose this event right now.  Combined with `total_time_cost_days`, the policy can compute the marginal value of a partial observation versus waiting for a cleaner window.

#### Global features — 9 named + N population-bin features, shape `(G,)`

Mission-level state that is the same for all K candidate slots.

**Named features (indices 0–8):**

| # | Feature | Source | Notes |
|---|---|---|---|
| 0 | `fraction_elapsed` | `clock.fraction_elapsed` | Mission time consumed [0,1] |
| 1 | `tier1_fraction` | `tier1_completed / total_targets` | T1-complete fraction [0,1] |
| 2 | `tier2_fraction` | `tier2_completed / total_targets` | T2-complete fraction [0,1] |
| 3 | `tier3_fraction` | `tier3_completed / total_targets` | T3-complete fraction [0,1] |
| 4 | `used_science_fraction` | `used_science_time / mission_length` | Science-time budget used [0,1] |
| 5 | `used_slew_fraction` | `used_slew_time / mission_length` | Slew-time budget used [0,1] |
| 6 | `used_idle_fraction` | `used_idle_time / mission_length` | Idle-time budget used [0,1]; non-zero signals scheduling inefficiency |
| 7 | `n_observations_norm` | raw count ÷ 5 000 | Cumulative observation count |
| 8 | `n_completed_targets_norm` | targets at `max_tier` / total | Fraction of catalogue "used up" and masked |

> **Removed**: `n_missed_norm` (constant 0 after action-mask feasibility fix).
> **Added**: `used_idle_fraction` — the agent can see when it is accumulating idle time and learn to avoid it.

**Population-bin features (indices 9 … G−1):**

One feature per population bin with `≥ min_bin_targets` targets (default **10**).  Each value is:

```
bin_fraction[b] = observations_made_in_bin_b / targets_in_bin_b
```

Normalised **per-bin** so a rare bin at 50 % coverage has the same magnitude as a common bin at 50 % coverage.  Bins with fewer than `min_bin_targets` targets are excluded (constant-zero features waste model capacity).

With the default catalogue (814 targets, 56 unique bins), `min_bin_targets = 10` retains **17 bins**, giving **G = 26** global features total.

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

`step_result` is always present after a valid action and includes: `tier_before`, `tier_after`, `tier_completed`, `missed`, `obs_duration_days`, `block_duration_days`, `slew_days`, `idle_days`, `total_cost_days`, `progress_before`, `progress_after`, `science_weight`, `population_bin`, `period`, `host_id`.  These are passed directly to `compute_reward` to avoid re-querying the state.

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
reward = tier_bonus + progress_shaping + efficiency_bonus
       + coverage_potential + unique_host + comparative
       − idle_penalty − miss_penalty
```

| Component | Type | Formula | Default weight |
|---|---|---|---|
| **Idle penalty** | Dense | `−idle_penalty_per_day × idle_days` | `idle_penalty_per_day`=0.005 |
| **Tier completion bonus** | Sparse | `tier_weight × science_weight × diversity_mult` fired when `tier_after > tier_before` | T1=3.0, T2=10.0, T3=30.0 |
| **Progress shaping** | Dense | `progress_weight × Δprogress × science_weight × diversity_mult × near_boost` | `progress_weight`=0.05 |
| **Efficiency reward** | Dense | `efficiency_weight × obs_duration / total_cost` | `efficiency_weight`=0.0 (see default YAML) |
| **Coverage potential** | Dense (fires on T1) | `coverage_weight × [U_pop(s_{t+1}) − U_pop(s_t)]` where `U_pop = Σ_b min(q_b/quota, 1)` | `coverage_weight`=2.0 |
| **Unique host bonus** | Sparse | `unique_host_weight` fired when a new planetary *system* reaches T1 for the first time | `unique_host_weight`=0.5 |
| **Comparative planetology bonus** | Sparse | `comparative_weight × min(n_siblings, 3)` when a T1 target shares a host with existing T1+ siblings | `comparative_weight`=0.3 |
| **Rarity / difficulty** | Dense | `rarity_weight × (period/period_ref)² / tier_worked` | `rarity_weight`=0.5 |
| **Missed-event penalty** | Sparse | `−miss_penalty` (arrived after `window_end`) | `miss_penalty`=0.1 |

> **Invalid-action penalty** (−`invalid_action_penalty`) is applied directly in `ArielEnv.step()` before `_compute_reward`; the clock does not advance.  With `MaskablePPO` this almost never fires.

**Near-completion boost** — when `progress_in_tier ≥ near_completion_threshold` (default 0.7), the progress reward is multiplied by `near_completion_scale` (default 3.0).

### Diversity multiplier (`_diversity_multiplier`)

Every science-facing component (tier bonus + progress shaping) is scaled by a live coverage-diversity multiplier:

```
observed_fraction = bin_tier1_completed / bin_total_targets   ∈ [0, 1]
diversity_mult    = 1 + (max_mult − 1) × max(0, 1 − observed_fraction)
                 ∈ [1, max_mult]    (default max_mult = 5.0)
```

- A bin with **nothing observed** yet → multiplier = **5.0** (maximum boost).
- A bin that is **fully saturated** → multiplier = **1.0** (no boost, no penalty).
- Computed live each step from `MissionState.population_bin_counts` and `_bin_totals`.

### Coverage potential reward (`compute_coverage_potential`)

The marginal signal that fires when a new Tier-1 completion advances population coverage:

```
U_pop(s) = Σ_b  min(q_b / quota_per_bin, 1)

where q_b = current T1+ completions in bin b
      quota_per_bin = desired coverage per bin (default 5)

r_coverage = coverage_weight × max(0, U_pop(s_{t+1}) − U_pop(s_t))
```

Once a bin reaches its quota, extra observations in that bin stop producing coverage reward.  This is the mechanism that naturally drives breadth across the catalogue.

### Science weight floor

`science_weight` is the inverse-bin-frequency rarity score.  Without a floor, the most common population bin receives `science_weight = 0`, making its tier completions worthless.  The floor remaps weights as:

```
science_weight' = floor + (1 − floor) × science_weight_normalised
```

Default `science_weight_floor = 0.3`: the most common bin receives weight 0.3, the rarest bin receives 1.0.

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
| Prioritise T3 over T2 over T1 within a target | T3 weight (30) > T2 (10) > T1 (3); cumulative so T3 earns 3+10+30=43 × scale total |
| Finish targets that are almost done | `near_completion_scale=3` when `progress_in_tier ≥ 0.7` |
| Cover the full population space at T1 | Coverage potential U_pop + milestone bonuses + quadratic terminal bonus |
| Reach *new* population bins | Diversity multiplier up to 5× for unseen bins |
| Don't miss rare long-period targets | Rarity bonus ∝ (period/365)² / tier_worked |
| Spread across planetary *systems* (not just bins) | Unique-host bonus for first T1 in each stellar system |
| Reward comparative planetology | Comparative bonus when a T1 target has T1+ siblings on the same host |
| Penalise idle waiting | `idle_penalty_per_day` on time spent waiting before block starts |
| Penalise wasted slew time | Efficiency reward = obs_duration / total_cost |
| All weights configurable without code changes | `RewardConfig` dataclass, overrideable in YAML or `configs/reward/default.yaml` |

### RewardConfig reference

See `configs/reward/default.yaml` for the canonical defaults.  All weights are configurable in YAML without code changes.

```yaml
reward:
  # --- sparse tier completion ---
  tier1_completion:           3.0
  tier2_completion:           10.0
  tier3_completion:           30.0

  # --- dense progress shaping ---
  progress_weight:            0.05
  near_completion_threshold:  0.7
  near_completion_scale:      3.0

  # --- diversity multiplier ---
  diversity_multiplier_max:   5.0   # unseen bin is max_mult× more attractive

  # --- population coverage potential U_pop ---
  coverage_quota_per_bin:     5     # desired T1+ observations per bin
  coverage_weight:            2.0

  # --- science weight floor ---
  science_weight_floor:       0.3   # prevents most-common bin from getting weight 0

  # --- time efficiency ---
  efficiency_weight:          0.0   # implicit via idle_penalty + obs/total_cost
  idle_penalty_per_day:       0.005 # per-day cost for waiting before block starts

  # --- host diversity ---
  unique_host_weight:         0.5   # first T1 in a new planetary system
  comparative_weight:         0.3   # T1 when host already has T1+ sibling(s)

  # --- rarity ---
  rarity_weight:              0.5
  rarity_period_ref_days:     365.0

  # --- coverage milestones (one-shot per episode) ---
  t1_milestone_fractions:     [0.25, 0.5, 0.75, 0.90, 1.0]
  t1_milestone_bonus:         50.0  # 100% milestone pays 2×

  # --- terminal bonus ---
  t1_terminal_weight:         300.0
  t1_terminal_power:          2.0   # quadratic: near-complete coverage most valuable

  # --- penalties ---
  miss_penalty:               0.1
  invalid_action_penalty:     0.5
```

---

## Layer 7 — RL Agents  ✅ Implemented

All agents are trained with **MaskablePPO** from `sb3-contrib`, which enforces action-validity at every rollout step without penalising the loss function for masked actions.

| Module | Description |
|---|---|
| `agents/ppo_masked.py` | Masked-env factory: `make_masked_env()`, `make_training_envs()` |
| `agents/rl_agent.py` | `RLAgentWrapper` — adapts a trained SB3 model to `BaselineAgent` interface |
| `agents/policies/event_attention_policy.py` | `ArielTransformerPolicy` — Top-K full self-attention over K event tokens |
| `agents/policies/full_set_isab_policy.py` | `FullSetISABPolicy` — ISAB Set Transformer over all N planet tokens (O(N·m)) |
| `agents/policies/full_set_attention_policy.py` | `FullSetSelfAttentionPolicy` — full O(N²) self-attention ablation over all N planet tokens |
| `agents/policies/isab_modules.py` | `MAB`, `ISAB`, `PMA` — Set Transformer primitives (Lee et al. 2019) |
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

A simple sanity-check policy — flattens the `(K × 18)` event matrix and the global vector, processes them through a shared MLP, and outputs independent logit and value heads.

```
obs["events"]  (K×18) ──┐
                         ├─ flatten → shared_mlp(256→256) → policy_head (K,) + value_head (1,)
obs["global"]  (G,)   ──┘
```

Action logits are clipped to `−∞` for masked positions before the softmax.

#### Policy architectures — three-way comparison

Three policies are available, targeting different action spaces and computational trade-offs:

| Policy | File | Action space | Token = | Attention | Global in actor | CLI flag |
|---|---|---|---|---|---|---|
| `ArielTransformerPolicy` | `event_attention_policy.py` | `topk` | 1 candidate event | O(K²) full | via CLS prepend | `--policy transformer` |
| `FullSetSelfAttentionPolicy` | `full_set_attention_policy.py` | `full_set` | 1 active planet | O(N²) full | ✅ broadcast + MLP | `--policy full_set_attention` |
| `FullSetISABPolicy` | `full_set_isab_policy.py` | `full_set` | 1 active planet | O(N·m) ISAB | ✅ broadcast + MLP | `--policy full_set_isab` |

The `ArielTransformerPolicy` remains the Top-K baseline — it is not replaced.

**Core invariants for full_set policies:**
1. One token = one active planet (completed planets are removed, not merely masked).
2. An action means "slew towards this planet now."
3. Each token's associated event is the **first reachable** opportunity — the first event whose block does not expire before the telescope arrives.  All per-planet features (immediate + future) are anchored to this event; `event_2`, `event_3` are the subsequent occurrences after it.
4. Possible-but-poor choices remain as valid actions — discouraged by learned value/reward, not unnecessary masking.  Only physically infeasible actions (observation would exceed `mission_end`) are masked.
5. Padding tokens (indices `n_active … N_max-1`) are always zero-vectors and always masked False.
6. Global mission state conditions **both actor and critic** in all full_set policies.
7. Runtime insertion of new targets mid-episode is **not yet supported** (deferred future work).  The dynamic set covers only removal of targets from the fixed initial catalogue.

#### `ArielTransformerPolicy` (`policies/event_attention_policy.py`)

The Top-K baseline.  Each of the K candidate events is treated as a token; a transformer encoder with multi-head self-attention processes the full set simultaneously, enabling the policy to reason about relative priorities across all candidates in a single pass.

```
obs["events"]  (K×18) → event_proj(18→d_model)  ─┐
                                                    ├─ [CLS | e_1 | … | e_K]
obs["global"]  (G,)   → global_proj(G→d_model)  ──┘  (prepend CLS from global embedding)
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

#### `FullSetSelfAttentionPolicy` and `FullSetISABPolicy` (active-planet set architectures)

Both policies share the same high-level structure — only the planet encoder differs.

```
obs["planets"]  (N_max × n_pf)
  │  (rows 0…n_active-1: real active planets; rows n_active…N_max-1: zero padding)
  │
  ▼  planet_proj(n_pf → d_model)
 tokens  (N_max, d_model)
  │
  ▼  [ISAB stack]  or  [TransformerEncoder]    — padding mask applied
 contextualised_tokens  (N_max, d_model)
  │
  ├──── Actor head ─────────────────────────────────────────┐
  │      global_proj_actor(G → d)                           │
  │      g_expand  = broadcast to (N_max, d)                │
  │      actor_in  = cat([token, g_expand]) → (N_max, 2d)   │
  │      actor_head: Linear(2d→d) → ReLU → Linear(d→1)      │
  │      logits (N_max,)  ← masked_fill(pad∪action_mask, -∞)│
  └─────────────────────────────────────────────────────────┘
  │
  └──── Critic head ────────────────────────────────────────┐
         PMA(d, n_heads, k=1): set → summary (d,)           │
         global_proj_critic(G → d)                          │
         critic_in = cat([summary, g_critic]) → (2d,)       │
         value_mlp: Linear(2d→d) → ReLU → Linear(d→1)       │
         value (1,)                                          │
         ──────────────────────────────────────────────────-┘
```

Key differences:
- **FullSetSelfAttentionPolicy**: uses `nn.TransformerEncoder` (O(N²)) — good ablation baseline, cheaper to implement.
- **FullSetISABPolicy**: uses `ISAB` layers with `m` inducing points (O(N·m)) — scales to N_max ≈ 2000 without O(N²) memory cost.  PMA critic is shared.

Global mission features condition both actor and critic — logits change when the global state changes even if all planet tokens are identical.

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
# Top-K Transformer (default)
python src/ariel_rl/scripts/train_agent.py \
    --policy transformer \
    --action-type topk \
    --total-timesteps 500_000 \
    --n-envs 4 \
    --run-name topk_run \
    --device auto

# Full-Set ISAB (2000 planet tokens, ISAB attention)
python src/ariel_rl/scripts/train_agent.py \
    --policy full_set_isab \
    --action-type full_set \
    --n-max 2000 \
    --total-timesteps 500_000 \
    --n-envs 4 \
    --run-name isab_run \
    --device auto
```

**Outputs** written to `outputs/<run_name>/`:

| File | Contents |
|---|---|
| `model.zip` | Trained `MaskablePPO` model (SB3 format) |
| `progress.csv` | Per-rollout: timestep, episode reward/length, value loss, policy loss, entropy, KL |
| `reward_config.yaml` | Snapshot of the reward config used for this run |
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

**`TrainingLoggerCallback`** captures rollout metrics from SB3's `logger` at each `on_rollout_end` call and writes them to `progress.csv` for offline analysis.

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
│       ├── event_attention_policy.py   ← ArielTransformerNet + ArielTransformerPolicy (Top-K)
│       ├── full_set_isab_policy.py     ← FullSetISABNet + FullSetISABPolicy (ISAB, O(N·m))
│       ├── full_set_attention_policy.py← FullSetSelfAttentionNet + FullSetSelfAttentionPolicy (ablation)
│       ├── isab_modules.py             ← MAB, ISAB, PMA (Set Transformer primitives)
│       └── mlp_scorer.py               ← ArielMlpNet + ArielMlpPolicy (sanity check)
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
    ├── progress.csv                ← per-rollout training metrics
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
