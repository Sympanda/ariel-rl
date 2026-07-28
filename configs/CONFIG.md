# Ariel RL — Configuration Reference

All environment parameters live in a frozen dataclass hierarchy (`EnvConfig`)
defined in `src/ariel_rl/utils/config.py`.  Config files are YAML; unknown keys
are silently ignored and missing keys fall back to code defaults.

Load a config:

```python
from ariel_rl.utils.config import load_env_config, default_env_config

cfg = load_env_config("configs/env/simple.yaml")   # from file
cfg = default_env_config()                          # all defaults, no file needed
```

Override specific fields at runtime (e.g. for curriculum):

```python
import dataclasses
cfg = dataclasses.replace(cfg.mission, lifetime_days=365, max_tier_cap=1)
```

Or via the training CLI:

```bash
python -m ariel_rl.scripts.train_agent \
    --lifetime-days 365 --max-tier-cap 1 --run-name curriculum_t1
```

---

## Status legend

| Symbol | Meaning |
|---|---|
| ✅ Active | Field is read by the environment / reward / simulator at runtime |
| ⚠️  Partial | Field is read but only affects one specific mode or edge case |
| ❌ Dead | Field exists in the dataclass but is never read by runtime code |

---

## `MissionConfig`

```yaml
mission:
  start_bjd:           2462867.5   # ~2029-01-01
  lifetime_days:       1278.375    # 3.5-year mission
  max_tier_cap:        3
  cost_factor:         2.5         # ❌ dead
  overhead_days_per_obs: 0.0       # ❌ dead
```

| Field | Default | Status | Effect |
|---|---|---|---|
| `start_bjd` | `2462867.5` | ✅ | BJD start of the event table and mission clock. All window times are relative to this. |
| `lifetime_days` | `1278.375` | ✅ | Mission duration. The event table is generated up to `start_bjd + lifetime_days`; episodes terminate when the clock exceeds this. **Primary curriculum lever.** |
| `max_tier_cap` | `3` | ✅ | Global ceiling on tier achievement. `1` = T1-only run; `2` = exclude T3. Targets already at their cap are **masked from the action space**. Useful for ablation studies. |
| `cost_factor` | `2.5` | ❌ Dead | Observation cost multiplier (`2.5 × T14 / 86400 days`). Written into the config for documentation purposes but observation costs are baked into the target table **at build time** from `schemas.COST_FACTOR` — changing this in YAML has no effect at run time. |
| `overhead_days_per_obs` | `0.0` | ❌ Dead | Intended per-observation operational overhead. The `MissionClock` has its own internal `overhead_days` parameter with a separate default. Wiring this up is a future task. |

---

## `SlewConfig`

```yaml
slew:
  rate_deg_per_min:   1.0      # placeholder — true Ariel value TBD
  min_slew_seconds:   120.0    # 2-min guide-star settle floor
  max_slew_seconds:   7200.0   # 2-hr cap
```

| Field | Default | Status | Effect |
|---|---|---|---|
| `rate_deg_per_min` | `1.0` | ✅ | Degrees per minute the telescope can slew. **The most uncertain physical parameter** — the true Ariel slew performance is not yet published. Increase to test how much slew cost is limiting performance. |
| `min_slew_seconds` | `120` | ✅ | Minimum slew time regardless of angular separation (guide-star acquisition floor). |
| `max_slew_seconds` | `7200` | ✅ | Maximum slew time cap. Prevents pathological cross-sky slews from dominating the budget. |

---

## `ActionConfig`

```yaml
action:
  type: topk             # "topk" | "target" | "full_set"
  topk:
    k: 50
    sort_by: window_mid  # ❌ dead
  target:
    include_completed: false
  full_set:
    include_completed: false
    cache_static: true
```

| Field | Default | Status | Effect |
|---|---|---|---|
| `type` | `"topk"` | ✅ | Selects the action space. `"topk"` — K next events; `"target"` — all N targets (env picks next event per target); `"full_set"` — all N targets with full per-planet feature matrix. |
| `topk.k` | `50` | ✅ | Size of the candidate window. The transformer sees a `(K × 18)` events matrix. Smaller K = narrower view but faster; larger K = wider scheduling horizon. |
| `topk.sort_by` | `"window_mid"` | ❌ Dead | Backends always sort by `window_mid` internally; this value is never passed through. |
| `target.include_completed` | `False` | ⚠️ Partial | Only applies when `type="target"` or `"full_set"`. If `False`, targets that have reached `max_tier` are masked out. |
| `full_set.cache_static` | `True` | ✅ | Pre-compute static planet features at `reset()` and recompute only dynamic features each step. Saves ~2 ms/step on the full catalogue. |

> **Default backend**: `DynamicBackend` — no event table needed, infinite mission horizon.
> All three action types work with `DynamicBackend`.  `TableBackend` is deprecated.

---

## `ObservationConfig`

```yaml
observation:
  normalise: true
  include_population_bin_fractions: true
  min_bin_targets: 10
  event_features:
    - slew_time_days
    - window_urgency_norm
    - duration_days
    - total_time_cost_days
    - progress_in_tier
    - obs_remaining_next_tier_norm
    - base_science_value
    - science_weight
    - planet_radius_norm
    - planet_temperature_norm
    - planet_mass_norm
    - stellar_temperature_norm
    - stellar_metallicity
    - tier_goal_norm
    - event_type_binary
    - days_to_block_end_norm
  global_features:
    - fraction_elapsed
    - tier1_fraction
    - tier2_fraction
    - tier3_fraction
    - used_science_fraction
    - used_slew_fraction
    - n_observations_norm
    - n_completed_targets_norm
```

| Field | Default | Status | Effect |
|---|---|---|---|
| `normalise` | `True` | ✅ | Apply per-feature normalisation. Should always be `True` for RL training. |
| `include_population_bin_fractions` | `True` | ✅ | Append one feature per population bin to `obs["global"]`. Gives the agent live diversity feedback. Set `False` to reduce obs size. |
| `min_bin_targets` | `10` | ✅ | Exclude bins with fewer than this many targets. Prevents empty/near-zero features from wasting model capacity. With the default catalogue, `10` retains 17 bins → `G = 25` global features. |
| `event_features` | All 18 (see below) | ✅ | Subset or reorder per-event features. Changing this directly changes the input dimension to the policy network — you must retrain after any change. |
| `global_features` | All 9 (see below) | ✅ | Subset or reorder mission-state scalar features. Same caveat — retrain after changing. |

### Per-event features (`obs["events"]`, shape `K × 18`)

Each of the K candidate events contributes one row.  Rows beyond the real event count are zero-padded (masked actions).

**Static features** (fixed per target across the mission):

| Feature | Normalisation | Description |
|---|---|---|
| `base_science_value` | already [0,1] | Catalogue SNR-derived intrinsic value. |
| `science_weight` | already [0,1] | Inverse-bin-frequency rarity weight (with `science_weight_floor` applied). |
| `planet_radius_norm` | ÷ 20 R⊕ | Planet radius. |
| `planet_temperature_norm` | ÷ 3000 K | Equilibrium temperature. |
| `planet_mass_norm` | ÷ 4000 M⊕ | Planet mass. |
| `stellar_temperature_norm` | ÷ 10000 K | Host star T_eff. |
| `stellar_metallicity` | ÷ 1.5 dex | [Fe/H]. Negative values allowed; clipped to [−3, 3]. |
| `tier_goal_norm` | ÷ 3 | Which tier this event contributes toward. |
| `event_type_binary` | — | 0 = transit, 1 = eclipse. |

**Dynamic features** (update every step):

| Feature | Normalisation | Description |
|---|---|---|
| `slew_time_days` | ÷ 2-hr cap | Angular slew cost from current pointing. Core scheduling signal. |
| `window_urgency_norm` | already [0,1] | `(t_now − window_start) / window_duration`. 0 = just opened, →1 = closing. |
| `duration_days` | ÷ 1 day | Raw transit/eclipse duration T₁₄. |
| `block_duration_days` | ÷ 1 day | Full observation block = 2.5 × T₁₄; the authoritative clock advance per observation. |
| `total_time_cost_days` | ÷ 3 days | `slew + idle + effective_fraction × block_duration`. Uses the tier-capped effective duration, so cost shrinks as a target nears tier completion. |
| `capture_fraction` | already [0,1] | **Fraction of the observation block capturable if chosen now.** 1.0 = arrive before block_start (full), <1 = arrive mid-block (partial). |
| `progress_in_tier` | already [0,1] | Equivalent obs fraction completed toward the **next** tier boundary (float). |
| `obs_remaining_next_tier_norm` | ÷ per-target max | Equivalent obs still needed (float), normalised for cross-target comparability. |
| `days_to_block_end_norm` | ÷ 5 days | `block_end − t_now` where `block_end = mid + 1.25 × T₁₄`. The correct scheduling deadline; an event is still capturable until block_end, not just window_end. |

> **Removed** (vs original design): `wait_time_days` (83% zeros → superseded by `window_urgency_norm`), `is_valid` (constant 1 → replaced by `days_to_block_end_norm`), `days_to_window_end_norm` (used raw transit end; replaced by `days_to_block_end_norm`).

### Global features (`obs["global"]`, shape `G = 26`)

**Named features (indices 0–8):**

| Feature | Description |
|---|---|
| `fraction_elapsed` | Mission time consumed [0,1] |
| `tier1_fraction` | T1-complete targets / total [0,1] |
| `tier2_fraction` | T2-complete targets / total [0,1] |
| `tier3_fraction` | T3-complete targets / total [0,1] |
| `used_science_fraction` | Science time / mission length [0,1] |
| `used_slew_fraction` | Slew time / mission length [0,1] |
| `used_idle_fraction` | Idle/wait time / mission length [0,1] |
| `n_observations_norm` | Cumulative observation count ÷ 5000 |
| `n_completed_targets_norm` | Targets fully at `max_tier` / total [0,1] |

**Population-bin fractions (indices 9–25):** `observations_in_bin / targets_in_bin` per bin (bins with ≥ `min_bin_targets` targets; default 17 bins).

> **Removed**: `n_missed_norm` (constant 0 after action-mask feasibility fix).
> **Added**: `used_idle_fraction` — lets the agent see how much mission time it is losing to idle waiting.

---

## `RewardConfig`

The canonical defaults live in `configs/reward/default.yaml` and are automatically saved to `outputs/<run_name>/reward_config.yaml` at training start for reproducibility.

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
  diversity_multiplier_max:   5.0

  # --- population coverage potential U_pop ---
  coverage_quota_per_bin:     5
  coverage_weight:            2.0

  # --- science weight floor ---
  science_weight_floor:       0.3

  # --- time efficiency ---
  efficiency_weight:          0.0
  idle_penalty_per_day:       0.005

  # --- host diversity ---
  unique_host_weight:         0.5
  comparative_weight:         0.3

  # --- rarity ---
  rarity_weight:              0.5
  rarity_period_ref_days:     365.0

  # --- milestones / terminal ---
  t1_milestone_fractions:     [0.25, 0.5, 0.75, 0.90, 1.0]
  t1_milestone_bonus:         50.0
  t1_terminal_weight:         300.0
  t1_terminal_power:          2.0

  # --- penalties ---
  miss_penalty:               0.1
  invalid_action_penalty:     0.5
```

| Field | Default | Status | Effect |
|---|---|---|---|
| `tier1_completion` | `3.0` | ✅ | Sparse reward when a target crosses the T1 boundary. Scaled by `science_weight × diversity_mult`. |
| `tier2_completion` | `10.0` | ✅ | Same for T2. |
| `tier3_completion` | `30.0` | ✅ | Same for T3. Full T1→T2→T3 earns `(3+10+30) × scale = 43 × scale` total. |
| `progress_weight` | `0.05` | ✅ | Dense per-step shaping proportional to `Δprogress_in_tier`. Kept small — the tier bonuses dominate. |
| `near_completion_threshold` | `0.7` | ✅ | `progress_in_tier` above which the near-finish boost fires. |
| `near_completion_scale` | `3.0` | ✅ | Multiplier on progress reward near a tier boundary. |
| `diversity_multiplier_max` | `5.0` | ✅ | Maximum diversity multiplier for a completely unseen bin. At 5.0 an unseen bin is 5× more attractive than a saturated one. |
| `coverage_quota_per_bin` | `5` | ✅ | Desired T1+ observations per population bin for the U_pop coverage signal. Once a bin hits its quota, extra observations no longer earn coverage reward. |
| `coverage_weight` | `2.0` | ✅ | Scale on the marginal `U_pop(s_{t+1}) − U_pop(s_t)` coverage signal. |
| `science_weight_floor` | `0.3` | ✅ | Minimum science weight after rarity normalisation. Prevents the most common bin from receiving weight 0. |
| `efficiency_weight` | `0.0` | ✅ | Scale on `obs_duration / total_cost`. Set to a positive value to explicitly reward efficient scheduling. Default 0 — the idle penalty covers this implicitly. |
| `idle_penalty_per_day` | `0.005` | ✅ | Per-day cost for waiting before the observation block starts (arrived early). Encourages the agent not to lock on to a target well before its window. |
| `unique_host_weight` | `0.5` | ✅ | Bonus fired the first time any target in a new planetary *system* reaches T1. |
| `comparative_weight` | `0.3` | ✅ | Bonus fired when a T1 target shares a host star with at least one existing T1+ target (comparative planetology). Scales with number of siblings (capped at 3×). |
| `rarity_weight` | `0.5` | ✅ | Per-observation bonus: `rarity_weight × (period/period_ref)² / tier_worked`. |
| `rarity_period_ref_days` | `365.0` | ✅ | Period mapped to `difficulty = 1.0`. |
| `t1_milestone_fractions` | `[0.25, 0.5, 0.75, 0.9, 1.0]` | ✅ | T1 coverage fractions that each trigger a one-shot bonus (fires at most once per episode). |
| `t1_milestone_bonus` | `50.0` | ✅ | Value per milestone. 100% milestone pays double. |
| `t1_terminal_weight` | `300.0` | ✅ | Scale of the end-of-episode bonus `weight × (t1_fraction)^power`. |
| `t1_terminal_power` | `2.0` | ✅ | Quadratic default: near-complete T1 coverage is disproportionately rewarded. |
| `subtract_random_baseline` | `false` | ✅ | If `true`, subtract `random_baseline_per_step` every valid step to centre reward around random-agent performance. |
| `random_baseline_per_step` | `4.0` | ✅ (when flag on) | Calibrate from a short random-agent run: `total_reward / n_steps`. |
| `miss_penalty` | `0.1` | ✅ | Subtracted when the agent arrives after `window_end`. |
| `invalid_action_penalty` | `0.5` | ⚠️ | Applied by `ArielEnv.step()` directly; rarely fires with `MaskablePPO`. |

---

## Existing config files

| File | Purpose | Status |
|---|---|---|
| `env/simple.yaml` | topk K=50, full 3.5-year mission | ⚠️ Outdated feature names (see below) |
| `env/full.yaml` | target action space (all N targets) | ⚠️ Outdated feature names |
| `env/with_visibility.yaml` | topk, faster slew for sensitivity tests | ⚠️ Outdated feature names |
| `env/with_slew.yaml` | (empty placeholder) | — |
| `reward/balanced_population.yaml` | (empty placeholder) | — |
| `reward/chemical_consensus.yaml` | (empty placeholder) | — |
| `reward/tier_count.yaml` | (empty placeholder) | — |
| `agent/attention_policy.yaml` | (empty placeholder) | — |
| `agent/greedy_baseline.yaml` | (empty placeholder) | — |
| `agent/ppo_masked.yaml` | (empty placeholder) | — |
| `experiment/debug.yaml` | (empty placeholder) | — |
| `experiment/train_full.yaml` | (empty placeholder) | — |
| `experiment/train_small.yaml` | (empty placeholder) | — |

> **Outdated feature names**: The three env YAMLs were written before the
> observation audit.  They list `wait_time_days`, `is_valid`, and `n_missed_norm`
> which have been removed from the observation builder.  They also list old reward
> keys (`diversity_bonus`, `missed_event_penalty`, `time_efficiency_bonus`) that
> don't exist in `RewardConfig`.  **The code defaults (`default_env_config()`) are
> always authoritative.** When using `--config`, prefer the feature lists shown
> in the `ObservationConfig` section above.

---

## Useful experiment recipes

### Curriculum stage 1 — T1 only, short episode

```yaml
mission:
  lifetime_days: 365
  max_tier_cap: 1
```

```bash
python -m ariel_rl.scripts.train_agent \
    --lifetime-days 365 --max-tier-cap 1 --run-name curriculum_t1
```

### Reduce action space

```yaml
action:
  type: topk
  topk:
    k: 25   # narrower window, faster per-step
```

### Pure sparse reward (remove dense shaping)

```yaml
reward:
  progress_weight: 0.0
  efficiency_weight: 0.0
```

### Stronger terminal signal (focus on survey completion)

```yaml
reward:
  t1_terminal_weight: 100.0
  t1_terminal_power: 3.0
  t1_milestone_bonus: 50.0
```

### Faster slew sensitivity test

```yaml
slew:
  rate_deg_per_min: 5.0
  min_slew_seconds: 30.0
```
