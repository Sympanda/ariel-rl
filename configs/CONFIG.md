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
  type: topk             # "topk" | "target"
  topk:
    k: 50
    sort_by: window_mid  # ❌ dead
  target:
    include_completed: false
```

| Field | Default | Status | Effect |
|---|---|---|---|
| `type` | `"topk"` | ✅ | Selects the action space. `"topk"` presents the agent with the *K* next upcoming events; `"target"` presents all N targets and schedules the next event automatically. |
| `topk.k` | `50` | ✅ | Size of the candidate window. The transformer sees a `(K × 16)` events matrix. Smaller K = narrower view but faster; larger K = wider scheduling horizon. |
| `topk.sort_by` | `"window_mid"` | ❌ Dead | Intended to control candidate ordering but the backends always sort by `window_mid` internally; this value is never passed through. |
| `target.include_completed` | `False` | ⚠️ Partial | Only applies when `type="target"`. If `False`, targets that have reached `max_tier` are masked out of the action space. |

> **Note:** `type="target"` requires `TableBackend` (pre-computed events).
> Use `type="topk"` with `DynamicBackend` for training (no pre-computation needed,
> infinite horizon).

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
    - days_to_window_end_norm
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
| `event_features` | All 16 (see below) | ✅ | Subset or reorder per-event features. Changing this directly changes the input dimension to the policy network — you must retrain after any change. |
| `global_features` | All 8 (see below) | ✅ | Subset or reorder mission-state scalar features. Same caveat — retrain after changing. |

### Per-event features (`obs["events"]`, shape `K × 16`)

Each of the K candidate events contributes one row.  Rows beyond the real event count are zero-padded and correspond to masked actions.

| Feature | Normalisation | Description |
|---|---|---|
| `slew_time_days` | ÷ 2-hr cap | Angular slew cost from current pointing. Core scheduling signal. |
| `window_urgency_norm` | already [0,1] | `(t_now − window_start) / window_duration`. 0 = window just opened, →1 = closing. Replaces the old `wait_time_days`. |
| `duration_days` | ÷ 1 day | Transit / eclipse duration T₁₄. |
| `total_time_cost_days` | ÷ 1 day | `slew + duration` (excludes idle wait). |
| `progress_in_tier` | already [0,1] | Fraction of observations completed toward the **next** tier boundary. 0 for unvisited targets. |
| `obs_remaining_next_tier_norm` | ÷ per-target max | Observations still needed to reach the next tier, normalised so values are comparable across targets. |
| `base_science_value` | already [0,1] | Catalogue SNR-derived intrinsic value. Static per target. |
| `science_weight` | already [0,1] | Catalogue priority weight from rarity scoring. Static per target. |
| `planet_radius_norm` | ÷ 20 R⊕ | Physical feature for population diversity. |
| `planet_temperature_norm` | ÷ 3000 K | Equilibrium temperature. |
| `planet_mass_norm` | ÷ 4000 M⊕ | Planet mass. |
| `stellar_temperature_norm` | ÷ 10000 K | Host star T_eff. |
| `stellar_metallicity` | ÷ 1.5 dex | [Fe/H]. Can be negative; clipped to [−3, 3]. |
| `tier_goal_norm` | ÷ 3 | Which tier this observation contributes toward. |
| `event_type_binary` | — | 0 = transit, 1 = eclipse. |
| `days_to_window_end_norm` | ÷ 2 days | `(window_end − t_now)`. Absolute urgency; small = act now. Replaces the old `is_valid` (constant 1 after the mask fix). |

> **Removed features** (vs the old `simple.yaml`): `wait_time_days` (83 % zeros
> — superseded by `window_urgency_norm`) and `is_valid` (constant 1.0 after the
> action-mask feasibility fix — replaced by `days_to_window_end_norm`).

### Global features (`obs["global"]`, shape `G = 25`)

| Feature | Description |
|---|---|
| `fraction_elapsed` | Mission time consumed [0,1] |
| `tier1_fraction` | T1-complete targets / total [0,1] |
| `tier2_fraction` | T2-complete targets / total [0,1] |
| `tier3_fraction` | T3-complete targets / total [0,1] |
| `used_science_fraction` | Science time / mission length [0,1] |
| `used_slew_fraction` | Slew time / mission length [0,1] |
| `n_observations_norm` | Raw observation count ÷ 5000 |
| `n_completed_targets_norm` | Fraction of targets fully at `max_tier` [0,1] |
| + 17 population-bin fractions | `observations_in_bin / targets_in_bin` per bin (bins with ≥ `min_bin_targets` targets) |

> **Removed feature**: `n_missed_norm` (constant 0 after the action-mask
> feasibility fix; replaced by `n_completed_targets_norm`).

---

## `RewardConfig`

```yaml
reward:
  # Sparse tier completion bonuses
  tier1_completion: 1.0
  tier2_completion: 3.0
  tier3_completion: 10.0

  # Dense per-step shaping
  progress_weight:           0.3
  efficiency_weight:         0.5
  near_completion_threshold: 0.7
  near_completion_scale:     3.0

  # Rarity / difficulty bonus
  rarity_weight:          0.5
  rarity_period_ref_days: 365.0

  # Coverage milestone bonuses (one-shot per episode)
  t1_milestone_fractions: [0.25, 0.5, 0.75, 0.90, 1.0]
  t1_milestone_bonus: 20.0

  # Terminal episode bonus
  t1_terminal_weight: 50.0
  t1_terminal_power:  2.0

  # Random-baseline subtraction
  subtract_random_baseline: false
  random_baseline_per_step: 4.0

  # Penalties
  miss_penalty: 0.1
  invalid_action_penalty: 0.5   # ❌ dead — hardcoded -0.5 in ariel_env.py
```

| Field | Default | Status | Effect |
|---|---|---|---|
| `tier1_completion` | `1.0` | ✅ | Sparse reward when a target crosses the T1 boundary. Scaled by `science_weight × diversity_mult` at runtime. |
| `tier2_completion` | `3.0` | ✅ | Same for T2. Ratio T2/T1 = 3× incentivises going deeper on targets worth it. |
| `tier3_completion` | `10.0` | ✅ | Same for T3. A target taken T1→T2→T3 earns `(1+3+10) × scale = 14 × scale` total. |
| `progress_weight` | `0.3` | ✅ | Scale on the dense progress-shaping signal. Provides gradient in the long stretches between tier boundaries. Set to `0.0` for pure sparse reward (harder but less hand-engineered). |
| `near_completion_threshold` | `0.7` | ✅ | `progress_in_tier` above which the near-finish boost fires. |
| `near_completion_scale` | `3.0` | ✅ | Multiplier on progress reward when above threshold. Makes the final 30% of a tier worth 3× more per step — discourages abandoning almost-complete targets. |
| `efficiency_weight` | `0.5` | ✅ | Scale on `obs_duration / (obs + slew)`. Penalises long slews without requiring a separate explicit slew penalty. Set to `0.0` to remove. |
| `rarity_weight` | `0.5` | ✅ | Scale of the per-observation rarity bonus: `rarity_weight × (period / period_ref)² / tier_worked`. Set to `0.0` to disable. |
| `rarity_period_ref_days` | `365.0` | ✅ | Orbital period [days] that maps to `difficulty = 1.0`. Planets with `period ≥ period_ref` all get the maximum difficulty score. |
| `t1_milestone_fractions` | `[0.25, 0.5, 0.75, 0.9, 1.0]` | ✅ | T1 catalogue coverage fractions that each trigger a one-shot bonus. Fire at most once per episode. Directly incentivise breadth over the whole catalogue. |
| `t1_milestone_bonus` | `20.0` | ✅ | Value of each milestone bonus. The 100% milestone pays double (`2 × bonus = 40`). |
| `t1_terminal_weight` | `50.0` | ✅ | Scale of the end-of-episode bonus `weight × (t1_fraction)^power`. |
| `t1_terminal_power` | `2.0` | ✅ | Exponent on T1 coverage. Quadratic (`2.0`) makes the last 10% of coverage far more valuable than the first 10% — the agent is pushed hardest to complete the survey. Set to `1.0` for linear. |
| `subtract_random_baseline` | `false` | ✅ | If `true`, subtract `random_baseline_per_step` from every valid observation reward. Centres reward around zero for random-agent behaviour so PPO advantage estimates reflect improvement *over* random, not absolute magnitude. |
| `random_baseline_per_step` | `4.0` | ✅ (when flag is on) | Constant to subtract per valid step. Calibrate as `total_random_reward / n_valid_observations` for your chosen reward weights. With `sparse_dominant.yaml` weights use ~`0.5` instead of `4.0`. |
| `miss_penalty` | `0.1` | ✅ | Subtracted when the agent arrives after `window_end` (observation missed). |
| `invalid_action_penalty` | `0.5` | ❌ Dead | Defined in config but `ArielEnv.step()` hardcodes `-0.5` directly. Since `MaskablePPO` never sends an invalid action, this rarely fires anyway. |

### Reward magnitude reference

At typical episode length with `SmartGreedy`:

| Source | Approx. contribution |
|---|---|
| T1 completions (~200 targets) | 200 × 1.0 × ~1.5 scale ≈ **300** |
| T2/T3 completions | smaller (fewer completions in 3.5 years) |
| Progress shaping | **~50–100** (dense, many steps) |
| Efficiency reward | **~50–100** (dense, every step) |
| Rarity bonus | **~20–60** (dense; depends on period distribution) |
| Coverage milestones | up to **100** (5 × 20, if reached) |
| Terminal bonus | 50 × (0.25)² ≈ **3** (low T1 fraction in short episodes) |
| **Total typical range** | **~200–600** per episode |

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
