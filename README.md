# Ariel RL Target Selection

Reinforcement learning environment for optimising Ariel target selection and observation scheduling.

This project explores whether reinforcement learning can learn scientifically useful target-selection policies for the ESA Ariel mission. The agent selects exoplanet targets from the Ariel Mission Candidate Sample (MCS), spends observation time, advances the mission clock, and receives rewards based on population coverage, chemical-demographic balance, and observing efficiency.

The long-term goal is not simply to maximise the number of observations. The goal is to build a policy that produces a chemically and astrophysically balanced Ariel observing programme across planet populations, host-star types, temperatures, radii, masses, and other science-relevant bins.

## Motivation

Ariel is designed to conduct a large-scale atmospheric survey of transiting exoplanets. The mission target list is selected from a larger pool of possible targets, with observations organised into science tiers. A good observing programme must balance several competing objectives:

* observe many planets within a finite mission lifetime;
* cover a diverse population of exoplanets;
* avoid over-selecting easy but scientifically redundant targets;
* prioritise underrepresented regions of parameter space;
* account for tier-dependent observation costs;
* eventually respect time-dependent observability constraints such as transits, eclipses, ephemerides, and visibility windows.

This makes the problem a natural candidate for reinforcement learning, combinatorial optimisation, and scheduling methods.

## Project concept

The environment is based on the following working model.

At each decision step, the agent receives a snapshot of the current mission state. This state includes the current mission time, remaining observing budget, available target information, previous observations, and the current coverage of science bins.

The agent selects a target, and eventually a target-tier pair. The environment computes the required observing time, advances the mission clock to after the observation, updates the target catalogue state, and returns a reward.

In the first version, this is treated as a budgeted target-selection problem. Later versions will add proper scheduling dynamics, where the environment advances to the next feasible transit or eclipse event and masks targets that are not observable at the current time.

## Current status

Core infrastructure, realistic scheduling dynamics, and RL agent training are fully implemented.

| Component | Status |
|---|---|
| MCS data loading + preprocessing | ✅ |
| Gymnasium environment (`ArielEnv`) with action masking | ✅ |
| Five baseline scheduling heuristics | ✅ |
| Evaluation framework + 7 diagnostic plot types | ✅ |
| Mission clock (science / slew / idle / overhead time tracking) | ✅ |
| DynamicBackend: on-the-fly transit/eclipse windows, infinite horizon | ✅ |
| Slew model (haversine + rate cap) | ✅ |
| Feasibility-aware action masking (`block_end` threshold, partial-obs aware) | ✅ |
| Observation timing: slew immediately → idle → observe (2.5 × T₁₄ block) | ✅ |
| Partial observation model: mid-block arrivals give fractional progress (`capture_fraction`) | ✅ |
| Multi-component reward (tier, progress, coverage potential U_pop, unique host, comparative, idle penalty, milestones, terminal) | ✅ |
| Science weight floor + diversity multiplier | ✅ |
| Reward config saved per training run for reproducibility | ✅ |
| RL agents: MLP policy + Transformer policy (MaskablePPO) | ✅ |
| Training CLI with device auto-detection and post-training plots | ✅ |
| Full-set action space (all N targets with per-planet features) | ✅ |

The intended development path:

1. ✅ load and validate the Ariel MCS;
2. ✅ compute tier-dependent observation costs in days;
3. ✅ define target features and science bins;
4. ✅ implement a Gymnasium-style environment;
5. ✅ train simple baselines and RL agents;
6. ✅ compare RL policies against random, greedy, and optimisation baselines;
7. ✅ add realistic scheduling constraints (transit windows, ephemerides, slew model, idle tracking).

## Data

The main input data is the Ariel Mission Candidate Sample CSV.

Expected raw data location:

```text
data/raw/Ariel_MCS_Known_2025-08-18.csv
```

The dataset contains planet, host-star, observability, and Ariel tier information. The exact column names should be validated in `src/ariel_rl/data.py` rather than hardcoded throughout notebooks.

Important columns expected by the first prototype include:

```text
Planet Name
Planet Radius [R_Earth]
Planet Mass [M_Earth]
Planet Temperature [K]
Planet Period [days]
Star Temperature [K]
Star Metallicity
Star Age [Gyr]
Transit Duration T14 [s]
Tier 1 Observations
Tier 2 Observations
Tier 3 Observations
```

Column names may need to be adapted to the actual MCS release being used.

## Observation cost

Each action advances the mission clock by:

```text
total_time = slew_time + idle_time + block_duration + overhead

where:
  block_duration = 2.5 × T14_seconds / 86400  (COST_FACTOR × T14 in days)
  slew_time      = angular_separation / slew_rate  (clamped to [2 min, 2 hr])
  idle_time      = max(0, block_start − t_arrive)  (arrived before window)
```

The 2.5× factor accounts for the transit/eclipse itself plus the required out-of-transit baseline.  All three tiers share the same per-observation block cost; higher tiers cost more in total because they require more observations.

## RL formulation

### State

The agent observation at each step is a dict of two arrays:

**Per-event features** (`obs["events"]`, shape `K × 18`) — one row per candidate event:

*Static* (fixed per target): `base_science_value`, `science_weight`, `planet_radius`, `planet_temperature`, `planet_mass`, `stellar_temperature`, `stellar_metallicity`, `tier_goal`, `event_type`.

*Dynamic* (update each step): `slew_time_days`, `window_urgency_norm`, `duration_days`, `block_duration_days` (= 2.5 × T₁₄), `total_time_cost_days` (slew + idle + **effective_fraction** × block_dur + overhead; matches the physical clock advance), `capture_fraction` (fraction of block still capturable if chosen now; 1 = full, <1 = late arrival), `progress_in_tier`, `obs_remaining_next_tier_norm`, `days_to_block_end_norm` (time to scheduling deadline `block_end = mid + 1.25 × T₁₄`; replaces `days_to_window_end_norm`).

**Global mission features** (`obs["global"]`, shape `G = 26`) — mission-level summary the same for all K candidates:

`fraction_elapsed`, `tier1/2/3_fraction`, `used_science_fraction`, `used_slew_fraction`, `used_idle_fraction`, `n_observations_norm`, `n_completed_targets_norm`, + 17 per-population-bin coverage fractions.

### Action

```text
action ∈ {0, …, K−1}  (topk mode, default)
action ∈ {0, …, N−1}  (target / full_set mode)
```

Invalid actions are masked out at every step by the feasibility-aware action mask.  With `MaskablePPO` the agent only ever sees valid actions.

### Reward

The reward is a sum of scientifically motivated components (all configurable in YAML):

```text
reward =
    tier_completion_bonus          (sparse, per tier boundary crossed)
  + progress_shaping               (dense, per observation toward a tier)
  + coverage_potential_U_pop       (dense, marginal Σ_b min(q_b/quota, 1))
  + unique_host_bonus              (sparse, first T1 in each planetary system)
  + comparative_planetology_bonus  (sparse, T1 with T1+ sibling on same host)
  + rarity_bonus                   (dense, long-period targets)
  − idle_penalty                   (dense, per-day waiting before block starts)
  − miss_penalty                   (sparse, arrived after block_end — no capture possible)
  [+ milestone bonuses + terminal bonus]
```

Population bins: `planet_radius × planet_temperature × stellar_type` (3-dimensional grid, 17 bins with ≥ 10 targets each).

## Repository structure

```text
ariel-rl/
├── README.md
├── ARCHITECTURE.md              ← detailed layer-by-layer design notes
├── RL_DESIGN.md                 ← RL architecture design exploration
├── environment.yml              ← Conda environment (PyTorch, SB3, sb3-contrib, …)
├── pyproject.toml
├── data/
│   └── raw/
│       └── Ariel_MCS_Known_2025-08-18.csv
├── configs/
│   └── env/
│       ├── simple.yaml          ← topk K=50 (default)
│       ├── full.yaml            ← all-targets action space
│       └── with_visibility.yaml
├── scripts/
│   └── run_short_episode.py     ← compare baselines (+ optional RL model), save plots
├── src/
│   └── ariel_rl/
│       ├── data/                ← MCS loading, population bins, observation requirements
│       ├── simulator/           ← ephemeris, event backends, mission state
│       ├── envs/                ← ArielEnv (Gymnasium), observation builder, action mask
│       ├── baselines/           ← 5 scheduling heuristics
│       ├── evaluation/          ← metrics, coverage, compare_runs, plots
│       ├── rewards/             ← compute_reward, milestones, terminal bonus
│       ├── agents/              ← MaskablePPO setup, RLAgentWrapper, policies
│       │   ├── ppo_masked.py
│       │   ├── rl_agent.py
│       │   └── policies/
│       │       ├── event_attention_policy.py  ← ArielTransformerPolicy
│       │       └── mlp_scorer.py              ← ArielMlpPolicy
│       ├── scripts/
│       │   ├── train_agent.py   ← CLI training entrypoint
│       │   └── build_dataset.py
│       └── utils/
│           └── config.py
├── tests/                       ← 167 tests
├── outputs/                     ← created by train_agent.py
│   └── <run_name>/
│       ├── model.zip
│       ├── training_log.csv
│       └── plots/
└── plots/                       ← created by run_short_episode.py
    └── short_episode/           ← default output subdirectory
```

## Installation

Create and activate the Conda environment (recommended — handles PyTorch + MPS/CUDA):

```bash
conda env create -f environment.yml
conda activate ariel-rl
pip install -e .
```

Key dependencies:

| Package | Purpose |
|---|---|
| `gymnasium` | Environment interface |
| `stable-baselines3 ≥ 2.3` | PPO implementation |
| `sb3-contrib ≥ 2.3` | `MaskablePPO` (action-masked PPO) |
| `pytorch ≥ 2.3` | Policy networks; auto-uses MPS / CUDA / CPU |
| `pandas`, `numpy`, `scipy` | Data handling |
| `matplotlib` | Diagnostic plots |

## Basic usage

### Run baseline comparison (60-day episode)

```bash
python scripts/run_short_episode.py --days 60
# Plots saved to plots/short_episode/
```

To include a trained RL model in the comparison:

```bash
python scripts/run_short_episode.py \
    --days 60 \
    --model-path outputs/my_run/model.zip \
    --model-name MyTransformer \
    --out-dir plots/my_run_vs_baselines
```

### Train an RL agent

```bash
# Transformer policy (recommended)
python src/ariel_rl/scripts/train_agent.py \
    --policy transformer \
    --timesteps 500000 \
    --n-envs 4 \
    --run-name transformer_v1 \
    --device auto

# MLP sanity-check policy
python src/ariel_rl/scripts/train_agent.py \
    --policy mlp \
    --timesteps 200000 \
    --run-name mlp_baseline
```

Outputs are written to `outputs/<run-name>/`:
- `model.zip` — trained weights
- `training_log.csv` — per-rollout metrics
- `plots/` — training curves + evaluation episode diagnostic plots

### Run tests

```bash
pytest  # 140+ tests
```

## Baselines

Five scheduling heuristics are implemented in `src/ariel_rl/baselines/`.  They share the `BaselineAgent` interface and can be swapped with RL agents in any evaluation call.

| Agent | Strategy |
|---|---|
| `RandomValid` | Uniform random over valid actions |
| `GreedyValue` | Highest catalogue science value first |
| `GreedyBalanced` | Science weight × tier-urgency score |
| `EarliestDeadline` | Soonest window-close first |
| `SmartGreedy` | Science return per unit time cost (slew-aware) |

The gap between `SmartGreedy` (best heuristic) and the true reward function — which accounts for diversity multipliers, coverage milestones, and long-horizon planning — is the space RL is designed to exploit.

## RL agents

Two policies are implemented, both trained with MaskablePPO:

| Policy | Architecture | Use case |
|---|---|---|
| `ArielMlpPolicy` | Flat MLP over flattened obs | Sanity check / fast baseline |
| `ArielTransformerPolicy` | Transformer encoder (Pre-LN, 2 layers, 4 heads) over K event tokens | Primary RL policy |

The transformer treats each of the K candidate events as a token and uses self-attention to reason about relative priorities across all candidates simultaneously.  A CLS token seeded from global mission features serves as the critic input.

## Development roadmap

### ✅ Phase 1–3: Core environment (complete)

* MCS loading, population bins, tier costs.
* Gymnasium environment with action masking, DynamicBackend, mission clock.
* Slew model, idle tracking, 2.5 × T₁₄ observation blocks.
* Multi-component reward: tier completion, progress shaping, coverage potential U_pop, unique-host, comparative planetology, idle penalty, rarity, milestones, terminal bonus.
* Transformer and MLP policies with MaskablePPO.
* Per-run reward config saving for reproducibility.

### ✅ Phase 4: Partial observations + fractional progress (complete)

* Partial-observation model: arriving mid-block gives `capture_fraction = (block_end − t_arrive) / block_dur`.
* `obs_completed` is now a **float** — fractions accumulate toward integer tier thresholds.
* New `capture_fraction` observation feature (index 5 in event vector).
* Action mask updated to use `block_end` (not `window_end`) as miss cutoff — agents now see opportunities even when the raw transit has ended.
* `full_set` action mask is more permissive: omits the budget-fit check, letting the agent decide whether a partial capture is worthwhile.
* 22 new partial-observation tests covering Cases A, B, C, and accumulation.

### ✅ Phase 5: Full-set policy architecture (Set Transformer / ISAB)

* **FullSetISABPolicy** implemented — ISAB × 2 + PMA critic, O(N·m) attention scalable to ~2000 planets.
* **FullSetSelfAttentionPolicy** implemented — full O(N²) attention ablation for direct comparison.
* `N_max` padding: observation space is `(N_max, 28)` with zero-pad rows; action space `Discrete(N_max)`.
* `events_for_target()` added to `DynamicBackend` — single source of truth for eclipse/either targets.
* All three policies wired into `train_agent.py`:
  `--policy transformer` (Top-K), `--policy full_set_isab`, `--policy full_set_attention`.

Three-way comparison:

| Policy | Action space | Attention |
|---|---|---|
| `ArielTransformerPolicy` | Top-K events | O(K²) full |
| `FullSetSelfAttentionPolicy` | All N planets | O(N²) full |
| `FullSetISABPolicy` | All N planets | O(N·m) ISAB |

### 🔄 Phase 6: Policy improvement

* Curriculum training: T1-only short episodes → full 3.5-year mission.
* Offline pre-training from `SmartGreedy` rollouts → fine-tune with PPO.
* Ablation: `efficiency_weight`, `diversity_multiplier_max`, tier bonus ratios.
* Multi-seed evaluation with mean ± std reporting.

### 🔲 Phase 7: Science evaluation

* Compare selected samples against desired population distributions.
* Quantify coverage of planet radius, temperature, mass, density, stellar type, and metallicity.
* Stress-test under catalogue uncertainty, missing values, and ephemeris drift.
* Assess comparative-planetology and multi-planet-system characterisation.

## Notebooks

The notebooks are for exploration and debugging only. Core logic should live in `src/ariel_rl/`.

Recommended notebook roles:

```text
00_mcs_eda.ipynb
    Inspect the MCS, missing values, feature distributions, and cost distributions.

01_feature_bins.ipynb
    Prototype planet/star population bins.

02_reward_prototype.ipynb
    Test reward functions and visualise reward behaviour.

03_env_debug.ipynb
    Step through the environment manually.

04_baselines.ipynb
    Compare random, greedy, and optimisation baselines.

05_rl_training.ipynb
    Train and inspect RL agents.
```

## Important modelling choices

This project should avoid hard cuts during early development. Expensive or incomplete targets should usually remain in the environment, with the agent learning their trade-off through cost and reward.

Missing values should not be silently dropped. They should be handled explicitly using masks, imputation, missingness flags, or feature subsets.

Observation cost should always be tier-dependent.

The environment should report why actions are invalid. Silent invalid actions make debugging RL unnecessarily difficult.

The reward should be tested independently before training agents. A bad reward will produce a bad policy even if the RL algorithm is working correctly.

## Scientific references

Useful background material:

```text
Edwards & Tinetti, "The Ariel Target List: The Impact of TESS and the Potential for Characterising Multiple Planets Within a System", 2022.

ESA Ariel Definition Study Report, ESA/SCI(2020)1, 2020.

Ariel Mission Candidate Sample:
https://github.com/arielmission-space/Mission_Candidate_Sample
```

## Project status

Prototype research code. Interfaces, environment design, and reward definitions are expected to change.
