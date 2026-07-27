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

Core infrastructure and RL agent training are implemented and functional.

| Component | Status |
|---|---|
| MCS data loading + preprocessing | ✅ |
| Gymnasium environment (`ArielEnv`) with action masking | ✅ |
| Five baseline scheduling heuristics | ✅ |
| Evaluation framework + 7 diagnostic plot types | ✅ |
| Multi-component reward (tier, progress, diversity, milestones, terminal) | ✅ |
| RL agents: MLP policy + Transformer policy (MaskablePPO) | ✅ |
| Training CLI with device auto-detection and post-training plots | ✅ |
| Realistic time-dependent scheduling constraints | 🔲 Planned |

The intended development path was:

1. ✅ load and validate the Ariel MCS;
2. ✅ compute tier-dependent observation costs in days;
3. ✅ define target features and science bins;
4. ✅ implement a Gymnasium-style environment;
5. ✅ train simple baselines and RL agents;
6. ✅ compare RL policies against random, greedy, and optimisation baselines;
7. 🔲 add realistic scheduling constraints (transit windows, ephemerides, visibility).

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

Observation cost is stored in days.

For a target `i` and tier `k`:

```text
cost_days(i, k) =
    2.5 × transit_duration_T14_seconds(i) × n_observations(i, k) / 86400
```

The factor of `2.5` approximates the total observing window around a transit or eclipse event.

In the demo notebook, the cost was initially computed using Tier 2 observations only. In the actual environment, cost should be tier-dependent and should be computed from the selected action.

## RL formulation

### State

The first useful state representation should include:

```text
current mission day
remaining mission budget in days
target feature matrix
targets already selected
current science-bin counts
desired or reference science-bin counts
tier-completion state
valid action mask
```

Later versions should include:

```text
next observable event time
target visibility windows
transit/eclipse type
ephemeris uncertainty
revisit requirements
slew or operational overheads if modelled
```

### Action

Initial prototype:

```text
action = target_index
```

Preferred next version:

```text
action = (target_index, tier)
```

For large catalogues, the environment should use action masking so that the agent only chooses from currently valid targets or target-tier combinations.

### Reward

The reward should favour scientific coverage rather than raw observation count.

A first reward model:

```text
reward =
    diversity_gain
  + underrepresented_bin_bonus
  + tier_completion_bonus
  - time_cost_penalty
  - repeat_observation_penalty
  - invalid_action_penalty
```

Possible science bins:

```text
planet radius class
planet temperature class
planet mass or density class
host-star temperature class
host-star metallicity class
orbital period class
```

The central reward idea is chemical consensus: the agent should learn to distribute observations across scientifically meaningful populations rather than repeatedly selecting the cheapest or easiest targets.

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
pytest  # 167 tests
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

### Phase 1: Static target selection

* Load full MCS.
* Keep all targets unless there is a technical reason to exclude a row.
* Compute all observation costs in days.
* Build science bins.
* Implement reward based on diversity and budget use.
* Train simple agents on target-only actions.

### Phase 2: Tier-aware selection

* Extend the action space to target-tier pairs.
* Penalise repeated or inconsistent tier choices.
* Add separate Tier 1, Tier 2, and Tier 3 science objectives.
* Compare policies against greedy tier allocation.

### Phase 3: Time-evolving environment

* Add mission clock.
* Advance time after each observation.
* Track target states over mission time.
* Add event-based observations using transit and eclipse timing.
* Use action masks for currently observable targets.

### Phase 4: Realistic scheduling

* Add visibility windows.
* Add ephemeris constraints.
* Add revisit constraints.
* Add operational gaps or overheads.
* Evaluate whether RL remains useful compared with classical scheduling and optimisation methods.

### Phase 5: Science evaluation

* Compare final selected samples against desired population distributions.
* Analyse coverage of planet radius, temperature, mass, density, stellar type, and metallicity.
* Quantify whether the selected sample supports balanced atmospheric-demographic inference.
* Stress-test policies under catalogue uncertainty and missing values.

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
