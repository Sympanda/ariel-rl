# Ariel RL — Policy Design & Architecture Ideas

This document collects ideas for how to frame the reinforcement learning problem for Ariel mission scheduling.  The goal is to keep a running record of design trade-offs so decisions are documented before committing to an implementation.

---

## The Core Problem

At each step the telescope must pick **which target to observe next** from a set of currently-reachable candidates.  The decision depends on:

- **Local event info** — slew cost, transit window urgency, tier progress of this specific target
- **Global mission state** — time remaining, how many T1/T2/T3 completions so far, population bin coverage
- **Future opportunity** — what other windows are coming up, which targets will be permanently lost if not observed soon

The hard part is that decisions interact over time: choosing target A now might crowd out target B's only remaining window.  Greedy heuristics can't reason about this; RL can in principle.

---

## Idea 1 — Attention / Pointer Network (set-to-action)

**What it is:** Each of the *k* candidate events is a token.  A transformer encoder runs self-attention across tokens so candidates can see each other.  A CLS token seeded from the global mission state serves as the critic input and as a query over the event tokens.  The action is the argmax (or sample) of the resulting logit vector.

```
global (G=26,) ──► linear ──► CLS token (d_model)
                                    │
events (K×17)  ──► linear ──► token embeddings (K×d_model)
               ──► [CLS | e_1 | … | e_K]
               ──► Transformer encoder (Pre-LN, n_layers, n_heads)
                        ┌──────────────┴──────────────┐
                   tokens[1:]                     tokens[0]  (CLS)
                policy_head(K,)               value_head(1,)
             (per-token logits)              (scalar value)
               ──► mask invalid ──► softmax ──► π(a|s)
```

**Why it fits this problem:**
- The action space is a **set**, not a sequence.  Self-attention is permutation-equivariant, which is the right inductive bias.
- The existing observation dict (`events: (k, 16)`, `global: (64,)`) maps directly onto the architecture — no env changes needed.
- Action masking (already implemented) integrates as `-inf` logits before softmax.
- Directly inspired by the routing / scheduling literature (Kool et al., 2019 *Attention, Learn to Solve Routing Problems*; Nazari et al., 2018 for VRP).

**Trade-offs:**
- More parameters than an MLP → slower training, but still small (~500k–1M params for 2 layers, 128 dim).
- Harder to debug than a flat MLP at first.
- Requires writing a custom `ActorCriticPolicy` for SB3.

**Recommended algorithm:** PPO (handles masked discrete actions natively, well understood).

---

## Idea 2 — Plain MLP Baseline (flatten events)

**What it is:** Flatten the `events (K×17)` array → concatenate with `global (26,)` → standard MLP → logits.  Standard SB3 `MlpPolicy` with a custom feature extractor.

**Why it's worth doing first:**
- Takes an afternoon to implement.
- Acts as a sanity-check: if the MLP can't beat greedy baselines, the reward signal or obs space has a problem that needs fixing before adding architectural complexity.
- Once the training loop is stable, swapping the feature extractor to the attention version is straightforward.

**Limitation:** The MLP treats the 50×16 input as a flat vector — it loses the "each event is a structured item" inductive bias.  It might still learn something useful but will be sample-inefficient.

---

## Idea 3 — Graph Neural Network (targets as nodes, slews as edges)

**What it is:** Build a graph where nodes are candidate targets and edges carry slew-cost features.  A GNN propagates information across the graph before each decision.

```
target nodes (features per target)
    + edge features (slew_days between every pair)
    ──► GNN message passing (2–3 rounds)
    ──► node embeddings
    ──► global pooling ──► mission context
    ──► node scores ──► π(a|s)
```

**Why it could be powerful:**
- The slew structure IS a graph.  A target in the same sky region as the current pointing should look "cheap" — a GNN can learn this explicitly.
- Could capture "clusters" of targets that are spatially co-located and therefore cheap to visit together.

**Trade-offs:**
- Requires building a dynamic graph each step (the k=50 candidates change every step).
- Adds a dependency on PyG or DGL.
- Harder to implement and tune.
- The attention architecture (Idea 1) achieves much of this via self-attention on the event features (which already include slew cost), without needing an explicit graph.

**Verdict:** Interesting direction if the attention model plateaus, but not the first thing to try.

---

## Idea 4 — Hierarchical RL (region → target)

**What it is:** Two policies operating at different timescales:
- **High-level policy** — picks a *sky region* or *priority class* (e.g., "focus on T3-capable targets in the hot-Jupiter bin") on some multi-step horizon.
- **Low-level policy** — given the region/class constraint, picks the specific next observation greedily or with a small MLP.

**Why it could fit:**
- The problem has a natural hierarchy: macro decisions (which population to focus on for the next few days) vs micro decisions (which specific transit to take right now).
- Might be more interpretable — the high-level policy directly expresses scientific priorities.
- Could be easier to reward: the high-level policy gets a shaped reward based on weekly coverage statistics.

**Trade-offs:**
- Significantly more complex to implement (options framework, feudal networks, or HIRO).
- The low-level policy needs to be stable before the high-level one can learn.
- May be overkill for a k=50 flat action space.

**Verdict:** Worth thinking about if the flat policy struggles to learn good macro-level diversity behaviour.

---

## Idea 5 — Offline / Imitation Pre-training then Fine-tuning

**What it is:** 
1. **Behavioural cloning** — run the best baselines (SmartGreedy, EarliestDeadline) for thousands of episodes, collect (state, action) pairs, and train the RL policy to imitate them via supervised learning.
2. **Fine-tune with PPO** — use the pre-trained policy as a warm start, then run online RL to surpass the baselines.

**Why it helps:**
- Cold-start is the biggest problem in combinatorial RL.  A random policy wastes millions of steps learning that valid actions exist.
- Imitation gives the policy a strong prior (behave like SmartGreedy) before it has to explore.
- Well-established approach: AlphaStar used this, as does most modern decision transformer work.

**Trade-offs:**
- Need to store large offline datasets.
- Risk of the policy getting "stuck" in the imitation local optimum if the fine-tuning reward is not well shaped.
- Adds a two-phase training pipeline.

**Verdict:** High-value technique, especially because we have strong baselines.  Should be explored after the basic PPO loop is working.

---

## Idea 6 — Decision Transformer (sequence model)

**What it is:** Frame the entire mission as a sequence: `(return-to-go, state, action, return-to-go, state, action, ...)`.  Train a GPT-style causal transformer to predict the next action given the desired return-to-go and history.  At inference, set a high return-to-go target and let it autoregressively produce a schedule.

**Why it's interesting:**
- No RL training instability — it's a supervised sequence prediction problem.
- Can in principle reason about the entire mission history.
- Chen et al., 2021 (*Decision Transformer*) showed this works surprisingly well on offline RL benchmarks.

**Trade-offs:**
- Requires a large offline dataset of high-quality episodes to be useful.
- Inference is sequential (one token at a time) which could be slow for a long mission.
- Doesn't naturally handle the dynamic, stochastic nature of the observation window (transits are deterministic but the set of valid actions changes each step based on timing).
- Better suited for settings where you have many expert demonstrations already.

**Verdict:** Speculative but exciting.  Worth considering if we build a large corpus of baseline + RL rollouts.

---

## Idea 7 — Curriculum Learning

**What it is:** Rather than training on the full 814-target, 4-year mission from the start, gradually increase difficulty:

| Stage | Targets | Mission length | Max tier |
|-------|---------|----------------|----------|
| 1 | 50 targets | 30 days | T1 only |
| 2 | 200 targets | 60 days | T1 + T2 |
| 3 | 500 targets | 6 months | T1–T3 |
| 4 | Full 814 | 4 years | Full |

**Why it helps:**
- The reward signal is very sparse for T3 completions on a 4-year timeline.  A policy that never completes a T3 in early training receives no learning signal for it.
- Smaller problems are faster to simulate → faster iteration on reward/arch design.
- Well supported by the `max_tier_cap` and `lifetime_days` configs already in the env — no code changes needed.

**Verdict:** Almost certainly worth doing.  The env already has the config levers (`MissionConfig.lifetime_days`, `MissionConfig.max_tier_cap`).

---

## Recommended Implementation Order

```
1. ── Verify env + reward are well-shaped  ────────────────────  ✅ Done (baselines pass)
2. ── MLP policy with PPO (SB3, flat obs)  ────────────────────  ✅ Done — worse than all baselines
       └─ Established training loop, SB3 CSV logger, post-training plots
       └─ Confirmed it does NOT beat RandomValid — reward dominated by dense shaping,
          not sparse tier completions; validates that architecture matters
3. ── Attention / Transformer policy (Idea 1)  ─────────────────  ✅ Done (transformer_v1/v3, 3M steps)
       └─ Beats all baselines on science efficiency (+10 % over best greedy on 1Y run)
       └─ Matches T1/T2/T3 completion counts with top baselines
       └─ Highest T1 and T3 completion on 1Y run
       └─ Coverage gap: 0.88 vs 1.0 for 3 baselines — fixed in reward redesign
4. ── Reward redesign  ─────────────────────────────────────────  ✅ Done
       └─ Coverage potential U_pop, unique-host, comparative-planetology bonuses
       └─ Science weight floor (prevents zero-weight bins)
       └─ Diversity multiplier max 5× (was 2×)
       └─ Idle-time penalty; 2.5×T14 block duration enforced consistently
       └─ Reward config saved per run for reproducibility
5. ── Scheduling dynamics overhaul  ────────────────────────────  ✅ Done
       └─ Slew immediately, then idle, then observe
       └─ DynamicBackend as default (no pre-computed event table)
       └─ block_duration_days in event schema and observation features
       └─ used_idle_fraction in global obs
6. ── Curriculum: T1-only → full tiers  ────────────────────────  Next step
7. ── Offline pre-training from baselines  ─────────────────────  After curriculum settled
8. ── Full-set action space (all N targets)  ───────────────────  Partially implemented (full_set mode)
9. ── Hierarchical or GNN extensions  ──────────────────────────  Research direction
```

---

## Empirical Results — transformer_v1 (June 2026)

Trained for **3M timesteps**, MaskablePPO, d_model=128, n_heads=4, n_layers=3.
Full 3.5-year mission, `topk` K=50 action space, `DynamicBackend`.

### Training dynamics

| Signal | Observation | Interpretation |
|---|---|---|
| Episode reward | Slow climb from ~2k, wave-like, plateaus near end | Expected PPO behaviour on sparse+dense mixed reward |
| Episode length | Similar wave pattern to reward | Policy learning to use more of the budget efficiently |
| Entropy | Slow decay throughout | Policy consolidating — commits to efficiency strategy before fully exploring coverage |
| KL divergence | Near-flat with one early spike (self-corrected) | Stable training; spike = moment transformer first cracked chaining observations on a target |
| Value loss | Near-linear drop | Critic steadily improving; healthy sign |
| Policy loss | Small, noisy | Consistent with well-clipped PPO on a long-horizon problem |
| Starting reward | ~2k immediately | Same as weakest baseline — dense shaping (efficiency + progress) is always available; sparse tier bonuses take longer to discover |

### Evaluation results (1-year episode)

| Metric | Transformer | Best greedy |
|---|---|---|
| Science efficiency | **~10% higher than all baselines** | SmartGreedy (2nd) |
| T1 completion | **Highest** | GreedyBalanced |
| T3 completion | **Highest** | SmartGreedy |
| Bin coverage | 0.88 | 1.0 (3 baselines perfect) |
| Finishes early? | Yes — stops before budget exhausted | No |

The transformer finishes early on the 3.5Y run too — it exhausts its preferred (high-efficiency, nearby) targets before running out of time budget, then stops rather than picking up the remaining rare-bin targets it hasn't prioritised.

---

## Known Issue — Coverage Gap

**Root cause:** The transformer latched onto the dense efficiency reward and learned to cluster observations spatially. This is locally optimal for `efficiency_weight = 0.5` but misses the ~12% of population bins that are rare and spatially scattered.

The three baselines that achieved perfect coverage all have explicit diversity in their scoring formula (via `science_weight`). The transformer's diversity multiplier (max 2×) is not strong enough to overcome the ~10% efficiency gain from staying in a familiar sky region once entropy has decayed.

**Contributing factor:** T3 bonuses (10×) are much larger than T1 bonuses (1×). This incentivises going deep on a subset of targets rather than broad T1 coverage of the full catalogue.

### Strategies to fix coverage

**1. Per-bin first-contact bonus** *(requires new RewardConfig field + env logic)*
Fire a bonus the first time any target in a previously-unseen population bin reaches T1. Directly rewards visiting new parts of parameter space.
```yaml
per_bin_first_t1_bonus: 5.0   # fires once per bin per episode
```

**2. Stronger diversity multiplier**
Current multiplier range is `[1.0, 2.0]`. Increasing to `[1.0, 5.0]` makes rare/unseen bins 5× more attractive. Simple one-line change in `compute_reward.py`.

**3. Rebalance tier ratios toward breadth**
Current: T1=1, T2=3, T3=10. Flattening to T1=2, T2=4, T3=6 makes many T1 completions roughly equivalent in value to a few T3s. Retrain required.
```yaml
reward:
  tier1_completion: 2.0
  tier2_completion: 4.0
  tier3_completion: 6.0
```

**4. Curriculum: T1-only first**
Train with `--max-tier-cap 1` for 1–2M steps so the only path to reward is broad T1 coverage. Then fine-tune with full tiers. The env config lever already exists.

---

## Dynamic Target List — Transformer vs MLP vs Greedy

The Ariel target list will be revised as TESS and ground-based surveys discover new candidates. This has significant practical implications for policy architecture:

| | New targets added mid-mission | Feature count changes |
|---|---|---|
| **Greedy baselines** | ✅ Trivial — targets just appear in the candidate set at the next step | ✅ Trivial — score formula is hand-coded |
| **Transformer** | ✅ Easy — new event tokens are extra rows in the `(K×16)` input. The attention mechanism is permutation-equivariant and doesn't care about set size. Zero-padding already handles variable counts. Policy generalises immediately. | ⚠️ Retrain needed if feature count (16 or 25) changes, but not if only N_targets changes |
| **MLP** | ❌ Hard — input is `K×16` *flattened*, so if K or the feature count changes the input dimension changes and the model must be retrained from scratch | ❌ Same problem |

This is one of the strongest practical arguments for the transformer over MLP for this specific problem. A production Ariel scheduler will need to absorb catalogue updates without full retraining, and the transformer's set-input architecture supports this directly.

---

## Open Questions

- **k size**: How many candidates should the agent see? k=50 is a reasonable default but larger k gives more context at the cost of a larger token sequence.
- **Coverage vs efficiency trade-off**: Is there an optimal `efficiency_weight` that preserves the efficiency gains while still driving full coverage? Needs ablation.
- **Multi-seed evaluation**: RL results are noisy. Any comparison should run ≥5 seeds and report mean ± std.
- **What counts as "better than greedy"**: Proposed benchmark: T1 completion rate + bin coverage Gini index + science efficiency, all reported together.
- **Fine-tuning from transformer_v1**: Can we warm-start a coverage-focused run from the existing model rather than retraining from scratch?

---

*Last updated: July 2026*
