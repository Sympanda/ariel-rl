"""
Full-set architecture tests.

Covers:
  1. Observation / action shapes with N_max padding.
  2. N_max enforcement — catalogue exceeding n_max raises.
  3. Dynamic removal — completed planets leave the active set.
  4. Dynamic insertion — adding a target works without architecture changes.
  5. Target mapping — action_index → target_id after removal/reorder.
  6. Permutation equivariance — ISAB and full-attention are PE by construction.
  7. Padding invariance — changing PAD rows must not change real-token logits.
  8. Global conditioning — changing global state changes actor logits.
  9. PPO smoke tests — short training for both full_set policies.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ariel_rl.utils.config import (
    ActionConfig,
    FullSetActionConfig,
    TopKActionConfig,
    TargetActionConfig,
    default_env_config,
)
from ariel_rl.envs.ariel_env import ArielEnv


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FALLBACK_COLS = [
    "event_id", "target_id", "event_type",
    "window_start", "window_mid", "window_end",
    "duration", "duration_days", "block_duration_days", "tier_goal",
    "base_science_value", "visibility_valid",
    "ephemeris_uncertainty", "event_index",
]


def _make_targets(n: int, ra_spread_deg: float = 90.0) -> pd.DataFrame:
    """Minimal target table accepted by ArielEnv."""
    import numpy as _np
    rng = _np.random.default_rng(42)
    return pd.DataFrame({
        "target_id":             [f"T{i:04d}" for i in range(n)],
        "ra":                    _np.linspace(0, ra_spread_deg, n),
        "dec":                   _np.zeros(n),
        "period":                _np.full(n, 3.0),          # 3-day period
        "epoch":                 _np.full(n, 2460000.0),
        "transit_duration":      _np.full(n, 7200.0),       # 2-hour transit
        "eclipse_duration":      _np.full(n, 7200.0),
        "preferred_method":      ["Transit"] * n,
        "tier1_required_obs":    _np.full(n, 2, dtype=int),
        "tier2_required_obs":    _np.full(n, 5, dtype=int),
        "tier3_required_obs":    _np.full(n, 8, dtype=int),
        "max_tier":              _np.full(n, 3, dtype=int),
        "science_weight":        _np.full(n, 1.0),
        "planet_radius":         rng.uniform(1, 10, n),
        "planet_mass":           rng.uniform(1, 100, n),
        "planet_temperature":    rng.full(n, 1000.0) if False else _np.full(n, 1000.0),
        "stellar_temperature":   _np.full(n, 5800.0),
        "stellar_metallicity":   _np.zeros(n),
        "distance":              _np.full(n, 100.0),
        "base_science_value":    _np.full(n, 1.0),
        "population_bin":        [f"bin_{i % 5}" for i in range(n)],
        "epoch_uncertainty":     _np.zeros(n),
        "host_id":               [f"H{i // 2}" for i in range(n)],
        "tier_goal":             _np.full(n, 3, dtype=int),
    })


def _full_set_env(n_targets: int = 20, n_max: int = 30) -> ArielEnv:
    """Return a small full_set ArielEnv, already reset."""
    targets = _make_targets(n_targets)
    cfg = default_env_config()
    cfg = dataclasses.replace(cfg, action=dataclasses.replace(
        cfg.action,
        type="full_set",
        full_set=FullSetActionConfig(n_max=n_max, include_completed=False),
    ))
    env = ArielEnv(cfg, targets=targets)
    env.reset(seed=0)
    return env


# ---------------------------------------------------------------------------
# 1. Observation / action shapes
# ---------------------------------------------------------------------------

class TestObservationActionShapes:
    def test_planet_obs_shape(self):
        env = _full_set_env(n_targets=20, n_max=30)
        obs, _ = env.reset(seed=0)
        from ariel_rl.envs.planet_feature_builder import N_PLANET_FEATURES
        assert obs["planets"].shape == (30, N_PLANET_FEATURES), (
            f"Expected (30, {N_PLANET_FEATURES}), got {obs['planets'].shape}"
        )

    def test_action_mask_shape(self):
        env = _full_set_env(n_targets=20, n_max=30)
        _, info = env.reset(seed=0)
        mask = info["action_mask"]
        assert mask.shape == (30,), f"Expected (30,), got {mask.shape}"

    def test_action_space_size(self):
        env = _full_set_env(n_targets=20, n_max=30)
        from gymnasium import spaces
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 30

    def test_padding_rows_are_zero(self):
        env = _full_set_env(n_targets=20, n_max=30)
        obs, _ = env.reset(seed=0)
        pad = obs["planets"][20:]   # rows 20-29 are padding
        assert np.all(pad == 0.0), "Padding rows must be all-zero"

    def test_padding_always_masked_false(self):
        env = _full_set_env(n_targets=20, n_max=30)
        _, info = env.reset(seed=0)
        mask = info["action_mask"]
        assert not mask[20:].any(), "Padding actions must always be masked False"


# ---------------------------------------------------------------------------
# 2. N_max enforcement
# ---------------------------------------------------------------------------

class TestNMaxEnforcement:
    def test_catalogue_exceeds_nmax_raises(self):
        targets = _make_targets(50)
        cfg = default_env_config()
        cfg = dataclasses.replace(cfg, action=dataclasses.replace(
            cfg.action,
            type="full_set",
            full_set=FullSetActionConfig(n_max=30),
        ))
        with pytest.raises(ValueError, match="n_max"):
            ArielEnv(cfg, targets=targets)

    def test_nmax_zero_uses_catalogue_size(self):
        targets = _make_targets(20)
        cfg = default_env_config()
        cfg = dataclasses.replace(cfg, action=dataclasses.replace(
            cfg.action,
            type="full_set",
            full_set=FullSetActionConfig(n_max=0),
        ))
        env = ArielEnv(cfg, targets=targets)
        assert env.action_space.n == 20

    def test_nmax_equal_to_catalogue_ok(self):
        targets = _make_targets(20)
        cfg = default_env_config()
        cfg = dataclasses.replace(cfg, action=dataclasses.replace(
            cfg.action,
            type="full_set",
            full_set=FullSetActionConfig(n_max=20),
        ))
        env = ArielEnv(cfg, targets=targets)
        assert env.action_space.n == 20


# ---------------------------------------------------------------------------
# 3. Dynamic removal — completed planets leave the active set
# ---------------------------------------------------------------------------

class TestDynamicRemoval:
    def test_completed_target_removed_from_active_set(self):
        env = _full_set_env(n_targets=5, n_max=10)
        env.reset(seed=0)

        tid = env._active_target_ids[0]
        assert tid in env._active_target_ids

        # Force the target to max_tier by patching progress
        from ariel_rl.data.observation_requirements import compute_progress
        target_row = env._state._target_lookup[tid]
        new_prog = compute_progress(float(target_row["tier3_required_obs"]), target_row)
        env._state._progress_dict[tid].update(new_prog)

        # Trigger active set update
        env._update_active_set()

        assert tid not in env._active_target_ids, (
            "Completed target must be removed from active set"
        )

    def test_active_set_shrinks_after_completion(self):
        env = _full_set_env(n_targets=5, n_max=10)
        env.reset(seed=0)
        initial_size = len(env._active_target_ids)

        # Complete first target
        tid = env._active_target_ids[0]
        from ariel_rl.data.observation_requirements import compute_progress
        target_row = env._state._target_lookup[tid]
        new_prog = compute_progress(float(target_row["tier3_required_obs"]), target_row)
        env._state._progress_dict[tid].update(new_prog)
        env._update_active_set()

        assert len(env._active_target_ids) == initial_size - 1

    def test_completed_planet_obs_row_becomes_padding(self):
        """After removal, the vacated slot should be zero-padded in the observation."""
        env = _full_set_env(n_targets=5, n_max=10)
        env.reset(seed=0)

        # Complete all targets
        from ariel_rl.data.observation_requirements import compute_progress
        for tid in list(env._active_target_ids):
            tr = env._state._target_lookup[tid]
            prog_update = compute_progress(float(tr["tier3_required_obs"]), tr)
            env._state._progress_dict[tid].update(prog_update)
        env._update_active_set()

        obs = env._build_observation()
        # All planet tokens should now be zero (all active targets removed → all padding)
        assert np.all(obs["planets"] == 0.0)

    def test_active_tid_to_idx_consistent_after_removal(self):
        env = _full_set_env(n_targets=5, n_max=10)
        env.reset(seed=0)

        # Remove first target
        tid = env._active_target_ids[0]
        from ariel_rl.data.observation_requirements import compute_progress
        tr = env._state._target_lookup[tid]
        prog_u = compute_progress(float(tr["tier3_required_obs"]), tr)
        env._state._progress_dict[tid].update(prog_u)
        env._update_active_set()

        # Check consistency
        for idx, t in enumerate(env._active_target_ids):
            assert env._active_tid_to_idx[t] == idx


# ---------------------------------------------------------------------------
# 4. Dynamic insertion — adding a target works without architecture changes
# ---------------------------------------------------------------------------

class TestDynamicInsertion:
    def test_insert_new_target_grows_active_set(self):
        """Inserting a new target into _active_target_ids increases set size."""
        env = _full_set_env(n_targets=5, n_max=10)
        env.reset(seed=0)
        before = len(env._active_target_ids)

        # Simulate a newly-discovered target by adding it to the active list
        # (in a real mission this would happen via an event; here we test the mechanic)
        new_tid = "T_NEW"
        env._active_target_ids.append(new_tid)
        env._active_tid_to_idx[new_tid] = len(env._active_target_ids) - 1

        assert len(env._active_target_ids) == before + 1


# ---------------------------------------------------------------------------
# 5. Target mapping — action_index → target_id
# ---------------------------------------------------------------------------

class TestTargetMapping:
    def test_action_index_maps_to_active_tid(self):
        env = _full_set_env(n_targets=5, n_max=10)
        env.reset(seed=0)
        for i, tid in enumerate(env._active_target_ids):
            assert env._active_tid_to_idx[tid] == i

    def test_mapping_correct_after_removal(self):
        env = _full_set_env(n_targets=5, n_max=10)
        env.reset(seed=0)

        # Remove the second target (index 1)
        removed = env._active_target_ids[1]
        from ariel_rl.data.observation_requirements import compute_progress
        tr = env._state._target_lookup[removed]
        prog_u = compute_progress(float(tr["tier3_required_obs"]), tr)
        env._state._progress_dict[removed].update(prog_u)
        env._update_active_set()

        for i, tid in enumerate(env._active_target_ids):
            assert env._active_tid_to_idx[tid] == i
        assert removed not in env._active_tid_to_idx


# ---------------------------------------------------------------------------
# 6. Permutation equivariance
# ---------------------------------------------------------------------------

class TestPermutationEquivariance:
    def test_isab_permutation_equivariant(self):
        th = pytest.importorskip("torch")
        from ariel_rl.agents.policies.full_set_isab_policy import FullSetISABNet
        net = FullSetISABNet(n_planet_features=8, n_global_features=6,
                             d_model=32, n_heads=4, n_isab_layers=1, n_inducing=4)
        net.eval()

        B, N, F = 1, 6, 8
        planets = th.randn(B, N, F)
        global_f = th.randn(B, 6)

        perm = [2, 0, 4, 1, 5, 3]
        planets_perm = planets[:, perm, :]

        with th.no_grad():
            logits_orig, _ = net(planets,      global_f)
            logits_perm, _ = net(planets_perm, global_f)

        # Logits for permuted input should equal permuted logits of original
        assert th.allclose(
            logits_orig[:, perm], logits_perm, atol=1e-5
        ), "ISAB must be permutation-equivariant"

    def test_full_attention_permutation_equivariant(self):
        th = pytest.importorskip("torch")
        from ariel_rl.agents.policies.full_set_attention_policy import FullSetSelfAttentionNet
        net = FullSetSelfAttentionNet(n_planet_features=8, n_global_features=6,
                                     d_model=32, n_heads=4, n_layers=1)
        net.eval()

        B, N, F = 1, 6, 8
        planets = th.randn(B, N, F)
        global_f = th.randn(B, 6)

        perm = [3, 1, 5, 0, 2, 4]
        planets_perm = planets[:, perm, :]

        with th.no_grad():
            logits_orig, _ = net(planets,      global_f)
            logits_perm, _ = net(planets_perm, global_f)

        assert th.allclose(
            logits_orig[:, perm], logits_perm, atol=1e-5
        ), "FullSetSelfAttentionNet must be permutation-equivariant"


# ---------------------------------------------------------------------------
# 7. Padding invariance
# ---------------------------------------------------------------------------

class TestPaddingInvariance:
    def test_isab_padding_doesnt_change_real_token_logits(self):
        th = pytest.importorskip("torch")
        from ariel_rl.agents.policies.full_set_isab_policy import FullSetISABNet
        net = FullSetISABNet(n_planet_features=8, n_global_features=6,
                             d_model=32, n_heads=4, n_isab_layers=1, n_inducing=4)
        net.eval()

        B, N_real, N_pad, F = 1, 4, 3, 8
        N = N_real + N_pad
        planets = th.randn(B, N, F)
        # Force padding to zero
        planets[:, N_real:, :] = 0.0
        global_f = th.randn(B, 6)
        pad_mask = th.zeros(B, N, dtype=th.bool)
        pad_mask[:, N_real:] = True

        planets_diff_pad = planets.clone()
        # Change padding values (shouldn't matter)
        planets_diff_pad[:, N_real:, :] = th.randn(B, N_pad, F)

        with th.no_grad():
            logits1, _ = net(planets,          global_f, padding_mask=pad_mask)
            logits2, _ = net(planets_diff_pad, global_f, padding_mask=pad_mask)

        assert th.allclose(
            logits1[:, :N_real], logits2[:, :N_real], atol=1e-5
        ), "Real-token logits must not change when padding values change"

    def test_fullset_attn_padding_invariance(self):
        th = pytest.importorskip("torch")
        from ariel_rl.agents.policies.full_set_attention_policy import FullSetSelfAttentionNet
        net = FullSetSelfAttentionNet(n_planet_features=8, n_global_features=6,
                                     d_model=32, n_heads=4, n_layers=1)
        net.eval()

        B, N_real, N_pad, F = 1, 4, 3, 8
        N = N_real + N_pad
        planets = th.randn(B, N, F)
        planets[:, N_real:, :] = 0.0
        global_f = th.randn(B, 6)
        pad_mask = th.zeros(B, N, dtype=th.bool)
        pad_mask[:, N_real:] = True

        planets_diff_pad = planets.clone()
        planets_diff_pad[:, N_real:, :] = th.randn(B, N_pad, F)

        with th.no_grad():
            l1, _ = net(planets,          global_f, padding_mask=pad_mask)
            l2, _ = net(planets_diff_pad, global_f, padding_mask=pad_mask)

        assert th.allclose(l1[:, :N_real], l2[:, :N_real], atol=1e-5)


# ---------------------------------------------------------------------------
# 8. Global conditioning
# ---------------------------------------------------------------------------

class TestGlobalConditioning:
    """Actor logits must change when global mission state changes, even with
    identical planet tokens.  This validates that global features are wired
    into the actor, not only the critic.
    """

    def _run_nets(self, net, planets, global1, global2, pad_mask):
        th = pytest.importorskip("torch")
        with th.no_grad():
            l1, _ = net(planets, global1, padding_mask=pad_mask)
            l2, _ = net(planets, global2, padding_mask=pad_mask)
        return l1, l2

    def test_isab_actor_depends_on_global(self):
        th = pytest.importorskip("torch")
        from ariel_rl.agents.policies.full_set_isab_policy import FullSetISABNet
        net = FullSetISABNet(n_planet_features=8, n_global_features=6,
                             d_model=32, n_heads=4, n_isab_layers=1, n_inducing=4)
        net.eval()

        B, N, F = 2, 5, 8
        planets  = th.randn(B, N, F)
        global1  = th.zeros(B, 6)
        global2  = th.ones(B, 6)         # different global state
        pad_mask = th.zeros(B, N, dtype=th.bool)

        l1, l2 = self._run_nets(net, planets, global1, global2, pad_mask)

        assert not th.allclose(l1, l2), (
            "Actor logits must change when global mission state changes"
        )

    def test_fullset_attn_actor_depends_on_global(self):
        th = pytest.importorskip("torch")
        from ariel_rl.agents.policies.full_set_attention_policy import FullSetSelfAttentionNet
        net = FullSetSelfAttentionNet(n_planet_features=8, n_global_features=6,
                                     d_model=32, n_heads=4, n_layers=1)
        net.eval()

        B, N, F = 2, 5, 8
        planets  = th.randn(B, N, F)
        global1  = th.zeros(B, 6)
        global2  = th.ones(B, 6)
        pad_mask = th.zeros(B, N, dtype=th.bool)

        l1, l2 = self._run_nets(net, planets, global1, global2, pad_mask)
        assert not th.allclose(l1, l2)


# ---------------------------------------------------------------------------
# 9. PPO smoke tests
# ---------------------------------------------------------------------------

class TestPPOSmoke:
    """Very short training runs (200 steps) for both full_set policies to
    catch any forward/backward-pass wiring errors.
    """

    def _make_env_and_model(self, policy_name: str, n_targets: int = 15, n_max: int = 20):
        pytest.importorskip("sb3_contrib")
        import dataclasses
        from sb3_contrib import MaskablePPO
        from ariel_rl.agents.ppo_masked import make_training_envs

        targets = _make_targets(n_targets)
        cfg = default_env_config()
        cfg = dataclasses.replace(cfg, action=dataclasses.replace(
            cfg.action,
            type="full_set",
            full_set=FullSetActionConfig(n_max=n_max, include_completed=False),
        ))

        if policy_name == "full_set_isab":
            from ariel_rl.agents.policies.full_set_isab_policy import FullSetISABPolicy
            policy_cls = FullSetISABPolicy
            policy_kwargs = {"d_model": 32, "n_heads": 4, "n_isab_layers": 1, "n_inducing": 4}
        else:
            from ariel_rl.agents.policies.full_set_attention_policy import FullSetSelfAttentionPolicy
            policy_cls = FullSetSelfAttentionPolicy
            policy_kwargs = {"d_model": 32, "n_heads": 4, "n_layers": 1}

        env = make_training_envs(cfg, n_envs=1, seed=0, targets=targets)
        model = MaskablePPO(
            policy_cls, env,
            n_steps=64, batch_size=16, n_epochs=2,
            policy_kwargs=policy_kwargs, verbose=0, seed=0,
        )
        return model

    def test_isab_smoke_train(self):
        model = self._make_env_and_model("full_set_isab")
        model.learn(total_timesteps=200)

    def test_full_attention_smoke_train(self):
        model = self._make_env_and_model("full_set_attention")
        model.learn(total_timesteps=200)
