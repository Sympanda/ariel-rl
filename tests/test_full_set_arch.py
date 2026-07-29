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
# 4. Dynamic insertion — current limitation documented (Option B)
# ---------------------------------------------------------------------------

class TestDynamicInsertion:
    """Runtime insertion of genuinely new targets is *not* supported in the
    current implementation.  Completed-target removal from a fixed initial
    catalogue IS supported.  Runtime discovery is deferred as future work.

    This test class documents the boundary: naively appending to
    _active_target_ids does NOT make a target usable — it has no entry in
    _target_lookup, no progress, and no backend ephemeris.
    """

    def test_runtime_insertion_is_not_supported(self):
        """Naively appending a ghost target to _active_target_ids is unsafe.

        A genuinely new target needs to be added to the target catalogue,
        mission state, backend ephemeris, static feature cache, and active
        mapping simultaneously.  Without that, the environment raises an error
        or silently produces incorrect observations.

        This test documents the limitation: runtime discovery is deferred as
        future work.  The current dynamic set supports only *removal* of
        targets from the fixed initial catalogue.
        """
        env = _full_set_env(n_targets=5, n_max=10)
        env.reset(seed=0)

        ghost_tid = "T_GHOST"
        env._active_target_ids.append(ghost_tid)
        env._active_tid_to_idx[ghost_tid] = len(env._active_target_ids) - 1

        # Expect either a shape error (from static-cache mismatch) or incorrect
        # behaviour — either way, unsupported.
        with pytest.raises(Exception):
            env._build_observation()  # must fail — ghost not in catalogue or cache


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


# ---------------------------------------------------------------------------
# 10. Mission-end feasibility masking
# ---------------------------------------------------------------------------

class TestMissionEndFeasibility:
    """Regression tests for the can_fit check in full_set mode.

    Before fix: full_set used a permissive mask that skipped can_fit, so a
    target whose idle wait would push past mission_end could still be selected.
    After fix: can_fit is always checked using the tier-capped captured duration.
    """

    def _env_near_end(self, days_remaining: float, n_targets: int = 5) -> "ArielEnv":
        """Return a full_set env with the clock manually wound close to mission_end."""
        env = _full_set_env(n_targets=n_targets, n_max=20)
        env.reset(seed=0)
        # Advance the clock so only `days_remaining` are left
        mission_end = env._state.clock.mission_end
        env._state.clock._current_time = mission_end - days_remaining
        return env

    def test_target_beyond_mission_end_is_masked(self):
        """Target whose first event starts 5 days away, but mission ends tomorrow."""
        env = _full_set_env(n_targets=5, n_max=20)
        env.reset(seed=0)

        # Build a candidate table where the first event for target 0 is 5 days away
        # and the mission ends in 1 day.
        from ariel_rl.envs.action_mask import _mask_target
        import pandas as pd

        t_now = env._state.clock.current_time
        mission_end = env._state.clock.mission_end
        # Wind clock so only 1 day is left
        env._state.clock.current_time = mission_end - 1.0

        # Craft a candidate whose event is 5 days away (block_mid = t_now + 5)
        # so slew=0, idle=5 days, captured_dur≈0.1 days → total ≈ 5.1 > 1 remaining
        tid = env._active_target_ids[0]
        target = env._state._target_lookup[tid]
        t_now2 = env._state.clock.current_time
        far_mid = t_now2 + 5.0
        block_dur = 0.1
        row = {
            "event_id": 999, "target_id": tid, "event_type": "transit",
            "window_start": far_mid - block_dur / 2.0,
            "window_mid": far_mid,
            "window_end": far_mid + block_dur / 2.0,
            "duration": block_dur * 86400, "duration_days": block_dur,
            "block_duration_days": block_dur, "tier_goal": 1,
            "base_science_value": 1.0, "visibility_valid": True,
            "ephemeris_uncertainty": 0.0, "event_index": -1,
        }
        candidates = pd.DataFrame([row])

        from ariel_rl.utils.config import default_env_config
        import dataclasses
        cfg = default_env_config()
        cfg = dataclasses.replace(cfg, action=dataclasses.replace(
            cfg.action, type="full_set",
            full_set=FullSetActionConfig(n_max=20),
        ))
        mask = _mask_target(
            env._state, candidates,
            include_completed=False, permissive=False,
        )
        assert not mask[0], (
            "Target with event 5 days away must be masked when only 1 day remains"
        )

    def test_long_idle_still_feasible_if_fits_mission(self):
        """Target that requires 4 days idle but still completes before mission_end."""
        env = _full_set_env(n_targets=5, n_max=20)
        env.reset(seed=0)

        from ariel_rl.envs.action_mask import _mask_target
        import pandas as pd, dataclasses

        mission_end = env._state.clock.mission_end
        # Wind clock so 5 days remain
        env._state.clock.current_time = mission_end - 5.0
        tid = env._active_target_ids[0]

        t_now2 = env._state.clock.current_time
        # Event starts in 4 days, observation takes 0.1 day → total ≈ 4.1 days < 5 days
        mid = t_now2 + 4.0
        block_dur = 0.1
        row = {
            "event_id": 998, "target_id": tid, "event_type": "transit",
            "window_start": mid - block_dur / 2.0, "window_mid": mid,
            "window_end": mid + block_dur / 2.0,
            "duration": block_dur * 86400, "duration_days": block_dur,
            "block_duration_days": block_dur, "tier_goal": 1,
            "base_science_value": 1.0, "visibility_valid": True,
            "ephemeris_uncertainty": 0.0, "event_index": -1,
        }
        candidates = pd.DataFrame([row])

        cfg = default_env_config()
        cfg = dataclasses.replace(cfg, action=dataclasses.replace(
            cfg.action, type="full_set",
            full_set=FullSetActionConfig(n_max=20),
        ))
        mask = _mask_target(
            env._state, candidates,
            include_completed=False, permissive=False,
        )
        assert mask[0], (
            "Target requiring 4-day idle with 5 days remaining must remain selectable"
        )


# ---------------------------------------------------------------------------
# 11. Feature alignment: planet token event matches the executed action event
# ---------------------------------------------------------------------------

class TestFeatureAlignment:
    """Validate that per-planet features describe the same event that would
    actually execute if the agent selects that planet.

    event_1 (the action event) is the first REACHABLE event, not the first
    chronological one.  The future-event sequence (event_2, event_3) must
    come AFTER event_1, not before.
    """

    def test_dt_next_event_matches_candidate_event(self):
        """dt_next_event_norm should correspond to the event in the candidates table."""
        env = _full_set_env(n_targets=5, n_max=10)
        env.reset(seed=0)

        obs, _ = env.reset(seed=0)
        candidates = env._candidates

        from ariel_rl.envs.planet_feature_builder import (
            N_PLANET_FEATURES, PLANET_FEATURE_NAMES
        )
        # dt_next_event_norm is a specific feature; find its index
        try:
            dt_idx = PLANET_FEATURE_NAMES.index("dt_next_event_norm")
        except ValueError:
            pytest.skip("dt_next_event_norm not in PLANET_FEATURE_NAMES")

        t_now = env._state.clock.current_time
        n_active = len(env._active_target_ids)

        for i in range(min(n_active, 5)):
            if i >= len(candidates):
                break
            cand_mid = float(candidates.iloc[i]["window_mid"])
            if cand_mid <= 0:
                continue  # sentinel row

            expected_dt = max(0.0, cand_mid - t_now)
            # Get normalisation denominator (365.25 days by default)
            norm_val = 365.25
            feature_dt = obs["planets"][i, dt_idx] * norm_val

            # Allow 1% tolerance (normalisation clipping may reduce large values)
            expected_clipped = min(expected_dt, 365.25)
            assert abs(feature_dt - expected_clipped) < 0.01 * max(expected_clipped, 0.1) + 0.01, (
                f"Planet {i}: dt_next_event feature {feature_dt:.4f} does not match "
                f"candidate mid {expected_clipped:.4f} (t_now={t_now:.2f}, cand_mid={cand_mid:.2f})"
            )

    def test_future_events_ordered_after_action_event(self):
        """event_2 and event_3 must occur strictly after event_1."""
        env = _full_set_env(n_targets=5, n_max=10)
        env.reset(seed=0)

        from ariel_rl.envs.planet_feature_builder import PLANET_FEATURE_NAMES

        try:
            dt1_idx = PLANET_FEATURE_NAMES.index("dt_next_event_norm")
            dt2_idx = PLANET_FEATURE_NAMES.index("dt_second_event_norm")
            dt3_idx = PLANET_FEATURE_NAMES.index("dt_third_event_norm")
        except ValueError:
            pytest.skip("Multi-event dt features not in PLANET_FEATURE_NAMES")

        obs, _ = env.reset(seed=0)
        n_active = len(env._active_target_ids)
        norm = 365.25

        for i in range(min(n_active, 5)):
            dt1 = obs["planets"][i, dt1_idx] * norm
            dt2 = obs["planets"][i, dt2_idx] * norm
            dt3 = obs["planets"][i, dt3_idx] * norm
            assert dt2 >= dt1 - 0.001, (
                f"Planet {i}: event_2 dt={dt2:.3f} must be >= event_1 dt={dt1:.3f}"
            )
            assert dt3 >= dt2 - 0.001, (
                f"Planet {i}: event_3 dt={dt3:.3f} must be >= event_2 dt={dt2:.3f}"
            )
