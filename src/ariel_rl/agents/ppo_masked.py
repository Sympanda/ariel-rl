"""
Environment factory helpers for MaskablePPO training.

``MaskablePPO`` from sb3-contrib expects each environment to expose an
``action_masks()`` callable.  The ``ActionMasker`` wrapper provides this by
calling a user-supplied function against the unwrapped env at every step.

Typical usage
-------------
    from sb3_contrib import MaskablePPO
    from ariel_rl.agents.ppo_masked import make_training_envs
    from ariel_rl.agents.policies.event_attention_policy import ArielTransformerPolicy
    from ariel_rl.utils.config import load_env_config

    cfg = load_env_config("configs/env/simple.yaml")
    env = make_training_envs(cfg, n_envs=4, seed=42)

    model = MaskablePPO(
        ArielTransformerPolicy,
        env,
        policy_kwargs={"d_model": 128, "n_heads": 4, "n_layers": 2},
        verbose=1,
        tensorboard_log="runs/",
    )
    model.learn(total_timesteps=1_000_000)
    model.save("outputs/my_run/final_model")

Single-env usage (evaluation / debugging)
------------------------------------------
    from ariel_rl.agents.ppo_masked import make_masked_env

    env = make_masked_env(cfg)
    obs, info = env.reset()
    action_masks = env.action_masks()   # added by ActionMasker
    action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ariel_rl.envs import ArielEnv
from ariel_rl.utils.config import EnvConfig


def _get_action_mask(env: ArielEnv) -> np.ndarray:
    """
    Mask function supplied to ``ActionMasker``.

    Returns the current action validity array.  Falls back to all-valid
    before ``reset()`` is first called (ActionMasker may query this during
    VecEnv construction before any episode begins).
    """
    mask = env.action_mask
    if mask is None:
        return np.ones(env.n_actions, dtype=bool)
    return mask


def make_masked_env(
    config: EnvConfig,
    seed: int = 0,
    targets=None,
    events=None,
) -> ActionMasker:
    """
    Create a single ActionMasker-wrapped ArielEnv ready for MaskablePPO.

    Parameters
    ----------
    config : EnvConfig
        Environment configuration.
    seed : int
        Passed to env construction (not to reset — call ``env.reset(seed=seed)``
        separately for reproducible episodes).
    targets : pd.DataFrame, optional
        Pre-built target table (skip CSV loading if provided).
    events : pd.DataFrame, optional
        Pre-built event table (skip event generation if provided).

    Returns
    -------
    ActionMasker wrapping an ArielEnv.
    """
    kwargs = {"config": config}
    if targets is not None:
        kwargs["targets"] = targets
    if events is not None:
        kwargs["events"] = events

    env = ArielEnv(**kwargs)
    # ActionMasker must wrap ArielEnv directly so _get_action_mask receives an
    # ArielEnv instance (which has .action_mask).  Monitor goes on top: SB3's
    # DummyVecEnv.get_wrapper_attr() traverses the stack inward to find
    # ActionMasker.action_masks(), and Monitor still intercepts step() / reset()
    # to log ep_rew_mean / ep_len_mean.
    env = ActionMasker(env, _get_action_mask)
    return Monitor(env)


def make_training_envs(
    config: EnvConfig,
    n_envs: int = 1,
    seed: int = 0,
    targets=None,
    events=None,
) -> DummyVecEnv:
    """
    Create a ``DummyVecEnv`` of ``n_envs`` ActionMasker-wrapped ArielEnvs.

    Each environment gets a unique seed offset (``seed + i``) so episodes
    are independently randomised.

    Pre-built ``targets`` and ``events`` tables are shared across all envs
    (they are read-only; each env has its own ``MissionState`` copy).

    Parameters
    ----------
    config : EnvConfig
    n_envs : int
        Number of parallel environments.
    seed : int
        Base seed; env i uses ``seed + i``.
    targets, events : pd.DataFrame, optional
        Pre-built tables.  Pass these to avoid regenerating events N times.

    Returns
    -------
    DummyVecEnv suitable as the ``env`` argument to MaskablePPO.
    """
    def _make_fn(i: int) -> Callable:
        def _fn() -> ActionMasker:
            return make_masked_env(config, seed=seed + i, targets=targets, events=events)
        return _fn

    return DummyVecEnv([_make_fn(i) for i in range(n_envs)])
