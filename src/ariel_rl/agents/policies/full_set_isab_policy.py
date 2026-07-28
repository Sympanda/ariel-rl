"""
FullSetISABPolicy — Set Transformer actor-critic for the full-set action space.

Architecture
------------
The observation is a Dict with two arrays:

    "planets"  (N_max, n_pf)  — one row per target (padded to N_max)
    "global"   (n_gf,)        — mission-level state vector

Flow:

    planets (N_max, n_pf)  ──► Linear ──► planet tokens  (N_max, d_model)
                                                │
                                          ISAB × n_isab_layers    O(N·m)
                                                │
                              ┌─────────────────┴──────────────────┐
                              │                                     │
                         policy_head                           PMA (k=1)
                        (per-token linear)              ──► global summary (d_model)
                              │                              + global features
                         logits (N_max,)                        │
                         + action mask                       value_head
                              │                                  │
                        π(a|s)                              V(s) scalar

Distinct from ``ArielTransformerPolicy`` (Top-K full self-attention):
- Input tokens are planet features, not event features.
- ISAB replaces full self-attention → O(N·m) complexity instead of O(N²).
- Critic uses PMA (permutation-invariant set pooling) instead of [CLS] token.
- Designed for N_max ≈ 2000 planets without quadratic memory blowup.

Usage
-----
    from sb3_contrib import MaskablePPO
    from ariel_rl.agents.policies.full_set_isab_policy import FullSetISABPolicy

    model = MaskablePPO(
        FullSetISABPolicy,
        env,   # must use action_type="full_set"
        policy_kwargs={
            "d_model":     128,
            "n_heads":     4,
            "n_isab_layers": 2,
            "n_inducing":  32,
        },
    )

Note: keep ``ArielTransformerPolicy`` for the Top-K baseline.  Do not merge
or replace — they serve different action spaces and answer different questions.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from ariel_rl.agents.policies.isab_modules import ISAB, PMA


# ---------------------------------------------------------------------------
# Core network module
# ---------------------------------------------------------------------------

class FullSetISABNet(nn.Module):
    """
    ISAB-based actor-critic network for the full-set action space.

    Parameters
    ----------
    n_planet_features : int
        Number of features per planet token (N_PLANET_FEATURES).
    n_global_features : int
        Dimension of the global state vector.
    d_model : int
        Internal hidden dimension for all attention layers.
    n_heads : int
        Number of attention heads (must divide d_model).
    n_isab_layers : int
        Number of stacked ISAB blocks (2 recommended).
    n_inducing : int
        Number of inducing points per ISAB layer (32–64 typical).
    dropout : float
        Currently unused (attention layers don't apply dropout in on-policy RL).
    """

    def __init__(
        self,
        n_planet_features: int,
        n_global_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_isab_layers: int = 2,
        n_inducing: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Input projection: planet features → d_model
        self.planet_proj = nn.Linear(n_planet_features, d_model)
        # No positional encoding — planet tokens are permutation-invariant.

        # ISAB stack
        self.isab_layers = nn.ModuleList([
            ISAB(d_model, n_heads, n_inducing)
            for _ in range(n_isab_layers)
        ])

        # Actor head: per-token score → one logit per planet
        self.policy_head = nn.Linear(d_model, 1)

        # Critic: PMA reduces the set to a single summary vector,
        # concatenated with global features before the value MLP.
        self.pma = PMA(d_model, n_heads, k=1)
        self.global_proj = nn.Linear(n_global_features, d_model)
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_mlp[-1].weight, gain=1.0)

    def forward(
        self,
        planets: th.Tensor,                         # (B, N_max, n_pf)
        global_feat: th.Tensor,                      # (B, n_gf)
        padding_mask: Optional[th.Tensor] = None,   # (B, N_max) bool, True = pad
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns
        -------
        logits : (B, N_max) — raw per-planet action scores (before masking)
        values : (B,)       — scalar state-value estimate
        """
        # --- Embed planet tokens ---
        tokens = self.planet_proj(planets)          # (B, N, d)

        # --- ISAB layers ---
        for isab in self.isab_layers:
            tokens = isab(tokens, key_padding_mask=padding_mask)   # (B, N, d)

        # --- Actor: per-token logit ---
        logits = self.policy_head(tokens).squeeze(-1)              # (B, N)

        # --- Critic: PMA → global embed → value ---
        summary = self.pma(tokens, key_padding_mask=padding_mask)  # (B, 1, d)
        summary = summary.squeeze(1)                               # (B, d)
        g_embed = self.global_proj(global_feat)                    # (B, d)
        critic_in = th.cat([summary, g_embed], dim=-1)             # (B, 2d)
        values = self.value_mlp(critic_in).squeeze(-1)             # (B,)

        return logits, values


# ---------------------------------------------------------------------------
# SB3 MaskableActorCriticPolicy wrapper
# ---------------------------------------------------------------------------

class FullSetISABPolicy(MaskableActorCriticPolicy):
    """
    SB3 MaskablePPO-compatible policy using the ISAB set transformer.

    Expects observation_space to be a Dict with:
        "planets" : Box(N_max, n_planet_features)
        "global"  : Box(n_global_features,)

    Uses the same action-mask interface as ``ArielTransformerPolicy``.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        d_model: int = 128,
        n_heads: int = 4,
        n_isab_layers: int = 2,
        n_inducing: int = 32,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        # Pull out our kwargs before passing to super().__init__
        self._d_model       = d_model
        self._n_heads       = n_heads
        self._n_isab_layers = n_isab_layers
        self._n_inducing    = n_inducing
        self._dropout       = dropout

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Disable SB3's default feature extractor (we build our own)
            features_extractor_class=None,
            **{k: v for k, v in kwargs.items()
               if k not in ("features_extractor_class",)},
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """Construct the network and optimizer — called by super().__init__."""
        planet_shape  = self.observation_space["planets"].shape   # (N_max, n_pf)
        global_shape  = self.observation_space["global"].shape    # (n_gf,)
        n_planet_feat = planet_shape[-1]
        n_global_feat = global_shape[0]

        self.isab_net = FullSetISABNet(
            n_planet_features=n_planet_feat,
            n_global_features=n_global_feat,
            d_model=self._d_model,
            n_heads=self._n_heads,
            n_isab_layers=self._n_isab_layers,
            n_inducing=self._n_inducing,
            dropout=self._dropout,
        )

        self.action_dist = CategoricalDistribution(int(self.action_space.n))
        self.optimizer   = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs_to_tensors(
        self, obs: dict
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """Extract planet/global tensors and build the padding mask."""
        planets     = obs["planets"]   # (B, N_max, n_pf)
        global_feat = obs["global"]    # (B, n_gf)

        # Padding mask: a row is padding if *all* features are exactly zero.
        # This matches the zero-padding added by ArielEnv._candidates_full_set.
        pad_mask = (planets.abs().sum(dim=-1) == 0.0)   # (B, N_max) bool

        return planets, global_feat, pad_mask

    def _predict_logits_and_values(
        self,
        obs: dict,
        action_masks: Optional[th.Tensor],
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Run forward pass; return (logits, values) with masks applied."""
        planets, global_feat, pad_mask = self._obs_to_tensors(obs)
        logits, values = self.isab_net(planets, global_feat, padding_mask=pad_mask)

        # Force padding positions to -inf so they can never be sampled.
        if pad_mask.any():
            logits = logits.masked_fill(pad_mask, float("-inf"))

        # Apply SB3 action mask (completed targets, infeasible events, etc.)
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks.bool(), float("-inf"))

        return logits, values

    # ------------------------------------------------------------------
    # MaskableActorCriticPolicy interface
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: dict,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Returns (actions, values, log_probs)."""
        obs_t = {k: th.as_tensor(v).to(self.device) for k, v in obs.items()}

        mask_t = (
            th.as_tensor(action_masks, dtype=th.bool).to(self.device)
            if action_masks is not None else None
        )

        logits, values = self._predict_logits_and_values(obs_t, mask_t)
        dist = CategoricalDistribution(int(self.action_space.n))
        dist = dist.proba_distribution(action_logits=logits)

        actions   = dist.get_actions(deterministic=deterministic)
        log_probs = dist.log_prob(actions)

        return actions, values, log_probs

    def evaluate_actions(
        self,
        obs: dict,
        actions: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """Returns (values, log_probs, entropy) for PPO update."""
        logits, values = self._predict_logits_and_values(obs, action_masks)
        dist = CategoricalDistribution(int(self.action_space.n))
        dist = dist.proba_distribution(action_logits=logits)

        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()

        return values, log_probs, entropy

    def predict_values(self, obs: dict) -> th.Tensor:
        """Returns scalar state values (B,)."""
        obs_t = {k: th.as_tensor(v).to(self.device) for k, v in obs.items()}
        planets, global_feat, pad_mask = self._obs_to_tensors(obs_t)
        _, values = self.isab_net(planets, global_feat, padding_mask=pad_mask)
        return values.unsqueeze(-1)   # (B, 1) as expected by SB3

    def _predict(
        self,
        observation: dict,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> th.Tensor:
        """Greedy/stochastic prediction (inference only)."""
        actions, _, _ = self.forward(observation, deterministic, action_masks)
        return actions
