"""
FullSetSelfAttentionPolicy — full self-attention ablation for the full-set space.

Architecture
------------
Identical to ``FullSetISABPolicy`` EXCEPT that it uses standard full self-attention
(O(N²)) instead of ISAB's induced attention (O(N·m)).

This is an ablation to separate:
    benefit of seeing all N planets  (vs. top-K filtering)
FROM:
    effect of induced attention       (ISAB vs. full self-attention)

    planets (N_max, n_pf)  ──► Linear ──► planet tokens (N_max, d)
                                                │
                                    TransformerEncoder  (n_layers × self-attn + FFN)
                                    Pre-LN, no positional encoding
                                                │
                        ┌───────────────────────┴────────────────────────┐
                        │                                                 │
                   policy_head                                        PMA (k=1)
                  (per-token linear)                           ──► global summary
                        │                                         + global features
                   logits (N_max,)                                      │
                   + action mask                                    value_head
                        │                                               │
                  π(a|s)                                           V(s) scalar

Compare against:
    Top-K full attention         (ArielTransformerPolicy,      input = top-K events)
    Full-set full attention      (FullSetSelfAttentionPolicy,  input = all N planets)
    Full-set induced attention   (FullSetISABPolicy,           input = all N planets)

This comparison isolates:
    Full-set vs. Top-K          → does seeing all planets help?
    ISAB vs. full attn          → does O(N·m) induced attention match O(N²)?

Usage
-----
    from sb3_contrib import MaskablePPO
    from ariel_rl.agents.policies.full_set_attention_policy import FullSetSelfAttentionPolicy

    model = MaskablePPO(
        FullSetSelfAttentionPolicy,
        env,   # must use action_type="full_set"
        policy_kwargs={"d_model": 128, "n_heads": 4, "n_layers": 2},
    )
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from ariel_rl.agents.policies.isab_modules import PMA


# ---------------------------------------------------------------------------
# Core network module
# ---------------------------------------------------------------------------

class FullSetSelfAttentionNet(nn.Module):
    """Full self-attention set encoder with PMA critic.

    Same interface as FullSetISABNet but uses nn.TransformerEncoder (O(N²))
    instead of ISAB (O(N·m)).  For N_max ≤ ~800 the memory difference is
    negligible; for N_max ≈ 2000 ISAB is preferred.
    """

    def __init__(
        self,
        n_planet_features: int,
        n_global_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.planet_proj = nn.Linear(n_planet_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # Pre-LN
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="enable_nested_tensor")
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Actor: per-token logit conditioned on both token and global mission state
        self.global_proj_actor = nn.Linear(n_global_features, d_model)
        self.actor_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # PMA critic (consistent with ISAB policy for fair comparison)
        self.pma = PMA(d_model, n_heads, k=1)
        self.global_proj_critic = nn.Linear(n_global_features, d_model)
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
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_mlp[-1].weight, gain=1.0)

    def forward(
        self,
        planets: th.Tensor,
        global_feat: th.Tensor,
        padding_mask: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        tokens = self.planet_proj(planets)                              # (B, N, d)
        tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)  # (B, N, d)

        # Actor: per-token logit conditioned on global mission state
        N = tokens.shape[1]
        g_actor  = self.global_proj_actor(global_feat)                  # (B, d)
        g_expand = g_actor.unsqueeze(1).expand(-1, N, -1)               # (B, N, d)
        logits   = self.actor_head(th.cat([tokens, g_expand], dim=-1)).squeeze(-1)  # (B, N)

        # Critic: PMA + global
        summary  = self.pma(tokens, key_padding_mask=padding_mask).squeeze(1)  # (B, d)
        g_critic = self.global_proj_critic(global_feat)                  # (B, d)
        values   = self.value_mlp(th.cat([summary, g_critic], dim=-1)).squeeze(-1)

        return logits, values


# ---------------------------------------------------------------------------
# SB3 policy wrapper (mirrors FullSetISABPolicy)
# ---------------------------------------------------------------------------

class FullSetSelfAttentionPolicy(MaskableActorCriticPolicy):
    """Full self-attention ablation on the full-set planet observation space.

    Use this to compare against FullSetISABPolicy:
        Full-set + full attention  (this class)
        Full-set + ISAB            (FullSetISABPolicy)

    Keep ArielTransformerPolicy unchanged as the Top-K baseline.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            **kwargs,
        )

        n_pf = observation_space["planets"].shape[-1]
        n_gf = observation_space["global"].shape[0]

        self.attn_net = FullSetSelfAttentionNet(
            n_planet_features=n_pf,
            n_global_features=n_gf,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _obs_to_tensors(self, obs):
        planets     = obs["planets"]
        global_feat = obs["global"]
        pad_mask    = (planets.abs().sum(dim=-1) == 0.0)
        return planets, global_feat, pad_mask

    def _predict_logits_and_values(self, obs, action_masks):
        planets, global_feat, pad_mask = self._obs_to_tensors(obs)
        logits, values = self.attn_net(planets, global_feat, padding_mask=pad_mask)
        if pad_mask.any():
            logits = logits.masked_fill(pad_mask, float("-inf"))
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks.bool(), float("-inf"))
        return logits, values

    def forward(self, obs, deterministic=False, action_masks=None):
        obs_t  = {k: th.as_tensor(v).to(self.device) for k, v in obs.items()}
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

    def evaluate_actions(self, obs, actions, action_masks=None):
        logits, values = self._predict_logits_and_values(obs, action_masks)
        dist = CategoricalDistribution(int(self.action_space.n))
        dist = dist.proba_distribution(action_logits=logits)
        return values, dist.log_prob(actions), dist.entropy()

    def predict_values(self, obs):
        obs_t = {k: th.as_tensor(v).to(self.device) for k, v in obs.items()}
        planets, global_feat, pad_mask = self._obs_to_tensors(obs_t)
        _, values = self.attn_net(planets, global_feat, padding_mask=pad_mask)
        return values.unsqueeze(-1)

    def _predict(self, observation, deterministic=False, action_masks=None):
        actions, _, _ = self.forward(observation, deterministic, action_masks)
        return actions
