"""Latent-space policy and value networks.

Both operate directly on the world model's latent state ``z`` rather than the
raw (high-dimensional, noisy) observation.  Because the latent is compact and
denoised, these networks are small -- which is part of the motivation for doing
policy optimisation in latent space.

* :class:`LatentPolicy` -- categorical policy ``pi(a | z)`` for discrete actions.
* :class:`ValueNet`     -- state-value baseline ``V(z)`` used **only** by PPO.
  GRPO does not instantiate this network at all, which is exactly the
  parameter/compute saving the project sets out to quantify.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .ssm import mlp


class LatentPolicy(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = mlp(latent_dim, action_dim, hidden_dim, n_hidden=2, activation=nn.Tanh)
        self.action_dim = action_dim

    def forward(self, z) -> Categorical:
        logits = self.net(z)
        return Categorical(logits=logits)

    def act(self, z):
        """Sample an action; returns ``(action, log_prob)``."""
        dist = self.forward(z)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, z, action):
        """Return ``(log_prob, entropy)`` of ``action`` under the current policy."""
        dist = self.forward(z)
        return dist.log_prob(action), dist.entropy()

    @torch.no_grad()
    def greedy_action(self, z):
        return torch.argmax(self.net(z), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ValueNet(nn.Module):
    """State-value network V(z). Used by PPO only."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = mlp(latent_dim, 1, hidden_dim, n_hidden=2, activation=nn.Tanh)

    def forward(self, z):
        return self.net(z).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
