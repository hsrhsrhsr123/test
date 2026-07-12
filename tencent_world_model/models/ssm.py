"""Latent state-space model (SSM) for latent dynamics.

A lightweight, S4/Mamba-inspired recurrence used as the *dynamics core* of the
world model.  The next latent state is modelled as a **linear recurrence** plus
a **non-linear residual correction**::

    z_{t+1} = A z_t + B a_t + MLP_res([z_t, a_t])

* ``A`` (``latent_dim x latent_dim``) is a learnable state-transition matrix.
* ``B`` (``latent_dim x action_dim``) maps the (one-hot) action into latent
  space.
* ``MLP_res`` captures the non-linear part of the dynamics that a purely linear
  recurrence cannot represent.

The linear term gives the stable, well-conditioned long-range propagation that
makes multi-step latent rollouts trustworthy (the property we exploit for
long-horizon prediction and for model-based policy optimisation); the residual
adds the expressiveness needed for a genuinely non-linear environment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim, out_dim, hidden_dim, n_hidden=2, activation=nn.Tanh):
    layers = [nn.Linear(in_dim, hidden_dim), activation()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


class LatentSSM(nn.Module):
    """Linear latent recurrence with a learned non-linear residual."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 128,
                 residual_scale: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.residual_scale = residual_scale

        # Initialise A close to identity so the recurrence starts stable
        # (spectral radius ~1) rather than collapsing or exploding.
        self.A = nn.Parameter(torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim))
        self.B = nn.Parameter(0.01 * torch.randn(latent_dim, action_dim))
        self.residual_mlp = mlp(latent_dim + action_dim, latent_dim, hidden_dim)

    def _one_hot(self, action):
        # Already one-hot encoded floats: [..., action_dim].
        if action.is_floating_point() and action.shape[-1] == self.action_dim:
            return action
        # Otherwise treat as integer action indices, shape [B] or [B, 1].
        idx = action.long()
        if idx.dim() >= 2 and idx.shape[-1] == 1:
            idx = idx.squeeze(-1)
        return F.one_hot(idx, num_classes=self.action_dim).float()

    def step(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """One-step latent transition.

        ``z``: ``[B, latent_dim]``; ``action``: ``[B]`` indices or
        ``[B, action_dim]`` one-hot.  Returns ``z_next`` ``[B, latent_dim]``.
        """
        a = self._one_hot(action)
        linear = z @ self.A.t() + a @ self.B.t()
        residual = self.residual_mlp(torch.cat([z, a], dim=-1))
        return linear + self.residual_scale * residual

    def rollout(self, z0: torch.Tensor, action_seq: torch.Tensor):
        """Roll the recurrence forward over a sequence of actions.

        ``z0``: ``[B, latent_dim]``; ``action_seq``: ``[B, T]`` indices or
        ``[B, T, action_dim]`` one-hot.  Returns ``[B, T, latent_dim]`` of the
        predicted latents ``z_1 .. z_T``.
        """
        T = action_seq.shape[1]
        z = z0
        outs = []
        for t in range(T):
            a_t = action_seq[:, t]
            z = self.step(z, a_t)
            outs.append(z)
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def spectral_radius(self) -> float:
        """Largest |eigenvalue| of A -- a stability diagnostic for the linear core."""
        eig = torch.linalg.eigvals(self.A)
        return float(eig.abs().max().item())
