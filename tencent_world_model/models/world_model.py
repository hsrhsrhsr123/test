"""VAE-based Neural World Model.

Components
----------
1. **Encoder**   ``obs -> (mu, log_var)`` and a sampled latent ``z`` (VAE).
2. **Dynamics**  ``(z_t, a_t) -> z_{t+1}`` -- implemented by :class:`LatentSSM`.
3. **Reward**    ``z -> proximity`` in ``[0, 1]`` (the environment's per-state
   reward potential; sparsity is applied *outside* the model, at rollout time).
4. **Decoder**   ``z -> obs_reconstructed`` (used to validate latent quality
   and to measure long-range prediction error in observation space).

Training objective
-------------------
::

    L = L_recon + beta * L_KL + w_dyn * L_dyn + w_rew * L_reward

    L_recon  = || decode(z_t) - obs_t ||^2
    L_KL     = KL( q(z|obs) || N(0, I) )
    L_dyn    = || dynamics(z_t, a_t) - stop_grad(encode(obs_{t+1})) ||^2
    L_reward = || reward_head(z_t) - proximity_t ||^2

The dynamics target uses the *mean* encoding of the next observation with a
stop-gradient, i.e. the dynamics model is trained to predict where the encoder
maps the next observation (a stable, consistent latent target).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import LatentSSM, mlp


@dataclass
class WorldModelConfig:
    obs_dim: int = 16
    action_dim: int = 5
    latent_dim: int = 16
    hidden_dim: int = 128
    beta: float = 1e-3          # KL weight (small -> more informative latent)
    w_dyn: float = 1.0          # dynamics-consistency weight
    w_rew: float = 2.0          # reward-prediction weight
    reward_pos_weight: float = 4.0  # up-weights rare high-proximity samples in the reward loss
    ssm_hidden_dim: int = 128
    ssm_residual_scale: float = 1.0


class Encoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden_dim, hidden_dim, n_hidden=1, activation=nn.ReLU)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, obs):
        h = F.relu(self.trunk(obs))
        return self.mu(h), self.log_var(h).clamp(-8, 4)


class Decoder(nn.Module):
    def __init__(self, latent_dim, obs_dim, hidden_dim):
        super().__init__()
        self.net = mlp(latent_dim, obs_dim, hidden_dim, n_hidden=2, activation=nn.ReLU)

    def forward(self, z):
        return self.net(z)


class WorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.cfg = config
        self.encoder = Encoder(config.obs_dim, config.latent_dim, config.hidden_dim)
        self.decoder = Decoder(config.latent_dim, config.obs_dim, config.hidden_dim)
        self.dynamics = LatentSSM(config.latent_dim, config.action_dim,
                                  hidden_dim=config.ssm_hidden_dim,
                                  residual_scale=config.ssm_residual_scale)
        # Reward head predicts proximity in [0, 1] via a sigmoid.
        self.reward_head = mlp(config.latent_dim, 1, config.hidden_dim,
                               n_hidden=2, activation=nn.ReLU)

    # ------------------------------------------------------------------ VAE
    def encode(self, obs, sample: bool = True):
        mu, log_var = self.encoder(obs)
        if sample:
            std = torch.exp(0.5 * log_var)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z, mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def predict_next(self, z, action):
        return self.dynamics.step(z, action)

    def predict_reward(self, z):
        """Predicted per-state proximity in [0, 1]."""
        return torch.sigmoid(self.reward_head(z)).squeeze(-1)

    # -------------------------------------------------------------- training
    def compute_loss(self, obs, action, next_obs, proximity):
        """Compute the composite world-model loss on a batch of transitions.

        Shapes: ``obs``/``next_obs`` ``[B, obs_dim]``, ``action`` ``[B]``,
        ``proximity`` ``[B]``.  Returns ``(total_loss, metrics_dict)``.
        """
        z, mu, log_var = self.encode(obs, sample=True)

        # Reconstruction.
        recon = self.decode(z)
        l_recon = F.mse_loss(recon, obs)

        # KL divergence to N(0, I).
        l_kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # Dynamics consistency: predict the (mean) encoding of next_obs.
        with torch.no_grad():
            _, mu_next, _ = self.encode(next_obs, sample=False)
        z_pred = self.predict_next(z, action)
        l_dyn = F.mse_loss(z_pred, mu_next)

        # Reward (proximity) prediction from the (deterministic) latent mean --
        # this matches how the reward head is queried at rollout time, and
        # avoids VAE sampling noise corrupting the peaked proximity signal.
        # High-proximity (goal) states are rare, so we up-weight them; otherwise
        # MSE collapses to the low-proximity mean and the reward head never fires
        # near the goal (which would break sparse-reward latent rollouts).
        r_pred = self.predict_reward(mu)
        weight = 1.0 + self.cfg.reward_pos_weight * proximity
        l_rew = torch.mean(weight * (r_pred - proximity) ** 2)

        total = (l_recon + self.cfg.beta * l_kl
                 + self.cfg.w_dyn * l_dyn + self.cfg.w_rew * l_rew)
        metrics = {
            "loss": float(total.item()),
            "recon": float(l_recon.item()),
            "kl": float(l_kl.item()),
            "dyn": float(l_dyn.item()),
            "reward": float(l_rew.item()),
        }
        return total, metrics

    # ------------------------------------------------------------- rollouts
    @torch.no_grad()
    def multistep_predict_obs(self, obs0, action_seq):
        """Predict a sequence of observations by rolling out in latent space.

        ``obs0``: ``[obs_dim]`` or ``[B, obs_dim]``; ``action_seq``:
        ``[T]`` or ``[B, T]``.  Returns predicted observations ``[B, T, obs_dim]``
        (or ``[T, obs_dim]`` if a single trajectory was given).
        """
        single = obs0.dim() == 1
        if single:
            obs0 = obs0.unsqueeze(0)
            action_seq = action_seq.unsqueeze(0)
        _, z, _ = self.encode(obs0, sample=False)
        z_seq = self.dynamics.rollout(z, action_seq)  # [B, T, latent]
        B, T, _ = z_seq.shape
        obs_pred = self.decode(z_seq.reshape(B * T, -1)).reshape(B, T, -1)
        return obs_pred.squeeze(0) if single else obs_pred

    @torch.no_grad()
    def encode_np(self, obs_np, device="cpu"):
        """Convenience: encode a numpy observation array to latent means."""
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)
        _, mu, _ = self.encode(obs, sample=False)
        return mu.squeeze(0) if single else mu
