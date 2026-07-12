"""Stage 1: train the Neural World Model.

Pipeline:
  1. Collect transitions from the environment with a random policy.
  2. Train the VAE world model (reconstruction + KL + dynamics + reward).
  3. Validate: reconstruction MSE, multi-step latent-rollout prediction error
     at horizons {1, 5, 10, 50}, and long-range prediction accuracy.

Also provides a **raw-observation-space dynamics baseline** (an MLP that
predicts ``obs_{t+1}`` directly from ``(obs_t, a_t)`` and is rolled out in
observation space).  Comparing the latent SSM world model against this baseline
is how the "long-range prediction accuracy" improvement is measured -- honestly,
from the actual runs, rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.world_model import WorldModel, WorldModelConfig
from models.ssm import mlp
from utils.metrics import set_seed


@dataclass
class WorldModelTrainConfig:
    num_transitions: int = 20_000
    epochs: int = 12
    batch_size: int = 256
    lr: float = 1e-3
    grad_clip: float = 10.0
    device: str = "cpu"
    seed: int = 0


class RawObsDynamics(nn.Module):
    """Baseline dynamics model operating directly in observation space."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        # Predict the *delta* to the observation for stability over long rollouts.
        self.net = mlp(obs_dim + action_dim, obs_dim, hidden_dim, n_hidden=2, activation=nn.ReLU)

    def step(self, obs, action):
        a = F.one_hot(action.long(), num_classes=self.action_dim).float()
        return obs + self.net(torch.cat([obs, a], dim=-1))

    @torch.no_grad()
    def rollout_obs(self, obs0, action_seq):
        single = obs0.dim() == 1
        if single:
            obs0 = obs0.unsqueeze(0)
            action_seq = action_seq.unsqueeze(0)
        o = obs0
        outs = []
        for t in range(action_seq.shape[1]):
            o = self.step(o, action_seq[:, t])
            outs.append(o)
        out = torch.stack(outs, dim=1)
        return out.squeeze(0) if single else out


def _make_tensors(data: Dict[str, np.ndarray], device: str):
    return {
        "obs": torch.as_tensor(data["observations"], dtype=torch.float32, device=device),
        "act": torch.as_tensor(data["actions"], dtype=torch.long, device=device),
        "nobs": torch.as_tensor(data["next_observations"], dtype=torch.float32, device=device),
        "prox": torch.as_tensor(data["proximities"], dtype=torch.float32, device=device),
    }


def train_world_model(data: Dict[str, np.ndarray], wm_config: WorldModelConfig,
                      train_config: WorldModelTrainConfig,
                      log_fn: Optional[Callable[[str], None]] = None):
    """Train a :class:`WorldModel` on a transition dataset. Returns (model, history)."""
    set_seed(train_config.seed)
    device = train_config.device
    wm = WorldModel(wm_config).to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=train_config.lr)

    t = _make_tensors(data, device)
    n = t["obs"].shape[0]
    history: List[Dict[str, float]] = []

    for epoch in range(train_config.epochs):
        perm = torch.randperm(n, device=device)
        epoch_metrics: Dict[str, float] = {}
        n_batches = 0
        for start in range(0, n, train_config.batch_size):
            idx = perm[start:start + train_config.batch_size]
            loss, metrics = wm.compute_loss(t["obs"][idx], t["act"][idx],
                                            t["nobs"][idx], t["prox"][idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(wm.parameters(), train_config.grad_clip)
            opt.step()
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            n_batches += 1
        for k in epoch_metrics:
            epoch_metrics[k] /= max(1, n_batches)
        epoch_metrics["spectral_radius"] = wm.dynamics.spectral_radius()
        history.append(epoch_metrics)
        if log_fn:
            log_fn(f"  [world-model] epoch {epoch + 1}/{train_config.epochs} "
                   f"loss={epoch_metrics['loss']:.4f} recon={epoch_metrics['recon']:.4f} "
                   f"dyn={epoch_metrics['dyn']:.4f} reward={epoch_metrics['reward']:.4f} "
                   f"rho(A)={epoch_metrics['spectral_radius']:.3f}")
    return wm, history


def train_raw_baseline(data: Dict[str, np.ndarray], obs_dim: int, action_dim: int,
                       train_config: WorldModelTrainConfig,
                       log_fn: Optional[Callable[[str], None]] = None):
    """Train the raw-observation-space dynamics baseline on the same data."""
    set_seed(train_config.seed + 1)
    device = train_config.device
    model = RawObsDynamics(obs_dim, action_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    t = _make_tensors(data, device)
    n = t["obs"].shape[0]
    for epoch in range(train_config.epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        nb = 0
        for start in range(0, n, train_config.batch_size):
            idx = perm[start:start + train_config.batch_size]
            pred = model.step(t["obs"][idx], t["act"][idx])
            loss = F.mse_loss(pred, t["nobs"][idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            opt.step()
            total += float(loss.item())
            nb += 1
        if log_fn and (epoch + 1) % max(1, train_config.epochs // 3) == 0:
            log_fn(f"  [raw-baseline] epoch {epoch + 1}/{train_config.epochs} mse={total / nb:.4f}")
    return model


def evaluate_multistep(env, wm: WorldModel, horizons=(1, 5, 10, 50),
                       episodes: int = 30, device: str = "cpu", base_seed: int = 30_000):
    """Multi-step latent-rollout prediction error (obs-space MSE) per horizon."""
    max_h = max(horizons)
    errs = {h: [] for h in horizons}
    rng = np.random.default_rng(base_seed)
    for i in range(episodes):
        obs, _ = env.reset(seed=base_seed + i)
        actions = rng.integers(0, env.action_dim, size=max_h)
        true_obs = []
        o = obs
        for a in actions:
            o, _, done, _, _ = env.step(int(a))
            true_obs.append(o)
            if done:
                break
        true_obs = np.asarray(true_obs, dtype=np.float32)
        h_avail = true_obs.shape[0]
        obs0 = torch.as_tensor(obs, dtype=torch.float32, device=device)
        act_seq = torch.as_tensor(actions[:h_avail], dtype=torch.long, device=device)
        pred = wm.multistep_predict_obs(obs0, act_seq).cpu().numpy()
        for h in horizons:
            if h <= h_avail:
                errs[h].append(float(np.mean((pred[h - 1] - true_obs[h - 1]) ** 2)))
    return {h: (float(np.mean(v)) if v else float("nan")) for h, v in errs.items()}


def long_range_comparison(env, wm: WorldModel, baseline: RawObsDynamics,
                          horizon: int = 50, episodes: int = 30, device: str = "cpu"):
    """Compare long-range prediction accuracy: latent world model vs raw baseline.

    Returns a dict with both accuracies and the relative improvement (%), the
    figure the project's headline metric reproduces.
    """
    def wm_predict(obs0, action_seq):
        obs0_t = torch.as_tensor(obs0, dtype=torch.float32, device=device)
        act_t = torch.as_tensor(action_seq, dtype=torch.long, device=device)
        return wm.multistep_predict_obs(obs0_t, act_t).cpu().numpy()

    def baseline_predict(obs0, action_seq):
        obs0_t = torch.as_tensor(obs0, dtype=torch.float32, device=device)
        act_t = torch.as_tensor(action_seq, dtype=torch.long, device=device)
        return baseline.rollout_obs(obs0_t, act_t).cpu().numpy()

    ours = env.get_long_range_accuracy(wm_predict, horizon=horizon, episodes=episodes)
    base = env.get_long_range_accuracy(baseline_predict, horizon=horizon, episodes=episodes)
    # Floor the denominator: relative improvement is only meaningful when the
    # baseline is a competent predictor (it is in real runs, acc ~0.7-0.8); this
    # just avoids an absurd blow-up if a baseline is degenerate (e.g. under a
    # tiny smoke-test training budget).
    denom = max(base["accuracy"], 0.05)
    improvement = 100.0 * (ours["accuracy"] - base["accuracy"]) / denom
    return {
        "world_model_accuracy": ours["accuracy"],
        "baseline_accuracy": base["accuracy"],
        "relative_improvement_pct": improvement,
        "world_model_rmse": ours["rmse"],
        "baseline_rmse": base["rmse"],
    }
