"""Baseline comparison: PPO policy optimisation in latent space.

Standard PPO (clipped surrogate + GAE), operating on the world model's latent
state just like the GRPO trainer, so the two differ *only* in how they estimate
the advantage and update:

* PPO trains a **critic** ``V(z)`` and uses **GAE** for per-step advantages.
* PPO's loss adds a value term: ``L = L_policy + c1 * L_value - c2 * H``.

This is the value-based counterpart to the critic-free GRPO trainer.  Tracking
the critic's value-prediction error (explained variance) is what lets the
sparse-reward analysis show *why* PPO degrades when the reward is delayed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch

from models.policy import LatentPolicy, ValueNet
from utils.metrics import grad_norm, set_seed
from .latent_rollout import apply_reward_mode, sample_initial_latents, evaluate_in_env


@dataclass
class PPOConfig:
    num_trajectories: int = 128   # matched to GRPO's num_prompts * group_size
    horizon: int = 30
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    minibatch_size: int = 512
    grad_clip: float = 1.0
    reward_mode: str = "dense"
    success_threshold: float = 0.6
    device: str = "cpu"


class PPOLatentTrainer:
    def __init__(self, policy: LatentPolicy, value_net: ValueNet, world_model,
                 config: PPOConfig):
        self.policy = policy
        self.value_net = value_net
        self.wm = world_model
        self.cfg = config
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value_net.parameters()), lr=config.lr)
        for p in self.wm.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def collect_trajectories(self, z_init: torch.Tensor):
        cfg = self.cfg
        z = z_init
        states, actions, logps, proximities, values = [], [], [], [], []
        for _ in range(cfg.horizon):
            dist = self.policy(z)
            a = dist.sample()
            states.append(z)
            actions.append(a)
            logps.append(dist.log_prob(a))
            values.append(self.value_net(z))
            z = self.wm.predict_next(z, a)
            proximities.append(self.wm.predict_reward(z))
        last_value = self.value_net(z)  # bootstrap value of terminal latent z_T
        return {
            "states": torch.stack(states, dim=1),         # [B, T, latent]
            "actions": torch.stack(actions, dim=1),        # [B, T]
            "logp_old": torch.stack(logps, dim=1),         # [B, T]
            "proximity": torch.stack(proximities, dim=1),  # [B, T]
            "values": torch.stack(values, dim=1),          # [B, T]
            "last_value": last_value,                      # [B]
        }

    def compute_gae(self, batch):
        cfg = self.cfg
        rewards = apply_reward_mode(batch["proximity"], cfg.reward_mode,
                                    cfg.horizon, cfg.success_threshold)
        values = batch["values"]
        last_value = batch["last_value"]
        B, T = rewards.shape
        adv = torch.zeros_like(rewards)
        last_gae = torch.zeros(B, device=rewards.device)
        for t in reversed(range(T)):
            next_value = values[:, t + 1] if t < T - 1 else last_value
            delta = rewards[:, t] + cfg.gamma * next_value - values[:, t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * last_gae
            adv[:, t] = last_gae
        returns = adv + values
        mean_return = float((rewards.sum(dim=1)).mean().item())
        return adv, returns, rewards, mean_return

    def update(self, batch, advantages, returns):
        cfg = self.cfg
        B, T, _ = batch["states"].shape
        flat_states = batch["states"].reshape(B * T, -1)
        flat_actions = batch["actions"].reshape(B * T)
        flat_logp_old = batch["logp_old"].reshape(B * T)
        flat_adv = advantages.reshape(B * T)
        flat_ret = returns.reshape(B * T)
        # Normalise advantages (standard PPO practice).
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        N = B * T
        last = {}
        for _ in range(cfg.update_epochs):
            perm = torch.randperm(N, device=flat_states.device)
            for start in range(0, N, cfg.minibatch_size):
                idx = perm[start:start + cfg.minibatch_size]
                dist = self.policy(flat_states[idx])
                logp_new = dist.log_prob(flat_actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp_new - flat_logp_old[idx])
                a = flat_adv[idx]
                surr1 = ratio * a
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * a
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = self.value_net(flat_states[idx])
                value_loss = torch.nn.functional.mse_loss(value_pred, flat_ret[idx])

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                gnorm = grad_norm(self.policy)  # policy-only norm, comparable to GRPO
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    cfg.grad_clip)
                self.optimizer.step()
                last = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "grad_norm": gnorm,
                }

        # Critic quality: explained variance of value predictions over the batch.
        with torch.no_grad():
            v = self.value_net(flat_states)
            var_y = torch.var(flat_ret)
            ev = float(1.0 - torch.var(flat_ret - v) / (var_y + 1e-8))
            last["value_explained_variance"] = ev
            last["value_error"] = float(torch.nn.functional.mse_loss(v, flat_ret).item())
        return last

    def train(self, env, num_iterations: int, eval_every: int = 10,
              eval_episodes: int = 10, seed: int = 0,
              log_fn: Optional[Callable[[str], None]] = None):
        set_seed(seed)
        device = self.cfg.device
        history: Dict[str, list] = {
            "iteration": [], "mean_return": [], "policy_loss": [], "value_loss": [],
            "entropy": [], "grad_norm": [], "value_explained_variance": [],
            "value_error": [], "eval_iteration": [], "eval_return": [],
            "eval_proximity": [], "eval_success": [], "wallclock": [],
        }
        t_start = time.time()
        for it in range(num_iterations):
            z_init = sample_initial_latents(env, self.wm, self.cfg.num_trajectories,
                                            device=device, base_seed=50_000 + it * 200 + seed)
            batch = self.collect_trajectories(z_init)
            adv, returns, _, mean_return = self.compute_gae(batch)
            metrics = self.update(batch, adv, returns)

            history["iteration"].append(it)
            history["mean_return"].append(mean_return)
            history["policy_loss"].append(metrics["policy_loss"])
            history["value_loss"].append(metrics["value_loss"])
            history["entropy"].append(metrics["entropy"])
            history["grad_norm"].append(metrics["grad_norm"])
            history["value_explained_variance"].append(metrics["value_explained_variance"])
            history["value_error"].append(metrics["value_error"])
            history["wallclock"].append(time.time() - t_start)

            if (it + 1) % eval_every == 0 or it == num_iterations - 1:
                ev = evaluate_in_env(env, self.wm, self.policy, device, episodes=eval_episodes)
                history["eval_iteration"].append(it)
                history["eval_return"].append(ev["episodic_return"])
                history["eval_proximity"].append(ev["mean_proximity"])
                history["eval_success"].append(ev["success_rate"])
                if log_fn:
                    log_fn(f"  [PPO ] it {it + 1}/{num_iterations} "
                           f"mean_R={mean_return:.3f} ent={metrics['entropy']:.3f} "
                           f"|g|={metrics['grad_norm']:.3f} EV={metrics['value_explained_variance']:.2f} "
                           f"eval_prox={ev['mean_proximity']:.3f} eval_succ={ev['success_rate']:.2f}")
        history["total_time"] = time.time() - t_start
        history["policy_params"] = self.policy.count_parameters()
        history["total_params"] = self.policy.count_parameters() + self.value_net.count_parameters()
        return history
