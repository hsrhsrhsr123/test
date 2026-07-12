"""Stage 2: GRPO policy optimisation in latent space.

Group Relative Policy Optimisation (GRPO) -- the critic-free algorithm behind
DeepSeek-R1 / and the basis DAPO & VAPO build on -- adapted to model-based RL:

  1. **Sample.** For each initial latent state (a "prompt") sample ``G`` action
     sequences from the current policy.
  2. **Rollout.** Unroll every sequence through the world model's latent
     dynamics (the SSM) -- no real environment needed.
  3. **Reward.** Score each trajectory with the world model's reward head,
     under the task's reward-delivery mode.
  4. **Group-relative advantage.**
     ``A_i = (R_i - mean(R_group)) / (std(R_group) + eps)`` -- normalised
     *within* the group sharing an initial state.  No value network.
  5. **Clipped policy update** (PPO-style ratio clipping), advantage broadcast
     across the trajectory's timesteps.

Why this matters for the comparison: there is **no critic**, so there is no
value-estimation bias to corrupt the advantage under sparse rewards, and the
group normalisation makes the update invariant to reward scale.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from models.policy import LatentPolicy
from utils.metrics import grad_norm, discounted_returns, set_seed
from .latent_rollout import apply_reward_mode, sample_initial_latents, evaluate_in_env


@dataclass
class GRPOConfig:
    group_size: int = 8         # G: trajectories per initial state
    num_prompts: int = 16       # initial states sampled per iteration
    horizon: int = 30
    gamma: float = 0.99
    clip_eps: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 0.01
    update_epochs: int = 4
    grad_clip: float = 1.0
    adv_eps: float = 1e-6
    reward_mode: str = "dense"
    success_threshold: float = 0.6
    device: str = "cpu"


class GRPOLatentTrainer:
    def __init__(self, policy: LatentPolicy, world_model, config: GRPOConfig):
        self.policy = policy
        self.wm = world_model
        self.cfg = config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
        # World model is frozen during policy optimisation.
        for p in self.wm.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------- sampling
    @torch.no_grad()
    def collect_trajectories(self, z_init: torch.Tensor):
        """Roll out ``G`` trajectories per initial latent state in latent space.

        ``z_init``: ``[P, latent]``.  Returns dict of tensors with a leading
        batch dim ``B = P * G`` (groups are contiguous blocks of ``G``).
        """
        cfg = self.cfg
        P = z_init.shape[0]
        G = cfg.group_size
        # Repeat each prompt G times -> [P*G, latent]; groups are contiguous.
        z = z_init.repeat_interleave(G, dim=0)
        states, actions, logps, proximities = [], [], [], []
        for _ in range(cfg.horizon):
            dist = self.policy(z)
            a = dist.sample()
            lp = dist.log_prob(a)
            states.append(z)
            actions.append(a)
            logps.append(lp)
            z = self.wm.predict_next(z, a)
            proximities.append(self.wm.predict_reward(z))
        return {
            "states": torch.stack(states, dim=1),          # [B, T, latent]
            "actions": torch.stack(actions, dim=1),         # [B, T]
            "logp_old": torch.stack(logps, dim=1),          # [B, T]
            "proximity": torch.stack(proximities, dim=1),   # [B, T]
            "num_prompts": P,
        }

    # -------------------------------------------------------- advantages
    def compute_group_advantages(self, proximity: torch.Tensor, num_prompts: int):
        """Group-relative advantage from trajectory returns.

        Returns ``(advantages[B], mean_return, reward_stream[B,T])``.
        """
        cfg = self.cfg
        rewards = apply_reward_mode(proximity, cfg.reward_mode, cfg.horizon,
                                    cfg.success_threshold)
        returns = discounted_returns(rewards, cfg.gamma)  # [B, T]
        R = returns[:, 0]                                  # total discounted return per traj
        G = cfg.group_size
        Rg = R.view(num_prompts, G)
        mean = Rg.mean(dim=1, keepdim=True)
        std = Rg.std(dim=1, keepdim=True)
        adv = ((Rg - mean) / (std + cfg.adv_eps)).view(-1)  # [B]
        # Fraction of groups whose return spread is non-trivial -- i.e. groups
        # that actually produce a usable group-relative gradient.  This is the
        # quantity that stays healthy under sparse rewards (no critic needed).
        signal_frac = float((std.squeeze(-1) > 1e-4).float().mean().item())
        return adv, float(R.mean().item()), rewards, signal_frac

    # ------------------------------------------------------------- update
    def update_policy(self, batch, advantages):
        cfg = self.cfg
        states = batch["states"]      # [B, T, latent]
        actions = batch["actions"]    # [B, T]
        logp_old = batch["logp_old"]  # [B, T]
        B, T, _ = states.shape
        adv = advantages.unsqueeze(1).expand(B, T)  # broadcast over time

        flat_states = states.reshape(B * T, -1)
        flat_actions = actions.reshape(B * T)

        last = {}
        for _ in range(cfg.update_epochs):
            dist = self.policy(flat_states)
            logp_new = dist.log_prob(flat_actions).reshape(B, T)
            entropy = dist.entropy().reshape(B, T).mean()

            ratio = torch.exp(logp_new - logp_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            loss = policy_loss - cfg.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            gnorm = grad_norm(self.policy)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
            self.optimizer.step()

            with torch.no_grad():
                approx_kl = float((logp_old - logp_new).mean().item())
            last = {
                "policy_loss": float(policy_loss.item()),
                "entropy": float(entropy.item()),
                "grad_norm": gnorm,
                "approx_kl": approx_kl,
            }
        return last

    # ------------------------------------------------------------- driver
    def train(self, env, num_iterations: int, eval_every: int = 10,
              eval_episodes: int = 10, seed: int = 0,
              log_fn: Optional[Callable[[str], None]] = None):
        set_seed(seed)
        device = self.cfg.device
        history: Dict[str, list] = {
            "iteration": [], "mean_return": [], "policy_loss": [], "entropy": [],
            "grad_norm": [], "approx_kl": [], "group_signal_frac": [],
            "eval_iteration": [], "eval_return": [], "eval_proximity": [],
            "eval_success": [], "wallclock": [],
        }
        t_start = time.time()
        for it in range(num_iterations):
            z_init = sample_initial_latents(env, self.wm, self.cfg.num_prompts,
                                            device=device, base_seed=40_000 + it * 100 + seed)
            batch = self.collect_trajectories(z_init)
            adv, mean_return, _, signal_frac = self.compute_group_advantages(
                batch["proximity"], batch["num_prompts"])
            metrics = self.update_policy(batch, adv)

            history["iteration"].append(it)
            history["mean_return"].append(mean_return)
            history["policy_loss"].append(metrics["policy_loss"])
            history["entropy"].append(metrics["entropy"])
            history["grad_norm"].append(metrics["grad_norm"])
            history["approx_kl"].append(metrics["approx_kl"])
            history["group_signal_frac"].append(signal_frac)
            history["wallclock"].append(time.time() - t_start)

            if (it + 1) % eval_every == 0 or it == num_iterations - 1:
                ev = evaluate_in_env(env, self.wm, self.policy, device, episodes=eval_episodes)
                history["eval_iteration"].append(it)
                history["eval_return"].append(ev["episodic_return"])
                history["eval_proximity"].append(ev["mean_proximity"])
                history["eval_success"].append(ev["success_rate"])
                if log_fn:
                    log_fn(f"  [GRPO] it {it + 1}/{num_iterations} "
                           f"mean_R={mean_return:.3f} ent={metrics['entropy']:.3f} "
                           f"|g|={metrics['grad_norm']:.3f} eval_prox={ev['mean_proximity']:.3f} "
                           f"eval_succ={ev['success_rate']:.2f}")
        history["total_time"] = time.time() - t_start
        history["policy_params"] = self.policy.count_parameters()
        history["total_params"] = self.policy.count_parameters()  # GRPO: no critic
        return history
