"""Shared helpers for latent-space policy optimisation (GRPO and PPO).

Both trainers roll policies out **entirely inside the world model's latent
space** -- no real-environment interaction during optimisation.  The pieces
they share live here so the two algorithms differ only in their advantage
estimation and update rule, which is exactly the comparison we care about.
"""

from __future__ import annotations

import numpy as np
import torch


def apply_reward_mode(proximity: torch.Tensor, mode: str, horizon: int,
                      success_threshold: float) -> torch.Tensor:
    """Turn a per-step proximity signal into a mode-specific reward stream.

    ``proximity``: ``[B, T]`` predicted proximity of the *resulting* state at
    each step.  Returns rewards ``[B, T]``.  The temporal delivery mirrors
    :meth:`SequencePredictionEnv._deliver_reward` so the quantity optimised in
    latent space matches what the real environment pays out.
    """
    B, T = proximity.shape
    if mode == "dense":
        return proximity
    if mode == "sparse":
        rewards = torch.zeros_like(proximity)
        rewards[:, -1] = proximity[:, -1] * horizon
        return rewards
    if mode == "very_sparse":
        success = (proximity >= success_threshold).float()
        # +1 exactly at the first step the goal region is entered.
        csum = torch.cumsum(success, dim=1)
        first_hit = ((csum == 1) & (success > 0)).float()
        return first_hit
    raise ValueError(mode)


@torch.no_grad()
def sample_initial_latents(env, world_model, num_states: int, device: str = "cpu",
                           base_seed: int = 40_000) -> torch.Tensor:
    """Reset the real env ``num_states`` times and encode the start observations."""
    obs_list = []
    for i in range(num_states):
        obs, _ = env.reset(seed=base_seed + i)
        obs_list.append(obs)
    obs = torch.as_tensor(np.asarray(obs_list), dtype=torch.float32, device=device)
    _, mu, _ = world_model.encode(obs, sample=False)
    return mu


def make_latent_policy_fn(world_model, policy, device: str = "cpu", greedy: bool = True):
    """Wrap (world_model, latent policy) into a ``policy_fn(obs)->action`` for the real env."""
    def policy_fn(obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, mu, _ = world_model.encode(obs_t, sample=False)
            if greedy:
                a = policy.greedy_action(mu)
            else:
                a, _ = policy.act(mu)
        return int(a.item())
    return policy_fn


def evaluate_in_env(env, world_model, policy, device: str = "cpu",
                    episodes: int = 10, base_seed: int = 10_000):
    """Evaluate the current policy in the *real* environment (mode-agnostic metrics)."""
    policy_fn = make_latent_policy_fn(world_model, policy, device, greedy=True)
    return env.evaluate_policy(policy_fn, episodes=episodes, base_seed=base_seed)
