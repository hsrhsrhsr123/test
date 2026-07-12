"""GRPO vs PPO comparison orchestrator.

Trains a shared world model, then runs GRPO and PPO in its latent space under
identical conditions (same initial policy architecture, same rollout budget per
iteration, same seeds) and records everything the analysis modules need:
per-iteration return / loss / entropy / gradient-norm curves, real-environment
evaluation curves, wall-clock time, parameter counts, and (for PPO) the
critic's explained variance.

The public entry points are :func:`train_shared_world_model`,
:func:`run_algorithm`, and :func:`run_comparison`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from envs.sequence_prediction import (SequencePredictionEnv,
                                       SequencePredictionConfig,
                                       collect_transitions)
from models.world_model import WorldModel, WorldModelConfig
from models.policy import LatentPolicy, ValueNet
from training.train_world_model import (train_world_model, train_raw_baseline,
                                         WorldModelTrainConfig, evaluate_multistep,
                                         long_range_comparison)
from training.grpo_latent import GRPOLatentTrainer, GRPOConfig
from training.ppo_latent import PPOLatentTrainer, PPOConfig
from utils.metrics import set_seed


def make_env(reward_mode: str, horizon: int, seed: int = 0,
             overrides: Optional[dict] = None) -> SequencePredictionEnv:
    kwargs = dict(reward_mode=reward_mode, horizon=horizon, seed=seed)
    if overrides:
        kwargs.update(overrides)
    return SequencePredictionEnv(SequencePredictionConfig(**kwargs))


def train_shared_world_model(reward_mode: str, horizon: int, latent_dim: int,
                             wm_train: WorldModelTrainConfig,
                             env_overrides: Optional[dict] = None,
                             train_baseline: bool = True,
                             log_fn: Optional[Callable[[str], None]] = None):
    """Collect data and train the world model (+ optional raw baseline).

    Returns ``(world_model, baseline_or_None, env, info_dict)``.
    """
    env = make_env(reward_mode, horizon, seed=0, overrides=env_overrides)
    data = collect_transitions(env, wm_train.num_transitions, seed=wm_train.seed)
    wm_cfg = WorldModelConfig(obs_dim=env.observation_dim, action_dim=env.action_dim,
                              latent_dim=latent_dim)
    wm, wm_history = train_world_model(data, wm_cfg, wm_train, log_fn=log_fn)
    baseline = None
    if train_baseline:
        baseline = train_raw_baseline(data, env.observation_dim, env.action_dim,
                                      wm_train, log_fn=log_fn)
    # Multi-step error needs an env whose horizon covers the longest horizon (50).
    ms_env = make_env(reward_mode, horizon=max(52, horizon), seed=0, overrides=env_overrides)
    multistep = evaluate_multistep(ms_env, wm, horizons=(1, 5, 10, 50),
                                   episodes=30, device=wm_train.device)
    info = {"wm_history": wm_history, "multistep_error": multistep,
            "data_proximity_mean": float(np.mean(data["proximities"])),
            "reward_mode": reward_mode}
    return wm, baseline, env, info


def run_algorithm(algo: str, env, world_model, num_iterations: int, seed: int,
                  hyper: dict, eval_every: int = 10, eval_episodes: int = 15,
                  log_fn: Optional[Callable[[str], None]] = None) -> Dict:
    """Run one algorithm (``'grpo'`` or ``'ppo'``) with one seed. Returns history."""
    set_seed(seed)
    latent_dim = world_model.cfg.latent_dim
    action_dim = world_model.cfg.action_dim
    reward_mode = env.cfg.reward_mode
    horizon = env.cfg.horizon
    success_threshold = env.cfg.success_threshold

    policy = LatentPolicy(latent_dim, action_dim, hidden_dim=hyper.get("policy_hidden", 64))
    if algo == "grpo":
        cfg = GRPOConfig(
            group_size=hyper.get("group_size", 8),
            num_prompts=hyper.get("num_prompts", 16),
            horizon=horizon, gamma=hyper.get("gamma", 0.99),
            clip_eps=hyper.get("clip_eps", 0.2), lr=hyper.get("lr", 3e-4),
            entropy_coef=hyper.get("entropy_coef", 0.01),
            update_epochs=hyper.get("update_epochs", 4),
            reward_mode=reward_mode, success_threshold=success_threshold)
        trainer = GRPOLatentTrainer(policy, world_model, cfg)
    elif algo == "ppo":
        value_net = ValueNet(latent_dim, hidden_dim=hyper.get("policy_hidden", 64))
        # Match PPO's rollout budget to GRPO's (num_prompts * group_size trajectories).
        num_traj = hyper.get("num_prompts", 16) * hyper.get("group_size", 8)
        cfg = PPOConfig(
            num_trajectories=num_traj, horizon=horizon, gamma=hyper.get("gamma", 0.99),
            gae_lambda=hyper.get("gae_lambda", 0.95), clip_eps=hyper.get("clip_eps", 0.2),
            lr=hyper.get("lr", 3e-4), value_coef=hyper.get("value_coef", 0.5),
            entropy_coef=hyper.get("entropy_coef", 0.01),
            update_epochs=hyper.get("update_epochs", 4),
            reward_mode=reward_mode, success_threshold=success_threshold)
        trainer = PPOLatentTrainer(policy, value_net, world_model, cfg)
    else:
        raise ValueError(algo)

    return trainer.train(env, num_iterations, eval_every=eval_every,
                         eval_episodes=eval_episodes, seed=seed, log_fn=log_fn)


def run_comparison(env, world_model, num_iterations: int, seeds: List[int],
                   hyper: dict, algos=("grpo", "ppo"), eval_every: int = 10,
                   eval_episodes: int = 15,
                   log_fn: Optional[Callable[[str], None]] = None) -> Dict[str, List[Dict]]:
    """Run every algorithm over every seed. Returns ``{algo: [history_per_seed]}``."""
    results: Dict[str, List[Dict]] = {a: [] for a in algos}
    for algo in algos:
        for seed in seeds:
            if log_fn:
                log_fn(f"  running {algo.upper()} seed={seed} "
                       f"({env.cfg.reward_mode}, {num_iterations} iters)")
            hist = run_algorithm(algo, env, world_model, num_iterations, seed, hyper,
                                 eval_every=eval_every, eval_episodes=eval_episodes,
                                 log_fn=log_fn)
            results[algo].append(hist)
    return results


def aggregate_eval_curves(histories: List[Dict], key: str = "eval_proximity"):
    """Stack a per-seed evaluation curve into ``(iterations, mean, std)`` arrays."""
    iters = np.asarray(histories[0]["eval_iteration"], dtype=np.float64)
    stacked = np.asarray([h[key] for h in histories], dtype=np.float64)  # [seeds, points]
    return iters, stacked.mean(axis=0), stacked.std(axis=0)
