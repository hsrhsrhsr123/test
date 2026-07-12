"""CartPole environment wrapper (secondary / classic-control sanity check).

Wraps ``gymnasium``'s ``CartPole-v1`` so it exposes the *same* interface as
:class:`~envs.sequence_prediction.SequencePredictionEnv`.  In particular it
derives a per-state **proximity** signal (how centred & upright the pole is)
which the three reward modes are built on top of:

``dense``        proximity every step (close to the classic +1/step reward);
``sparse``       cumulative proximity only at the final step;
``very_sparse``  a single ``+1`` if the pole is balanced for the whole horizon.

The packaged experiments run on ``SequencePredictionEnv`` (fixed horizon, no
early termination -> clean pure-latent rollouts).  CartPole is provided as a
familiar classic-control cross-check for data collection and real-environment
evaluation; because it can terminate early, latent rollouts here treat the
episode as a fixed horizon (an approximation documented in the report).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    _HAS_GYM = True
except Exception:  # pragma: no cover - gymnasium is a listed dependency
    _HAS_GYM = False


@dataclass
class CartpoleLatentConfig:
    horizon: int = 200
    reward_mode: str = "dense"
    seed: Optional[int] = 0
    # thresholds used to build the proximity signal
    x_threshold: float = 2.4
    theta_threshold: float = 0.2095  # ~12 degrees in radians
    success_threshold: float = 0.5


class CartpoleLatentEnv:
    """Gymnasium CartPole with a proximity-based, sparsity-configurable reward."""

    def __init__(self, config: Optional[CartpoleLatentConfig] = None, **kwargs):
        if not _HAS_GYM:
            raise ImportError("gymnasium is required for CartpoleLatentEnv")
        if config is None:
            config = CartpoleLatentConfig(**kwargs)
        self.cfg = config
        assert self.cfg.reward_mode in ("dense", "sparse", "very_sparse")

        self._env = gym.make("CartPole-v1")
        self.observation_dim = int(self._env.observation_space.shape[0])
        self.action_dim = int(self._env.action_space.n)
        self.t = 0
        self._last_obs = None
        self._balanced_all = True

    def seed(self, seed: int) -> None:
        self.cfg.seed = seed

    def _proximity(self, obs: np.ndarray) -> float:
        x, _, theta, _ = obs
        px = max(0.0, 1.0 - abs(x) / self.cfg.x_threshold)
        pth = max(0.0, 1.0 - abs(theta) / self.cfg.theta_threshold)
        return float(px * pth)

    def reset(self, seed: Optional[int] = None):
        s = self.cfg.seed if seed is None else seed
        obs, info = self._env.reset(seed=s)
        self.t = 0
        self._balanced_all = True
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return self._last_obs, {"proximity": self._proximity(self._last_obs)}

    def step(self, action: int):
        obs, _, terminated, truncated, info = self._env.step(int(action))
        obs = np.asarray(obs, dtype=np.float32)
        self.t += 1
        proximity = self._proximity(obs)
        if proximity < self.cfg.success_threshold:
            self._balanced_all = False
        done = terminated or truncated or (self.t >= self.cfg.horizon)

        mode = self.cfg.reward_mode
        if mode == "dense":
            reward = proximity
        elif mode == "sparse":
            reward = proximity * self.cfg.horizon if done else 0.0
        else:  # very_sparse
            reward = 1.0 if (done and self._balanced_all and self.t >= self.cfg.horizon) else 0.0

        self._last_obs = obs
        info = {"proximity": proximity, "reached_goal": self._balanced_all,
                "true_state": obs.copy()}
        return obs, float(reward), bool(done), False, info

    def rollout(self, policy_fn, seed: Optional[int] = None):
        obs, info = self.reset(seed=seed)
        observations, actions, rewards, next_obs, proximities = [], [], [], [], []
        done = False
        while not done:
            action = int(policy_fn(obs))
            nobs, reward, done, _, info = self.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_obs.append(nobs)
            proximities.append(info["proximity"])
            obs = nobs
        proximities = np.asarray(proximities, dtype=np.float32)
        return {
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_observations": np.asarray(next_obs, dtype=np.float32),
            "proximities": proximities,
            "episodic_return": float(np.sum(rewards)),
            "mean_proximity": float(np.mean(proximities)),
            "success": float(len(proximities)),  # steps survived
        }

    def evaluate_policy(self, policy_fn, episodes: int = 10, base_seed: int = 10_000):
        rets, proxs, steps = [], [], []
        for i in range(episodes):
            r = self.rollout(policy_fn, seed=base_seed + i)
            rets.append(r["episodic_return"])
            proxs.append(r["mean_proximity"])
            steps.append(len(r["actions"]))
        return {
            "episodic_return": float(np.mean(rets)),
            "mean_proximity": float(np.mean(proxs)),
            "success_rate": float(np.mean(steps) / self.cfg.horizon),
        }


def collect_transitions(env: CartpoleLatentEnv, num_transitions: int,
                        policy_fn=None, seed: int = 0):
    rng = np.random.default_rng(seed)
    obs_buf, act_buf, nobs_buf, rew_buf, prox_buf, next_prox_buf, done_buf = [], [], [], [], [], [], []
    obs, info = env.reset(seed=seed)
    cur_prox = info["proximity"]
    collected = 0
    while collected < num_transitions:
        action = int(rng.integers(0, env.action_dim)) if policy_fn is None else int(policy_fn(obs))
        nobs, reward, done, _, info = env.step(action)
        obs_buf.append(obs)
        act_buf.append(action)
        nobs_buf.append(nobs)
        rew_buf.append(reward)
        prox_buf.append(cur_prox)
        next_prox_buf.append(info["proximity"])
        done_buf.append(float(done))
        obs = nobs
        cur_prox = info["proximity"]
        collected += 1
        if done:
            obs, info = env.reset(seed=seed + collected)
            cur_prox = info["proximity"]
    return {
        "observations": np.asarray(obs_buf, dtype=np.float32),
        "actions": np.asarray(act_buf, dtype=np.int64),
        "next_observations": np.asarray(nobs_buf, dtype=np.float32),
        "rewards": np.asarray(rew_buf, dtype=np.float32),
        "proximities": np.asarray(prox_buf, dtype=np.float32),
        "next_proximities": np.asarray(next_prox_buf, dtype=np.float32),
        "dones": np.asarray(done_buf, dtype=np.float32),
    }
