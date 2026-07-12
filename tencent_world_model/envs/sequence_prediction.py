"""Sequence-prediction / controllable-dynamics environment.

This is the primary environment for the project.  It is a *controllable
non-linear dynamical system*: an agent applies discrete control actions to
steer a low-dimensional latent state towards a goal region.  Crucially, the
agent never observes the true low-dimensional state directly -- it only sees a
*high-dimensional, noisy projection* of it.  That mismatch is what motivates
learning a compact latent world model and doing policy optimisation in the
learned latent space (dimensionality reduction + denoising).

Design goals
------------
1. **Actions matter.**  Unlike a pure supervised "predict the next token"
   task, the control action genuinely changes the state trajectory, so it is a
   meaningful RL problem for comparing GRPO vs PPO.
2. **Latent structure.**  The true dynamics live in ``state_dim`` dimensions
   but observations live in ``obs_dim > state_dim`` dimensions with additive
   noise, so a good encoder is rewarded.
3. **A single knob for reward sparsity.**  ``reward_mode`` selects how the
   underlying per-state proximity signal is *delivered in time*:

   ``dense``        proximity reward every step;
   ``sparse``       proximity reward only on the final step;
   ``very_sparse``  a single ``+1`` the first time the goal region is reached.

   The *underlying* quantity (proximity to the goal) is identical across
   modes -- only the temporal delivery changes.  This isolates the
   credit-assignment difficulty (which hurts a bootstrapped critic) from the
   difficulty of learning the reward *function* (which does not).

The environment follows a minimal Gym-like API (``reset`` / ``step``) but is
self-contained and dependency-free so it is trivial to batch and reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SequencePredictionConfig:
    state_dim: int = 4          # dimensionality of the true (hidden) state
    obs_dim: int = 48           # dimensionality of the observation (>> state_dim)
    num_actions: int = 5        # number of discrete control actions
    horizon: int = 30           # episode length
    obs_noise: float = 0.12     # std of observation noise (motivates denoising)
    process_noise: float = 0.02  # std of process (dynamics) noise
    reward_mode: str = "dense"  # {'dense','sparse','very_sparse'}
    success_threshold: float = 0.6  # proximity above which the goal counts as reached
    proximity_scale: float = 0.8    # length-scale of the proximity kernel
    nonlinearity: float = 0.4   # strength of the non-linear term in the dynamics
    seed: Optional[int] = 0


class SequencePredictionEnv:
    """A controllable non-linear dynamical system with configurable reward sparsity."""

    def __init__(self, config: Optional[SequencePredictionConfig] = None, **kwargs):
        if config is None:
            config = SequencePredictionConfig(**kwargs)
        self.cfg = config
        assert self.cfg.reward_mode in ("dense", "sparse", "very_sparse")
        assert self.cfg.obs_dim >= self.cfg.state_dim

        self._rng = np.random.default_rng(self.cfg.seed)

        # --- Fixed structural parameters of the environment -------------------
        # These are drawn once (with a *separate* deterministic RNG) so that the
        # dynamics are identical regardless of the episode seed.  This makes the
        # world model learnable: the environment is a single fixed system.
        struct_rng = np.random.default_rng(12345)
        d = self.cfg.state_dim
        a = self.cfg.num_actions

        # Linear state-transition matrix, slightly contractive & stable.
        A = struct_rng.normal(0, 1, size=(d, d))
        # Normalise spectral radius to be < 1 for stability.
        eigmax = np.max(np.abs(np.linalg.eigvals(A)))
        self.A = 0.9 * A / (eigmax + 1e-8)

        # Each discrete action maps to a control vector in state space.
        self.control_vectors = struct_rng.normal(0, 0.6, size=(a, d))

        # Random projection observation matrix (state -> observation).
        W = struct_rng.normal(0, 1, size=(self.cfg.obs_dim, d))
        # Orthonormalise columns so information is preserved.
        q, _ = np.linalg.qr(W)
        self.W = q[:, :d] if q.shape[1] >= d else W / np.linalg.norm(W, axis=0, keepdims=True)

        # A fixed goal in state space.
        self.goal = struct_rng.normal(0, 0.5, size=(d,))

        # Non-linear "attractor" centres that bend the flow of the dynamics.
        self.attractors = struct_rng.normal(0, 1.0, size=(3, d))

        self.state = np.zeros(d, dtype=np.float64)
        self.t = 0
        self._reached_goal = False

        # Spaces (kept simple, Gym-compatible attributes).
        self.observation_dim = self.cfg.obs_dim
        self.action_dim = self.cfg.num_actions

    # ------------------------------------------------------------------ core
    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None, near_goal: bool = False):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if near_goal:
            # Start near the goal -- used during data collection so the reward
            # head observes the full proximity range (see collect_transitions).
            self.state = self.goal + self._rng.normal(0, 0.4, size=(self.cfg.state_dim,))
        else:
            # Start away from the goal so there is something to solve.
            self.state = self._rng.normal(0, 1.0, size=(self.cfg.state_dim,))
        self.t = 0
        self._reached_goal = False
        return self._observe(), {"proximity": self._proximity(self.state)}

    def _nonlinear_term(self, s: np.ndarray) -> np.ndarray:
        """A smooth non-linear bending of the flow toward/around attractors."""
        out = np.zeros_like(s)
        for c in self.attractors:
            diff = c - s
            dist2 = np.sum(diff ** 2)
            out += diff * np.exp(-dist2) * np.tanh(np.sum(s))
        return self.cfg.nonlinearity * out

    def _dynamics(self, s: np.ndarray, action: int) -> np.ndarray:
        control = self.control_vectors[action]
        s_next = self.A @ s + control + self._nonlinear_term(s)
        s_next = s_next + self._rng.normal(0, self.cfg.process_noise, size=s.shape)
        # Keep the state bounded so trajectories don't explode.
        return np.clip(s_next, -5.0, 5.0)

    def _clean_observe(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Noise-free observation (the true underlying signal W @ state)."""
        s = self.state if state is None else state
        return (self.W @ s).astype(np.float32)

    def _observe(self) -> np.ndarray:
        obs = self.W @ self.state
        obs = obs + self._rng.normal(0, self.cfg.obs_noise, size=obs.shape)
        return obs.astype(np.float32)

    def _proximity(self, s: np.ndarray) -> float:
        """Per-state proximity to the goal in [0, 1] (1 == at the goal)."""
        dist2 = np.sum((s - self.goal) ** 2)
        return float(np.exp(-dist2 / (2 * self.cfg.proximity_scale ** 2)))

    def step(self, action: int):
        assert 0 <= action < self.cfg.num_actions
        self.state = self._dynamics(self.state, action)
        self.t += 1

        proximity = self._proximity(self.state)
        newly_reached = (not self._reached_goal) and (proximity >= self.cfg.success_threshold)
        if proximity >= self.cfg.success_threshold:
            self._reached_goal = True

        done = self.t >= self.cfg.horizon
        reward = self._deliver_reward(proximity, newly_reached, done)

        info = {
            "proximity": proximity,
            "reached_goal": self._reached_goal,
            "true_state": self.state.copy(),
        }
        return self._observe(), reward, done, False, info

    def _deliver_reward(self, proximity: float, newly_reached: bool, done: bool) -> float:
        """Convert the underlying proximity signal into a mode-specific reward."""
        mode = self.cfg.reward_mode
        if mode == "dense":
            return proximity
        if mode == "sparse":
            # Only the terminal step carries the (scaled) proximity signal.
            return proximity * self.cfg.horizon if done else 0.0
        if mode == "very_sparse":
            return 1.0 if newly_reached else 0.0
        raise ValueError(mode)

    # -------------------------------------------------------------- rollouts
    def rollout(self, policy_fn, seed: Optional[int] = None):
        """Roll out ``policy_fn(obs) -> action`` for one episode in the *real* env.

        Returns a dict of trajectory arrays plus mode-agnostic evaluation
        metrics (mean proximity, success rate) that are comparable across
        reward modes.
        """
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
            "episodic_return": float(np.sum(rewards)),      # mode-specific
            "mean_proximity": float(np.mean(proximities)),  # mode-agnostic
            "success": float(np.max(proximities) >= self.cfg.success_threshold),
        }

    def evaluate_policy(self, policy_fn, episodes: int = 10, base_seed: int = 10_000):
        """Average mode-agnostic metrics over several episodes."""
        rets, proxs, succ = [], [], []
        for i in range(episodes):
            r = self.rollout(policy_fn, seed=base_seed + i)
            rets.append(r["episodic_return"])
            proxs.append(r["mean_proximity"])
            succ.append(r["success"])
        return {
            "episodic_return": float(np.mean(rets)),
            "mean_proximity": float(np.mean(proxs)),
            "success_rate": float(np.mean(succ)),
        }

    def get_long_range_accuracy(self, predict_fn, horizon: int = 50,
                                episodes: int = 20, base_seed: int = 20_000,
                                target: str = "clean"):
        """Long-range prediction accuracy of a model that predicts observations.

        ``predict_fn(obs0, action_seq) -> predicted_obs_seq`` returns the model's
        predicted observations for ``horizon`` steps given the *noisy* initial
        observation and a sequence of actions.  Predictions are scored against
        the **clean underlying signal** ``W @ true_state`` (``target='clean'``,
        the default) -- i.e. how well the model recovers the true trajectory,
        which fairly rewards a denoising latent model -- or against the raw
        noisy observations (``target='noisy'``).  Reports ``1 - normalised-RMSE``
        aggregated over the horizon (higher is better).
        """
        errs = []
        base_norms = []
        for i in range(episodes):
            obs, _ = self.reset(seed=base_seed + i)
            action_seq = self._rng.integers(0, self.cfg.num_actions, size=horizon)
            target_obs = []
            o = obs
            for a in action_seq:
                o, _, done, _, info = self.step(int(a))
                if target == "clean":
                    target_obs.append(self._clean_observe(info["true_state"]))
                else:
                    target_obs.append(o)
                if done:
                    break
            target_obs = np.asarray(target_obs, dtype=np.float32)
            h = target_obs.shape[0]
            pred_obs = np.asarray(predict_fn(obs, action_seq[:h]), dtype=np.float32)
            pred_obs = pred_obs[:h]
            errs.append(np.mean((pred_obs - target_obs) ** 2))
            base_norms.append(np.mean(target_obs ** 2))
        rmse = float(np.sqrt(np.mean(errs)))
        base = float(np.sqrt(np.mean(base_norms))) + 1e-8
        # Normalised accuracy in [0, 1]: 1 means perfect, 0 means as-bad-as
        # predicting zeros.
        accuracy = max(0.0, 1.0 - rmse / base)
        return {"accuracy": accuracy, "rmse": rmse, "baseline_norm": base}


def collect_transitions(env: SequencePredictionEnv, num_transitions: int,
                        policy_fn=None, seed: int = 0, goal_biased_fraction: float = 0.5):
    """Collect ``(obs, action, next_obs, reward, proximity)`` transitions.

    Uses a random policy by default (good for world-model pre-training because
    it covers the state space broadly).  A fraction of episodes are reset
    *near the goal* so the reward head sees the full proximity range -- this
    plays the role of the "partially trained policy" data in the plan and is
    what makes the reward model usable for sparse-reward latent rollouts.
    Returns numpy arrays.
    """
    rng = np.random.default_rng(seed)
    obs_buf, act_buf, nobs_buf, rew_buf, prox_buf, next_prox_buf, done_buf = [], [], [], [], [], [], []
    near = rng.random() < goal_biased_fraction
    obs, info = env.reset(seed=seed, near_goal=near)
    cur_prox = info["proximity"]  # proximity of the state that ``obs`` represents
    collected = 0
    while collected < num_transitions:
        if policy_fn is None:
            action = int(rng.integers(0, env.cfg.num_actions))
        else:
            action = int(policy_fn(obs))
        nobs, reward, done, _, info = env.step(action)
        obs_buf.append(obs)
        act_buf.append(action)
        nobs_buf.append(nobs)
        rew_buf.append(reward)
        prox_buf.append(cur_prox)             # proximity aligned to ``obs`` (s_t)
        next_prox_buf.append(info["proximity"])  # proximity aligned to ``nobs`` (s_{t+1})
        done_buf.append(float(done))
        obs = nobs
        cur_prox = info["proximity"]
        collected += 1
        if done:
            near = rng.random() < goal_biased_fraction
            obs, info = env.reset(seed=seed + collected, near_goal=near)
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
