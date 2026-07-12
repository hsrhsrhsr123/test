"""Numpy metrics for preference-optimization experiments.

These are deliberately framework-free (numpy only) so the analysis and the
CPU toy engine can share them without importing torch. Everything is defined on
*categorical* distributions over a finite candidate set per prompt, which is the
setting of the tabular scale-model in ``toy_engine.py``. The formulas are the
standard ones and transfer directly to the token-level HF setting.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# distribution helpers                                                         #
# --------------------------------------------------------------------------- #
def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


# --------------------------------------------------------------------------- #
# KL divergence                                                                #
# --------------------------------------------------------------------------- #
def categorical_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """KL(p || q) for categorical rows. Returns per-row KL (shape [...,] ).

    ``p`` and ``q`` are probability arrays whose last axis is the category axis.
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return (p * (np.log(p) - np.log(q))).sum(axis=-1)


# --------------------------------------------------------------------------- #
# off-policy diagnostics                                                       #
# --------------------------------------------------------------------------- #
def importance_weights(
    pi_probs: np.ndarray,
    mu_probs: np.ndarray,
    samples: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """w = pi(y) / mu(y) for y drawn from the *behaviour* policy mu.

    ``pi_probs`` / ``mu_probs`` are [P, k] categorical distributions per prompt,
    ``samples`` is an integer array [N] of category indices with a companion
    prompt-index; here we assume ``samples`` indexes into a flattened
    (prompt, category) selection provided by the caller as a tuple. For the
    common per-prompt case use :func:`importance_weights_per_prompt`.
    """
    raise NotImplementedError("use importance_weights_per_prompt")


def importance_weights_per_prompt(
    pi_probs: np.ndarray,
    mu_probs: np.ndarray,
    prompt_idx: np.ndarray,
    sample_idx: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Importance ratios pi(y|x)/mu(y|x) for a batch of sampled (x, y).

    ``prompt_idx`` and ``sample_idx`` are integer arrays of equal length N giving
    the prompt row and the sampled category for each drawn example.
    """
    pi = pi_probs[prompt_idx, sample_idx]
    mu = mu_probs[prompt_idx, sample_idx]
    return pi / np.clip(mu, eps, None)


def effective_sample_size(weights: np.ndarray, eps: float = 1e-12) -> float:
    """Kish effective sample size:  ESS = (sum w)^2 / sum(w^2).

    Ranges in [1, N]. A value near N means the importance weights are ~uniform
    (pi close to mu). A collapse toward 1 means a few samples dominate — the
    hallmark of off-policy distribution shift.
    """
    w = np.asarray(weights, dtype=np.float64)
    denom = (w ** 2).sum()
    if denom < eps:
        return 0.0
    return float((w.sum() ** 2) / denom)


def normalized_ess(weights: np.ndarray) -> float:
    """ESS as a fraction of N, in [0, 1]."""
    n = len(weights)
    if n == 0:
        return 0.0
    return effective_sample_size(weights) / n


# --------------------------------------------------------------------------- #
# reward / quality metrics                                                     #
# --------------------------------------------------------------------------- #
def expected_reward(policy_probs: np.ndarray, true_rewards: np.ndarray) -> float:
    """E_{y ~ pi(.|x)}[ true_reward(x, y) ], averaged over prompts."""
    return float((policy_probs * true_rewards).sum(axis=-1).mean())


def win_rate(
    policy_probs: np.ndarray,
    baseline_probs: np.ndarray,
    true_rewards: np.ndarray,
) -> float:
    """P( reward(y_pi) > reward(y_base) ) with ties counted as 0.5.

    Computed *exactly* from the categorical distributions (no sampling): for each
    prompt we form the outer product of the two response distributions and sum the
    mass where the policy's response has strictly higher true reward, plus half
    the tie mass. Averaged over prompts.
    """
    P, k = policy_probs.shape
    wins = np.zeros(P)
    for p in range(P):
        r = true_rewards[p]
        gt = (r[:, None] > r[None, :]).astype(np.float64)   # [k_pi, k_base]
        eq = (r[:, None] == r[None, :]).astype(np.float64)
        joint = policy_probs[p][:, None] * baseline_probs[p][None, :]
        wins[p] = (joint * (gt + 0.5 * eq)).sum()
    return float(wins.mean())


def reward_margin_stats(reward_chosen: np.ndarray, reward_rejected: np.ndarray) -> dict:
    """Summary of the chosen-minus-rejected reward gap distribution."""
    margin = reward_chosen - reward_rejected
    return {
        "mean": float(margin.mean()),
        "std": float(margin.std()),
        "median": float(np.median(margin)),
        "p10": float(np.percentile(margin, 10)),
        "p90": float(np.percentile(margin, 90)),
        "frac_negative": float((margin < 0).mean()),
    }


__all__ = [
    "softmax",
    "log_softmax",
    "categorical_kl",
    "importance_weights_per_prompt",
    "effective_sample_size",
    "normalized_ess",
    "expected_reward",
    "win_rate",
    "reward_margin_stats",
]
