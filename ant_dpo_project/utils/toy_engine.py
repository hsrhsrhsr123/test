"""A dependency-light tabular DPO engine (numpy only).

Why this exists
---------------
Full DPO on a 0.5B model needs a GPU and hours. To *study the algorithms*
(three-stage curriculum, iterative vs direct, off-policy shift) we don't need a
language model at all — we need a policy over a response space where every
quantity of interest (KL, expected reward, importance weights, ESS) is
computable in closed form. This module provides exactly that: a **tabular
scale-model** of DPO that reproduces the same dynamics in seconds on CPU.

The mapping to the real thing is one-to-one:

    real DPO                              tabular scale-model
    ------------------------------------  ------------------------------------
    prompt x                              row p in {0 .. P-1}
    response y (a token sequence)         category index in {0 .. k-1}
    log pi_theta(y|x)  (sum of token lp)  log softmax(theta[p])[y]
    log pi_ref(y|x)                       log softmax(ref_logits[p])[y]
    reward model score r(x, y)            rm_score[p, y]  (= true reward + noise)
    on-policy sampling                    sampling categories ~ pi(.|p)

Because the policy is an explicit categorical, the loss *gradients* are written
analytically (no autodiff dependency). The loss math is identical to the torch
functions in ``dpo_losses.py`` — this is the numpy mirror referenced there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .metrics import (
    softmax,
    log_softmax,
    categorical_kl,
    effective_sample_size,
    expected_reward,
    win_rate,
)


# --------------------------------------------------------------------------- #
# environment                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class ToyPreferenceEnv:
    """A synthetic preference environment.

    Attributes
    ----------
    true_rewards : [P, k]   latent ground-truth quality of each candidate.
    rm_scores    : [P, k]   reward-model observations (= true_rewards + noise).
    ref_logits   : [P, k]   logits of the (frozen) reference / SFT policy.
    """

    true_rewards: np.ndarray
    rm_scores: np.ndarray
    ref_logits: np.ndarray

    @property
    def n_prompts(self) -> int:
        return self.true_rewards.shape[0]

    @property
    def n_candidates(self) -> int:
        return self.true_rewards.shape[1]

    @property
    def ref_probs(self) -> np.ndarray:
        return softmax(self.ref_logits, axis=-1)


def make_toy_env(
    n_prompts: int = 400,
    n_candidates: int = 8,
    reward_spread: float = 1.5,
    rm_noise: float = 0.35,
    ref_suboptimality: float = 0.8,
    seed: int = 0,
) -> ToyPreferenceEnv:
    """Build a synthetic environment.

    ``ref_suboptimality`` controls how *misaligned* the reference policy is with
    true reward: 0 => the SFT policy already puts its mass on the best responses
    (nothing to learn); large => the SFT policy is nearly uniform / anti-correlated
    (lots of headroom for DPO). A positive-but-imperfect value is the realistic
    regime and creates the off-policy pressure we want to study.
    """
    rng = np.random.default_rng(seed)
    true_rewards = rng.normal(0.0, reward_spread, size=(n_prompts, n_candidates))
    # reward model = true reward corrupted by observation noise.
    rm_scores = true_rewards + rng.normal(0.0, rm_noise, size=true_rewards.shape)
    # reference policy: partially aligned with true reward, plus its own idiosyncrasy.
    ref_logits = ref_suboptimality * true_rewards + rng.normal(
        0.0, 1.0, size=true_rewards.shape
    )
    return ToyPreferenceEnv(true_rewards, rm_scores, ref_logits)


# --------------------------------------------------------------------------- #
# policy                                                                        #
# --------------------------------------------------------------------------- #
class TabularPolicy:
    """A per-prompt categorical policy, initialised at the reference."""

    def __init__(self, ref_logits: np.ndarray):
        self.ref_logits = ref_logits.copy()
        self.theta = ref_logits.copy()  # start pi_theta = pi_ref

    @property
    def probs(self) -> np.ndarray:
        return softmax(self.theta, axis=-1)

    @property
    def logprobs(self) -> np.ndarray:
        return log_softmax(self.theta, axis=-1)

    @property
    def ref_logprobs(self) -> np.ndarray:
        return log_softmax(self.ref_logits, axis=-1)

    @property
    def ref_probs(self) -> np.ndarray:
        return softmax(self.ref_logits, axis=-1)

    def implicit_reward(self, beta: float) -> np.ndarray:
        """h(y) = beta * (log pi_theta(y|x) - log pi_ref(y|x)),  shape [P, k]."""
        return beta * (self.logprobs - self.ref_logprobs)

    def sample(self, rng: np.random.Generator, n_per_prompt: int = 1) -> np.ndarray:
        """Draw category indices ~ pi(.|p).  Returns [P, n_per_prompt]."""
        probs = self.probs
        P, k = probs.shape
        out = np.empty((P, n_per_prompt), dtype=np.int64)
        for p in range(P):
            out[p] = rng.choice(k, size=n_per_prompt, p=probs[p])
        return out


# --------------------------------------------------------------------------- #
# analytic gradients:   dL/dtheta[p]  from  dL/dm[p]                            #
# --------------------------------------------------------------------------- #
def _backprop_through_softmax(dL_dm: np.ndarray, probs: np.ndarray, beta: float) -> np.ndarray:
    """Chain dL/dm (implicit-reward grad) back to the policy logits theta.

    m_i = beta*(log pi(y_i) - log pi_ref(y_i)),  log pi(y_i)=theta_i - logsumexp(theta).
    d log pi(y_i)/d theta_k = delta_ik - pi_k, and dm_i/d log pi(y_i) = beta.

        dL/dtheta_k = beta * ( dL/dm_k  -  pi_k * sum_i dL/dm_i )

    Vectorised over prompts. ``dL_dm`` and ``probs`` are [P, k].
    """
    g = beta * dL_dm
    return g - probs * g.sum(axis=-1, keepdims=True)


# ---- Stage 0/1: (weighted) pairwise ---------------------------------------- #
def dpo_pair_value_grad(
    logprobs: np.ndarray,
    ref_logprobs: np.ndarray,
    chosen: np.ndarray,
    rejected: np.ndarray,
    weights: Optional[np.ndarray],
    beta: float,
):
    """Value and dL/dm for a batch of single pairs (one per prompt).

    ``chosen`` / ``rejected`` are [P] category indices. Returns (loss_scalar,
    dL_dm [P, k]).
    """
    P, k = logprobs.shape
    rows = np.arange(P)
    m = beta * (logprobs - ref_logprobs)  # [P, k]
    m_c = m[rows, chosen]
    m_r = m[rows, rejected]
    z = m_c - m_r
    # numerically stable log-sigmoid
    loss_per = -_log_sigmoid(z)
    if weights is not None:
        loss_per = weights * loss_per
    loss = loss_per.mean()

    s = _sigmoid(z)                # sigma(m_c - m_r)
    coeff = -(1.0 - s)             # dL/dz  = -(1 - sigma(z))
    if weights is not None:
        coeff = weights * coeff
    dL_dm = np.zeros_like(m)
    dL_dm[rows, chosen] += coeff
    dL_dm[rows, rejected] -= coeff
    dL_dm /= P                     # because loss is a mean over prompts
    return loss, dL_dm


# ---- Stage 2: rank (all pairs from a sorted list) -------------------------- #
def rank_value_grad(
    logprobs: np.ndarray,
    ref_logprobs: np.ndarray,
    order: np.ndarray,
    beta: float,
    position_discount: float = 1.0,
):
    """Value and dL/dm for position-weighted all-pairs ranking loss.

    ``order`` is [P, n] giving, for each prompt, candidate indices sorted
    best -> worst by reward-model score.
    """
    P, n = order.shape
    m_full = beta * (logprobs - ref_logprobs)          # [P, k]
    dL_dm = np.zeros_like(m_full)
    total_loss = 0.0

    i = np.arange(n)
    disc = position_discount ** i                       # [n]
    # weight for pair (i<j): disc[i] * (j - i)
    W = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            W[a, b] = disc[a] * (b - a)
    Wsum = W.sum()

    for p in range(P):
        idx = order[p]
        m = m_full[p, idx]                              # [n], rank order
        # logits[a,b] = m[a] - m[b]
        diff = m[:, None] - m[None, :]
        s = _sigmoid(diff)
        loss_mat = -_log_sigmoid(diff)
        total_loss += (W * loss_mat).sum() / Wsum
        # dloss/dm[a] from pair (a,b): -W*(1-s[a,b]); dloss/dm[b]: +W*(1-s)
        coeff = -(1.0 - s) * W                          # only upper triangle nonzero
        grad_m = coeff.sum(axis=1) - coeff.sum(axis=0)  # sum over partners
        grad_m /= Wsum
        dL_dm[p, idx] += grad_m

    total_loss /= P
    dL_dm /= P
    return total_loss, dL_dm


# ---- Stage 3: listwise (Plackett-Luce / ListMLE) --------------------------- #
def list_value_grad(
    logprobs: np.ndarray,
    ref_logprobs: np.ndarray,
    order: np.ndarray,
    beta: float,
):
    """Value and dL/dm for the ListMLE (Plackett-Luce) listwise loss.

    ``order`` is [P, n], best -> worst by reward-model score.
    """
    P, n = order.shape
    m_full = beta * (logprobs - ref_logprobs)
    dL_dm = np.zeros_like(m_full)
    total_loss = 0.0

    for p in range(P):
        idx = order[p]
        f = m_full[p, idx]                              # [n], rank order (best first)
        # suffix logsumexp S_i = logsumexp(f[i:])
        S = _reverse_logcumsumexp(f)                    # [n]
        total_loss += float(-(f - S).sum())
        # P(k | i) = exp(f_k - S_i) for k >= i, gathered into [n,n] lower-in-suffix
        # dL/df_k = sum_{i<=k} exp(f_k - S_i) - 1
        grad_f = np.empty(n)
        # precompute exp(f_k - S_i) only for i<=k
        for kk in range(n):
            contrib = np.exp(f[kk] - S[: kk + 1]).sum()
            grad_f[kk] = contrib - 1.0
        dL_dm[p, idx] += grad_f

    total_loss /= P
    dL_dm /= P
    return total_loss, dL_dm


# --------------------------------------------------------------------------- #
# training loop                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class TrainHistory:
    step: list = field(default_factory=list)
    loss: list = field(default_factory=list)
    kl: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    win_rate: list = field(default_factory=list)
    ess: list = field(default_factory=list)          # off-policy ESS vs behaviour policy


def _evaluate(policy: TabularPolicy, env: ToyPreferenceEnv, behaviour_probs: np.ndarray):
    pi = policy.probs
    kl = categorical_kl(pi, policy.ref_probs).mean()
    rew = expected_reward(pi, env.true_rewards)
    wr = win_rate(pi, policy.ref_probs, env.true_rewards)
    # ESS of importance weights pi/behaviour over the full candidate support,
    # weighted by how often behaviour visits each candidate.
    ratio = pi / np.clip(behaviour_probs, 1e-12, None)
    w = (behaviour_probs * ratio).reshape(-1)         # this is just pi flattened;
    # ESS must reflect variance of the *ratio* under the behaviour dist:
    ess = _weighted_ess(ratio, behaviour_probs)
    return float(kl), float(rew), float(wr), float(ess)


def _weighted_ess(ratio: np.ndarray, behaviour_probs: np.ndarray) -> float:
    """ESS proxy = 1 / E_mu[w^2] * (E_mu[w])^2, over the categorical support.

    Under behaviour mu, E_mu[w] = 1 and E_mu[w^2] = sum_y mu(y) (pi(y)/mu(y))^2 =
    sum_y pi(y)^2/mu(y). Normalised to [0,1] by averaging per-prompt.
    """
    mu = np.clip(behaviour_probs, 1e-12, None)
    second_moment = (mu * ratio ** 2).sum(axis=-1)     # E_mu[w^2] per prompt
    ess_frac = 1.0 / np.clip(second_moment, 1e-12, None)
    return float(ess_frac.mean())


def train_tabular_dpo(
    env: ToyPreferenceEnv,
    stage: str = "weight",
    beta: float = 0.1,
    lr: float = 0.5,
    steps: int = 300,
    weight_temperature: float = 1.0,
    position_discount: float = 0.9,
    list_size: int = 6,
    behaviour_probs: Optional[np.ndarray] = None,
    init_theta: Optional[np.ndarray] = None,
    order: Optional[np.ndarray] = None,
    start_step: int = 0,
    seed: int = 0,
    eval_every: int = 10,
) -> tuple[TabularPolicy, TrainHistory]:
    """Train a tabular policy with one of the DPO variants.

    ``stage`` in {"dpo", "weight", "rank", "list"}. ``behaviour_probs`` is the
    distribution the preference data was sampled from (defaults to the reference,
    i.e. the Direct-DPO setting); iterative DPO passes the current policy here.
    ``order`` optionally overrides the per-prompt candidate ranking (used by
    iterative DPO to feed round-specific on-policy samples); ``start_step`` offsets
    the recorded step index so multi-round runs concatenate cleanly.
    """
    rng = np.random.default_rng(seed)
    policy = TabularPolicy(env.ref_logits)
    if init_theta is not None:
        policy.theta = init_theta.copy()
    if behaviour_probs is None:
        behaviour_probs = env.ref_probs

    P, k = env.true_rewards.shape
    if order is None:
        order = np.argsort(-env.rm_scores, axis=1)      # best -> worst by RM score
        n = min(list_size, k)
        order = order[:, :n]
    else:
        n = order.shape[1]

    # pairwise data: chosen = rank-0, rejected sampled from the rest (by RM order)
    chosen = order[:, 0]
    rejected = order[:, -1]
    # reward-gap confidence weights (batch softmax over RM margins)
    rm_gap = env.rm_scores[np.arange(P), chosen] - env.rm_scores[np.arange(P), rejected]
    w = softmax((rm_gap / weight_temperature)[None, :], axis=-1)[0] * P

    history = TrainHistory()
    for step in range(steps + 1):
        lp = policy.logprobs
        rlp = policy.ref_logprobs
        if stage == "dpo":
            loss, dL_dm = dpo_pair_value_grad(lp, rlp, chosen, rejected, None, beta)
        elif stage == "weight":
            loss, dL_dm = dpo_pair_value_grad(lp, rlp, chosen, rejected, w, beta)
        elif stage == "rank":
            loss, dL_dm = rank_value_grad(lp, rlp, order, beta, position_discount)
        elif stage == "list":
            loss, dL_dm = list_value_grad(lp, rlp, order, beta)
        else:
            raise ValueError(f"unknown stage {stage!r}")

        grad_theta = _backprop_through_softmax(dL_dm, policy.probs, beta)
        # The losses use a mean-over-prompts reduction, but each prompt has its own
        # independent policy row theta[p], so that 1/P factor would dilute every
        # per-prompt update P-fold. Rescale by P to recover full-strength, P-independent
        # per-prompt gradient descent (lr is then the true per-prompt step size).
        policy.theta -= lr * P * grad_theta

        if step % eval_every == 0:
            kl, rew, wr, ess = _evaluate(policy, env, behaviour_probs)
            history.step.append(step + start_step)
            history.loss.append(float(loss))
            history.kl.append(kl)
            history.reward.append(rew)
            history.win_rate.append(wr)
            history.ess.append(ess)

    return policy, history


# --------------------------------------------------------------------------- #
# numerically stable primitives                                                #
# --------------------------------------------------------------------------- #
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def _reverse_logcumsumexp(f: np.ndarray) -> np.ndarray:
    """S_i = log sum_{j>=i} exp(f_j), computed stably."""
    n = len(f)
    S = np.empty(n)
    running = -np.inf
    for i in range(n - 1, -1, -1):
        running = np.logaddexp(running, f[i])
        S[i] = running
    return S


__all__ = [
    "ToyPreferenceEnv",
    "make_toy_env",
    "TabularPolicy",
    "TrainHistory",
    "train_tabular_dpo",
    "dpo_pair_value_grad",
    "rank_value_grad",
    "list_value_grad",
]
