"""Iterative / Online DPO — and its comparison against Direct DPO.

Direct DPO trains once on a *fixed* dataset sampled from the reference policy
``pi_ref``. As ``pi_theta`` moves away from ``pi_ref`` during training, that data
becomes increasingly **off-policy**: the responses the policy would now generate
are under-represented in the training set, importance weights fan out, and the
effective sample size collapses (see ``analysis/off_policy_shift.py``).

Iterative DPO breaks the run into rounds and *re-samples on-policy* between them:

    Round 1: sample responses from pi_ref  -> build pairs -> DPO -> pi_1
    Round 2: sample responses from pi_1    -> build pairs -> DPO -> pi_2
    Round 3: sample responses from pi_2    -> build pairs -> DPO -> pi_3

Each round's data is drawn from the current policy, so the importance weights
re-anchor to ~1 at every round boundary and the ESS resets. The cost is repeated
generation + reward scoring.

This module runs the comparison on the tabular engine (CPU, seconds) so the
dynamics are reproducible without a GPU, and documents the real-model loop in
:func:`real_model_iterative_loop` for the ``trl`` path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils.toy_engine import ToyPreferenceEnv, TabularPolicy, train_tabular_dpo, TrainHistory


# --------------------------------------------------------------------------- #
# on-policy candidate construction                                             #
# --------------------------------------------------------------------------- #
def sample_on_policy_order(policy_probs, rm_scores, list_size, rng):
    """For each prompt: draw ``list_size`` candidates ~ policy, sort by RM score.

    This is the tabular analogue of "generate N responses from the current policy,
    then rank them with the reward model". As the policy sharpens onto good
    responses, later rounds increasingly compare *good* candidates to each other.
    """
    P, k = policy_probs.shape
    n = min(list_size, k)
    order = np.empty((P, n), dtype=np.int64)
    for p in range(P):
        cand = rng.choice(k, size=n, replace=False, p=policy_probs[p])
        cand = cand[np.argsort(-rm_scores[p, cand])]     # best -> worst by RM
        order[p] = cand
    return order


# --------------------------------------------------------------------------- #
# Direct DPO                                                                    #
# --------------------------------------------------------------------------- #
def run_direct_dpo(env: ToyPreferenceEnv, stage: str = "weight", total_steps: int = 300,
                   beta: float = 0.1, lr: float = 0.5, list_size: int = 6,
                   seed: int = 0, eval_every: int = 10):
    """Single-shot DPO on data sampled once from the reference policy."""
    return train_tabular_dpo(
        env, stage=stage, beta=beta, lr=lr, steps=total_steps,
        list_size=list_size, behaviour_probs=env.ref_probs,
        seed=seed, eval_every=eval_every)


# --------------------------------------------------------------------------- #
# Iterative DPO                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class IterativeResult:
    policy: TabularPolicy
    history: TrainHistory
    round_boundaries: list = field(default_factory=list)   # step index at each round start
    round_summaries: list = field(default_factory=list)    # per-round end metrics


def run_iterative_dpo(env: ToyPreferenceEnv, stage: str = "weight", n_rounds: int = 3,
                      steps_per_round: int = 100, beta: float = 0.1, lr: float = 0.5,
                      list_size: int = 6, seed: int = 0, eval_every: int = 10):
    """Multi-round DPO with on-policy resampling between rounds."""
    rng = np.random.default_rng(seed)
    merged = TrainHistory()
    boundaries, summaries = [], []
    theta = env.ref_logits.copy()
    policy = TabularPolicy(env.ref_logits)

    for r in range(n_rounds):
        policy.theta = theta
        # behaviour = current policy; data = on-policy samples ranked by RM
        behaviour = policy.probs.copy()
        order = sample_on_policy_order(behaviour, env.rm_scores, list_size, rng)
        boundaries.append(r * steps_per_round)

        policy, hist = train_tabular_dpo(
            env, stage=stage, beta=beta, lr=lr, steps=steps_per_round,
            list_size=list_size, behaviour_probs=behaviour, init_theta=theta,
            order=order, start_step=r * steps_per_round, seed=seed + r,
            eval_every=eval_every)
        theta = policy.theta.copy()

        for k in ("step", "loss", "kl", "reward", "win_rate", "ess"):
            getattr(merged, k).extend(getattr(hist, k))
        summaries.append({
            "round": r + 1,
            "kl": hist.kl[-1], "reward": hist.reward[-1],
            "win_rate": hist.win_rate[-1], "ess": hist.ess[-1],
        })

    return IterativeResult(policy, merged, boundaries, summaries)


# --------------------------------------------------------------------------- #
# real-model loop (documentation / trl path)                                   #
# --------------------------------------------------------------------------- #
def real_model_iterative_loop(base_model_name="Qwen/Qwen2.5-0.5B",
                              reward_model_name="OpenAssistant/reward-model-deberta-v3-large-v2",
                              prompts=None, n_rounds=3, n_samples=6):
    """Reference implementation of iterative DPO on real HF models (needs a GPU).

    Pseudocode of the loop this project's tabular engine emulates::

        policy = load(base_model_name); ref = frozen_copy(policy)
        rm = load(reward_model_name)
        for r in range(n_rounds):
            gens = {x: policy.generate(x, num_return_sequences=n_samples) for x in prompts}
            scores = {x: rm.score(x, gens[x]) for x in prompts}
            pairs  = build_pairs_from_ranking(gens, scores)   # data/build_preference_pairs
            pairs  = margin_filter(reward_filter(pairs))      # data/reward_filter
            trainer = WeightedDPOTrainer(policy, ref, pairs)  # or Rank/List trainer
            trainer.train()                                   # one round
            # ref stays frozen at the ORIGINAL SFT model (standard DPO KL anchor)
        return policy
    """
    raise NotImplementedError(
        "real_model_iterative_loop is a documented reference; run the tabular "
        "comparison via run_iterative_dpo / run_pipeline.py on CPU, or wire this "
        "up with trl + a GPU. See the docstring for the exact loop.")


__all__ = ["sample_on_policy_order", "run_direct_dpo", "run_iterative_dpo",
           "IterativeResult", "real_model_iterative_loop"]
