"""Core preference-optimization loss functions (sequence-level).

Every loss here is a *pure function of log-probabilities* — it takes
``log pi_theta(y|x)`` and ``log pi_ref(y|x)`` for a set of responses and returns
a scalar loss. That keeps the math in one place and lets it be reused by:

  * the ``trl``-based trainers in ``training/`` (real HF models, token-level
    log-probs summed to the sequence level), and
  * the dependency-light numpy engine in ``utils/toy_engine.py`` (which mirrors
    the *same* math for fast CPU experiments — see the note at the bottom).

The three training stages are a curriculum over *how much of the ranking signal
we expose to the loss*:

    Stage 1  Weighted-DPO : one pair, weighted by reward-model confidence
    Stage 2  Rank-DPO     : all C(n,2) pairs from a ranked list, position-weighted
    Stage 3  List-DPO     : the whole ranking as a single listwise (Plackett-Luce)
                            objective

All functions operate on the DPO "implicit reward"

    h_theta(y) = beta * ( log pi_theta(y|x) - log pi_ref(y|x) ).

Requires torch. The numpy mirror lives in ``toy_engine.py`` so that the demo
pipeline can run with numpy alone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Stage 0 / baseline: vanilla DPO                                             #
# --------------------------------------------------------------------------- #
def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """Standard Direct Preference Optimization loss (Rafailov et al., 2023).

    L = -E[ log sigma( beta * ( (logpi_w - logpi_ref_w) - (logpi_l - logpi_ref_l) ) ) ]

    All arguments are 1-D tensors of per-example *sequence* log-probabilities.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    losses = -F.logsigmoid(logits)
    return _reduce(losses, reduction)


# --------------------------------------------------------------------------- #
# Stage 1: Weighted-DPO                                                        #
# --------------------------------------------------------------------------- #
def weighted_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    weights: torch.Tensor,
    beta: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """DPO where each pair carries a confidence weight.

    ``weights`` is expected to be a non-negative tensor, normalised to sum to the
    batch size (so the effective learning rate matches vanilla DPO). See
    :func:`confidence_weights` for the reward-gap softmax weighting used in
    ``training/weight_dpo.py``.

    L = -E[ w_i * log sigma( beta * (...) ) ]
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    losses = -weights * F.logsigmoid(logits)
    return _reduce(losses, reduction)


def confidence_weights(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-pair weights = batch-softmax over the reward-model margin.

    A large ``reward_chosen - reward_rejected`` gap means the reward model is
    confident the preference is real, so that pair should dominate the update;
    tiny gaps are near-ties and get down-weighted. We softmax the margins across
    the batch and renormalise to sum to ``batch_size`` so that the *average*
    weight is 1 (keeps the effective step size comparable to unweighted DPO).
    """
    margins = (reward_chosen - reward_rejected) / temperature
    w = torch.softmax(margins, dim=0)
    return w * w.numel()


# --------------------------------------------------------------------------- #
# Stage 2: Rank-DPO (all pairs from a ranked list, position-weighted)         #
# --------------------------------------------------------------------------- #
def rank_dpo_loss(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    beta: float = 0.1,
    position_discount: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Pairwise ranking loss over a *sorted* list of responses.

    ``policy_logps`` / ``ref_logps`` are 1-D tensors of length ``n`` for a single
    prompt, already sorted best -> worst by reward-model score
    (index 0 is the best response).

    We form every ordered pair (i, j) with i < j (so response i should be
    preferred to response j) and sum the DPO pairwise losses. Pairs whose rank
    positions are far apart / near the top of the list are weighted more:

        w_ij = (position_discount ** i) * (rank_gap = j - i)

    ``position_discount = 1.0`` recovers uniform pair weights; values < 1 apply a
    DCG-style emphasis on the top of the ranking.
    """
    n = policy_logps.shape[0]
    if n < 2:
        return policy_logps.new_zeros(())

    h = beta * (policy_logps - ref_logps)  # implicit rewards, shape [n]
    idx = torch.arange(n, device=policy_logps.device)
    i, j = torch.meshgrid(idx, idx, indexing="ij")
    mask = i < j                                   # upper triangle: i preferred to j
    logits = h.unsqueeze(1) - h.unsqueeze(0)       # h_i - h_j
    pair_losses = -F.logsigmoid(logits)

    disc = position_discount ** i.to(h.dtype)
    rank_gap = (j - i).to(h.dtype)
    weights = disc * rank_gap
    weights = weights * mask

    total = (pair_losses * weights).sum()
    denom = weights.sum().clamp_min(1e-8)
    if reduction == "sum":
        return total
    return total / denom


# --------------------------------------------------------------------------- #
# Stage 3: List-DPO (listwise / Plackett-Luce, a.k.a. ListMLE)               #
# --------------------------------------------------------------------------- #
def list_dpo_loss(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    beta: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """Listwise DPO using the Plackett-Luce / ListMLE objective.

    ``policy_logps`` / ``ref_logps`` are length-``n`` tensors for one prompt,
    sorted best -> worst. Treating the implicit reward ``f(y) = beta * (logpi -
    logpi_ref)`` as a Plackett-Luce score, the probability of the *whole* observed
    ranking factorises as a product of softmaxes over shrinking suffixes:

        L_list = - sum_i [ f(y_i) - log sum_{j >= i} exp( f(y_j) ) ]

    Unlike Rank-DPO this optimises the consistency of the entire ordering jointly
    (each term normalises over all not-yet-placed responses), rather than treating
    the C(n,2) pairs as independent.
    """
    n = policy_logps.shape[0]
    if n < 2:
        return policy_logps.new_zeros(())

    f = beta * (policy_logps - ref_logps)          # [n], already in rank order
    # log sum_{j>=i} exp(f_j) via reverse cumulative logsumexp.
    f_rev = torch.flip(f, dims=[0])
    logcumsumexp_rev = torch.logcumsumexp(f_rev, dim=0)
    suffix_logsumexp = torch.flip(logcumsumexp_rev, dims=[0])  # [n]

    per_position = f - suffix_logsumexp            # log P(y_i placed at step i)
    losses = -per_position
    return _reduce(losses, reduction)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _reduce(losses: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    if reduction == "none":
        return losses
    raise ValueError(f"unknown reduction: {reduction!r}")


__all__ = [
    "dpo_loss",
    "weighted_dpo_loss",
    "confidence_weights",
    "rank_dpo_loss",
    "list_dpo_loss",
]
