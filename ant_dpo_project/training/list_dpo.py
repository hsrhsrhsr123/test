"""Stage 3 — List-DPO trainer.

Idea
----
Rank-DPO treats the C(n,2) pairs as *independent* constraints. List-DPO instead
optimises the whole ranking jointly with a Plackett-Luce / ListMLE objective:

    L_list = - sum_i [ f(y_i) - log sum_{j>=i} exp( f(y_j) ) ],
    f(y) = beta * ( log pi_theta(y|x) - log pi_ref(y|x) ),   list sorted best->worst.

Each term is the log-probability, under the PL model, of placing response ``y_i``
first among the not-yet-placed responses. This couples the responses: raising one
response's score lowers the target for all the others simultaneously, which yields
a more globally consistent ordering than summing pairwise terms.

Why this is Stage 3 (not Stage 1): the listwise objective is the strongest but
also the least forgiving signal. The Weight -> Rank -> List curriculum warms the
policy up on easy, low-variance pair signals before exposing it to the full
listwise geometry (see README, "Why three stages").
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    _HAVE_HF = True
except Exception:  # pragma: no cover
    torch = None
    _HAVE_HF = False

from ..utils import dpo_losses
from .rank_dpo import RankDPOTrainer


@dataclass
class ListDPOConfig:
    beta: float = 0.1
    list_size: int = 6
    lora_r: int = 16


class ListDPOTrainer(RankDPOTrainer):
    """Same forward pass as Rank-DPO, listwise (Plackett-Luce) loss instead."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pol, ref = self._list_logps(model, inputs)
        losses = []
        for p in range(pol.shape[0]):
            losses.append(dpo_losses.list_dpo_loss(
                pol[p], ref[p], beta=self.cfg.beta, reduction="sum"))
        loss = torch.stack(losses).mean()
        return (loss, {"policy_logps": pol}) if return_outputs else loss


__all__ = ["ListDPOConfig", "ListDPOTrainer"]
