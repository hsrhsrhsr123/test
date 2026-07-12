"""Stage 2 — Rank-DPO trainer.

Idea
----
For a single prompt we sample several responses (4-8), score them with the reward
model, and sort them best -> worst. From that ranking we form *all* C(n,2) ordered
pairs and sum position-weighted pairwise DPO losses:

    L = - sum_{i<j} w_ij * log sigma( beta * ( h(y_i) - h(y_j) ) ),
    h(y) = log pi_theta(y|x) - log pi_ref(y|x),
    w_ij = position_discount^i * (j - i)        # DCG-style: top-of-list pairs matter more

This is a richer signal than a single (chosen, rejected) pair — one prompt now
contributes up to 28 constraints (for n=8) instead of 1.

Because trl's ``DPOTrainer`` is hard-wired to two responses, Rank/List-DPO are
implemented as a small custom ``transformers.Trainer`` that forwards the whole
list of responses and calls the shared loss in ``utils/dpo_losses.py``. Guarded
imports keep the module readable without the GPU stack; the CPU demo lives in
``utils/toy_engine.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
    from transformers import Trainer
    _HAVE_HF = True
except Exception:  # pragma: no cover
    torch = None
    Trainer = object
    _HAVE_HF = False

from ..utils import dpo_losses  # pure functions; import is cheap


@dataclass
class RankDPOConfig:
    beta: float = 0.1
    position_discount: float = 0.9
    list_size: int = 6
    lora_r: int = 16


# --------------------------------------------------------------------------- #
# log-prob utility                                                             #
# --------------------------------------------------------------------------- #
def sequence_logprobs(model, input_ids, attention_mask, loss_mask):
    """Sum of token log-probs over the response span for each sequence.

    ``loss_mask`` is 1 on response tokens, 0 on prompt/pad tokens. Returns a
    tensor of shape [batch].
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:].clone()
    mask = loss_mask[:, 1:]
    logps = torch.gather(F.log_softmax(logits, dim=-1), 2,
                         labels.unsqueeze(-1)).squeeze(-1)
    return (logps * mask).sum(dim=-1)


# --------------------------------------------------------------------------- #
# trainer                                                                      #
# --------------------------------------------------------------------------- #
class RankDPOTrainer(Trainer):
    """Custom trainer applying the position-weighted all-pairs ranking loss.

    Each dataset row is expected to yield, after collation, tensors for a *list*
    of ``list_size`` responses per prompt, already sorted best -> worst by RM
    score. The reference model is frozen.
    """

    def __init__(self, *args, ref_model=None, cfg: RankDPOConfig = RankDPOConfig(), **kwargs):
        if not _HAVE_HF:
            raise RuntimeError("transformers/torch not available — GPU stack required.")
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        self.cfg = cfg

    def _list_logps(self, model, batch):
        """Return [P, n] policy and ref sequence-logprobs for the response lists."""
        P, n = batch["list_shape"]
        ids = batch["input_ids"]            # [P*n, T]
        am = batch["attention_mask"]
        lm = batch["loss_mask"]
        pol = sequence_logprobs(model, ids, am, lm).view(P, n)
        with torch.no_grad():
            ref = sequence_logprobs(self.ref_model, ids, am, lm).view(P, n)
        return pol, ref

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pol, ref = self._list_logps(model, inputs)
        losses = []
        for p in range(pol.shape[0]):
            losses.append(dpo_losses.rank_dpo_loss(
                pol[p], ref[p], beta=self.cfg.beta,
                position_discount=self.cfg.position_discount))
        loss = torch.stack(losses).mean()
        return (loss, {"policy_logps": pol}) if return_outputs else loss


__all__ = ["RankDPOConfig", "RankDPOTrainer", "sequence_logprobs"]
