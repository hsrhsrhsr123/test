"""Stage 1 — Weighted-DPO trainer (extends trl's ``DPOTrainer``).

Idea
----
Not every preference pair is equally trustworthy. Weighted-DPO scales each pair's
loss by the reward model's *confidence* in that preference — the softmax over the
batch of reward-model margins ``r(chosen) - r(rejected)``:

    L = - E[ w_i * log sigma( beta * ( (logpi_w - logpi_ref_w) - (logpi_l - logpi_ref_l) ) ) ]
    w_i = softmax_batch( (r_chosen_i - r_rejected_i) / weight_temperature ) * batch_size

Confident pairs (large margin) dominate the update; near-ties are damped. The
weights are renormalised to average 1 so the effective step size matches vanilla
DPO.

Runtime note
------------
This is the *real* training path: it needs ``trl`` + ``transformers`` + a GPU and
a model such as ``Qwen/Qwen2.5-0.5B`` with LoRA. When ``trl`` is not installed the
module still imports (so it can be read / type-checked), but instantiating the
trainer will fail. For a CPU-only demonstration of the identical loss dynamics,
see ``utils/toy_engine.py`` and ``run_pipeline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

try:  # real dependency, guarded so the module is importable without a GPU stack
    import torch
    from trl import DPOTrainer
    _HAVE_TRL = True
except Exception:  # pragma: no cover
    torch = None
    DPOTrainer = object
    _HAVE_TRL = False


@dataclass
class WeightedDPOConfig:
    beta: float = 0.1
    weight_temperature: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    learning_rate: float = 5e-6
    num_train_epochs: float = 1.0


class WeightedDPOTrainer(DPOTrainer):
    """DPOTrainer that weights each pair by reward-model confidence.

    The per-batch reward margins are expected on the batch under the key
    ``reward_margin`` (attach them with :class:`WeightedDPOCollator` below, which
    wraps trl's default collator and carries the extra column through).
    """

    def __init__(self, *args, weight_temperature: float = 1.0, **kwargs):
        if not _HAVE_TRL:
            raise RuntimeError(
                "trl/transformers not available — this trainer needs the GPU stack. "
                "Use the tabular engine in utils/toy_engine.py for CPU demos.")
        super().__init__(*args, **kwargs)
        self.weight_temperature = weight_temperature
        self._batch_weights = None

    # -- weight computation ------------------------------------------------- #
    def _confidence_weights(self, margins: "torch.Tensor") -> "torch.Tensor":
        w = torch.softmax(margins / self.weight_temperature, dim=0)
        return w * w.numel()  # renormalise so mean(w) == 1

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        # stash this batch's confidence weights for dpo_loss() to consume
        margins = batch.get("reward_margin")
        if margins is not None:
            if not torch.is_tensor(margins):
                margins = torch.as_tensor(margins, dtype=torch.float32)
            self._batch_weights = self._confidence_weights(
                margins.to(self.accelerator.device))
        else:
            self._batch_weights = None
        return super().get_batch_loss_metrics(model, batch, train_eval)

    # -- inject weights into the per-example DPO loss ----------------------- #
    def dpo_loss(self, policy_chosen_logps, policy_rejected_logps,
                 reference_chosen_logps, reference_rejected_logps, *args, **kwargs):
        losses, chosen_rewards, rejected_rewards = super().dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps, *args, **kwargs)
        if self._batch_weights is not None:
            w = self._batch_weights.to(losses.device)
            if w.shape == losses.shape:
                losses = losses * w
        return losses, chosen_rewards, rejected_rewards


class WeightedDPOCollator:
    """Wrap trl's DPO collator to carry a per-example ``reward_margin`` tensor."""

    def __init__(self, base_collator, margin_key: str = "reward_margin"):
        self.base = base_collator
        self.margin_key = margin_key

    def __call__(self, features):
        margins = [f.get(self.margin_key, 0.0) for f in features]
        batch = self.base(features)
        if _HAVE_TRL:
            batch[self.margin_key] = torch.tensor(margins, dtype=torch.float32)
        return batch


def build_trainer(model, ref_model, tokenizer, train_dataset,
                  cfg: WeightedDPOConfig = WeightedDPOConfig(), **trainer_kwargs):
    """Convenience constructor wiring LoRA + the weighted trainer.

    ``train_dataset`` rows must contain ``prompt``, ``chosen``, ``rejected`` and a
    numeric ``reward_margin`` column (= RM score of chosen minus rejected).
    """
    if not _HAVE_TRL:
        raise RuntimeError("trl not available")
    from peft import LoraConfig
    from trl import DPOConfig

    peft_config = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                             lora_dropout=0.05, task_type="CAUSAL_LM",
                             target_modules="all-linear")
    dpo_config = DPOConfig(beta=cfg.beta, learning_rate=cfg.learning_rate,
                           num_train_epochs=cfg.num_train_epochs,
                           remove_unused_columns=False, **trainer_kwargs)
    trainer = WeightedDPOTrainer(
        model=model, ref_model=ref_model, args=dpo_config,
        train_dataset=train_dataset, processing_class=tokenizer,
        peft_config=peft_config, weight_temperature=cfg.weight_temperature)
    # preserve the reward_margin column through collation
    trainer.data_collator = WeightedDPOCollator(trainer.data_collator)
    return trainer


__all__ = ["WeightedDPOConfig", "WeightedDPOTrainer", "WeightedDPOCollator",
           "build_trainer"]
