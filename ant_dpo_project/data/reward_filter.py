"""Reward-model scoring and quality filtering of preference pairs.

Why filter at all?
------------------
DPO treats every ``(chosen, rejected)`` pair as ground truth. If the pair is a
near-tie, or if the "chosen" is actually worse, that noise flows straight into
the gradient. A reward model lets us (a) *re-derive* chosen/rejected from a
consistent scorer and (b) drop low-signal pairs. Two strategies are provided:

1. **Top-k filtering** — for each prompt, take the RM's top-k responses as the
   ``chosen`` pool and the bottom ones as ``rejected``; keeps only high-confidence
   extremes.
2. **Margin filtering** — keep pairs whose ``reward_chosen - reward_rejected``
   exceeds a threshold ``tau``.

The RM itself is pluggable: a real HF ``AutoModelForSequenceClassification``
(e.g. ``OpenAssistant/reward-model-deberta-v3-large-v2``) when available, else the
synthetic reward observations produced by the feedback simulator.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# reward-model scoring                                                         #
# --------------------------------------------------------------------------- #
class RewardModel:
    """Thin wrapper around a HF sequence-classification reward model.

    Instantiation is lazy and guarded: if torch/transformers or the weights are
    unavailable, :meth:`score` raises and callers fall back to synthetic scores.
    """

    def __init__(self, model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2"):
        from transformers import (AutoModelForSequenceClassification,  # noqa
                                  AutoTokenizer)
        import torch  # noqa

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def score(self, prompts, responses, batch_size: int = 8) -> np.ndarray:
        scores = []
        for i in range(0, len(prompts), batch_size):
            bp = prompts[i:i + batch_size]
            br = responses[i:i + batch_size]
            enc = self.tok(bp, br, truncation=True, padding=True,
                           max_length=512, return_tensors="pt")
            with self.torch.no_grad():
                out = self.model(**enc).logits.squeeze(-1)
            scores.extend(out.tolist())
        return np.asarray(scores)


def score_pairs(pairs: pd.DataFrame, model_name: str | None = None) -> pd.DataFrame:
    """Attach ``reward_chosen`` / ``reward_rejected`` columns.

    If ``model_name`` is given and loadable, use a real RM; otherwise assume the
    columns already hold synthetic RM observations (from the feedback simulator).
    """
    if model_name is None:
        if "reward_chosen" not in pairs.columns:
            raise ValueError("no RM scores present and no model_name given")
        return pairs
    try:
        rm = RewardModel(model_name)
        pairs = pairs.copy()
        pairs["reward_chosen"] = rm.score(list(pairs["prompt"]), list(pairs["chosen"]))
        pairs["reward_rejected"] = rm.score(list(pairs["prompt"]), list(pairs["rejected"]))
        print(f"[reward_filter] scored {len(pairs)} pairs with {model_name}")
    except Exception as e:
        print(f"[reward_filter] real RM unavailable ({e}); using existing scores.")
    return pairs


# --------------------------------------------------------------------------- #
# filtering strategies                                                         #
# --------------------------------------------------------------------------- #
def top_k_filter(pairs: pd.DataFrame, k: int = 2, per_prompt_col: str = "prompt_id"):
    """Keep pairs where chosen is in the RM top-k *and* rejected in the bottom-k.

    Requires per-candidate structure (``chosen_idx``/``rejected_idx``). For each
    prompt, ranks the candidates by reward and keeps only pairs pitting a top-k
    response against a bottom-k one.
    """
    if not {"chosen_idx", "rejected_idx"}.issubset(pairs.columns):
        # fall back to a global top/bottom split by reward
        hi = pairs["reward_chosen"] >= pairs["reward_chosen"].quantile(0.5)
        lo = pairs["reward_rejected"] <= pairs["reward_rejected"].quantile(0.5)
        return pairs[hi & lo].reset_index(drop=True)

    keep = []
    for pid, grp in pairs.groupby(per_prompt_col):
        # build per-candidate reward view for this prompt
        rc = grp.set_index("chosen_idx")["reward_chosen"].to_dict()
        rr = grp.set_index("rejected_idx")["reward_rejected"].to_dict()
        rewards = {**rr, **rc}
        ranked = sorted(rewards, key=rewards.get, reverse=True)
        top = set(ranked[:k])
        bot = set(ranked[-k:])
        m = grp["chosen_idx"].isin(top) & grp["rejected_idx"].isin(bot)
        keep.append(grp[m])
    return pd.concat(keep).reset_index(drop=True) if keep else pairs.iloc[:0]


def margin_filter(pairs: pd.DataFrame, tau: float = 0.5) -> pd.DataFrame:
    margin = pairs["reward_chosen"] - pairs["reward_rejected"]
    return pairs[margin > tau].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# quality report                                                               #
# --------------------------------------------------------------------------- #
def quality_report(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Compare pair quality before/after filtering.

    ``label_error_rate`` = fraction of pairs where the RM-preferred response is
    actually *worse* by true reward (only available in synthetic mode). A good
    filter should drive this down.
    """
    def stats(df):
        margin = df["reward_chosen"] - df["reward_rejected"]
        d = {
            "n_pairs": int(len(df)),
            "margin_mean": float(margin.mean()) if len(df) else 0.0,
            "margin_median": float(margin.median()) if len(df) else 0.0,
            "margin_p10": float(np.percentile(margin, 10)) if len(df) else 0.0,
        }
        if {"true_reward_chosen", "true_reward_rejected"}.issubset(df.columns) and len(df):
            true_margin = df["true_reward_chosen"] - df["true_reward_rejected"]
            d["label_error_rate"] = float((true_margin < 0).mean())
        return d

    return {"before": stats(before), "after": stats(after),
            "retention": float(len(after) / max(1, len(before)))}


def format_report(rep: dict) -> str:
    b, a = rep["before"], rep["after"]
    lines = [
        "## Reward-model filtering report", "",
        f"| metric | before | after |",
        f"|---|---|---|",
        f"| pairs | {b['n_pairs']} | {a['n_pairs']} |",
        f"| margin mean | {b['margin_mean']:.3f} | {a['margin_mean']:.3f} |",
        f"| margin p10 | {b['margin_p10']:.3f} | {a['margin_p10']:.3f} |",
    ]
    if "label_error_rate" in b:
        lines.append(f"| label error rate | {b['label_error_rate']:.3f} | "
                     f"{a['label_error_rate']:.3f} |")
    lines.append(f"| retention | — | {rep['retention']*100:.1f}% |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Reward-model filtering of preference pairs")
    ap.add_argument("--in", dest="inp", default="results/preference_pairs.parquet")
    ap.add_argument("--strategy", choices=["topk", "margin"], default="margin")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--model-name", default=None)
    args = ap.parse_args()

    pairs = (pd.read_parquet(args.inp) if args.inp.endswith(".parquet")
             else pd.read_csv(args.inp))
    pairs = score_pairs(pairs, args.model_name)
    after = (top_k_filter(pairs, args.k) if args.strategy == "topk"
             else margin_filter(pairs, args.tau))
    print(format_report(quality_report(pairs, after)))


if __name__ == "__main__":
    main()
