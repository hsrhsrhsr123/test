"""KL-divergence analysis:  KL(pi_theta || pi_ref) over training and per-prompt.

DPO's whole premise is *bounded* movement away from the reference policy — the
implicit KL penalty (weight 1/beta) keeps the policy from collapsing onto the
reward model's blind spots. Two views:

  * **Trajectory** — how KL grows over training steps, Direct vs Iterative. Direct
    DPO keeps pushing against stale off-policy data; iterative DPO re-anchors each
    round, so its KL grows in controlled steps.
  * **Per-prompt histogram** — KL is not uniform across prompts. A long right tail
    means a few prompts moved a lot (often where the reference was most wrong).
"""

from __future__ import annotations

import numpy as np

from ..utils import visualization as viz
from ..utils.metrics import categorical_kl


def per_prompt_kl(policy_probs: np.ndarray, ref_probs: np.ndarray) -> np.ndarray:
    return categorical_kl(policy_probs, ref_probs)


def plot_kl_analysis(direct_hist, iterative_hist, boundaries, final_policy_probs,
                     ref_probs, path: str):
    """Two-panel figure: KL trajectory (left) + per-prompt KL histogram (right)."""
    import matplotlib.pyplot as plt
    viz.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # -- left: KL over steps --------------------------------------------------
    ax = axes[0]
    viz.line(ax, direct_hist.step, direct_hist.kl, "direct")
    viz.line(ax, iterative_hist.step, iterative_hist.kl, "iterative")
    for b in boundaries[1:]:
        ax.axvline(b, color=viz.color("iterative"), ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\mathrm{KL}(\pi_\theta \,\|\, \pi_{ref})$")
    ax.set_title("KL divergence over training")
    ax.legend()
    ax.annotate("iterative re-anchors\nat each round", xy=(boundaries[1], 0),
                xytext=(0.42, 0.82), textcoords="axes fraction", fontsize=9,
                color=viz.color("iterative"))

    # -- right: per-prompt KL histogram --------------------------------------
    ax = axes[1]
    kl = per_prompt_kl(final_policy_probs, ref_probs)
    ax.hist(kl, bins=40, color=viz.color("weight"), alpha=0.8, edgecolor="white")
    ax.axvline(kl.mean(), color="#111827", ls="--", lw=1.5,
               label=f"mean = {kl.mean():.3f}")
    ax.set_xlabel(r"per-prompt $\mathrm{KL}(\pi_\theta \,\|\, \pi_{ref})$")
    ax.set_ylabel("# prompts")
    ax.set_title("Per-prompt KL distribution (final policy)")
    ax.legend()

    fig.suptitle("KL-divergence analysis", fontsize=14, fontweight="bold")
    return viz.savefig(fig, path)


def summarize(direct_hist, iterative_hist) -> dict:
    return {
        "direct_final_kl": float(direct_hist.kl[-1]),
        "iterative_final_kl": float(iterative_hist.kl[-1]),
        "direct_max_kl": float(np.max(direct_hist.kl)),
        "iterative_max_kl": float(np.max(iterative_hist.kl)),
    }


if __name__ == "__main__":
    # standalone smoke run on a fresh synthetic world
    from ..utils.toy_engine import make_toy_env
    from ..training.iterative_dpo import run_direct_dpo, run_iterative_dpo
    env = make_toy_env(seed=0)
    dp, dh = run_direct_dpo(env, stage="weight", total_steps=300)
    it = run_iterative_dpo(env, stage="weight", n_rounds=3, steps_per_round=100)
    out = plot_kl_analysis(dh, it.history, it.round_boundaries,
                           dp.probs, env.ref_probs, "results/figures/kl_divergence_curves.png")
    print("wrote", out, summarize(dh, it.history))
