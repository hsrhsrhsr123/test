"""Reward-distribution analysis across the three training stages.

Two questions:

  * **Where does the reward mass move?** For each stage's policy we look at the
    distribution over prompts of the expected true reward ``E_{y~pi}[r(x,y)]``.
    Successive stages should shift this distribution rightward (and, ideally,
    tighten it) as the ranking signal gets richer.
  * **Does the chosen/rejected margin matter?** We bucket training pairs by their
    reward-model margin and measure how much the policy actually improved on those
    prompts — motivating margin-based filtering (small-margin pairs carry little
    usable signal).
"""

from __future__ import annotations

import numpy as np

from ..utils import visualization as viz
from ..utils.metrics import expected_reward


def per_prompt_reward(policy_probs: np.ndarray, true_rewards: np.ndarray) -> np.ndarray:
    return (policy_probs * true_rewards).sum(axis=-1)


def plot_reward_distributions(env, stage_policies: dict, pairs_df, path: str):
    """Three-panel figure.

    ``stage_policies`` maps stage-name ("reference"/"weight"/"rank"/"list") ->
    policy probs [P,k]. ``pairs_df`` carries reward_chosen/reward_rejected columns.
    """
    import matplotlib.pyplot as plt
    viz.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # -- panel 1: expected-reward violins per stage --------------------------
    data = {name: per_prompt_reward(probs, env.true_rewards)
            for name, probs in stage_policies.items()}
    viz.violin(axes[0], data, ylabel=r"$E_{y\sim\pi}[\,r(x,y)\,]$",
               title="Expected reward by stage")

    # -- panel 2: chosen vs rejected reward gap ------------------------------
    ax = axes[1]
    if pairs_df is not None and "reward_chosen" in pairs_df.columns:
        margin = (pairs_df["reward_chosen"] - pairs_df["reward_rejected"]).values
        ax.hist(margin, bins=40, color=viz.color("weight"), alpha=0.85,
                edgecolor="white")
        ax.axvline(0, color=viz.color("rejected"), ls="--", lw=1.5, label="tie")
        ax.axvline(np.median(margin), color="#111827", ls=":", lw=1.5,
                   label=f"median = {np.median(margin):.2f}")
        ax.set_xlabel("reward(chosen) - reward(rejected)")
        ax.set_ylabel("# pairs")
        ax.set_title("Preference margin distribution")
        ax.legend()

    # -- panel 3: margin vs realised improvement -----------------------------
    ax = axes[2]
    ref = per_prompt_reward(env.ref_probs, env.true_rewards)
    final_name = "list" if "list" in stage_policies else list(stage_policies)[-1]
    final = per_prompt_reward(stage_policies[final_name], env.true_rewards)
    improvement = final - ref
    # per-prompt reward-model margin between best & worst candidate
    rm_margin = env.rm_scores.max(axis=1) - env.rm_scores.min(axis=1)
    ax.scatter(rm_margin, improvement, s=14, alpha=0.4, color=viz.color("rank"))
    if len(rm_margin) > 2:
        b, a = np.polyfit(rm_margin, improvement, 1)
        xs = np.linspace(rm_margin.min(), rm_margin.max(), 50)
        ax.plot(xs, a + b * xs, color="#111827", lw=1.8,
                label=f"slope = {b:.3f}, r = {np.corrcoef(rm_margin, improvement)[0,1]:.2f}")
        ax.legend()
    ax.set_xlabel("per-prompt RM margin (best - worst)")
    ax.set_ylabel("reward improvement (final - ref)")
    ax.set_title("Margin vs realised gain")

    fig.suptitle("Reward-distribution analysis", fontsize=14, fontweight="bold")
    return viz.savefig(fig, path)


def summarize(env, stage_policies: dict) -> dict:
    return {name: float(expected_reward(probs, env.true_rewards))
            for name, probs in stage_policies.items()}


if __name__ == "__main__":
    from ..utils.toy_engine import make_toy_env, train_tabular_dpo
    env = make_toy_env(seed=0)
    pols = {"reference": env.ref_probs}
    for st in ("weight", "rank", "list"):
        pol, _ = train_tabular_dpo(env, stage=st, steps=300)
        pols[st] = pol.probs
    out = plot_reward_distributions(env, pols, None,
                                    "results/figures/reward_distribution_comparison.png")
    print("wrote", out, summarize(env, pols))
