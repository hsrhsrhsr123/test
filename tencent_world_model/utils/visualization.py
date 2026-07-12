"""Plotting helpers that turn experiment results into the report figures.

Matplotlib only (seaborn is optional and only used for styling if present).
Every function takes already-computed results and writes a PNG.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import moving_average, rolling_std

try:
    import seaborn as sns  # noqa: F401
    sns.set_theme(style="whitegrid")
except Exception:
    plt.style.use("default")

GRPO_COLOR = "#2166ac"   # blue
PPO_COLOR = "#d6604d"    # red/orange
ALGO_COLORS = {"grpo": GRPO_COLOR, "ppo": PPO_COLOR}
ALGO_LABELS = {"grpo": "GRPO (critic-free)", "ppo": "PPO (with critic)"}


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _mean_std(histories: List[Dict], key: str):
    stacked = np.asarray([h[key] for h in histories], dtype=np.float64)
    return stacked.mean(axis=0), stacked.std(axis=0)


def _eval_iters(histories: List[Dict]):
    return np.asarray(histories[0]["eval_iteration"], dtype=np.float64)


def _train_iters(histories: List[Dict]):
    return np.asarray(histories[0]["iteration"], dtype=np.float64)


# --------------------------------------------------------------------- figures
def plot_reward_curves(results: Dict[str, List[Dict]], out_path: str,
                       metric: str = "eval_proximity",
                       title: str = "GRPO vs PPO -- latent-space policy learning",
                       ylabel: str = "Goal proximity (real env)"):
    """Core figure: evaluation curves with mean +/- std bands over seeds."""
    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo, hists in results.items():
        x = _eval_iters(hists)
        mean, std = _mean_std(hists, metric)
        c = ALGO_COLORS.get(algo, None)
        ax.plot(x, mean, label=ALGO_LABELS.get(algo, algo), color=c, lw=2)
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.2)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_training_stability(results: Dict[str, List[Dict]], out_path: str):
    """Entropy, gradient norm, and reward-curve smoothness (rolling std)."""
    _ensure_dir(out_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) policy entropy
    for algo, hists in results.items():
        x = _train_iters(hists)
        mean, std = _mean_std(hists, "entropy")
        c = ALGO_COLORS.get(algo)
        axes[0].plot(x, mean, color=c, label=ALGO_LABELS.get(algo, algo), lw=1.8)
        axes[0].fill_between(x, mean - std, mean + std, color=c, alpha=0.15)
    axes[0].set_title("(a) Policy entropy\n(collapse detection)")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Entropy"); axes[0].legend()

    # (b) gradient norm
    for algo, hists in results.items():
        x = _train_iters(hists)
        mean, std = _mean_std(hists, "grad_norm")
        c = ALGO_COLORS.get(algo)
        axes[1].plot(x, mean, color=c, label=ALGO_LABELS.get(algo, algo), lw=1.8)
        axes[1].fill_between(x, mean - std, mean + std, color=c, alpha=0.15)
    axes[1].set_title("(b) Policy gradient norm\n(explosion/vanishing)")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("||grad||"); axes[1].legend()

    # (c) reward smoothness: rolling std of the mean training return
    for algo, hists in results.items():
        x = _train_iters(hists)
        mean_ret, _ = _mean_std(hists, "mean_return")
        rstd = rolling_std(mean_ret, window=10)
        axes[2].plot(x, rstd, color=ALGO_COLORS.get(algo),
                     label=ALGO_LABELS.get(algo, algo), lw=1.8)
    axes[2].set_title("(c) Return volatility\n(rolling std, lower = smoother)")
    axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("Rolling std of return"); axes[2].legend()

    fig.suptitle("Training stability: GRPO vs PPO", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_sparse_reward_comparison(results_by_mode: Dict[str, Dict[str, List[Dict]]],
                                  out_path: str, metric: str = "eval_proximity"):
    """One panel per reward mode (dense / sparse / very_sparse)."""
    _ensure_dir(out_path)
    modes = list(results_by_mode.keys())
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 4.2), squeeze=False)
    axes = axes[0]
    for ax, mode in zip(axes, modes):
        results = results_by_mode[mode]
        for algo, hists in results.items():
            x = _eval_iters(hists)
            mean, std = _mean_std(hists, metric)
            c = ALGO_COLORS.get(algo)
            ax.plot(x, mean, color=c, label=ALGO_LABELS.get(algo, algo), lw=2)
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.2)
        ax.set_title(f"{mode} reward")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Goal proximity (real env)")
        ax.legend(fontsize=8)
    fig.suptitle("Reward sparsity: where critic-free GRPO helps", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_efficiency(results: Dict[str, List[Dict]], out_path: str,
                                threshold: float, metric: str = "eval_proximity"):
    """Time/iterations to a performance threshold, and parameter counts."""
    _ensure_dir(out_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    algos = list(results.keys())
    colors = [ALGO_COLORS.get(a) for a in algos]

    # (a) iterations to threshold (per seed -> mean/std)
    iters_to_thr = {}
    for algo, hists in results.items():
        vals = []
        for h in hists:
            curve = np.asarray(h[metric])
            x = np.asarray(h["eval_iteration"])
            hit = np.where(curve >= threshold)[0]
            vals.append(x[hit[0]] if len(hit) else np.nan)
        iters_to_thr[algo] = vals
    means = [np.nanmean(iters_to_thr[a]) if np.any(~np.isnan(iters_to_thr[a])) else np.nan for a in algos]
    stds = [np.nanstd(iters_to_thr[a]) if np.any(~np.isnan(iters_to_thr[a])) else 0 for a in algos]
    axes[0].bar(algos, means, yerr=stds, color=colors, alpha=0.85, capsize=5)
    axes[0].set_title(f"(a) Iterations to reach\nproximity={threshold:.2f}")
    axes[0].set_ylabel("Iterations (lower = faster)")

    # (b) wall-clock to threshold
    time_to_thr = []
    for algo, hists in results.items():
        vals = []
        for h in hists:
            curve = np.asarray(h[metric]); x = np.asarray(h["eval_iteration"])
            wc = np.asarray(h["wallclock"])
            hit = np.where(curve >= threshold)[0]
            if len(hit):
                it = int(x[hit[0]])
                vals.append(wc[min(it, len(wc) - 1)])
            else:
                vals.append(np.nan)
        time_to_thr.append((algo, np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan))
    axes[1].bar([a for a, _ in time_to_thr], [v for _, v in time_to_thr],
                color=colors, alpha=0.85)
    axes[1].set_title(f"(b) Wall-clock to reach\nproximity={threshold:.2f}")
    axes[1].set_ylabel("Seconds (lower = faster)")

    # (c) parameter counts (policy vs total incl. critic)
    pol_params = [results[a][0]["policy_params"] for a in algos]
    tot_params = [results[a][0]["total_params"] for a in algos]
    xpos = np.arange(len(algos)); w = 0.35
    axes[2].bar(xpos - w / 2, pol_params, w, label="policy", color="#8c8c8c")
    axes[2].bar(xpos + w / 2, tot_params, w, label="total (incl. critic)", color=colors)
    axes[2].set_xticks(xpos); axes[2].set_xticklabels(algos)
    axes[2].set_title("(c) Trainable parameters\n(GRPO has no critic)")
    axes[2].set_ylabel("# parameters"); axes[2].legend(fontsize=8)

    fig.suptitle("Convergence efficiency & model size", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_latent_space(latents: np.ndarray, values: np.ndarray, out_path: str,
                      method: str = "pca", value_label: str = "goal proximity"):
    """2-D projection (PCA or t-SNE) of latent states coloured by a value."""
    _ensure_dir(out_path)
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            n = min(len(latents), 2000)
            idx = np.random.default_rng(0).choice(len(latents), n, replace=False)
            emb = TSNE(n_components=2, init="pca", perplexity=30,
                       random_state=0).fit_transform(latents[idx])
            values = values[idx]
        except Exception:
            method = "pca"
    if method != "tsne":
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2, random_state=0).fit_transform(latents)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=values, cmap="viridis", s=8, alpha=0.7)
    fig.colorbar(sc, ax=ax, label=value_label)
    ax.set_title(f"Latent space ({method.upper()}) coloured by {value_label}")
    ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_world_model_quality(multistep_error: Dict[int, float], out_path: str,
                             recon_true: Optional[np.ndarray] = None,
                             recon_pred: Optional[np.ndarray] = None,
                             long_range: Optional[Dict] = None):
    """Multi-step prediction error + (optional) reconstruction sample + long-range bars."""
    _ensure_dir(out_path)
    n_panels = 1 + (recon_true is not None) + (long_range is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.2), squeeze=False)
    axes = axes[0]
    i = 0

    horizons = sorted(multistep_error.keys())
    errs = [multistep_error[h] for h in horizons]
    axes[i].plot(horizons, errs, "o-", color="#2166ac", lw=2)
    axes[i].set_title("(a) Multi-step latent-rollout error")
    axes[i].set_xlabel("Prediction horizon (steps)")
    axes[i].set_ylabel("Observation MSE")
    i += 1

    if recon_true is not None and recon_pred is not None:
        axes[i].plot(recon_true, label="true obs", color="#333333", lw=1.5)
        axes[i].plot(recon_pred, label="reconstruction", color="#d6604d", lw=1.5, ls="--")
        axes[i].set_title("(b) Reconstruction sample")
        axes[i].set_xlabel("observation dimension"); axes[i].set_ylabel("value")
        axes[i].legend(fontsize=8)
        i += 1

    if long_range is not None:
        labels = ["latent\nworld model", "raw-obs\nbaseline"]
        vals = [long_range["world_model_accuracy"], long_range["baseline_accuracy"]]
        bars = axes[i].bar(labels, vals, color=["#2166ac", "#8c8c8c"], alpha=0.85)
        axes[i].set_ylim(0, 1)
        axes[i].set_title(f"(c) Long-range (H) prediction accuracy\n"
                          f"+{long_range['relative_improvement_pct']:.1f}% vs baseline")
        axes[i].set_ylabel("accuracy (1 - norm. RMSE)")
        for b, v in zip(bars, vals):
            axes[i].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")

    fig.suptitle("World-model quality", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_group_size_ablation(group_results: Dict[int, List[Dict]], out_path: str,
                             metric: str = "eval_proximity"):
    """GRPO performance vs group size G."""
    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    gs = sorted(group_results.keys())
    finals, errs = [], []
    for g in gs:
        vals = [h[metric][-1] for h in group_results[g]]
        finals.append(np.mean(vals)); errs.append(np.std(vals))
    ax.errorbar([str(g) for g in gs], finals, yerr=errs, marker="o", lw=2,
                color="#2166ac", capsize=5)
    ax.set_title("GRPO: effect of group size G")
    ax.set_xlabel("group size G")
    ax.set_ylabel("final goal proximity (real env)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
