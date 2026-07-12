"""Sparse-reward analysis: why critic-free GRPO holds up as rewards get sparser.

For each reward mode (dense / sparse / very_sparse) we compare the final
performance of GRPO and PPO and inspect the *mechanism*:

* **PPO** relies on a critic.  As the reward is delayed, the value target
  becomes high-variance and hard to bootstrap, so the critic's **explained
  variance** falls -- and a bad critic means a biased advantage.
* **GRPO** needs only that *some* trajectories in a group beat others.  We track
  the **group-signal fraction**: the share of groups whose return spread is
  non-trivial (so the group-relative advantage is informative).  As long as this
  stays high, GRPO keeps getting a usable gradient without any value estimate.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _final_mean_std(hists: List[Dict], key: str = "eval_proximity"):
    vals = np.asarray([h[key][-1] for h in hists], dtype=np.float64)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def _mean_last_quarter(hists: List[Dict], key: str):
    """Mean over the last quarter of a per-iteration metric, averaged over seeds."""
    out = []
    for h in hists:
        if key not in h or len(h[key]) == 0:
            continue
        arr = np.asarray(h[key], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            out.append(np.mean(arr[-max(1, len(arr) // 4):]))
    return float(np.mean(out)) if out else float("nan")


def analyze(results_by_mode: Dict[str, Dict[str, List[Dict]]],
            metric: str = "eval_proximity") -> Dict[str, Dict]:
    """Per-mode comparison and mechanism diagnostics."""
    table = {}
    for mode, results in results_by_mode.items():
        entry = {}
        if "grpo" in results:
            g_mean, g_std = _final_mean_std(results["grpo"], metric)
            entry["grpo_final"] = g_mean
            entry["grpo_final_std"] = g_std
            entry["grpo_group_signal_frac"] = _mean_last_quarter(results["grpo"], "group_signal_frac")
        if "ppo" in results:
            p_mean, p_std = _final_mean_std(results["ppo"], metric)
            entry["ppo_final"] = p_mean
            entry["ppo_final_std"] = p_std
            entry["ppo_value_explained_variance"] = _mean_last_quarter(results["ppo"], "value_explained_variance")
        if "grpo" in results and "ppo" in results:
            entry["grpo_minus_ppo"] = entry["grpo_final"] - entry["ppo_final"]
        table[mode] = entry
    return table


def summarize(table: Dict[str, Dict]) -> str:
    lines = ["Sparse-reward summary (final goal proximity, mean over seeds):",
             f"{'mode':13s} {'GRPO':>8s} {'PPO':>8s} {'GRPO-PPO':>9s} "
             f"{'PPO val.EV':>11s} {'GRPO signal':>12s}"]
    for mode, e in table.items():
        lines.append(
            f"{mode:13s} {e.get('grpo_final', float('nan')):8.3f} "
            f"{e.get('ppo_final', float('nan')):8.3f} "
            f"{e.get('grpo_minus_ppo', float('nan')):9.3f} "
            f"{e.get('ppo_value_explained_variance', float('nan')):11.2f} "
            f"{e.get('grpo_group_signal_frac', float('nan')):12.2f}")
    lines.append("(PPO val.EV: critic explained variance -- falls as reward sparsifies.")
    lines.append(" GRPO signal: fraction of groups with a usable return spread.)")
    return "\n".join(lines)
