"""Convergence-efficiency analysis: GRPO vs PPO.

Measures how quickly each algorithm reaches a target performance and what it
costs in parameters and wall-clock time:

1. **Iterations to threshold** -- first evaluation iteration reaching the target.
2. **Wall-clock to threshold** -- seconds of training to the same target.
3. **Sample/compute efficiency** -- area under the evaluation curve (AUC).
4. **Model size** -- policy vs total parameters (GRPO has no critic).
5. **Per-update speed** -- mean wall-clock per training iteration.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from utils.metrics import area_under_curve


def _iters_to_threshold(h: Dict, threshold: float, metric: str):
    curve = np.asarray(h[metric], dtype=np.float64)
    x = np.asarray(h["eval_iteration"], dtype=np.float64)
    hit = np.where(curve >= threshold)[0]
    return (float(x[hit[0]]), int(x[hit[0]])) if len(hit) else (np.nan, None)


def analyze_algorithm(histories: List[Dict], threshold: float,
                      metric: str = "eval_proximity") -> Dict:
    iters, times, aucs = [], [], []
    for h in histories:
        it_val, it_idx = _iters_to_threshold(h, threshold, metric)
        iters.append(it_val)
        wc = np.asarray(h["wallclock"], dtype=np.float64)
        if it_idx is not None and len(wc):
            times.append(float(wc[min(it_idx, len(wc) - 1)]))
        else:
            times.append(np.nan)
        aucs.append(area_under_curve(h[metric]))
    total_times = [h["total_time"] for h in histories]
    n_iters = [len(h["iteration"]) for h in histories]
    per_update = [t / max(1, n) for t, n in zip(total_times, n_iters)]

    def _nanmean(a):  # nan-safe: returns nan (no warning) when everything is nan
        a = np.asarray(a, dtype=np.float64)
        finite = a[np.isfinite(a)]
        return float(finite.mean()) if len(finite) else float("nan")

    def _nanstd(a):
        a = np.asarray(a, dtype=np.float64)
        finite = a[np.isfinite(a)]
        return float(finite.std()) if len(finite) else float("nan")

    return {
        "iters_to_threshold_mean": _nanmean(iters),
        "iters_to_threshold_std": _nanstd(iters),
        "time_to_threshold_mean": _nanmean(times),
        "auc_mean": float(np.mean(aucs)),
        "policy_params": histories[0]["policy_params"],
        "total_params": histories[0]["total_params"],
        "per_update_time_mean": float(np.mean(per_update)),
        "total_time_mean": float(np.mean(total_times)),
        "reached_fraction": float(np.mean([np.isfinite(i) for i in iters])),
    }


def compare_convergence(results: Dict[str, List[Dict]], threshold: float,
                        metric: str = "eval_proximity") -> Dict[str, Dict]:
    return {algo: analyze_algorithm(hists, threshold, metric)
            for algo, hists in results.items()}


def summarize(conv: Dict[str, Dict], threshold: float) -> str:
    lines = [f"Convergence summary (target proximity = {threshold:.2f}):",
             f"{'algo':6s} {'iters->thr':>11s} {'time->thr(s)':>13s} {'AUC':>7s} "
             f"{'params':>8s} {'s/update':>9s} {'reached':>8s}"]
    for algo, c in conv.items():
        it = c["iters_to_threshold_mean"]
        tt = c["time_to_threshold_mean"]
        lines.append(
            f"{algo:6s} {it:11.1f} {tt:13.2f} {c['auc_mean']:7.3f} "
            f"{c['total_params']:8d} {c['per_update_time_mean']:9.4f} "
            f"{c['reached_fraction']*100:6.0f}%")
    # highlight parameter saving
    if "grpo" in conv and "ppo" in conv:
        saving = 100.0 * (conv["ppo"]["total_params"] - conv["grpo"]["total_params"]) / conv["ppo"]["total_params"]
        lines.append(f"GRPO uses {saving:.1f}% fewer trainable params than PPO (no critic).")
    return "\n".join(lines)
