"""Training-stability analysis: GRPO vs PPO.

Quantifies four stability facets from the per-seed training histories:

1. **Return smoothness** -- mean rolling-std of the training-return curve
   (lower = smoother learning).
2. **Policy collapse** -- the minimum policy entropy reached (a very low value
   flags premature determinism / collapse).
3. **Gradient behaviour** -- mean and max of the policy gradient norm
   (explosion / vanishing).
4. **Robustness across seeds** -- spread of final performance and a simple
   "crash" count (runs that ended NaN / far below the seed median).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from utils.metrics import rolling_std


def _finite(x):
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def analyze_algorithm(histories: List[Dict], metric: str = "eval_proximity") -> Dict:
    """Compute stability statistics for one algorithm across seeds."""
    return_volatility, min_entropy, mean_gnorm, max_gnorm, final_perf = [], [], [], [], []
    crashes = 0
    for h in histories:
        ret = _finite(h["mean_return"])
        return_volatility.append(float(np.mean(rolling_std(ret, window=10))) if len(ret) else np.nan)
        ent = _finite(h["entropy"])
        min_entropy.append(float(np.min(ent)) if len(ent) else np.nan)
        g = _finite(h["grad_norm"])
        mean_gnorm.append(float(np.mean(g)) if len(g) else np.nan)
        max_gnorm.append(float(np.max(g)) if len(g) else np.nan)
        fp = h[metric][-1]
        final_perf.append(fp)
        # crash heuristic: non-finite loss/grad, or NaN final performance
        if (not np.isfinite(fp)) or len(_finite(h["grad_norm"])) < len(h["grad_norm"]):
            crashes += 1

    final_perf = np.asarray(final_perf, dtype=np.float64)
    med = np.nanmedian(final_perf)
    # additionally count runs that collapsed to <25% of the median as "failed"
    failed = int(np.sum(final_perf < 0.25 * med)) if np.isfinite(med) and med > 0 else 0
    return {
        "return_volatility_mean": float(np.nanmean(return_volatility)),
        "min_entropy_mean": float(np.nanmean(min_entropy)),
        "grad_norm_mean": float(np.nanmean(mean_gnorm)),
        "grad_norm_max": float(np.nanmax(max_gnorm)),
        "final_perf_mean": float(np.nanmean(final_perf)),
        "final_perf_std": float(np.nanstd(final_perf)),
        "crash_count": crashes,
        "failed_runs": failed,
        "num_seeds": len(histories),
    }


def compare_stability(results: Dict[str, List[Dict]], metric: str = "eval_proximity") -> Dict[str, Dict]:
    """Return per-algorithm stability statistics."""
    return {algo: analyze_algorithm(hists, metric) for algo, hists in results.items()}


def summarize(stability: Dict[str, Dict]) -> str:
    """Human-readable summary table."""
    lines = ["Stability summary (mean over seeds):",
             f"{'algo':6s} {'final':>8s} {'±std':>7s} {'volatility':>11s} "
             f"{'min_ent':>8s} {'|g|mean':>8s} {'|g|max':>8s} {'failed':>7s}"]
    for algo, s in stability.items():
        lines.append(
            f"{algo:6s} {s['final_perf_mean']:8.3f} {s['final_perf_std']:7.3f} "
            f"{s['return_volatility_mean']:11.4f} {s['min_entropy_mean']:8.3f} "
            f"{s['grad_norm_mean']:8.3f} {s['grad_norm_max']:8.3f} "
            f"{s['failed_runs']:3d}/{s['num_seeds']:<3d}")
    return "\n".join(lines)
