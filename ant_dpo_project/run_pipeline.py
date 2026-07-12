"""Main entry point: run the full three-stage DPO pipeline end-to-end.

    build preference pairs  ->  reward filter  ->  clean
        ->  train Stage 1/2/3 (Weight -> Rank -> List)
        ->  Direct vs Iterative DPO
        ->  analysis figures + experiment report

By default this runs the **tabular (CPU) backend** so the whole thing completes in
seconds and produces real, reproducible numbers and figures without a GPU. The
same modules expose the ``trl``/``transformers`` training path for the real
0.5B-model runs (see ``training/`` and ``configs/experiment_configs.yaml``).

Usage:
    python -m ant_dpo_project.run_pipeline
    python ant_dpo_project/run_pipeline.py --config ant_dpo_project/configs/experiment_configs.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# make the package importable whether run as a script or a module
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from ant_dpo_project.data.build_preference_pairs import (  # noqa: E402
    simulate_implicit_feedback, margin_filter as build_margin_filter)
from ant_dpo_project.data import reward_filter as rf  # noqa: E402
from ant_dpo_project.data.data_cleaning_pipeline import clean_pairs  # noqa: E402
from ant_dpo_project.utils.toy_engine import train_tabular_dpo  # noqa: E402
from ant_dpo_project.utils.metrics import win_rate  # noqa: E402
from ant_dpo_project.utils import visualization as viz  # noqa: E402
from ant_dpo_project.training.iterative_dpo import (  # noqa: E402
    run_direct_dpo, run_iterative_dpo)
from ant_dpo_project.analysis import kl_divergence_analysis as kla  # noqa: E402
from ant_dpo_project.analysis import reward_distribution as rda  # noqa: E402
from ant_dpo_project.analysis import off_policy_shift as ops  # noqa: E402

STAGE_SEQUENCE = ["weight", "rank", "list"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def three_stage_performance_figure(rows, path):
    """Grouped bar chart of win-rate / reward / KL across reference + 3 stages."""
    import matplotlib.pyplot as plt
    viz.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    names = [r["stage"] for r in rows]
    colors = [viz.color(n) for n in names]
    labels = [viz.label(n).replace(" · ", "\n") for n in names]

    for ax, (key, title) in zip(axes, [
        ("win_rate", "Win rate vs reference"),
        ("reward", "Expected true reward"),
        ("kl", r"KL($\pi_\theta\|\pi_{ref}$)"),
    ]):
        vals = [r[key] for r in rows]
        ax.bar(range(len(names)), vals, color=colors, edgecolor="white", alpha=0.9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].axhline(0.5, color="#9CA3AF", ls="--", lw=1)
    fig.suptitle("Three-stage performance (Weight -> Rank -> List)",
                 fontsize=14, fontweight="bold")
    return viz.savefig(fig, path)


def main():
    ap = argparse.ArgumentParser(description="Run the three-stage DPO pipeline")
    ap.add_argument("--config", default=os.path.join(_HERE, "configs",
                                                     "experiment_configs.yaml"))
    ap.add_argument("--results-dir", default=os.path.join(_HERE, "results"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 0)
    fig_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 70)
    print("STEP 1 — build preference pairs from simulated implicit feedback")
    print("=" * 70)
    dcfg = cfg["data"]
    fb = simulate_implicit_feedback(
        n_prompts=dcfg["n_prompts"], n_candidates=dcfg["n_candidates"],
        n_users=dcfg["n_users"], reward_spread=dcfg["reward_spread"],
        rm_noise=dcfg["rm_noise"], ref_suboptimality=dcfg["ref_suboptimality"],
        seed=seed)
    env = fb.to_env()
    raw_pairs = fb.pairs
    print(f"  simulated {len(raw_pairs)} raw preference pairs "
          f"from {dcfg['n_prompts']} prompts x {dcfg['n_candidates']} candidates")

    print("\n" + "=" * 70)
    print("STEP 2 — reward-model filtering")
    print("=" * 70)
    scored = rf.score_pairs(raw_pairs, cfg["reward_filter"].get("model_name"))
    if cfg["reward_filter"]["strategy"] == "topk":
        filtered = rf.top_k_filter(scored, cfg["reward_filter"]["top_k"])
    else:
        filtered = rf.margin_filter(scored, cfg["reward_filter"]["tau"])
    rfreport = rf.quality_report(scored, filtered)
    print(rf.format_report(rfreport))

    print("\n" + "=" * 70)
    print("STEP 3 — data cleaning")
    print("=" * 70)
    clean, creport = clean_pairs(
        filtered, min_len=cfg["cleaning"]["min_len"],
        max_len=cfg["cleaning"]["max_len"],
        simhash_threshold=cfg["cleaning"]["simhash_threshold"])
    print(creport.to_markdown())
    with open(os.path.join(args.results_dir, "data_cleaning_report.md"), "w") as f:
        f.write(creport.to_markdown() + "\n")

    print("\n" + "=" * 70)
    print("STEP 4 — three-stage training (Weight -> Rank -> List)")
    print("=" * 70)
    scfg = cfg["stages"]
    stage_policies = {"reference": env.ref_probs}
    perf_rows = [{
        "stage": "reference", "win_rate": 0.5,
        "reward": float((env.ref_probs * env.true_rewards).sum(-1).mean()),
        "kl": 0.0,
    }]
    # Curriculum: each stage warm-starts from the previous stage's policy, so the
    # ranking signal grows in complexity (single weighted pair -> all pairs ->
    # full listwise) on top of an already-improved policy.
    warm_theta = None
    for st in STAGE_SEQUENCE:
        kwargs = dict(stage=st, beta=scfg["beta"], lr=scfg["lr"],
                      steps=scfg["steps"], list_size=scfg["list_size"], seed=seed,
                      init_theta=warm_theta)
        if st == "weight":
            kwargs["weight_temperature"] = scfg["weight"]["weight_temperature"]
        if st == "rank":
            kwargs["position_discount"] = scfg["rank"]["position_discount"]
        policy, hist = train_tabular_dpo(env, **kwargs)
        warm_theta = policy.theta.copy()
        stage_policies[st] = policy.probs
        wr = win_rate(policy.probs, env.ref_probs, env.true_rewards)
        perf_rows.append({"stage": st, "win_rate": wr,
                          "reward": hist.reward[-1], "kl": hist.kl[-1]})
        print(f"  {viz.label(st):22s}  win_rate={wr:.3f}  "
              f"reward={hist.reward[-1]:+.3f}  KL={hist.kl[-1]:.3f}")

    print("\n" + "=" * 70)
    print("STEP 5 — Direct vs Iterative DPO")
    print("=" * 70)
    icfg = cfg["iterative"]
    direct = run_direct_dpo(env, stage=icfg["stage"],
                            total_steps=icfg["n_rounds"] * icfg["steps_per_round"],
                            beta=scfg["beta"], lr=scfg["lr"],
                            list_size=scfg["list_size"], seed=seed)
    itr = run_iterative_dpo(env, stage=icfg["stage"], n_rounds=icfg["n_rounds"],
                            steps_per_round=icfg["steps_per_round"], beta=scfg["beta"],
                            lr=scfg["lr"], list_size=scfg["list_size"], seed=seed)
    direct_wr = win_rate(direct[0].probs, env.ref_probs, env.true_rewards)
    iter_wr = win_rate(itr.policy.probs, env.ref_probs, env.true_rewards)
    print(f"  Direct    : win_rate={direct_wr:.3f}  reward={direct[1].reward[-1]:+.3f}  "
          f"KL={direct[1].kl[-1]:.3f}  ESS={direct[1].ess[-1]:.2f}N")
    print(f"  Iterative : win_rate={iter_wr:.3f}  reward={itr.history.reward[-1]:+.3f}  "
          f"KL={itr.history.kl[-1]:.3f}  ESS={itr.history.ess[-1]:.2f}N")

    print("\n" + "=" * 70)
    print("STEP 6 — analysis figures")
    print("=" * 70)
    f1 = three_stage_performance_figure(perf_rows,
                                        os.path.join(fig_dir, "three_stage_performance.png"))
    f2 = rda.plot_reward_distributions(env, stage_policies, clean,
                                       os.path.join(fig_dir, "reward_distribution_comparison.png"))
    f3 = kla.plot_kl_analysis(direct[1], itr.history, itr.round_boundaries,
                              itr.policy.probs, env.ref_probs,
                              os.path.join(fig_dir, "kl_divergence_curves.png"))
    f4 = ops.plot_off_policy_analysis(direct, itr, env,
                                      os.path.join(fig_dir, "off_policy_shift_analysis.png"))
    for f in (f1, f2, f3, f4):
        print("  wrote", f)

    ops_conclusions = ops.conclusions(direct, itr, env)

    print("\n" + "=" * 70)
    print("STEP 7 — experiment report")
    print("=" * 70)
    report = build_report(cfg, perf_rows, direct, direct_wr, itr, iter_wr,
                          rfreport, creport, ops_conclusions)
    report_path = os.path.join(args.results_dir, "experiment_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    with open(os.path.join(args.results_dir, "reward_filter_report.md"), "w") as f:
        f.write(rf.format_report(rfreport) + "\n")
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump({"three_stage": perf_rows,
                   "direct": {"win_rate": direct_wr, "reward": direct[1].reward[-1],
                              "kl": direct[1].kl[-1], "ess": direct[1].ess[-1]},
                   "iterative": {"win_rate": iter_wr, "reward": itr.history.reward[-1],
                                 "kl": itr.history.kl[-1], "ess": itr.history.ess[-1]},
                   "off_policy": ops_conclusions}, f, indent=2)
    print("  wrote", report_path)
    print("\nDONE. See results/ for report + figures.")


def build_report(cfg, perf_rows, direct, direct_wr, itr, iter_wr,
                 rfreport, creport, ops_conclusions):
    def row(r):
        return f"| {viz.label(r['stage'])} | {r['win_rate']:.3f} | {r['reward']:+.3f} | {r['kl']:.3f} |"
    lines = [
        "# Three-Stage DPO — Experiment Report", "",
        "_Generated by `run_pipeline.py` on the tabular (CPU) backend. Numbers are "
        "real outputs of the analytic DPO engine; the identical losses drive the "
        "`trl` training path in `training/`._", "",
        "## 1. Three-stage performance (Weight → Rank → List)", "",
        "| stage | win rate vs ref | expected reward | KL(π‖π_ref) |",
        "|---|---|---|---|",
        *[row(r) for r in perf_rows],
        "",
        "Each stage adds ranking signal: Stage 1 weights a single pair by RM "
        "confidence, Stage 2 uses all C(n,2) pairs of a ranked list, Stage 3 "
        "optimises the whole list with a Plackett–Luce objective.", "",
        "## 2. Direct vs Iterative DPO", "",
        "| method | win rate | reward | KL | final ESS |",
        "|---|---|---|---|---|",
        f"| Direct DPO | {direct_wr:.3f} | {direct[1].reward[-1]:+.3f} | "
        f"{direct[1].kl[-1]:.3f} | {direct[1].ess[-1]:.2f} N |",
        f"| Iterative DPO ({cfg['iterative']['n_rounds']} rounds) | {iter_wr:.3f} | "
        f"{itr.history.reward[-1]:+.3f} | {itr.history.kl[-1]:.3f} | "
        f"{itr.history.ess[-1]:.2f} N |",
        "",
        "## 3. Off-policy distribution shift — key finding", "",
        f"> {ops_conclusions['verdict']}", "",
        "| diagnostic | value |",
        "|---|---|",
        f"| Direct final importance-weight variance | {ops_conclusions['direct_final_weight_variance']:.3f} |",
        f"| Direct final ESS (fraction of N) | {ops_conclusions['direct_final_ess_frac']:.3f} |",
        f"| Step Direct ESS first crosses 0.5 N | {ops_conclusions['direct_ess_cross_0.5_step']} |",
        f"| Iterative final ESS (fraction of N) | {ops_conclusions['iterative_final_ess_frac']:.3f} |",
        "",
        "## 4. Data pipeline", "",
        f"Reward-model filtering retained {rfreport['retention']*100:.1f}% of pairs; "
        f"cleaning retained {100*creport.final/max(1,creport.initial):.1f}%. See "
        "`reward_filter_report.md` and `data_cleaning_report.md`.", "",
        "## 5. Figures", "",
        "- `figures/three_stage_performance.png`",
        "- `figures/reward_distribution_comparison.png`",
        "- `figures/kl_divergence_curves.png`",
        "- `figures/off_policy_shift_analysis.png`", "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
