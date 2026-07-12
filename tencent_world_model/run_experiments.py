#!/usr/bin/env python3
"""End-to-end experiment runner for the Neural World Model + latent GRPO/PPO project.

Stages
------
1. Train a shared world model per reward mode (+ a raw-observation baseline).
2. Run GRPO vs PPO in latent space over several seeds, per reward mode.
3. World-model quality: multi-step prediction error, long-range accuracy vs the
   raw baseline, reconstruction sample, latent-space embedding.
4. GRPO group-size ablation (G = 4, 8, 16, ...).
5. Long-horizon stress test (T = 100 / 200).
6. Generate all figures and a data-driven technical report.

Usage
-----
    python run_experiments.py --profile quick        # default, ~10-20 min CPU
    python run_experiments.py --profile full          # ~1-2 h CPU
    python run_experiments.py --profile quick --only figures   # re-plot from saved results
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from typing import Dict

import numpy as np
import torch
import yaml

from envs.sequence_prediction import (SequencePredictionEnv, SequencePredictionConfig,
                                       collect_transitions)
from training.train_world_model import (WorldModelTrainConfig, long_range_comparison)
from training.grpo_vs_ppo import (train_shared_world_model, run_comparison,
                                   run_algorithm, make_env)
from utils import visualization as viz
from analysis import stability_analysis, convergence_analysis, sparse_reward_analysis

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_config(profile: str) -> Dict:
    with open(os.path.join(HERE, "configs", "experiment_configs.yaml")) as f:
        cfg = yaml.safe_load(f)
    prof = cfg["profiles"][profile]
    merged = dict(cfg["common"])
    merged["reward_modes"] = cfg["reward_modes"]
    merged["profile"] = profile
    merged.update(prof)
    return merged


def hyper_from_cfg(cfg: Dict) -> Dict:
    return {k: cfg[k] for k in ["gamma", "clip_eps", "lr", "entropy_coef",
                                "update_epochs", "group_size", "num_prompts",
                                "gae_lambda", "value_coef", "policy_hidden"]}


def wm_train_cfg(cfg: Dict, seed: int = 0) -> WorldModelTrainConfig:
    w = cfg["world_model"]
    return WorldModelTrainConfig(num_transitions=w["num_transitions"], epochs=w["epochs"],
                                 batch_size=w["batch_size"], lr=w["lr"], seed=seed)


# --------------------------------------------------------------------- stages
def run_all(cfg: Dict) -> Dict:
    hyper = hyper_from_cfg(cfg)
    horizon = cfg["horizon"]
    latent_dim = cfg["latent_dim"]
    seeds = cfg["rl"]["seeds"]
    num_iters = cfg["rl"]["num_iterations"]
    eval_every = cfg["rl"]["eval_every"]
    eval_eps = cfg["eval_episodes"]

    out = {"config": cfg, "results_by_mode": {}, "wm_info": {}}

    # --- Stages 1+2: per reward mode, train WM then compare GRPO/PPO ----------
    world_models = {}
    envs = {}
    baselines = {}
    for mode in cfg["reward_modes"]:
        log(f"=== Reward mode: {mode} -- training world model ===")
        wm, baseline, env, info = train_shared_world_model(
            mode, horizon, latent_dim, wm_train_cfg(cfg), train_baseline=(mode == "dense"),
            log_fn=log)
        world_models[mode] = wm
        envs[mode] = env
        baselines[mode] = baseline
        out["wm_info"][mode] = info
        log(f"    world-model multi-step error: {info['multistep_error']}")

        log(f"=== Reward mode: {mode} -- GRPO vs PPO ({len(seeds)} seeds x {num_iters} iters) ===")
        results = run_comparison(env, wm, num_iters, seeds, hyper, eval_every=eval_every,
                                 eval_episodes=eval_eps, log_fn=log)
        out["results_by_mode"][mode] = results

    # --- Stage 3: world-model quality (dense model) --------------------------
    log("=== World-model quality (long-range prediction vs raw baseline) ===")
    dense_env = envs["dense"]
    dense_wm = world_models["dense"]
    lr_env = make_env("dense", horizon=60, seed=0)  # long enough for H=50 eval
    long_range = long_range_comparison(lr_env, dense_wm, baselines["dense"],
                                       horizon=50, episodes=40)
    out["long_range"] = long_range
    log(f"    long-range (H=50) accuracy: world_model={long_range['world_model_accuracy']:.3f} "
        f"baseline={long_range['baseline_accuracy']:.3f} "
        f"improvement={long_range['relative_improvement_pct']:+.1f}%")

    # reconstruction sample + latent embedding data
    recon_true, recon_pred, latents, prox = latent_and_recon(dense_env, dense_wm)
    out["recon_sample"] = {"true": recon_true.tolist(), "pred": recon_pred.tolist()}
    out["latent_data"] = {"latents": latents, "proximity": prox}

    # --- Stage 4: GRPO group-size ablation (dense) ---------------------------
    log("=== GRPO group-size ablation (dense) ===")
    ablation = {}
    for G in cfg["ablation"]["group_sizes"]:
        hyp = dict(hyper)
        hyp["group_size"] = G
        # keep total trajectory budget ~constant: prompts * G ~ const
        hyp["num_prompts"] = max(4, (hyper["num_prompts"] * hyper["group_size"]) // G)
        hists = []
        for seed in seeds:
            hists.append(run_algorithm("grpo", dense_env, dense_wm, num_iters, seed, hyp,
                                       eval_every=eval_every, eval_episodes=eval_eps))
        ablation[G] = hists
        finals = [h["eval_proximity"][-1] for h in hists]
        log(f"    G={G}: final proximity {np.mean(finals):.3f} +/- {np.std(finals):.3f}")
    out["ablation"] = ablation

    # --- Stage 5: long-horizon stress test (dense) ---------------------------
    long_h = cfg["long_horizon"]
    log(f"=== Long-horizon stress test (T={long_h}, dense) ===")
    long_env = make_env("dense", horizon=long_h, seed=0)
    long_hyper = dict(hyper)
    lh_results = run_comparison(long_env, dense_wm, num_iters, seeds[:max(2, len(seeds) // 2)],
                                long_hyper, eval_every=eval_every, eval_episodes=eval_eps,
                                log_fn=log)
    # add horizon into each trainer's config is handled by env.cfg.horizon
    out["long_horizon_results"] = lh_results
    out["long_horizon"] = long_h

    return out


def latent_and_recon(env, wm, n: int = 2500):
    """Collect observations, encode them, and return a recon sample + latent embedding data."""
    data = collect_transitions(env, n, seed=777)
    obs = torch.as_tensor(data["observations"], dtype=torch.float32)
    with torch.no_grad():
        _, mu, _ = wm.encode(obs, sample=False)
        recon = wm.decode(mu)
    latents = mu.numpy()
    prox = data["proximities"]
    # a single reconstruction example
    return (data["observations"][0], recon.numpy()[0], latents, prox)


# --------------------------------------------------------------------- figures
def make_figures(out: Dict, cfg: Dict):
    os.makedirs(FIG_DIR, exist_ok=True)
    modes = cfg["reward_modes"]
    thr = cfg["convergence_threshold"]

    # core reward curves (dense)
    viz.plot_reward_curves(out["results_by_mode"]["dense"],
                           os.path.join(FIG_DIR, "grpo_vs_ppo_reward_curves.png"))
    # stability (dense)
    viz.plot_training_stability(out["results_by_mode"]["dense"],
                                os.path.join(FIG_DIR, "training_stability_comparison.png"))
    # sparse reward comparison (all modes)
    viz.plot_sparse_reward_comparison(out["results_by_mode"],
                                      os.path.join(FIG_DIR, "sparse_reward_comparison.png"))
    # convergence efficiency (dense)
    viz.plot_convergence_efficiency(out["results_by_mode"]["dense"],
                                    os.path.join(FIG_DIR, "convergence_efficiency.png"), thr)
    # latent space
    ld = out["latent_data"]
    viz.plot_latent_space(np.asarray(ld["latents"]), np.asarray(ld["proximity"]),
                          os.path.join(FIG_DIR, "latent_space_visualization.png"),
                          method="pca")
    # world-model quality
    viz.plot_world_model_quality(
        {int(k): v for k, v in out["wm_info"]["dense"]["multistep_error"].items()},
        os.path.join(FIG_DIR, "world_model_prediction_quality.png"),
        recon_true=np.asarray(out["recon_sample"]["true"]),
        recon_pred=np.asarray(out["recon_sample"]["pred"]),
        long_range=out["long_range"])
    # group-size ablation
    viz.plot_group_size_ablation(out["ablation"],
                                 os.path.join(FIG_DIR, "grpo_group_size_ablation.png"))
    # long-horizon
    viz.plot_reward_curves(out["long_horizon_results"],
                           os.path.join(FIG_DIR, "long_horizon_comparison.png"),
                           title=f"Long-horizon (T={out['long_horizon']}) GRPO vs PPO")
    log(f"    figures written to {FIG_DIR}")


# --------------------------------------------------------------------- report
def make_report(out: Dict, cfg: Dict):
    from report import build_report  # local module
    build_report(out, cfg, RESULTS_DIR)


def save_results(out: Dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # latents can be large -> store separately as npz, keep pickle lean
    with open(os.path.join(RESULTS_DIR, "raw_results.pkl"), "wb") as f:
        pickle.dump(out, f)
    log(f"    raw results saved to {os.path.join(RESULTS_DIR, 'raw_results.pkl')}")


def load_results() -> Dict:
    with open(os.path.join(RESULTS_DIR, "raw_results.pkl"), "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="quick", choices=["smoke","quick","full"])
    ap.add_argument("--only", default="all", choices=["all", "figures", "report"],
                    help="'figures'/'report' reuse saved results")
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    cfg = load_config(args.profile)

    if args.only in ("figures", "report"):
        out = load_results()
        cfg = out.get("config", cfg)
    else:
        t0 = time.time()
        out = run_all(cfg)
        save_results(out)
        log(f"experiments finished in {time.time() - t0:.1f}s")

    if args.only in ("all", "figures"):
        log("generating figures ...")
        make_figures(out, cfg)
    if args.only in ("all", "report"):
        log("generating report ...")
        make_report(out, cfg)
    log("done.")


if __name__ == "__main__":
    main()
