# Neural World Model + Latent-Space GRPO

Train a **Neural World Model**, then optimise a control policy **entirely inside
its learned latent space** using **GRPO** (Group Relative Policy Optimisation,
critic-free), and compare head-to-head against **PPO** (with a value critic).
The project quantifies *where* a critic-free, group-relative method wins and
where a value-based method wins — with stability, convergence, and sparse-reward
analyses, plus a long-range prediction study.

```
observation (high-dim, noisy)
        │  VAE encoder
        ▼
   latent state z ──►  latent SSM dynamics  ──►  reward head (proximity)
        │  (S4/Mamba-style: linear recurrence + non-linear residual)
        ▼
   latent policy π(a│z)  ──►  GRPO / PPO update  (rollouts happen in latent space, no env)
```

---

## Why this design

1. **Why RL in latent space, not observation space?** The observation is a
   high-dimensional, noisy projection of a small hidden state. Encoding to a
   compact, denoised latent means the policy/critic are tiny, the dynamics are
   more nearly linear (easier long-range rollout), and model-based rollouts are
   cheap — so the agent can be trained on imagined latent trajectories without
   touching the real environment.

2. **GRPO vs PPO.** GRPO drops the value network entirely and computes a
   **group-relative advantage**: sample `G` action sequences from the same
   initial state, and score each by how far its return is above/below the group
   mean. This means (a) **fewer parameters** (no critic), (b) **no
   value-estimation bias** — which is exactly what hurts a bootstrapped critic
   under sparse/delayed reward, and (c) **scale-invariance** via group
   normalisation. The cost is `G` rollouts per state.

3. **The SSM.** Latent dynamics are `z_{t+1} = A z_t + B a_t + MLP(z_t,a_t)` — a
   stable linear core (well-conditioned long-horizon propagation) plus a
   non-linear residual for expressiveness. This is the same intuition as
   S4/Mamba: keep the long-range part linear and stable.

4. **Relation to DAPO / VAPO.** GRPO is the critic-free foundation both build
   on (DAPO adds decoupled clipping + dynamic sampling). This repo is a
   controlled check that the *value-free* mechanism itself is sound in a latent
   world model.

---

## Quickstart

```bash
pip install -r requirements.txt

# Fast, CPU-friendly end-to-end run (~10-20 min). Produces every figure + report.
python run_experiments.py --profile quick

# Paper-scale (10 seeds, more iterations; ~1-2 h CPU)
python run_experiments.py --profile full

# Re-plot / re-report from already-saved results (no re-training)
python run_experiments.py --profile quick --only figures
python run_experiments.py --profile quick --only report
```

Outputs land in [`results/`](results/): `technical_report.md`, `metrics.json`,
and eight figures in `results/figures/`.

---

## Project structure

```
tencent_world_model/
├── models/
│   ├── world_model.py   # VAE encoder + latent dynamics (SSM) + reward head + decoder
│   ├── ssm.py           # LatentSSM: linear recurrence + non-linear residual
│   └── policy.py        # LatentPolicy (actor) and ValueNet (critic, PPO-only)
├── training/
│   ├── train_world_model.py  # Stage 1: collect data + train world model (+ raw baseline)
│   ├── grpo_latent.py        # Stage 2: GRPO in latent space (critic-free)
│   ├── ppo_latent.py         # Baseline: PPO in latent space (GAE + critic)
│   ├── latent_rollout.py     # shared latent-rollout / reward-mode helpers
│   └── grpo_vs_ppo.py        # comparison orchestrator
├── analysis/
│   ├── stability_analysis.py     # entropy collapse, grad norms, return volatility, crashes
│   ├── convergence_analysis.py   # iters/time-to-threshold, params, per-update speed
│   └── sparse_reward_analysis.py # critic explained-variance vs group-signal fraction
├── envs/
│   ├── sequence_prediction.py    # main env: controllable non-linear system (dense/sparse/very_sparse)
│   └── cartpole_latent.py        # classic-control cross-check (gymnasium CartPole)
├── utils/
│   ├── metrics.py         # seeding, moving averages, grad norm, returns, AUC
│   └── visualization.py   # all figure generators
├── configs/experiment_configs.yaml
├── run_experiments.py     # end-to-end runner (stages 1-6 + figures + report)
├── report.py              # data-driven technical-report generator
└── results/               # generated figures, report, metrics
```

---

## What each experiment measures

| Figure | Question |
|---|---|
| `grpo_vs_ppo_reward_curves.png` | Do both learn in latent space? (dense, mean±std over seeds) |
| `sparse_reward_comparison.png` | dense vs sparse vs very_sparse — where does critic-free help? |
| `training_stability_comparison.png` | entropy collapse, gradient norm, return volatility |
| `convergence_efficiency.png` | iterations & wall-clock to threshold; parameter counts |
| `world_model_prediction_quality.png` | multi-step error, reconstruction, long-range accuracy vs baseline |
| `latent_space_visualization.png` | PCA of latent states coloured by goal proximity |
| `grpo_group_size_ablation.png` | effect of GRPO group size G |
| `long_horizon_comparison.png` | GRPO vs PPO at long horizon (T=100/200) |

The **long-range prediction** study compares the latent world model against a
raw-observation-space dynamics model at predicting the *clean underlying signal*
`H` steps ahead. Both train on identical noisy data; the VAE bottleneck denoises
as an emergent inductive bias. The reported improvement is measured from the run
(see `results/technical_report.md`), not asserted — it is environment-dependent.

---

## The environment

`envs/sequence_prediction.py` is a controllable non-linear dynamical system.
A hidden state `s ∈ ℝ⁴` is driven by discrete control actions toward a goal;
the agent sees `obs = W·s + noise ∈ ℝ⁴⁸`. A single `reward_mode` knob sets how
the underlying goal-proximity signal is *delivered in time*:

- `dense` — proximity reward every step;
- `sparse` — proximity reward only on the final step;
- `very_sparse` — a single `+1` the first time the goal region is entered.

The underlying quantity is identical across modes, so the experiments isolate
the **credit-assignment difficulty** (which hurts a bootstrapped critic) from
the difficulty of learning the reward *function* (which does not).

---

## Reproducibility notes

- CPU-only; all seeds are set via `utils.metrics.set_seed`.
- `results/` in this repo was generated with `--profile quick`; `--profile full`
  tightens the error bands.
- Honest caveat: with a well-fit reward model and enough iterations, GRPO and
  PPO reach similar asymptotic performance here; GRPO's advantage in this study
  is efficiency, simplicity, and robustness to reward sparsity, not a higher
  ceiling. See §10 of the technical report.
