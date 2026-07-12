# Three-Stage DPO Preference-Optimization System

A complete, runnable pipeline for **Direct Preference Optimization (DPO)** that
implements a three-stage *curriculum* over preference signals, an iterative
(on-policy) variant, and the analysis tooling to reason about **off-policy
distribution shift** — the failure mode that decides when plain DPO is enough and
when you must go iterative.

```
implicit feedback ──▶ reward-model filtering ──▶ cleaning
      │
      ▼
  Stage 1  Weight-DPO   (confidence-weighted pair)
  Stage 2  Rank-DPO     (all pairs of a ranked list, position-weighted)
  Stage 3  List-DPO     (whole list, Plackett–Luce / ListMLE)
      │
      ▼
  Direct vs Iterative DPO  ──▶  KL / reward / off-policy-shift analysis
```

## Why this repo runs anywhere

Studying these *algorithms* does not require a 0.5B model or a GPU. The core loss
math (Weighted / Rank / List DPO) is a pure function of sequence log-probabilities,
so it is implemented once and driven by **two backends**:

* **`utils/toy_engine.py`** — a numpy **tabular scale-model** where the policy is an
  explicit categorical over a candidate set. Every quantity of interest (KL,
  expected reward, importance weights, ESS) is available in closed form, so the
  full pipeline runs in **~8 seconds on CPU** and produces *real, reproducible*
  numbers and figures. Its analytic gradients are checked against finite
  differences (`tests/test_losses.py`).
* **`training/*.py`** — the **real training path**: `trl.DPOTrainer` subclasses and
  custom listwise trainers for `Qwen/Qwen2.5-0.5B` / `SmolLM2-360M` with LoRA. These
  need a GPU + `trl`/`transformers`; they import (and type-check) without them but
  raise on instantiation. They call the *same* loss functions as the toy engine —
  verified identical in `tests/test_losses.py` (torch vs numpy agree to 1e-5).

All headline results below are the actual output of `python -m ant_dpo_project.run_pipeline`.

## Results (tabular backend, seed 0)

**Three-stage curriculum** — each stage warm-starts from the previous and adds
ranking signal:

| stage | win rate vs ref | expected reward | KL(π‖π_ref) |
|---|---|---|---|
| Reference (SFT) | 0.500 | +0.903 | 0.000 |
| Stage 1 · Weight-DPO | 0.662 | +1.582 | 0.455 |
| Stage 2 · Rank-DPO | 0.748 | +1.852 | 0.822 |
| Stage 3 · List-DPO | **0.800** | **+2.007** | 1.362 |

**Direct vs Iterative DPO** — comparable reward, very different estimator health:

| method | win rate | reward | KL | final ESS |
|---|---|---|---|---|
| Direct DPO | 0.719 | +1.776 | 0.769 | 0.46 N |
| Iterative DPO (3 rounds) | 0.716 | +1.769 | 0.733 | **0.90 N** |

Direct DPO's effective sample size decays monotonically and crosses 0.5 N at
step 250; iterative re-anchors it to ~1 N every round. See
`results/figures/off_policy_shift_analysis.png`.

## The four things this project is built to explain

**1. Why three stages (Weight → Rank → List) and not just List-DPO?**
*Curriculum learning over the ranking signal.* The listwise (Plackett–Luce)
objective is the richest but least forgiving: every response's target depends on
every other, so early in training — when the policy is still near a mediocre
reference — its gradients are high-variance. Warming up on the easy, low-variance
signal first (one confidence-weighted pair), then all pairs of a list, then the
full listwise loss, gives a smoother optimization path. In the run above the
curriculum climbs 0.662 → 0.748 → 0.800 win rate with monotonic reward gains.

**2. How do you *locate* off-policy distribution shift?**
Two coupled diagnostics on the importance weight `w = π_θ(y)/μ(y)` for data drawn
from behaviour policy `μ`:
* **weight variance** `Var_μ[w] = Σ_y π²/μ − 1` — blows up as π_θ leaves μ;
* **effective sample size** `ESS = (Σw)²/Σw²` — collapses toward 1.
`analysis/off_policy_shift.py` tracks both over training and reports the step at
which normalized ESS first drops below 0.5 N. That step is the honest "your data is
now stale" signal.

**3. What does Iterative DPO actually solve?**
Direct DPO trains on a *fixed* dataset sampled once from `π_ref`; as `π_θ` moves,
that data is increasingly off-policy (weights fan out, ESS collapses). Iterative
DPO re-samples on-policy every round (`μ ← π_current`), so the importance weights
re-anchor to ~1 and ESS resets — at the cost of repeated generation + reward
scoring. The payoff is **estimator robustness**, not necessarily higher reward in
mild regimes (here reward is within ±0.01 but ESS is 2× healthier). It matters most
when you train long/hard (large β, many steps).

**4. Why is reward-model filtering important?**
DPO trusts every pair as ground truth. Small-margin pairs are near-ties dominated by
reward-model noise; feeding them in injects label noise. In this run, margin
filtering cut the pair set to 79% but dropped the **label error rate 8× (0.040 →
0.005)** — the fraction of pairs where the "chosen" is actually worse by true reward.
`reward_distribution.py` also shows a positive correlation (r≈0.25) between a
prompt's reward-model margin and its realised reward gain, i.e. big-margin pairs are
where the learning actually happens.

## Layout

```
ant_dpo_project/
├── data/
│   ├── build_preference_pairs.py   # simulate 500k-DAU implicit feedback → (prompt, chosen, rejected, strength)
│   ├── reward_filter.py            # RM scoring + top-k / margin filtering + quality report
│   └── data_cleaning_pipeline.py   # exact+SimHash dedup, length/language/toxicity filters
├── training/
│   ├── weight_dpo.py               # Stage 1: WeightedDPOTrainer(DPOTrainer)
│   ├── rank_dpo.py                 # Stage 2: listwise-capable custom Trainer, all-pairs ranking loss
│   ├── list_dpo.py                 # Stage 3: Plackett–Luce (ListMLE) listwise loss
│   └── iterative_dpo.py            # Direct vs Iterative comparison (+ real-model loop reference)
├── analysis/
│   ├── kl_divergence_analysis.py   # KL over training + per-prompt KL histogram
│   ├── reward_distribution.py      # per-stage reward violins, margin distribution, margin↔gain
│   └── off_policy_shift.py         # importance-weight hist, ESS trajectories, verdict
├── utils/
│   ├── dpo_losses.py               # torch loss fns (the real training path)
│   ├── toy_engine.py               # numpy tabular DPO engine (CPU demo) + analytic gradients
│   ├── metrics.py                  # KL, ESS, importance weights, win rate
│   └── visualization.py            # one consistent, colour-blind-safe plotting style
├── configs/experiment_configs.yaml
├── tests/test_losses.py            # finite-difference gradient checks + torch/numpy agreement
├── run_pipeline.py                 # main entry point
└── results/                        # generated report + figures (committed)
```

## Running

```bash
pip install -r requirements.txt          # numpy/pandas/matplotlib/seaborn/scipy/pyyaml (+ torch optional)

# full pipeline on CPU (~8s): data → filter → clean → 3 stages → direct/iterative → figures + report
python -m ant_dpo_project.run_pipeline

# individual modules also run standalone, e.g.
python -m ant_dpo_project.data.build_preference_pairs --n-prompts 400 --n-users 500000
python -m ant_dpo_project.analysis.off_policy_shift

# correctness checks (gradient checks + torch/numpy loss agreement)
python ant_dpo_project/tests/test_losses.py
```

### Real 0.5B training (GPU)

Install `torch transformers trl peft accelerate datasets`, then wire the trainers in
`training/` to a base model. `training/iterative_dpo.py::real_model_iterative_loop`
documents the exact generate → score → rebuild-pairs → DPO loop the tabular engine
emulates. Config lives under the `model` / `lora` / `train` blocks of
`configs/experiment_configs.yaml`.

## Notes on the synthetic data

The `synthetic` backend is a *controlled* stand-in: latent per-response rewards, a
noisy reward model, and a partially-aligned reference policy, with users
click/dwell/select/skip stochastically as a function of true reward. It exists so
the algorithmic dynamics are reproducible without a model. `build_preference_pairs.py`
also has a best-effort `hf` backend that streams UltraFeedback / HH-RLHF when
`datasets` + network are available. (Templated synthetic responses are intentionally
repetitive, which is why the SimHash near-dedup stage drops a large fraction — on
real data that stage is far gentler.)
