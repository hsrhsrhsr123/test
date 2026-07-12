"""Build preference pairs from (simulated) implicit user feedback.

Story
-----
An assistant serving ~500k DAU logs a torrent of *implicit* signals — clicks,
dwell time, which candidate the user picked, which they skipped. None of these is
a clean preference label, but in aggregate they reveal which response the
population prefers. This module turns that signal into the
``(prompt, chosen, rejected, preference_strength)`` quadruples DPO consumes, and
applies **margin-based filtering** so only pairs with a real reward gap survive.

Two backends
------------
* ``--backend synthetic`` (default, always works): a generative model of the
  feedback funnel. Every candidate has a latent true reward; users click / dwell /
  select / skip stochastically as a function of it. Aggregating many impressions
  per prompt yields a noisy ``preference_strength`` — noisier when traffic is thin,
  tight when traffic is heavy (the 500k-DAU regime).
* ``--backend hf`` (best-effort): stream UltraFeedback / HH-RLHF from the Hub if
  ``datasets`` and network are available. Falls back to synthetic with a warning.

The synthetic backend is also the single source of the world used by the rest of
the pipeline: :meth:`FeedbackData.to_env` exports the arrays the tabular trainer
and the analysis modules consume, so data-building, filtering, training and
analysis all describe *the same* population.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---- synthetic text scaffolding ------------------------------------------- #
_TOPICS = [
    "reset my password", "a refund for a late order", "explain gradient descent",
    "plan a 3-day trip to Kyoto", "debug a segfault in C", "write a cover letter",
    "summarize this contract", "improve my resume bullet", "translate a phrase",
    "recommend a good book", "fix a slow SQL query", "name a startup",
]
_STYLES = [
    "Here is a clear, step-by-step answer:", "Sure — the short version is:",
    "Let me walk you through it.", "Honestly, it depends, but broadly:",
    "Quick answer:", "I'd suggest the following approach:",
]
_FILLERS = [
    "First, check the settings panel.", "Consider the trade-offs carefully.",
    "This usually resolves the issue.", "Make sure to verify each step.",
    "You may need to restart afterwards.", "Documentation covers the edge cases.",
]
# deliberately planted noise so data_cleaning has real work to do
_TOXIC = ["you idiot, just", "this is stupid,"]
_NON_EN = ["Voici la réponse en français:", "これは日本語の回答です:"]


@dataclass
class FeedbackData:
    """Container for the simulated world + the derived preference table."""

    prompts: list           # length P
    responses: np.ndarray   # object array [P, k] of response strings
    true_rewards: np.ndarray  # [P, k]
    rm_scores: np.ndarray     # [P, k]  (set later by reward_filter; here = noisy obs)
    ref_logits: np.ndarray    # [P, k]
    pairs: pd.DataFrame       # the (prompt, chosen, rejected, strength, ...) table

    def to_env(self):
        """Export a :class:`ToyPreferenceEnv` for the tabular trainer."""
        from ..utils.toy_engine import ToyPreferenceEnv
        return ToyPreferenceEnv(self.true_rewards, self.rm_scores, self.ref_logits)


# --------------------------------------------------------------------------- #
# synthetic backend                                                            #
# --------------------------------------------------------------------------- #
def _make_response_text(rng, topic, quality, planted):
    """Craft a response string whose *length/style* loosely tracks quality."""
    style = _STYLES[rng.integers(len(_STYLES))]
    n_fill = int(np.clip(1 + quality + rng.normal(0, 1), 0, 5))
    body = " ".join(rng.choice(_FILLERS, size=max(1, n_fill)))
    text = f"To {topic}: {style} {body}"
    if planted == "toxic":
        text = f"{_TOXIC[rng.integers(len(_TOXIC))]} {text}"
    elif planted == "non_en":
        text = f"{_NON_EN[rng.integers(len(_NON_EN))]} {text}"
    elif planted == "short":
        text = "ok"
    elif planted == "long":
        text = text + (" " + body) * 12
    return text


def simulate_implicit_feedback(
    n_prompts: int = 400,
    n_candidates: int = 8,
    n_users: int = 500_000,
    reward_spread: float = 1.5,
    rm_noise: float = 0.35,
    ref_suboptimality: float = 0.8,
    user_rationality: float = 1.2,
    planted_noise_frac: float = 0.06,
    seed: int = 0,
) -> FeedbackData:
    """Simulate the feedback funnel and emit preference pairs.

    ``n_users`` scales the number of impressions per prompt (traffic). More
    traffic -> lower-variance ``preference_strength`` estimates.
    """
    rng = np.random.default_rng(seed)

    true_rewards = rng.normal(0.0, reward_spread, size=(n_prompts, n_candidates))
    rm_scores = true_rewards + rng.normal(0.0, rm_noise, size=true_rewards.shape)
    ref_logits = ref_suboptimality * true_rewards + rng.normal(
        0.0, 1.0, size=true_rewards.shape)

    # impressions per prompt scale with DAU: more traffic -> more impressions per
    # prompt -> lower-variance preference_strength estimates. Capped for runtime.
    impressions = int(np.clip(n_users // max(1, n_prompts), 60, 1500))

    prompts, responses = [], np.empty((n_prompts, n_candidates), dtype=object)
    rows = []
    for p in range(n_prompts):
        topic = _TOPICS[rng.integers(len(_TOPICS))]
        prompts.append(f"How do I {topic}?")
        # plant a little noise into some candidates
        for c in range(n_candidates):
            planted = "clean"
            if rng.random() < planted_noise_frac:
                planted = rng.choice(["toxic", "non_en", "short", "long"])
            responses[p, c] = _make_response_text(rng, topic, true_rewards[p, c], planted)

        # --- implicit feedback: many impressions of random candidate pairs ---
        r = true_rewards[p]
        # aggregate pairwise "wins" from stochastic user choices
        a = rng.integers(0, n_candidates, size=impressions)
        b = rng.integers(0, n_candidates, size=impressions)
        valid = a != b
        a, b = a[valid], b[valid]
        # Bradley-Terry choice: P(pick a) = sigmoid(rationality*(r_a - r_b))
        pick_a_prob = 1.0 / (1.0 + np.exp(-user_rationality * (r[a] - r[b])))
        picked_a = rng.random(len(a)) < pick_a_prob

        # tally wins per unordered pair
        win = {}
        cnt = {}
        for ai, bi, pa in zip(a, b, picked_a):
            key = (ai, bi) if ai < bi else (bi, ai)
            cnt[key] = cnt.get(key, 0) + 1
            winner = ai if pa else bi
            win[key] = win.get(key, 0) + (1 if winner == key[0] else 0)

        for (x, y), n_imp in cnt.items():
            if n_imp < 5:
                continue
            frac_x = win[(x, y)] / n_imp                # empirical P(x preferred)
            # strength = |2*frac - 1| in [0,1]; direction sets chosen/rejected
            strength = abs(2 * frac_x - 1)
            if frac_x >= 0.5:
                ci, ri = x, y
            else:
                ci, ri = y, x
            # dwell-time signal (auxiliary): higher reward -> longer dwell
            dwell_c = max(0.0, r[ci] + rng.normal(0, 0.5))
            dwell_r = max(0.0, r[ri] + rng.normal(0, 0.5))
            rows.append({
                "prompt_id": p,
                "prompt": prompts[p],
                "chosen": responses[p, ci],
                "rejected": responses[p, ri],
                "chosen_idx": int(ci),
                "rejected_idx": int(ri),
                "preference_strength": float(strength),
                "n_impressions": int(n_imp),
                "reward_chosen": float(rm_scores[p, ci]),
                "reward_rejected": float(rm_scores[p, ri]),
                "true_reward_chosen": float(r[ci]),
                "true_reward_rejected": float(r[ri]),
                "dwell_chosen": float(dwell_c),
                "dwell_rejected": float(dwell_r),
                "signal": "explicit_choice" if strength > 0.4 else "weak_dwell",
            })

    pairs = pd.DataFrame(rows)
    return FeedbackData(prompts, responses, true_rewards, rm_scores, ref_logits, pairs)


def margin_filter(pairs: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Keep only pairs whose reward-model margin exceeds ``threshold``.

    Small-margin pairs are near-ties: the preference signal is weak and often
    just reward-model noise, which injects label noise into DPO.
    """
    margin = pairs["reward_chosen"] - pairs["reward_rejected"]
    return pairs[margin > threshold].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# best-effort HF backend                                                       #
# --------------------------------------------------------------------------- #
def load_hf_pairs(dataset: str = "ultrafeedback", max_rows: int = 5000) -> pd.DataFrame:
    """Load a real preference dataset from the Hub, or raise for fallback."""
    from datasets import load_dataset  # optional dep; raises if missing

    if dataset == "ultrafeedback":
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized",
                          split=f"train_prefs[:{max_rows}]")
        prompt_key, ch, rj = "prompt", "chosen", "rejected"
    else:  # hh-rlhf
        ds = load_dataset("Anthropic/hh-rlhf", split=f"train[:{max_rows}]")
        prompt_key, ch, rj = None, "chosen", "rejected"

    rows = []
    for i, ex in enumerate(ds):
        chosen = ex[ch][-1]["content"] if isinstance(ex[ch], list) else ex[ch]
        rejected = ex[rj][-1]["content"] if isinstance(ex[rj], list) else ex[rj]
        prompt = ex.get(prompt_key, "") if prompt_key else ""
        rows.append({"prompt_id": i, "prompt": prompt, "chosen": chosen,
                     "rejected": rejected, "preference_strength": 1.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Build preference pairs from implicit feedback")
    ap.add_argument("--backend", choices=["synthetic", "hf"], default="synthetic")
    ap.add_argument("--dataset", choices=["ultrafeedback", "hh"], default="ultrafeedback")
    ap.add_argument("--n-prompts", type=int, default=400)
    ap.add_argument("--n-candidates", type=int, default=8)
    ap.add_argument("--n-users", type=int, default=500_000)
    ap.add_argument("--margin-threshold", type=float, default=0.5)
    ap.add_argument("--out", default="results/preference_pairs.parquet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.backend == "hf":
        try:
            pairs = load_hf_pairs(args.dataset)
            print(f"[build] loaded {len(pairs)} pairs from HF {args.dataset}")
        except Exception as e:
            print(f"[build] HF backend unavailable ({e}); falling back to synthetic.")
            args.backend = "synthetic"

    if args.backend == "synthetic":
        fb = simulate_implicit_feedback(
            n_prompts=args.n_prompts, n_candidates=args.n_candidates,
            n_users=args.n_users, seed=args.seed)
        pairs = fb.pairs
        print(f"[build] simulated {len(pairs)} raw pairs "
              f"from {args.n_prompts} prompts x {args.n_candidates} candidates "
              f"({args.n_users:,} DAU)")

    kept = margin_filter(pairs, args.margin_threshold) if "reward_chosen" in pairs else pairs
    print(f"[build] margin-filter (tau={args.margin_threshold}): "
          f"{len(pairs)} -> {len(kept)} pairs "
          f"({100*len(kept)/max(1,len(pairs)):.1f}% kept)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    try:
        kept.to_parquet(args.out)
    except Exception:
        args.out = args.out.replace(".parquet", ".csv")
        kept.to_csv(args.out, index=False)
    print(f"[build] wrote {args.out}")


if __name__ == "__main__":
    main()
