"""Correctness tests: numerical gradient checks + torch/numpy loss agreement.

Run with:  python -m pytest ant_dpo_project/tests/ -q
or simply: python ant_dpo_project/tests/test_losses.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ant_dpo_project.utils import toy_engine as te
from ant_dpo_project.utils.metrics import log_softmax


def _finite_diff_grad(loss_fn, theta, eps=1e-6):
    """Central-difference gradient of loss_fn(theta) w.r.t. theta."""
    grad = np.zeros_like(theta)
    it = np.nditer(theta, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = theta[idx]
        theta[idx] = orig + eps
        lp = loss_fn(theta)
        theta[idx] = orig - eps
        lm = loss_fn(theta)
        theta[idx] = orig
        grad[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    return grad


def _analytic_grad(theta, ref_logits, beta, kind, order=None, chosen=None,
                   rejected=None, weights=None, position_discount=0.9):
    lp = log_softmax(theta, axis=-1)
    rlp = log_softmax(ref_logits, axis=-1)
    if kind == "pair":
        _, dL_dm = te.dpo_pair_value_grad(lp, rlp, chosen, rejected, weights, beta)
    elif kind == "rank":
        _, dL_dm = te.rank_value_grad(lp, rlp, order, beta, position_discount)
    elif kind == "list":
        _, dL_dm = te.list_value_grad(lp, rlp, order, beta)
    probs = np.exp(lp)
    return te._backprop_through_softmax(dL_dm, probs, beta)


def test_gradients():
    rng = np.random.default_rng(1)
    P, k = 5, 6
    ref_logits = rng.normal(size=(P, k))
    theta = rng.normal(size=(P, k))
    beta = 0.1
    rm = rng.normal(size=(P, k))
    order = np.argsort(-rm, axis=1)
    chosen = order[:, 0]
    rejected = order[:, -1]
    weights = rng.uniform(0.5, 1.5, size=P)

    checks = {
        "pair(weighted)": lambda th: te.dpo_pair_value_grad(
            log_softmax(th, -1), log_softmax(ref_logits, -1), chosen, rejected, weights, beta)[0],
        "rank": lambda th: te.rank_value_grad(
            log_softmax(th, -1), log_softmax(ref_logits, -1), order, beta, 0.9)[0],
        "list": lambda th: te.list_value_grad(
            log_softmax(th, -1), log_softmax(ref_logits, -1), order, beta)[0],
    }
    analytic = {
        "pair(weighted)": _analytic_grad(theta, ref_logits, beta, "pair",
                                         chosen=chosen, rejected=rejected, weights=weights),
        "rank": _analytic_grad(theta, ref_logits, beta, "rank", order=order),
        "list": _analytic_grad(theta, ref_logits, beta, "list", order=order),
    }
    for name, loss_fn in checks.items():
        num = _finite_diff_grad(loss_fn, theta.copy())
        ana = analytic[name]
        max_err = np.abs(num - ana).max()
        rel = max_err / (np.abs(num).max() + 1e-12)
        print(f"{name:16s}  max_abs_err={max_err:.2e}  rel_err={rel:.2e}")
        assert rel < 1e-4, f"gradient mismatch for {name}: rel={rel}"
    print("OK: all analytic gradients match finite differences.")


def test_torch_numpy_agreement():
    """The torch losses in dpo_losses.py must match the numpy engine values."""
    try:
        import torch
        from ant_dpo_project.utils import dpo_losses as dl
    except Exception as e:  # pragma: no cover
        print(f"skipping torch agreement test ({e})")
        return
    rng = np.random.default_rng(3)
    n = 6
    pol = rng.normal(size=n)
    ref = rng.normal(size=n)
    beta = 0.2

    # list loss: numpy engine returns the per-list Plackett-Luce NLL summed over
    # positions (the standard ListMLE convention); the torch fn does too with
    # reduction="sum".
    np_val, _ = te.list_value_grad(pol[None, :], ref[None, :], np.arange(n)[None, :], beta)
    t_val = dl.list_dpo_loss(torch.tensor(pol), torch.tensor(ref), beta=beta,
                             reduction="sum").item()
    print(f"list  numpy={np_val:.6f}  torch={t_val:.6f}")
    assert abs(np_val - t_val) < 1e-5

    # rank loss (uniform position discount => matches with position_discount=1.0)
    np_val, _ = te.rank_value_grad(pol[None, :], ref[None, :], np.arange(n)[None, :], beta, 1.0)
    t_val = dl.rank_dpo_loss(torch.tensor(pol), torch.tensor(ref), beta=beta,
                             position_discount=1.0).item()
    print(f"rank  numpy={np_val:.6f}  torch={t_val:.6f}")
    assert abs(np_val - t_val) < 1e-5
    print("OK: torch and numpy losses agree.")


if __name__ == "__main__":
    test_gradients()
    test_torch_numpy_agreement()
    print("\nAll tests passed.")
