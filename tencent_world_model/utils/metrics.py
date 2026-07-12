"""Metric helpers shared across training and analysis."""

from __future__ import annotations

import random
from typing import List, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def moving_average(x: Sequence[float], window: int = 10) -> np.ndarray:
    """Causal moving average (same length as input)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    out = np.empty_like(x)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = (cumsum[i + 1] - cumsum[lo]) / (i + 1 - lo)
    return out


def rolling_std(x: Sequence[float], window: int = 10) -> np.ndarray:
    """Rolling standard deviation -- a smoothness / stability proxy."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = np.std(x[lo:i + 1]) if i >= lo else 0.0
    return out


def grad_norm(model: torch.nn.Module) -> float:
    """Global L2 norm of the gradients currently stored on ``model``."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.data.norm(2).item()) ** 2
    return total ** 0.5


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discounted-to-go returns for ``rewards`` of shape ``[B, T]`` -> ``[B, T]``."""
    B, T = rewards.shape
    returns = torch.zeros_like(rewards)
    running = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        running = rewards[:, t] + gamma * running
        returns[:, t] = running
    return returns


def mean_std(values: List[float]):
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std())


def area_under_curve(y: Sequence[float]) -> float:
    """Normalised AUC of a learning curve (mean value) -- a simple summary of
    both speed and final performance."""
    y = np.asarray(y, dtype=np.float64)
    return float(y.mean()) if len(y) else 0.0


def steps_to_threshold(curve: Sequence[float], threshold: float) -> int:
    """First index at which ``curve`` reaches ``threshold`` (``-1`` if never)."""
    for i, v in enumerate(curve):
        if v >= threshold:
            return i
    return -1
