"""Plain helper functions shared by tests (kept out of conftest to avoid double import)."""

from __future__ import annotations

import numpy as np


def random_unit(n: int, d: int, seed: int = 0) -> np.ndarray:
    """``n`` random unit vectors in ``d`` dimensions (float32, C-contiguous)."""
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return np.ascontiguousarray(x / np.linalg.norm(x, axis=1, keepdims=True))
