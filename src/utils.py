"""Small shared helpers: seeding, device / thread selection, logging, normalisation."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once (ASCII-only format, safe for cp1251 consoles)."""
    # Third-party libraries are chatty at INFO level.
    for noisy in ("matplotlib", "PIL", "urllib3", "httpx", "httpx2", "httpcore", "huggingface_hub", "alembic", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
    root.addHandler(handler)


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is a hard dependency, but keep utils importable without it
        logger.debug("torch not available; skipped torch seeding")


def resolve_device(device: str = "auto") -> str:
    """Return ``"cuda"`` when requested/available, otherwise ``"cpu"``.

    The environment variable ``SNEAKER_SNIPER_DEVICE`` overrides ``"auto"``.
    """
    import torch

    if device == "auto":
        device = os.environ.get("SNEAKER_SNIPER_DEVICE", "auto")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available - using CPU")
        return "cpu"
    return device


def configure_threads(n_threads: int = 4) -> None:
    """Cap intra-op threads of torch and faiss (several processes share the machine)."""
    import faiss
    import torch

    torch.set_num_threads(n_threads)
    faiss.omp_set_num_threads(n_threads)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    r"""Row-wise L2 normalisation returning a C-contiguous float32 array.

    .. math:: \hat x_i = x_i / \max(\lVert x_i \rVert_2, \varepsilon)

    After normalisation the inner product equals cosine similarity:
    :math:`\langle \hat a, \hat b\rangle = \cos(a, b)`.
    """
    arr = np.asarray(x, dtype=np.float32)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[None, :]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    out = arr / np.maximum(norms, eps)
    out = np.ascontiguousarray(out, dtype=np.float32)
    return out[0] if squeeze else out


def relative_to_repo(path: Path) -> str:
    """Return ``path`` relative to the repo root if possible (POSIX separators)."""
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_repo_path(path: str | Path) -> Path:
    """Inverse of :func:`relative_to_repo`."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)
