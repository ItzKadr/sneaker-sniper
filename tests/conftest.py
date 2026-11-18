"""Shared fixtures: tiny in-memory catalogs and a random-init (offline) embedder."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # tests must never download

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import src  # noqa: E402,F401
from src.dataset import Catalog, build_demo_catalog  # noqa: E402
from src.embedder import Embedder, EmbedderConfig  # noqa: E402


@pytest.fixture(scope="session")
def tiny_catalog() -> Catalog:
    """2 models x 2 colourways = 4 SKUs rendered at 64 px."""
    return build_demo_catalog(n_models=2, colourways_per_model=(2, 2), image_size=64, seed=3)


@pytest.fixture(scope="session")
def tiny_embedder() -> Embedder:
    """DINOv2 ViT-S/14 architecture, random weights, 28 px input (2x2 patches), CPU."""
    torch.manual_seed(0)  # deterministic random init
    return Embedder(EmbedderConfig(pretrained=False, image_size=28, batch_size=8, device="cpu", use_amp=False))


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)

