from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.metric_learning import (
    HardNegativeBatchSampler,
    HeadConfig,
    MetricTrainConfig,
    ProjectionHead,
    ValidationSet,
    apply_head,
    batch_hard_triplet_loss,
    compare_raw_vs_head,
    info_nce_loss,
    load_head,
    mine_hard_negatives,
    save_head,
    train_projection_head,
)
from tests.helpers import random_unit


def test_info_nce_hand_computed() -> None:
    eye = torch.eye(2)
    # logits = I / 1 -> CE per row = log(1 + e^-1)
    assert info_nce_loss(eye, eye, temperature=1.0).item() == pytest.approx(math.log(1 + math.exp(-1)), abs=1e-6)
    # perfect alignment + low temperature -> ~0 loss; swapped positives -> large loss
    assert info_nce_loss(eye, eye, temperature=0.01).item() < 1e-6
    assert info_nce_loss(eye, eye.flip(0), temperature=0.1).item() > 5


def test_batch_hard_triplet_hand_computed() -> None:
    emb = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.8, 0.6], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1, 1])
    d = lambda c: math.sqrt(2 - 2 * c)  # noqa: E731  euclidean distance of unit vectors
    # anchors 0/3: hardest pos d(cos=.6), hardest neg d(cos=.8); anchors 1/2: pos d(.6), neg d(.96)
    expected = (2 * (d(0.6) - d(0.8) + 0.2) + 2 * (d(0.6) - d(0.96) + 0.2)) / 4
    assert batch_hard_triplet_loss(emb, labels, margin=0.2).item() == pytest.approx(expected, abs=1e-5)
    # well separated clusters -> zero loss
    sep = torch.tensor([[1.0, 0.0], [0.99, 0.141], [0.0, 1.0], [0.141, 0.99]])
    sep = F.normalize(sep, dim=1)
    assert batch_hard_triplet_loss(sep, labels, margin=0.2).item() == pytest.approx(0.0, abs=1e-6)


def test_heads_output_unit_vectors_and_residual_identity_init() -> None:
    x = torch.from_numpy(random_unit(6, 32, seed=1))
    res = ProjectionHead(32, HeadConfig("residual_mlp", hidden_dim=16)).eval()
    assert torch.allclose(res(x), x, atol=1e-6)  # W2 = 0 -> identity on unit inputs
    lin = ProjectionHead(32, HeadConfig("linear")).eval()
    assert torch.allclose(lin(x), x, atol=1e-6)  # identity init
    for cfg in (HeadConfig("linear", out_dim=8), HeadConfig("mlp", hidden_dim=16, out_dim=12)):
        y = ProjectionHead(32, cfg).eval()(x)
        assert y.shape == (6, cfg.out_dim) and torch.allclose(y.norm(dim=1), torch.ones(6), atol=1e-5)
    with pytest.raises(ValueError):
        ProjectionHead(32, HeadConfig("residual_mlp", out_dim=8))
    with pytest.raises(ValueError):
        ProjectionHead(32, HeadConfig("transformer"))


def test_mine_hard_negatives_and_sampler() -> None:
    e1, e2, e3 = np.eye(3, dtype=np.float32)
    close = (e1 + 0.1 * e2) / np.linalg.norm(e1 + 0.1 * e2)
    emb = np.stack([e1, close, e2, e3]).astype(np.float32)
    nn_ids = mine_hard_negatives(emb, 1)
    assert nn_ids[0, 0] == 1 and nn_ids[1, 0] == 0 and (nn_ids[:, 0] != np.arange(4)).all()

    sampler = HardNegativeBatchSampler(n_items=20, batch_size=8, hard_ratio=1.0)
    sampler.update_neighbors(np.array([[(i + 1) % 20, (i + 2) % 20] for i in range(20)]))
    batch = sampler.sample(np.random.default_rng(0))
    assert len(batch) == 8 and len(set(batch.tolist())) == 8
    seed = int(batch[0])
    assert {(seed + 1) % 20, (seed + 2) % 20} <= set(batch.tolist())  # neighbours share the batch


def _toy_problem(n: int = 24, d: int = 16, views: int = 4, seed: int = 0):
    """Refs + noisy 'photos' whose nuisance is a fixed shared direction (like a colour cast)."""
    rng = np.random.default_rng(seed)
    refs = random_unit(n, d, seed=seed)
    nuisance = random_unit(1, d, seed=seed + 99)[0]
    v = refs[:, None, :] + 0.6 * nuisance + 0.25 * rng.standard_normal((n, views, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=2, keepdims=True)
    return refs, v.astype(np.float32)


@pytest.mark.parametrize("loss", ["infonce", "triplet"])
def test_training_reduces_loss_and_keeps_best_epoch(loss: str) -> None:
    refs, views = _toy_problem()
    val = ValidationSet(refs, np.arange(len(refs)), views[:, 0], np.arange(len(refs)))
    cfg = MetricTrainConfig(loss=loss, epochs=25, batch_size=12, lr=3e-3, patience=30, device="cpu", seed=0)
    res = train_projection_head(refs, views, val, HeadConfig(hidden_dim=32, dropout=0.0), cfg)
    h = res.history
    assert h["epoch"].iloc[0] == 0 and len(h) == 26
    trained = h[h["epoch"] > 0]["loss"]
    assert trained.iloc[-5:].mean() < trained.iloc[:3].mean()
    assert res.best_val_recall1 >= h.loc[h["epoch"] == 0, "val_recall@1"].iloc[0]
    out = apply_head(res.head, refs)
    assert out.dtype == np.float32 and out.flags["C_CONTIGUOUS"] and np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-5)


def test_compare_and_persistence(tmp_path: Path) -> None:
    refs, views = _toy_problem(n=10)
    head = ProjectionHead(16, HeadConfig(hidden_dim=8))
    cmp = compare_raw_vs_head(refs, np.arange(10), views[:, 0], np.arange(10), head, ks=(1, 5))
    df = cmp.to_frame()
    assert df["embedding"].tolist() == ["raw backbone", "projection head"]
    assert cmp.delta()["recall@1"] == pytest.approx(0.0, abs=1e-9)  # identity-init head == raw
    loaded = load_head(save_head(head, tmp_path / "head.pt"))
    assert np.allclose(apply_head(loaded, refs), apply_head(head, refs), atol=1e-6)
    with pytest.raises(ValueError):
        train_projection_head(refs, views[:5], None, HeadConfig(), MetricTrainConfig(epochs=1))
