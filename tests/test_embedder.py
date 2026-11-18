from __future__ import annotations

import numpy as np
import pytest

import src.embedder as embedder_mod
from src.dataset import Catalog
from src.embedder import Embedder, EmbedderConfig, patch_size_from_name, snap_to_multiple
from src.utils import l2_normalize


def test_patch_parsing_and_snapping() -> None:
    assert patch_size_from_name("vit_small_patch14_dinov2.lvd142m") == 14
    assert patch_size_from_name("resnet50") is None
    assert snap_to_multiple(224, 14) == 224
    assert snap_to_multiple(230, 14) == 224
    assert snap_to_multiple(5, 14) == 14


def test_embeddings_are_unit_float32_contiguous(tiny_embedder: Embedder, tiny_catalog: Catalog) -> None:
    x = tiny_embedder.embed(tiny_catalog.images)
    assert x.shape == (tiny_catalog.n_skus, 384)
    assert x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-5)
    assert np.allclose(tiny_embedder.embed(tiny_catalog.images[:2], batch_size=1), x[:2], atol=1e-5)
    assert tiny_embedder.embed_one(tiny_catalog.images[0]).shape == (384,)
    assert tiny_embedder.embed([]).shape == (0, 384)


def test_input_size_follows_snapped_image_size() -> None:
    emb = Embedder(EmbedderConfig(pretrained=False, image_size=30, device="cpu"))
    assert emb.image_size == 28
    assert tuple(emb.data_config["input_size"]) == (3, 28, 28)
    assert emb.data_config["mean"] == pytest.approx((0.485, 0.456, 0.406))
    desc = emb.describe()
    assert desc["pretrained_loaded"] is False and desc["embedding_dim"] == 384


def test_cnn_backbone_without_img_size_kwarg(tiny_catalog: Catalog) -> None:
    emb = Embedder(EmbedderConfig(model_name="resnet18", pretrained=False, image_size=32, device="cpu"))
    x = emb.embed(tiny_catalog.images[:2])
    assert x.shape == (2, 512)
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-5)


def test_download_failure_falls_back_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    real = embedder_mod._create_model

    def fake(model_name: str, pretrained: bool, image_size: int | None):
        if pretrained:
            raise OSError("simulated: no internet")
        return real(model_name, pretrained, image_size)

    monkeypatch.setattr(embedder_mod, "_create_model", fake)
    with pytest.warns(RuntimeWarning, match="RANDOM weights"):
        emb = Embedder(EmbedderConfig(pretrained=True, image_size=28, device="cpu"))
    assert emb.pretrained_loaded is False and "no internet" in (emb.warning or "")
    with pytest.raises(OSError):
        Embedder(EmbedderConfig(pretrained=True, image_size=28, device="cpu", allow_fallback=False))


def test_l2_normalize_handles_zero_and_1d() -> None:
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    y = l2_normalize(x)
    assert y.dtype == np.float32 and np.allclose(y[0], [0.6, 0.8]) and np.allclose(y[1], 0.0)
    assert np.allclose(l2_normalize(np.array([0.0, 2.0])), [0.0, 1.0])
