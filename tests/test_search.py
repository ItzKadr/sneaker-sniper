from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.embedder import Embedder
from src.metric_learning import HeadConfig, ProjectionHead
from src.search import SearchEngine, build_demo_engine, load_engine_from_artifacts, stability_profile
from tests.helpers import random_unit


@pytest.fixture(scope="module")
def engine(tiny_embedder: Embedder) -> SearchEngine:
    return build_demo_engine(tiny_embedder, n_models=2, colourways=(2, 2), image_size=64, seed=1)


def test_engine_encode_and_search(engine: SearchEngine, tiny_embedder: Embedder) -> None:
    assert engine.source == "demo" and engine.spaces == ["raw"]
    vec = engine.encode(engine.gallery_image(2), "raw")
    assert vec.shape == (tiny_embedder.dim,) and np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)
    # Random-init features are nearly collinear, so test the search logic on separated vectors.
    gallery = random_unit(engine.catalog.n_skus, tiny_embedder.dim, seed=3)
    eng = SearchEngine(tiny_embedder, engine.catalog, gallery)
    out = eng.search(gallery[2], "raw", k=3, nprobe=1, ef_search=16)
    assert set(out) == {"flat", "ivf", "hnsw"}
    exact = out["flat"]
    assert len(exact.hits) == 3 and exact.hits[0].gallery_id == 2  # a vector finds itself
    assert exact.hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert exact.hits[0].sku_id == engine.catalog.sku_ids[2]
    assert [h.rank for h in exact.hits] == [1, 2, 3]
    assert all(r.latency_ms >= 0 for r in out.values())


def test_engine_with_head_has_two_spaces(tiny_embedder: Embedder, engine: SearchEngine) -> None:
    head = ProjectionHead(tiny_embedder.dim, HeadConfig(hidden_dim=16))
    eng = SearchEngine(tiny_embedder, engine.catalog, engine.gallery_raw, head)
    assert eng.spaces == ["raw", "head"]
    assert np.allclose(eng.gallery("head"), eng.gallery("raw"), atol=1e-5)  # identity-init head
    q, labels = eng.eval_queries("head")
    assert q.shape == (engine.catalog.n_skus, tiny_embedder.dim) and len(labels) == engine.catalog.n_skus


def test_stability_profile(engine: SearchEngine) -> None:
    df = stability_profile(engine, engine.gallery_image(0), "raw")
    assert {"rotation", "crop", "exposure", "blur", "jpeg", "color_temp"} == set(df["factor"])
    level0 = df[(df["factor"] == "rotation") & (df["level"] == 0)]["cosine"].iloc[0]
    assert level0 == pytest.approx(1.0, abs=1e-4)  # no distortion -> identical embedding
    assert df["cosine"].between(-1.0001, 1.0001).all()


def test_missing_artifacts_reported(tmp_path: Path, tiny_embedder: Embedder) -> None:
    eng, reason = load_engine_from_artifacts(tmp_path, tiny_embedder)
    assert eng is None and "manifest" in reason
