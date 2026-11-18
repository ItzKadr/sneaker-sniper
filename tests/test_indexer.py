from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluator import overlap_at_k
from src.indexer import (
    HNSWParams,
    IVFParams,
    FaissIndex,
    as_faiss_array,
    auto_nlist,
    build_flat_index,
    build_hnsw_index,
    build_index,
    build_index_suite,
    build_ivf_index,
)
from tests.helpers import random_unit


def test_flat_matches_numpy_bruteforce() -> None:
    x = random_unit(300, 16, seed=1)
    q = random_unit(7, 16, seed=2)
    scores, ids = build_flat_index(x).search(q, 5)
    sims = q @ x.T
    ref_ids = np.argsort(-sims, axis=1)[:, :5]
    assert np.array_equal(ids, ref_ids)
    assert np.allclose(scores, np.take_along_axis(sims, ref_ids, axis=1), atol=1e-5)


def test_ivf_with_all_cells_probed_is_exact() -> None:
    x = random_unit(800, 16, seed=3)
    q = random_unit(20, 16, seed=4)
    ivf = build_ivf_index(x, IVFParams(nlist=8, nprobe=8))
    _, exact = build_flat_index(x).search(q, 10)
    _, approx = ivf.search(q, 10)
    assert overlap_at_k(exact, approx, 10) == pytest.approx(1.0)
    assert ivf.params["nlist"] == 8 and ivf.train_time_s >= 0


def test_hnsw_high_ef_recovers_exact_neighbours() -> None:
    x = random_unit(1500, 16, seed=5)
    q = random_unit(30, 16, seed=6)
    hnsw = build_hnsw_index(x, HNSWParams(M=16, ef_construction=100, ef_search=256))
    _, exact = build_flat_index(x).search(q, 10)
    _, approx = hnsw.search(q, 10)
    assert overlap_at_k(exact, approx, 10) >= 0.95


def test_search_param_setters() -> None:
    x = random_unit(500, 8, seed=7)
    ivf = build_ivf_index(x, IVFParams(nlist=10, nprobe=2))
    ivf.set_search_param(999)
    assert ivf.get_search_param() == 10  # clamped to nlist
    hnsw = build_hnsw_index(x, HNSWParams(M=8, ef_search=16))
    hnsw.set_search_param(128)
    assert hnsw.get_search_param() == 128 and hnsw.search_param_name == "efSearch"
    flat = build_flat_index(x)
    flat.set_search_param(5)
    assert flat.get_search_param() is None and flat.search_param_name is None


def test_save_load_roundtrip(tmp_path: Path) -> None:
    x = random_unit(400, 12, seed=8)
    q = random_unit(5, 12, seed=9)
    for kind, idx in build_index_suite(x, IVFParams(nlist=6, nprobe=3), HNSWParams(M=8)).items():
        loaded = FaissIndex.load(idx.save(tmp_path / f"{kind}.faiss"))
        assert loaded.kind == kind and loaded.ntotal == 400
        assert np.array_equal(idx.search(q, 5)[1], loaded.search(q, 5)[1])
    assert FaissIndex.load(tmp_path / "hnsw.faiss").params["M"] == 8


def test_memory_and_padding() -> None:
    import faiss

    x = random_unit(600, 32, seed=10)
    flat = build_flat_index(x)
    assert flat.memory_bytes() == 600 * 32 * 4
    hnsw = build_hnsw_index(x, HNSWParams(M=16))
    assert hnsw.memory_bytes() > flat.memory_bytes()  # graph links cost memory
    ivf = build_ivf_index(x, IVFParams(nlist=8))
    for idx in (flat, ivf, hnsw):  # analytic size == serialised size up to small headers
        serialised = faiss.serialize_index(idx.index).nbytes
        assert abs(serialised - idx.memory_bytes()) / serialised < 0.02
    x = x[:100]
    flat = build_flat_index(x)
    scores, ids = flat.search(x[:2], 150)
    assert ids.shape == (2, 150) and (ids[:, 100:] == -1).all()


def test_auto_nlist_and_validation() -> None:
    assert auto_nlist(10) == 1
    assert auto_nlist(10_000) == 256  # min(4*sqrt(1e4)=400, 10000//39=256)
    assert auto_nlist(1_000_000) == 4000
    with pytest.raises(ValueError):
        as_faiss_array(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        build_index("lsh", random_unit(10, 4))
    with pytest.raises(ValueError):
        build_flat_index(random_unit(10, 4)).search(random_unit(1, 5), 1)
