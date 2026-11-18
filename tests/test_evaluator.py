from __future__ import annotations

import numpy as np
import pytest

from src.evaluator import (
    BenchmarkConfig,
    evaluate_retrieval,
    ids_to_labels,
    latency_stats,
    make_distractors,
    overlap_at_k,
    pairwise_cosine_distance,
    retrieval_metrics,
    run_benchmark,
    self_cosine,
)
from src.indexer import HNSWParams, IVFParams
from tests.helpers import random_unit


def test_recall_and_mrr_hand_computed() -> None:
    retrieved = np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1], [4, 5, 6]])
    truth = np.array([1, 1, 1, 1])  # first hits at ranks 1, 2, 3, never
    m = retrieval_metrics(retrieved, truth, ks=(1, 2, 3))
    assert m.recall == {1: 0.25, 2: 0.5, 3: 0.75}
    assert m.mrr == pytest.approx((1 + 1 / 2 + 1 / 3 + 0) / 4)
    assert m.as_dict()["mrr@3"] == pytest.approx(0.458333, abs=1e-6)


def test_ids_to_labels_and_overlap() -> None:
    gallery_labels = np.array([10, 11, 12])
    assert ids_to_labels(np.array([[2, -1, 0]]), gallery_labels).tolist() == [[12, -1, 10]]
    assert overlap_at_k(np.array([[0, 1, 2]]), np.array([[2, 5, 0]]), 3) == pytest.approx(2 / 3)
    assert overlap_at_k(np.array([[0, 1], [4, 5]]), np.array([[1, 0], [6, 7]]), 2) == pytest.approx(0.5)


def test_latency_percentiles_hand_computed() -> None:
    s = latency_stats([1.0, 2.0, 3.0, 4.0])
    # h = (n-1) q: p50 -> 1.5 -> 2.5 ; p95 -> 2.85 -> 3 + 0.85 * (4 - 3) = 3.85
    assert s.p50_ms == pytest.approx(2.5) and s.p95_ms == pytest.approx(3.85) and s.mean_ms == pytest.approx(2.5)
    with pytest.raises(ValueError):
        latency_stats([])


def test_cosine_helpers() -> None:
    v = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    d = pairwise_cosine_distance(v)
    assert np.allclose(np.diag(d), 0.0, atol=1e-6) and np.allclose(d, d.T)
    assert d[0, 1] == pytest.approx(1.0) and d[0, 2] == pytest.approx(1 - np.sqrt(0.5), abs=1e-6)
    assert self_cosine(np.array([1.0, 0.0]), np.array([3.0, 3.0])) == pytest.approx(np.sqrt(0.5), abs=1e-6)


@pytest.mark.parametrize("mode", ["gaussian", "random", "perturbed"])
def test_distractors_are_unit_vectors(mode: str) -> None:
    ref = random_unit(50, 24, seed=1) + 0.5  # anisotropic: shared mean direction
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    d = make_distractors(1000, ref, mode=mode, seed=0, chunk=300)
    assert d.shape == (1000, 24) and d.dtype == np.float32
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)
    if mode == "perturbed":
        assert (d @ ref.T).max(axis=1).min() >= 0.5 - 1e-4  # cos(distractor, its base) == 0.5
    if mode == "gaussian":
        mean_cos = self_cosine(d.mean(axis=0), ref.mean(axis=0))
        assert mean_cos > 0.9  # follows the data distribution
    if mode == "random":
        assert abs(self_cosine(d.mean(axis=0), ref.mean(axis=0))) < 0.5


def test_distractor_validation() -> None:
    with pytest.raises(ValueError):
        make_distractors(5, random_unit(5, 4), mode="adversarial")
    assert make_distractors(0, random_unit(5, 4)).shape == (0, 4)
    buf = np.zeros((12, 3, 4), dtype=np.float32)[:, 0, :].copy()  # preallocated destination
    res = make_distractors(12, random_unit(5, 4), out=buf)
    assert res is buf and np.allclose(np.linalg.norm(buf, axis=1), 1.0, atol=1e-5)
    with pytest.raises(ValueError):
        make_distractors(10, random_unit(5, 4), out=np.zeros((9, 4), dtype=np.float32))


def test_evaluate_retrieval_perfect_when_queries_are_gallery() -> None:
    g = random_unit(40, 16, seed=3)
    labels = np.arange(40)
    m = evaluate_retrieval(g, labels, g, labels, ks=(1, 5))
    assert m.recall[1] == 1.0 and m.mrr == 1.0


def test_run_benchmark_rows_and_invariants() -> None:
    gallery = random_unit(120, 16, seed=11)
    labels = np.arange(120)
    queries = gallery[:30] + 0.05 * random_unit(30, 16, seed=12)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    cfg = BenchmarkConfig(
        n_distractors=600, warmup=2, nprobe_values=(1, 4), ef_search_values=(16, 64),
        ivf=IVFParams(nlist=8, nprobe=4), hnsw=HNSWParams(M=8, ef_construction=40, ef_search=16),
    )
    res = run_benchmark(gallery, labels, queries, labels[:30], cfg)
    df = res.to_frame()
    assert set(df["kind"]) == {"flat", "ivf", "hnsw"}
    assert df.groupby("kind")["is_default"].sum().to_dict() == {"flat": 1, "hnsw": 1, "ivf": 1}
    flat = df[df["kind"] == "flat"].iloc[0]
    assert flat["recall_at_1"] == 1.0 and flat["overlap_at_k"] == 1.0
    assert (df["p50_ms"] <= df["p95_ms"] + 1e-9).all() and (df["ntotal"] == 720).all()
    assert df["overlap_at_k"].between(0, 1).all() and (df["memory_mb"] > 0).all()
    assert len(res.default_rows()) == 3
