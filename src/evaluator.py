r"""Retrieval quality, ANN agreement and latency benchmarking.

Metrics
-------
For query :math:`q` with true SKU :math:`y_q` and ranked results :math:`r_1, r_2, \dots`
let :math:`\text{rank}_q` be the 1-based position of the first result with SKU
:math:`y_q` (:math:`\infty` if absent from the top-:math:`K_{max}`).

* Recall@K  :math:`= \frac{1}{|Q|}\sum_q \mathbb{1}[\text{rank}_q \le K]` (hit rate - there is
  exactly one relevant SKU per query).
* MRR@K     :math:`= \frac{1}{|Q|}\sum_q 1/\text{rank}_q` (0 when not in top-:math:`K_{max}`).
* overlap@K :math:`= \frac{1}{|Q|}\sum_q |E_K(q) \cap A_K(q)| / K` - agreement of an ANN
  index :math:`A` with exact search :math:`E` (label-free "ANN recall").
* Latency   per-query wall clock of ``index.search`` with batch size 1, after a
  warm-up; reported as percentiles p50 (median) and p95 (tail).

Distractors
-----------
To make latency differences visible, the catalog can be padded with **synthetic
distractor vectors** (default 50k). They are *not images* - just unit vectors - and
carry label ``-1`` so they can only hurt recall. Modes:

* ``gaussian``  - samples of a Gaussian fitted to the real embeddings (mean + covariance),
  then L2-normalised: same anisotropic geometry as real data ("unseen products").
* ``random``    - isotropic unit vectors (uniform on the sphere): easy to reject, no structure.
* ``perturbed`` - :math:`\hat x = \alpha b + \sqrt{1-\alpha^2} u` with :math:`b` a real
  embedding and :math:`u \perp b` random, so :math:`\cos(\hat x, b) = \alpha` exactly.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .indexer import (
    FaissIndex,
    HNSWParams,
    IVFParams,
    as_faiss_array,
    build_flat_index,
    build_hnsw_index,
    build_ivf_index,
)
from .utils import l2_normalize

logger = logging.getLogger(__name__)

DISTRACTOR_MODES: tuple[str, ...] = ("gaussian", "random", "perturbed")


# ------------------------------------------------------------------ retrieval
@dataclass
class RetrievalMetrics:
    recall: dict[int, float]
    mrr: float
    depth: int
    n_queries: int

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        out = {f"{prefix}recall@{k}": float(v) for k, v in sorted(self.recall.items())}
        out[f"{prefix}mrr@{self.depth}"] = float(self.mrr)
        return out


def ids_to_labels(ids: np.ndarray, gallery_labels: np.ndarray) -> np.ndarray:
    """Map FAISS ids to SKU labels; id ``-1`` (empty slot) -> label ``-1``."""
    ids = np.asarray(ids, dtype=np.int64)
    gallery_labels = np.asarray(gallery_labels, dtype=np.int64)
    out = np.full(ids.shape, -1, dtype=np.int64)
    valid = ids >= 0
    out[valid] = gallery_labels[ids[valid]]
    return out


def first_hit_rank(retrieved_labels: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """0-based position of the first correct label per query, ``-1`` if none."""
    hits = np.asarray(retrieved_labels) == np.asarray(true_labels)[:, None]
    return np.where(hits.any(axis=1), hits.argmax(axis=1), -1)


def retrieval_metrics(
    retrieved_labels: np.ndarray, true_labels: np.ndarray, ks: Sequence[int] = (1, 5, 10)
) -> RetrievalMetrics:
    """Recall@K and MRR@depth from a ``(n_queries, depth)`` matrix of retrieved labels."""
    retrieved_labels = np.asarray(retrieved_labels)
    if retrieved_labels.ndim != 2:
        raise ValueError("retrieved_labels must be 2-D (n_queries, depth)")
    depth = retrieved_labels.shape[1]
    ranks = first_hit_rank(retrieved_labels, true_labels)
    found = ranks >= 0
    recall = {int(k): float(np.mean(found & (ranks < k))) for k in ks}
    rr = np.divide(1.0, ranks + 1.0, out=np.zeros(len(ranks), dtype=np.float64), where=found)
    return RetrievalMetrics(recall, float(rr.mean()) if len(rr) else 0.0, depth, len(ranks))


def evaluate_retrieval(
    gallery: np.ndarray,
    gallery_labels: np.ndarray,
    queries: np.ndarray,
    query_labels: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
) -> RetrievalMetrics:
    """Exact cosine retrieval (FAISS FlatIP) of ``queries`` against ``gallery``."""
    index = build_flat_index(gallery)
    _, ids = index.search(queries, max(ks))
    return retrieval_metrics(ids_to_labels(ids, gallery_labels), query_labels, ks)


def overlap_at_k(reference_ids: np.ndarray, candidate_ids: np.ndarray, k: int) -> float:
    """Mean ``|top-k(reference) & top-k(candidate)| / k`` over queries (ids ``-1`` ignored)."""
    ref = np.asarray(reference_ids)[:, :k]
    cand = np.asarray(candidate_ids)[:, :k]
    match = (ref[:, :, None] == cand[:, None, :]) & (ref[:, :, None] >= 0)
    inter = match.any(axis=2).sum(axis=1)
    denom = np.maximum((ref >= 0).sum(axis=1), 1)
    return float(np.mean(inter / denom)) if len(ref) else 0.0


def self_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors (normalised defensively)."""
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def pairwise_cosine_distance(vectors: np.ndarray) -> np.ndarray:
    r"""Matrix :math:`D_{ij} = 1 - \cos(v_i, v_j)` in ``[0, 2]``."""
    v = l2_normalize(np.asarray(vectors))
    return np.clip(1.0 - v @ v.T, 0.0, 2.0)


# -------------------------------------------------------------------- latency
@dataclass
class LatencyStats:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    n: int


def latency_stats(samples_ms: Sequence[float]) -> LatencyStats:
    r"""Percentiles with linear interpolation (numpy default):
    :math:`p_q = x_{(\lfloor h \rfloor)} + (h - \lfloor h \rfloor)(x_{(\lfloor h \rfloor + 1)} - x_{(\lfloor h \rfloor)})`,
    :math:`h = (n-1) q / 100` on the sorted sample."""
    arr = np.asarray(samples_ms, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("no latency samples")
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return LatencyStats(float(p50), float(p95), float(p99), float(arr.mean()), int(arr.size))


def measure_latency(
    index: FaissIndex,
    queries: np.ndarray,
    k: int,
    warmup: int = 20,
    max_queries: int | None = None,
    repeats: int = 1,
) -> LatencyStats:
    """Single-query (batch size 1) search latency in milliseconds.

    The first ``warmup`` searches are discarded (cold caches, lazy allocations, OpenMP
    thread-pool spin-up would otherwise inflate the tail). The query set is replayed
    ``repeats`` times so that p95 rests on enough samples to be stable.
    """
    q = as_faiss_array(queries)
    n = len(q) if max_queries is None else min(len(q), max_queries)
    for i in range(min(warmup, len(q))):
        index.search(q[i : i + 1], k)
    samples = np.empty(n * max(1, repeats), dtype=np.float64)
    j = 0
    for _ in range(max(1, repeats)):
        for i in range(n):
            t0 = time.perf_counter_ns()
            index.search(q[i : i + 1], k)
            samples[j] = (time.perf_counter_ns() - t0) / 1e6
            j += 1
    return latency_stats(samples)


def batch_ms_per_query(index: FaissIndex, queries: np.ndarray, k: int) -> float:
    """Amortised milliseconds per query when all queries are searched in one call."""
    q = as_faiss_array(queries)
    t0 = time.perf_counter()
    index.search(q, k)
    return (time.perf_counter() - t0) * 1000.0 / max(1, len(q))


# ----------------------------------------------------------------- distractors
def make_distractors(
    n: int,
    reference: np.ndarray,
    mode: str = "gaussian",
    seed: int = 0,
    perturb_cos: float = 0.5,
    chunk: int = 50_000,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Generate ``n`` synthetic unit-norm distractor vectors (see module docstring).

    ``out`` may be a preallocated ``(n, d)`` float32 view (e.g. the tail of the index
    matrix) so that no extra copy of 100k+ vectors is made.
    """
    if mode not in DISTRACTOR_MODES:
        raise ValueError(f"unknown distractor mode '{mode}', expected {DISTRACTOR_MODES}")
    ref = as_faiss_array(reference)
    d = ref.shape[1]
    if out is None:
        out = np.empty((max(0, n), d), dtype=np.float32)
    elif out.shape != (max(0, n), d) or out.dtype != np.float32:
        raise ValueError(f"out must be a float32 array of shape {(n, d)}")
    if n <= 0:
        return out
    rng = np.random.default_rng(seed)
    if mode == "gaussian":
        mean = ref.mean(axis=0)
        centred = ref - mean
        # cov = V diag(s^2/(m-1)) V^T  ->  sample = mean + z diag(s/sqrt(m-1)) V^T
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        scale = (s / np.sqrt(max(len(ref) - 1, 1)))[:, None] * vt
        # A small isotropic floor keeps samples full-rank when N_catalog < d.
        iso = 0.05 * float(np.sqrt(np.mean(s**2) / max(len(ref) - 1, 1) / d))
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        if mode == "random":
            block = rng.standard_normal((m, d), dtype=np.float32)
        elif mode == "gaussian":
            z = rng.standard_normal((m, scale.shape[0]), dtype=np.float32)
            block = mean + z @ scale.astype(np.float32) + iso * rng.standard_normal((m, d), dtype=np.float32)
        else:  # perturbed
            base = l2_normalize(ref[rng.integers(0, len(ref), size=m)])
            u = rng.standard_normal((m, d), dtype=np.float32)
            u -= (u * base).sum(axis=1, keepdims=True) * base
            u = l2_normalize(u)
            a = float(np.clip(perturb_cos, -1.0, 1.0))
            block = a * base + np.sqrt(1.0 - a * a) * u
        out[start : start + m] = l2_normalize(block)
    return out


# ------------------------------------------------------------------ benchmark
@dataclass
class BenchmarkConfig:
    ks: tuple[int, ...] = (1, 5, 10)
    n_distractors: int = 50_000
    distractor_mode: str = "gaussian"
    warmup: int = 20
    max_latency_queries: int | None = 1000
    latency_repeats: int = 3
    nprobe_values: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    ef_search_values: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
    ivf: IVFParams = field(default_factory=IVFParams)
    hnsw: HNSWParams = field(default_factory=HNSWParams)
    seed: int = 0


@dataclass
class BenchmarkRow:
    index: str
    kind: str
    param_name: str
    param_value: int
    is_default: bool
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    overlap_at_k: float
    p50_ms: float
    p95_ms: float
    mean_ms: float
    batch_ms_per_query: float
    build_time_s: float
    train_time_s: float
    memory_mb: float
    ntotal: int


@dataclass
class BenchmarkResult:
    rows: list[BenchmarkRow]
    n_gallery: int
    n_distractors: int
    n_queries: int
    dim: int
    distractor_mode: str
    k: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.rows])

    def default_rows(self) -> pd.DataFrame:
        df = self.to_frame()
        return df[df["is_default"]].reset_index(drop=True)


def _recall_value(metrics: RetrievalMetrics, k: int) -> float:
    return float(metrics.recall.get(k, float("nan")))


def evaluate_setting(
    index: FaissIndex,
    queries: np.ndarray,
    query_labels: np.ndarray,
    base_labels: np.ndarray,
    exact_ids: np.ndarray | None,
    cfg: BenchmarkConfig,
    memory_bytes: int,
    is_default: bool,
) -> BenchmarkRow:
    """Quality + latency of ``index`` at its *current* search parameter."""
    k = max(cfg.ks)
    _, ids = index.search(queries, k)
    metrics = retrieval_metrics(ids_to_labels(ids, base_labels), query_labels, cfg.ks)
    overlap = 1.0 if exact_ids is None else overlap_at_k(exact_ids, ids, k)
    lat = measure_latency(index, queries, k, cfg.warmup, cfg.max_latency_queries, cfg.latency_repeats)
    return BenchmarkRow(
        index=index.label,
        kind=index.kind,
        param_name=index.search_param_name or "-",
        param_value=int(index.get_search_param() or 0),
        is_default=is_default,
        recall_at_1=_recall_value(metrics, 1),
        recall_at_5=_recall_value(metrics, 5),
        recall_at_10=_recall_value(metrics, 10),
        mrr=metrics.mrr,
        overlap_at_k=overlap,
        p50_ms=lat.p50_ms,
        p95_ms=lat.p95_ms,
        mean_ms=lat.mean_ms,
        batch_ms_per_query=batch_ms_per_query(index, queries, k),
        build_time_s=index.build_time_s,
        train_time_s=index.train_time_s,
        memory_mb=memory_bytes / 2**20,
        ntotal=index.ntotal,
    )


def run_benchmark(
    gallery: np.ndarray,
    gallery_labels: np.ndarray,
    queries: np.ndarray,
    query_labels: np.ndarray,
    cfg: BenchmarkConfig | None = None,
) -> BenchmarkResult:
    """Build Exact / IVF / HNSW over catalog (+ distractors) and sweep nprobe / efSearch.

    Indexes are built and benchmarked one at a time and released afterwards, so peak
    RAM stays around 2x the size of one index.
    """
    cfg = cfg or BenchmarkConfig()
    k = max(cfg.ks)
    gallery = as_faiss_array(gallery)
    queries = as_faiss_array(queries)
    n_g = len(gallery)
    base = np.empty((n_g + cfg.n_distractors, gallery.shape[1]), dtype=np.float32)
    base[:n_g] = gallery
    make_distractors(cfg.n_distractors, gallery, cfg.distractor_mode, cfg.seed, out=base[n_g:])
    base_labels = np.concatenate([np.asarray(gallery_labels, dtype=np.int64), np.full(cfg.n_distractors, -1, dtype=np.int64)])
    logger.info(
        "Benchmark: %d catalog + %d %s distractors (d=%d), %d queries",
        len(gallery), cfg.n_distractors, cfg.distractor_mode, base.shape[1], len(queries),
    )
    rows: list[BenchmarkRow] = []

    flat = build_flat_index(base)
    _, exact_ids = flat.search(queries, k)
    rows.append(evaluate_setting(flat, queries, query_labels, base_labels, None, cfg, flat.memory_bytes(), True))
    del flat
    gc.collect()

    ivf = build_ivf_index(base, cfg.ivf)
    ivf_mem = ivf.memory_bytes()
    nlist = int(ivf.params["nlist"])
    default_nprobe = int(min(cfg.ivf.nprobe, nlist))
    for v in sorted({min(int(x), nlist) for x in cfg.nprobe_values} | {default_nprobe}):
        ivf.set_search_param(v)
        rows.append(evaluate_setting(ivf, queries, query_labels, base_labels, exact_ids, cfg, ivf_mem, v == default_nprobe))
    del ivf
    gc.collect()

    hnsw = build_hnsw_index(base, cfg.hnsw)
    hnsw_mem = hnsw.memory_bytes()
    for v in sorted({int(x) for x in cfg.ef_search_values} | {cfg.hnsw.ef_search}):
        hnsw.set_search_param(v)
        rows.append(evaluate_setting(hnsw, queries, query_labels, base_labels, exact_ids, cfg, hnsw_mem, v == cfg.hnsw.ef_search))
    del hnsw, base
    gc.collect()

    for r in rows:
        logger.info(
            "  %-15s %-8s=%-4s R@1=%.3f R@10=%.3f overlap@%d=%.3f p50=%.3fms p95=%.3fms mem=%.1fMB",
            r.index, r.param_name, r.param_value, r.recall_at_1, r.recall_at_10, k, r.overlap_at_k,
            r.p50_ms, r.p95_ms, r.memory_mb,
        )
    return BenchmarkResult(rows, len(gallery), cfg.n_distractors, len(queries), gallery.shape[1], cfg.distractor_mode, k)
