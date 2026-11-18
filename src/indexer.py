r"""FAISS index builders behind one small common interface.

All indexes use the **inner-product** metric on L2-normalised vectors, i.e. they
rank by cosine similarity :math:`s(q, x) = \langle q, x\rangle = \cos(q, x)`.

=========  =====================  ======================================================
kind       faiss class            idea / knobs
=========  =====================  ======================================================
``flat``   ``IndexFlatIP``        exact brute force, :math:`O(N d)` per query, no knobs
``ivf``    ``IndexIVFFlat``       k-means partitions space into ``nlist`` Voronoi cells;
                                  a query scans only the ``nprobe`` closest cells:
                                  :math:`\approx O(nlist\, d + \tfrac{nprobe}{nlist} N d)`
``hnsw``   ``IndexHNSWFlat``      multi-layer proximity graph with ``M`` links / node,
                                  built with beam ``efConstruction``; greedy search with
                                  beam ``efSearch``: :math:`\approx O(efSearch \cdot M \log N)`
=========  =====================  ======================================================
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)

INDEX_KINDS: tuple[str, ...] = ("flat", "ivf", "hnsw")
INDEX_LABELS: dict[str, str] = {"flat": "Exact (FlatIP)", "ivf": "IVF-Flat", "hnsw": "HNSW-Flat"}
SEARCH_PARAM: dict[str, str | None] = {"flat": None, "ivf": "nprobe", "hnsw": "efSearch"}


@dataclass
class IVFParams:
    nlist: int | None = None  # None -> auto_nlist(N)
    nprobe: int = 8
    spherical: bool = True  # spherical k-means (centroids re-normalised) - right for cosine
    max_train_points: int = 100_000
    seed: int = 0


@dataclass
class HNSWParams:
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 64


def as_faiss_array(x: np.ndarray) -> np.ndarray:
    """Validate and convert to the 2-D C-contiguous float32 layout FAISS requires."""
    arr = np.ascontiguousarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {arr.shape}")
    return arr


def auto_nlist(n: int) -> int:
    r"""Heuristic number of IVF cells: :math:`\approx 4\sqrt{N}`, capped so every cell gets
    at least 39 training points (FAISS warns below that)."""
    if n < 39:
        return 1
    return int(max(1, min(round(4 * math.sqrt(n)), n // 39)))


@dataclass
class FaissIndex:
    """A built FAISS index plus the metadata needed to benchmark and persist it."""

    kind: str
    index: faiss.Index
    dim: int
    build_time_s: float
    train_time_s: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    _keepalive: list[Any] = field(default_factory=list, repr=False)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    @property
    def label(self) -> str:
        return INDEX_LABELS.get(self.kind, self.kind)

    @property
    def search_param_name(self) -> str | None:
        return SEARCH_PARAM.get(self.kind)

    def set_search_param(self, value: int) -> None:
        """Set ``nprobe`` (IVF) or ``efSearch`` (HNSW); a no-op for the exact index."""
        if self.kind == "ivf":
            ivf = faiss.extract_index_ivf(self.index)
            ivf.nprobe = int(min(max(1, value), ivf.nlist))
            self.params["nprobe"] = int(ivf.nprobe)
        elif self.kind == "hnsw":
            self.index.hnsw.efSearch = int(max(1, value))
            self.params["ef_search"] = int(self.index.hnsw.efSearch)

    def get_search_param(self) -> int | None:
        if self.kind == "ivf":
            return int(faiss.extract_index_ivf(self.index).nprobe)
        if self.kind == "hnsw":
            return int(self.index.hnsw.efSearch)
        return None

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, ids)`` of shape ``(n_queries, k)``; missing slots have id ``-1``."""
        q = as_faiss_array(queries)
        if q.shape[1] != self.dim:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self.dim}")
        k_eff = max(1, min(int(k), self.ntotal))
        scores, ids = self.index.search(q, k_eff)
        if k_eff < k:  # pad so callers always get k columns
            pad = k - k_eff
            scores = np.hstack([scores, np.full((len(q), pad), -np.inf, dtype=np.float32)])
            ids = np.hstack([ids, np.full((len(q), pad), -1, dtype=np.int64)])
        return scores, ids

    def memory_bytes(self) -> int:
        """Bytes held by the index data structures, computed without copying anything.

        * flat: ``N d 4`` (float32 vectors)
        * ivf:  ``N d 4`` codes + ``N 8`` ids (inverted lists) + ``nlist d 4`` centroids
        * hnsw: ``N d 4`` vectors + neighbour links (int32) + levels (int32) + offsets (int64)

        ``faiss.serialize_index`` would give the same number (tested) but materialises a
        full copy, i.e. +300 MB at 200k x 384.
        """
        n, d = self.ntotal, self.dim
        if self.kind == "ivf":
            ivf = faiss.extract_index_ivf(self.index)
            return int(n * ivf.code_size + n * 8 + ivf.quantizer.ntotal * d * 4)
        if self.kind == "hnsw":
            h = self.index.hnsw
            return int(n * d * 4 + h.neighbors.size() * 4 + h.levels.size() * 4 + h.offsets.size() * 8)
        return int(n * d * 4)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FaissIndex":
        index = faiss.read_index(str(path))
        if isinstance(index, faiss.IndexHNSW):
            kind = "hnsw"
            params: dict[str, Any] = {
                "M": int(index.hnsw.nb_neighbors(1)),
                "ef_construction": int(index.hnsw.efConstruction),
                "ef_search": int(index.hnsw.efSearch),
            }
        elif isinstance(index, faiss.IndexIVF):
            kind = "ivf"
            ivf = faiss.extract_index_ivf(index)
            params = {"nlist": int(ivf.nlist), "nprobe": int(ivf.nprobe)}
        else:
            kind = "flat"
            params = {}
        return cls(kind=kind, index=index, dim=int(index.d), build_time_s=float("nan"), params=params)

    def describe(self) -> str:
        knob = self.search_param_name
        tail = f", {knob}={self.get_search_param()}" if knob else ""
        return f"{self.label} (N={self.ntotal}{tail})"


# ------------------------------------------------------------------ builders
def build_flat_index(x: np.ndarray) -> FaissIndex:
    """Exact inner-product index (the ground truth for ANN agreement)."""
    x = as_faiss_array(x)
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    return FaissIndex("flat", index, x.shape[1], time.perf_counter() - t0)


def build_ivf_index(x: np.ndarray, params: IVFParams | None = None) -> FaissIndex:
    """IVF-Flat with inner-product metric: k-means "training" + inverted lists."""
    params = params or IVFParams()
    x = as_faiss_array(x)
    n, d = x.shape
    nlist = params.nlist or auto_nlist(n)
    nlist = int(max(1, min(nlist, n)))
    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.cp.spherical = params.spherical
    index.cp.seed = params.seed
    # Tiny catalogs (< 39 points per cell) are legitimate in the demo; don't let faiss warn.
    index.cp.min_points_per_centroid = int(max(1, min(39, n // nlist)))
    rng = np.random.default_rng(params.seed)
    train = x if n <= params.max_train_points else x[rng.choice(n, params.max_train_points, replace=False)]
    index.train(train)
    train_time = time.perf_counter() - t0
    index.add(x)
    index.nprobe = int(min(params.nprobe, nlist))
    built = FaissIndex(
        "ivf", index, d, time.perf_counter() - t0, train_time,
        {"nlist": nlist, "nprobe": int(index.nprobe), "spherical": params.spherical},
    )
    built._keepalive.append(quantizer)  # the IVF index references the quantiser
    return built


def build_hnsw_index(x: np.ndarray, params: HNSWParams | None = None) -> FaissIndex:
    """HNSW graph over raw vectors with inner-product metric."""
    params = params or HNSWParams()
    x = as_faiss_array(x)
    d = x.shape[1]
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(d, params.M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = params.ef_construction
    index.add(x)
    index.hnsw.efSearch = params.ef_search
    return FaissIndex("hnsw", index, d, time.perf_counter() - t0, 0.0, asdict(params))


def build_index(
    kind: str, x: np.ndarray, ivf: IVFParams | None = None, hnsw: HNSWParams | None = None
) -> FaissIndex:
    """Factory: ``kind`` in :data:`INDEX_KINDS`."""
    if kind == "flat":
        return build_flat_index(x)
    if kind == "ivf":
        return build_ivf_index(x, ivf)
    if kind == "hnsw":
        return build_hnsw_index(x, hnsw)
    raise ValueError(f"unknown index kind '{kind}', expected one of {INDEX_KINDS}")


def build_index_suite(
    x: np.ndarray, ivf: IVFParams | None = None, hnsw: HNSWParams | None = None
) -> dict[str, FaissIndex]:
    """Build all three indexes over the same vectors."""
    suite = {kind: build_index(kind, x, ivf, hnsw) for kind in INDEX_KINDS}
    for idx in suite.values():
        logger.info("Built %s in %.3fs", idx.describe(), idx.build_time_s)
    return suite
