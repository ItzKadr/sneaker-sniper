"""Search engine facade used by the Streamlit app (and handy in a notebook-free REPL).

``SearchEngine`` = embedder + optional projection head + catalog + one index suite
(Exact / IVF / HNSW) per embedding space ("raw" backbone or "head"). Index suites
are built lazily and cached; saved FAISS files from ``train.py`` are reused when
compatible.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .dataset import Catalog, ImageLike, QuerySet, build_demo_catalog, load_image, make_synthetic_queries
from .distortion import SWEEP_LEVELS, apply_distortion, single_factor_params
from .embedder import Embedder
from .indexer import INDEX_KINDS, FaissIndex, HNSWParams, IVFParams, build_index_suite
from .metric_learning import ProjectionHead, apply_head, load_head
from .utils import resolve_repo_path

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    rank: int
    gallery_id: int
    label: int
    sku_id: str
    score: float


@dataclass
class SearchResponse:
    kind: str
    hits: list[SearchHit]
    ids: np.ndarray
    latency_ms: float


@dataclass
class SearchEngine:
    embedder: Embedder
    catalog: Catalog
    gallery_raw: np.ndarray
    head: ProjectionHead | None = None
    ivf: IVFParams = field(default_factory=IVFParams)
    hnsw: HNSWParams = field(default_factory=HNSWParams)
    source: str = "demo"
    query_raw: np.ndarray | None = None
    query_labels: np.ndarray | None = None
    preloaded: dict[str, dict[str, FaissIndex]] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    _gallery: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _indexes: dict[str, dict[str, FaissIndex]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------- spaces
    @property
    def spaces(self) -> list[str]:
        return ["raw", "head"] if self.head is not None else ["raw"]

    def project(self, raw: np.ndarray, space: str) -> np.ndarray:
        if space == "head":
            if self.head is None:
                raise ValueError("no projection head loaded")
            return apply_head(self.head, np.atleast_2d(raw))
        return np.ascontiguousarray(np.atleast_2d(raw), dtype=np.float32)

    def gallery(self, space: str) -> np.ndarray:
        if space not in self._gallery:
            self._gallery[space] = self.project(self.gallery_raw, space)
        return self._gallery[space]

    def indexes(self, space: str) -> dict[str, FaissIndex]:
        if space not in self._indexes:
            if space in self.preloaded and all(k in self.preloaded[space] for k in INDEX_KINDS):
                self._indexes[space] = self.preloaded[space]
            else:
                self._indexes[space] = build_index_suite(self.gallery(space), self.ivf, self.hnsw)
        return self._indexes[space]

    # ------------------------------------------------------------- queries
    def encode(self, image: ImageLike, space: str) -> np.ndarray:
        """Image -> unit vector in ``space`` (``(dim,)``)."""
        return self.project(self.embedder.embed_one(image), space)[0]

    def search(
        self, vector: np.ndarray, space: str, k: int = 5, nprobe: int | None = None, ef_search: int | None = None
    ) -> dict[str, SearchResponse]:
        out: dict[str, SearchResponse] = {}
        q = np.ascontiguousarray(np.atleast_2d(vector), dtype=np.float32)
        for kind, index in self.indexes(space).items():
            if kind == "ivf" and nprobe is not None:
                index.set_search_param(nprobe)
            if kind == "hnsw" and ef_search is not None:
                index.set_search_param(ef_search)
            index.search(q, k)  # warm-up so the displayed latency is not a cold start
            t0 = time.perf_counter_ns()
            scores, ids = index.search(q, k)
            latency = (time.perf_counter_ns() - t0) / 1e6
            hits = [
                SearchHit(r + 1, int(i), int(self.catalog.labels[i]), self.catalog.sku_ids[int(self.catalog.labels[i])], float(s))
                for r, (i, s) in enumerate(zip(ids[0], scores[0]))
                if i >= 0
            ]
            out[kind] = SearchResponse(kind, hits, ids[0], latency)
        return out

    def gallery_image(self, gallery_id: int) -> Image.Image:
        return load_image(self.catalog.images[gallery_id])

    def eval_queries(self, space: str, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
        """Query embeddings + labels for live benchmarks (saved ones or 1 synthetic photo per SKU)."""
        if self.query_raw is None or self.query_labels is None:
            qs: QuerySet = make_synthetic_queries(self.catalog, per_sku=1, strength=1.0, seed=seed)
            self.query_raw = self.embedder.embed(qs.images)
            self.query_labels = qs.labels
        return self.project(self.query_raw, space), self.query_labels


# --------------------------------------------------------------- factories
def build_demo_engine(
    embedder: Embedder, n_models: int = 30, colourways: tuple[int, int] = (3, 4), image_size: int = 224, seed: int = 7
) -> SearchEngine:
    """Cold-start engine: render a small catalog in memory and embed it."""
    catalog = build_demo_catalog(n_models, colourways, image_size, seed)
    gallery = embedder.embed(catalog.images)
    return SearchEngine(embedder, catalog, gallery, source="demo", info={"n_models": n_models})


def load_engine_from_artifacts(artifacts_dir: str | Path, embedder: Embedder) -> tuple[SearchEngine | None, str]:
    """Load the catalog/embeddings/head/indexes written by ``train.py``.

    Returns ``(engine, "")`` or ``(None, reason)`` when artifacts are missing or were
    produced with a different backbone configuration.
    """
    root = Path(artifacts_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None, "no artifacts/manifest.json (train.py has not been run yet)"
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        emb_meta = manifest["embedder"]
        mine = embedder.describe()
        for key in ("model_name", "image_size", "pretrained_loaded"):
            if emb_meta.get(key) != mine.get(key):
                return None, f"artifacts were built with {key}={emb_meta.get(key)!r}, current embedder has {mine.get(key)!r}"
        cat_df = pd.read_csv(root / "catalog.csv")
        paths = [resolve_repo_path(p) for p in cat_df["image_path"]]
        if not all(p.is_file() for p in paths[:: max(1, len(paths) // 20)]):
            return None, "catalog images referenced by artifacts are missing (was data/ deleted?)"
        sku_ids = list(dict.fromkeys(cat_df["sku_id"].astype(str)))
        sku_index = {s: i for i, s in enumerate(sku_ids)}
        groups_map = dict(zip(cat_df["sku_id"].astype(str), cat_df["model_id"].astype(str)))
        catalog = Catalog(
            sku_ids=sku_ids,
            images=list(paths),
            labels=np.array([sku_index[s] for s in cat_df["sku_id"].astype(str)]),
            groups=np.array([groups_map[s] for s in sku_ids], dtype=object),
            metadata=cat_df,
        )
        gallery_raw = np.load(root / "embeddings" / "catalog_raw.npy")
        head = load_head(root / "projection_head.pt") if (root / "projection_head.pt").is_file() else None
        q_raw = q_labels = None
        if (root / "embeddings" / "queries_raw.npy").is_file() and (root / "queries.csv").is_file():
            q_df = pd.read_csv(root / "queries.csv")
            q_raw = np.load(root / "embeddings" / "queries_raw.npy")
            q_labels = np.array([sku_index[s] for s in q_df["sku_id"].astype(str)])
        ivf = IVFParams(**manifest.get("ivf", {}))
        hnsw = HNSWParams(**manifest.get("hnsw", {}))
        preloaded: dict[str, dict[str, FaissIndex]] = {}
        space = manifest.get("index_space", "raw")
        idx_files = {k: root / "indexes" / f"catalog_{k}.faiss" for k in INDEX_KINDS}
        if all(p.is_file() for p in idx_files.values()):
            preloaded[space] = {k: FaissIndex.load(p) for k, p in idx_files.items()}
        engine = SearchEngine(
            embedder, catalog, gallery_raw, head, ivf, hnsw, source="artifacts",
            query_raw=q_raw, query_labels=q_labels, preloaded=preloaded,
            info={**manifest, "artifacts_dir": str(root.resolve())},
        )
        return engine, ""
    except Exception as exc:  # corrupted / partial artifacts must not break the app
        logger.warning("Could not load artifacts: %s", exc)
        return None, f"could not load artifacts ({type(exc).__name__}: {exc})"


# --------------------------------------------------------------- analysis
def stability_profile(engine: SearchEngine, image: Image.Image, space: str, seed: int = 0) -> pd.DataFrame:
    """Self-cosine stability ``cos(f(x), f(T_level(x)))`` for each single distortion factor."""
    clean = engine.encode(image, space)
    rows: list[tuple[str, float]] = []
    images: list[Image.Image] = []
    for factor, levels in SWEEP_LEVELS.items():
        for level in levels:
            rows.append((factor, level))
            images.append(apply_distortion(image, single_factor_params(factor, level, seed)))
    vecs = engine.project(engine.embedder.embed(images), space)
    cos = vecs @ clean
    return pd.DataFrame({"factor": [r[0] for r in rows], "level": [r[1] for r in rows], "cosine": cos})
