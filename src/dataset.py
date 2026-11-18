"""Catalog / query data: loader for user folders + synthetic catalog bootstrap.

Folder conventions
------------------
User-supplied catalog::

    <catalog_dir>/<sku_id>/*.jpg|png|webp     one or more reference images per SKU
    <queries_dir>/<sku_id>/*.jpg              optional real smartphone photos (labels = folder)

Synthetic catalog written by :func:`bootstrap_synthetic_dataset`::

    data/synthetic/catalog/<sku_id>/ref.jpg   clean studio product shot
    data/synthetic/queries/<sku_id>/q0.jpg    simulated phone photos of that SKU
    data/synthetic/metadata.csv               SKU attributes (model, colours, shape, ...)
    data/synthetic/manifest.json              generation config (used as a cache key)
"""

from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Sequence, Union

import numpy as np
import pandas as pd
from PIL import Image

from .distortion import DistortionParams, apply_distortion, sample_distortion_params
from .render import SneakerSpec, generate_catalog_specs, render_sneaker

logger = logging.getLogger(__name__)

ImageLike = Union[str, Path, Image.Image]
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
SYNTHETIC_FORMAT_VERSION = 1
DEFAULT_WORKERS = 4  # thread-pool size for rendering / distortion (machine is shared)


@dataclass
class Catalog:
    """Gallery of reference images. ``labels[i]`` indexes into ``sku_ids``."""

    sku_ids: list[str]
    images: list[ImageLike]
    labels: np.ndarray
    groups: np.ndarray  # (n_skus,) model / family id per SKU, used for leak-free splits
    metadata: pd.DataFrame | None = None
    root: Path | None = None

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.groups = np.asarray(self.groups)
        if len(self.images) != len(self.labels):
            raise ValueError("images and labels must have the same length")
        if len(self.groups) != len(self.sku_ids):
            raise ValueError("groups must have one entry per SKU")

    @property
    def n_skus(self) -> int:
        return len(self.sku_ids)

    def __len__(self) -> int:
        return len(self.images)

    def first_image_index(self) -> np.ndarray:
        """Index of the first gallery image of every SKU (shape ``(n_skus,)``)."""
        first = np.full(self.n_skus, -1, dtype=np.int64)
        for i in range(len(self.labels) - 1, -1, -1):
            first[self.labels[i]] = i
        return first


@dataclass
class QuerySet:
    """Query photos with their ground-truth SKU label (index into ``Catalog.sku_ids``)."""

    images: list[ImageLike]
    labels: np.ndarray

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels, dtype=np.int64)
        if len(self.images) != len(self.labels):
            raise ValueError("images and labels must have the same length")

    def __len__(self) -> int:
        return len(self.images)

    def subset(self, mask: np.ndarray) -> "QuerySet":
        idx = np.flatnonzero(mask)
        return QuerySet([self.images[i] for i in idx], self.labels[idx])


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """Parameters of the procedural catalog."""

    n_models: int = 100
    colourways_per_model: tuple[int, int] = (3, 5)
    queries_per_sku: int = 2
    image_size: int = 256
    distortion_strength: float = 1.0
    seed: int = 42

    def to_json_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["colourways_per_model"] = list(self.colourways_per_model)
        d["format_version"] = SYNTHETIC_FORMAT_VERSION
        return d


@dataclass
class SyntheticDataset:
    root: Path
    catalog: Catalog
    queries: QuerySet
    config: SyntheticDatasetConfig
    regenerated: bool = field(default=False)


# ------------------------------------------------------------------ image I/O
def load_image(item: ImageLike) -> Image.Image:
    """Open a path (or pass through a PIL image) as RGB."""
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    with Image.open(item) as im:
        return im.convert("RGB")


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


# -------------------------------------------------------------- folder loader
def load_catalog_folder(root: str | Path, metadata_csv: str | Path | None = None) -> Catalog:
    """Load ``root/<sku_id>/*.jpg`` into a :class:`Catalog`.

    If a metadata CSV with columns ``sku_id`` and ``model_id`` is found (explicit
    argument, or ``root/../metadata.csv``), ``model_id`` is used as the SKU group for
    leak-free train/test splits; otherwise every SKU is its own group.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Catalog folder not found: {root}. Expected <catalog>/<sku_id>/*.jpg")
    sku_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    sku_ids: list[str] = []
    images: list[ImageLike] = []
    labels: list[int] = []
    for d in sku_dirs:
        files = _list_images(d)
        if not files:
            logger.warning("SKU folder without images skipped: %s", d)
            continue
        label = len(sku_ids)
        sku_ids.append(d.name)
        images.extend(files)
        labels.extend([label] * len(files))
    if not sku_ids:
        raise FileNotFoundError(f"No images found under {root} (expected <sku_id>/*.jpg sub-folders)")

    meta_path = Path(metadata_csv) if metadata_csv else root.parent / "metadata.csv"
    metadata: pd.DataFrame | None = None
    groups = np.array(sku_ids, dtype=object)
    if meta_path.is_file():
        metadata = pd.read_csv(meta_path)
        if {"sku_id", "model_id"} <= set(metadata.columns):
            lookup = dict(zip(metadata["sku_id"].astype(str), metadata["model_id"].astype(str)))
            groups = np.array([lookup.get(s, s) for s in sku_ids], dtype=object)
            metadata = metadata.set_index("sku_id").reindex(sku_ids).reset_index()
    logger.info("Loaded catalog: %d SKUs, %d images from %s", len(sku_ids), len(images), root)
    return Catalog(sku_ids, images, np.array(labels), groups, metadata, root)


def load_query_folder(root: str | Path, catalog: Catalog) -> QuerySet:
    """Load ``root/<sku_id>/*.jpg`` query photos labelled by their folder name."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Query folder not found: {root}")
    index = {s: i for i, s in enumerate(catalog.sku_ids)}
    images: list[ImageLike] = []
    labels: list[int] = []
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        if d.name not in index:
            logger.warning("Query folder '%s' has no matching catalog SKU - skipped", d.name)
            continue
        files = _list_images(d)
        images.extend(files)
        labels.extend([index[d.name]] * len(files))
    if not images:
        raise FileNotFoundError(f"No query images matching catalog SKUs found under {root}")
    return QuerySet(images, np.array(labels))


# ------------------------------------------------------------ synthetic views
def distort_many(
    images: Sequence[Image.Image], params: Sequence[DistortionParams], n_workers: int = DEFAULT_WORKERS
) -> list[Image.Image]:
    """Apply :func:`apply_distortion` to many images on a thread pool.

    PIL and numpy release the GIL in their heavy kernels (resampling, blur, JPEG,
    array maths), so threads give a real speed-up without multiprocessing (which is
    spawn-based and slow to start on Windows). Order and results are deterministic
    because every image carries its own pre-sampled ``params`` (incl. seed).
    """
    if n_workers <= 1 or len(images) <= 1:
        return [apply_distortion(im, p) for im, p in zip(images, params)]
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(apply_distortion, images, params))


def iter_distorted_views(
    catalog: Catalog,
    sku_indices: Sequence[int],
    views_per_sku: int,
    strength: float = 1.0,
    seed: int = 0,
    chunk_size: int = 64,
    n_workers: int = DEFAULT_WORKERS,
) -> Iterator[tuple[np.ndarray, list[Image.Image]]]:
    """Yield ``(labels, images)`` chunks of simulated phone photos, generated lazily.

    Used for metric-learning training views (never written to disk) and for synthetic
    queries of user catalogs. Deterministic for a given ``seed`` (parameters are drawn
    sequentially, only the pixel work runs in parallel).
    """
    rng = np.random.default_rng(seed)
    first = catalog.first_image_index()
    jobs: list[tuple[int, Image.Image, DistortionParams]] = []

    def flush() -> tuple[np.ndarray, list[Image.Image]]:
        imgs = distort_many([j[1] for j in jobs], [j[2] for j in jobs], n_workers)
        return np.array([j[0] for j in jobs], dtype=np.int64), imgs

    for sku in sku_indices:
        clean = load_image(catalog.images[int(first[int(sku)])])
        for _ in range(views_per_sku):
            jobs.append((int(sku), clean, sample_distortion_params(rng, strength)))
            if len(jobs) >= chunk_size:
                yield flush()
                jobs = []
    if jobs:
        yield flush()


def make_synthetic_queries(
    catalog: Catalog, per_sku: int = 2, strength: float = 1.0, seed: int = 0
) -> QuerySet:
    """In-memory phone-photo queries for every SKU (for catalogs without real queries)."""
    images: list[ImageLike] = []
    labels: list[int] = []
    for lab, imgs in iter_distorted_views(catalog, range(catalog.n_skus), per_sku, strength, seed):
        images.extend(imgs)
        labels.extend(lab.tolist())
    return QuerySet(images, np.array(labels))


# --------------------------------------------------------- synthetic bootstrap
def build_demo_catalog(
    n_models: int = 30,
    colourways_per_model: tuple[int, int] = (3, 4),
    image_size: int = 224,
    seed: int = 7,
) -> Catalog:
    """Small fully in-memory synthetic catalog (used by the app on cold start)."""
    specs = generate_catalog_specs(n_models, colourways_per_model, seed=seed)
    images: list[ImageLike] = [render_sneaker(s, size=image_size) for s in specs]
    meta = pd.DataFrame([s.to_record() for s in specs])
    return Catalog(
        sku_ids=[s.sku_id for s in specs],
        images=images,
        labels=np.arange(len(specs)),
        groups=np.array([s.model_id for s in specs], dtype=object),
        metadata=meta,
        root=None,
    )


def _manifest_matches(manifest_path: Path, config: SyntheticDatasetConfig) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            stored = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    return stored.get("config") == config.to_json_dict() and bool(stored.get("complete"))


def bootstrap_synthetic_dataset(
    root: str | Path = "data/synthetic",
    config: SyntheticDatasetConfig | None = None,
    overwrite: bool = False,
    n_workers: int = DEFAULT_WORKERS,
) -> SyntheticDataset:
    """Render a procedural sneaker catalog + phone-photo queries to ``root`` (cached).

    If ``root/manifest.json`` records the same config, nothing is re-rendered.
    Each SKU gets one clean studio shot (``catalog/<sku>/ref.jpg``) and
    ``queries_per_sku`` simulated smartphone photos (``queries/<sku>/q<j>.jpg``)
    produced by :func:`src.distortion.apply_distortion` with random parameters.
    """
    config = config or SyntheticDatasetConfig()
    root = Path(root)
    manifest_path = root / "manifest.json"
    regenerated = False
    if overwrite or not _manifest_matches(manifest_path, config):
        regenerated = True
        logger.info(
            "Rendering synthetic catalog: %d models x %s colourways, %d queries/SKU -> %s",
            config.n_models, config.colourways_per_model, config.queries_per_sku, root,
        )
        if manifest_path.exists():
            manifest_path.unlink()
        for sub in ("catalog", "queries"):  # stale SKU folders from another config would leak in
            if (root / sub).is_dir():
                shutil.rmtree(root / sub)
        specs = generate_catalog_specs(config.n_models, config.colourways_per_model, seed=config.seed)
        rng = np.random.default_rng(config.seed + 1)
        query_params = [
            [sample_distortion_params(rng, config.distortion_strength) for _ in range(config.queries_per_sku)] for _ in specs
        ]

        def render_one(job: tuple[SneakerSpec, list[DistortionParams]]) -> None:
            spec, plist = job
            cat_dir = root / "catalog" / spec.sku_id
            q_dir = root / "queries" / spec.sku_id
            cat_dir.mkdir(parents=True, exist_ok=True)
            q_dir.mkdir(parents=True, exist_ok=True)
            clean = render_sneaker(spec, size=config.image_size)
            clean.save(cat_dir / "ref.jpg", quality=95)
            for j, params in enumerate(plist):
                apply_distortion(clean, params).save(q_dir / f"q{j}.jpg", quality=95)

        with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
            for i, _ in enumerate(pool.map(render_one, zip(specs, query_params))):
                if (i + 1) % 200 == 0:
                    logger.info("  rendered %d / %d SKUs", i + 1, len(specs))
        pd.DataFrame([s.to_record() for s in specs]).to_csv(root / "metadata.csv", index=False)
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump({"config": config.to_json_dict(), "n_skus": len(specs), "complete": True}, fh, indent=2)
    else:
        logger.info("Synthetic catalog cache hit: %s", root)

    catalog = load_catalog_folder(root / "catalog", root / "metadata.csv")
    queries = load_query_folder(root / "queries", catalog)
    return SyntheticDataset(root, catalog, queries, config, regenerated)


# ---------------------------------------------------------------------- splits
def split_skus_by_group(
    groups: np.ndarray,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Split SKU indices into train / val / test so that a *group* (model) never straddles splits.

    Holding out whole models means the projection head is evaluated on shoe shapes
    it has never seen - not just on unseen colourways of familiar shapes.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())), dtype=object)
    rng = np.random.default_rng(seed)
    perm = uniq[rng.permutation(len(uniq))]
    n = len(perm)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    if n >= 3:
        n_train = min(max(n_train, 1), n - 2)
        n_val = min(max(n_val, 1), n - n_train - 1)
    parts = {
        "train": set(perm[:n_train].tolist()),
        "val": set(perm[n_train : n_train + n_val].tolist()),
        "test": set(perm[n_train + n_val :].tolist()),
    }
    return {k: np.flatnonzero([g in v for g in groups.tolist()]) for k, v in parts.items()}
