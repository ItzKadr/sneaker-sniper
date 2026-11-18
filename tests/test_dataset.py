from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.dataset import (
    Catalog,
    SyntheticDatasetConfig,
    bootstrap_synthetic_dataset,
    iter_distorted_views,
    load_catalog_folder,
    load_query_folder,
    make_synthetic_queries,
    split_skus_by_group,
)


def _cfg(**kw) -> SyntheticDatasetConfig:
    base = dict(n_models=2, colourways_per_model=(2, 2), queries_per_sku=1, image_size=48, seed=5)
    base.update(kw)
    return SyntheticDatasetConfig(**base)


def test_bootstrap_writes_layout_and_caches(tmp_path: Path) -> None:
    ds = bootstrap_synthetic_dataset(tmp_path / "syn", _cfg())
    assert ds.regenerated
    assert ds.catalog.n_skus == 4 and len(ds.queries) == 4
    assert (tmp_path / "syn" / "metadata.csv").is_file()
    manifest = json.loads((tmp_path / "syn" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] and manifest["n_skus"] == 4
    assert len(set(ds.catalog.groups.tolist())) == 2  # model_id read from metadata.csv
    assert np.array_equal(np.sort(ds.queries.labels), np.arange(4))
    again = bootstrap_synthetic_dataset(tmp_path / "syn", _cfg())
    assert not again.regenerated and again.catalog.sku_ids == ds.catalog.sku_ids


def test_bootstrap_config_change_rerenders_without_stale_skus(tmp_path: Path) -> None:
    bootstrap_synthetic_dataset(tmp_path / "syn", _cfg(n_models=3))
    ds = bootstrap_synthetic_dataset(tmp_path / "syn", _cfg(n_models=1))
    assert ds.regenerated and ds.catalog.n_skus == 2


def _write(path: Path, colour: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), colour).save(path)


def test_load_user_catalog_and_queries(tmp_path: Path) -> None:
    _write(tmp_path / "cat" / "skuA" / "1.jpg", (255, 0, 0))
    _write(tmp_path / "cat" / "skuA" / "2.png", (250, 0, 0))
    _write(tmp_path / "cat" / "skuB" / "1.jpg", (0, 0, 255))
    (tmp_path / "cat" / "empty").mkdir()
    cat = load_catalog_folder(tmp_path / "cat")
    assert cat.sku_ids == ["skuA", "skuB"]
    assert cat.labels.tolist() == [0, 0, 1]
    assert cat.first_image_index().tolist() == [0, 2]
    assert cat.groups.tolist() == ["skuA", "skuB"]  # no metadata -> each SKU its own group

    _write(tmp_path / "q" / "skuB" / "x.jpg", (0, 0, 250))
    _write(tmp_path / "q" / "unknown" / "x.jpg", (0, 0, 0))
    qs = load_query_folder(tmp_path / "q", cat)
    assert qs.labels.tolist() == [1]

    with pytest.raises(FileNotFoundError):
        load_catalog_folder(tmp_path / "missing")


def test_distorted_views_order_and_count(tiny_catalog: Catalog) -> None:
    chunks = list(iter_distorted_views(tiny_catalog, [2, 0], views_per_sku=3, seed=1, chunk_size=4))
    labels = np.concatenate([c[0] for c in chunks])
    assert labels.tolist() == [2, 2, 2, 0, 0, 0]
    assert all(im.size == (64, 64) for c in chunks for im in c[1])
    qs = make_synthetic_queries(tiny_catalog, per_sku=2, seed=0)
    assert len(qs) == 2 * tiny_catalog.n_skus


def test_split_by_group_is_leak_free_and_complete() -> None:
    groups = np.array([f"m{i // 4}" for i in range(40)], dtype=object)  # 10 models x 4 SKUs
    splits = split_skus_by_group(groups, (0.6, 0.2, 0.2), seed=0)
    all_idx = np.concatenate(list(splits.values()))
    assert sorted(all_idx.tolist()) == list(range(40))
    seen = {name: set(groups[idx].tolist()) for name, idx in splits.items()}
    assert not (seen["train"] & seen["val"]) and not (seen["train"] & seen["test"]) and not (seen["val"] & seen["test"])
    assert len(seen["train"]) == 6 and len(seen["val"]) == 2 and len(seen["test"]) == 2
    again = split_skus_by_group(groups, (0.6, 0.2, 0.2), seed=0)
    assert all(np.array_equal(splits[k], again[k]) for k in splits)
    with pytest.raises(ValueError):
        split_skus_by_group(groups, (0.5, 0.2, 0.2))
