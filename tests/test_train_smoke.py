"""Tiny end-to-end run of train.py (random weights, 28 px, 1k distractors, no MLflow)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import train
from src.embedder import Embedder, EmbedderConfig
from src.render import generate_catalog_specs, render_sneaker
from src.search import load_engine_from_artifacts


def test_train_pipeline_end_to_end(tmp_path: Path) -> None:
    art = tmp_path / "artifacts"
    summary = train.main([
        "--n-models", "6", "--colourways", "2", "3", "--queries-per-sku", "1", "--views-per-sku", "2",
        "--no-pretrained", "--image-size", "28", "--epochs", "2", "--distractors", "1000",
        "--max-latency-queries", "20", "--device", "cpu", "--no-mlflow",
        "--data-dir", str(tmp_path / "data"), "--artifacts-dir", str(art),
    ])
    for rel in (
        "embeddings/catalog_raw.npy", "embeddings/queries_head.npy", "indexes/catalog_flat.faiss",
        "indexes/catalog_ivf.faiss", "indexes/catalog_hnsw.faiss", "projection_head.pt", "benchmark.csv",
        "head_comparison.csv", "training_history.csv", "catalog.csv", "queries.csv", "metrics.json",
        "manifest.json", "plots/recall_latency.png", "plots/training_history.png", "plots/head_comparison.png",
    ):
        assert (art / rel).is_file(), rel
    bench = pd.read_csv(art / "benchmark.csv")
    assert set(bench["kind"]) == {"flat", "ivf", "hnsw"} and (bench["ntotal"] > 1000).all()
    metrics = json.loads((art / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["pretrained_loaded"] is False and set(metrics["benchmark_defaults"]) == {"flat", "ivf", "hnsw"}
    assert summary["n_skus"] == metrics["n_skus"]
    emb = np.load(art / "embeddings" / "catalog_raw.npy")
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)

    # The app can load these artifacts with a matching embedder configuration.
    embedder = Embedder(EmbedderConfig(pretrained=False, image_size=28, device="cpu"))
    engine, reason = load_engine_from_artifacts(art, embedder)
    assert engine is not None, reason
    assert engine.source == "artifacts" and engine.spaces == ["raw", "head"]
    assert set(engine.indexes("head")) == {"flat", "ivf", "hnsw"}
    # ... and refuses them with a different backbone configuration.
    other = Embedder(EmbedderConfig(pretrained=False, image_size=42, device="cpu"))
    assert load_engine_from_artifacts(art, other)[0] is None


def test_train_with_user_catalog_folder(tmp_path: Path) -> None:
    """``--catalog-dir`` without real queries: phone photos are simulated in memory."""
    for spec in generate_catalog_specs(3, (2, 2), seed=9):
        out = tmp_path / "catalog" / spec.sku_id
        out.mkdir(parents=True)
        render_sneaker(spec, size=64).save(out / "ref.jpg")
    art = tmp_path / "artifacts"
    summary = train.main([
        "--catalog-dir", str(tmp_path / "catalog"), "--queries-per-sku", "1", "--views-per-sku", "2",
        "--no-pretrained", "--image-size", "28", "--epochs", "1", "--distractors", "200",
        "--max-latency-queries", "5", "--device", "cpu", "--no-mlflow", "--artifacts-dir", str(art),
    ])
    assert summary["data_source"] == "user catalog + simulated queries" and summary["n_skus"] == 6
    queries = pd.read_csv(art / "queries.csv")
    assert len(queries) == 6 and (queries["image_path"] == "<in-memory>").all()
    assert (art / "plots" / "examples.png").is_file()
