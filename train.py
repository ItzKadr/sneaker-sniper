"""End-to-end pipeline: catalog -> embeddings -> metric learning -> FAISS -> benchmark.

    python train.py                     # quick mode (~3-5 min first run, faster when cached)
    python train.py --mode full         # bigger catalog, 200k distractors
    python train.py --catalog-dir data/catalog --queries-dir data/queries   # your own photos
    python train.py --model resnet50    # supervised CNN backbone instead of DINOv2

Steps
  1. bootstrap (or load) the catalog + phone-photo queries
  2. split SKUs by *model* into train / val / test (no shape leaks between splits)
  3. embed catalog + queries with the frozen timm backbone (L2-normalised)
  4. render distorted training views of train SKUs, embed, train the projection head
  5. compare raw backbone vs head on held-out test SKUs (exact search, full catalog)
  6. benchmark Exact / IVF / HNSW on catalog + synthetic distractors (recall vs p50/p95)
  7. save artifacts/ (embeddings .npy, .faiss indexes, csv, json, png) + log to MLflow
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import src  # noqa: E402,F401  (sets env defaults before torch/faiss/mlflow import)
from src.dataset import (  # noqa: E402
    Catalog,
    QuerySet,
    SyntheticDatasetConfig,
    bootstrap_synthetic_dataset,
    iter_distorted_views,
    load_catalog_folder,
    load_image,
    load_query_folder,
    make_synthetic_queries,
    split_skus_by_group,
)
from src.embedder import DEFAULT_MODEL, Embedder, EmbedderConfig  # noqa: E402
from src.evaluator import (  # noqa: E402
    DISTRACTOR_MODES,
    BenchmarkConfig,
    ids_to_labels,
    overlap_at_k,
    retrieval_metrics,
    run_benchmark,
)
from src.indexer import HNSWParams, IVFParams, build_index_suite  # noqa: E402
from src.metric_learning import (  # noqa: E402
    HEAD_TYPES,
    LOSSES,
    HeadConfig,
    MetricTrainConfig,
    ValidationSet,
    apply_head,
    compare_raw_vs_head,
    save_head,
    train_projection_head,
)
from src.plots import plot_head_comparison, plot_recall_latency, plot_training_history, save_image_grid  # noqa: E402
from src.tracking import MLflowTracker  # noqa: E402
from src.utils import REPO_ROOT, configure_threads, relative_to_repo, seed_everything, setup_logging  # noqa: E402

logger = logging.getLogger("train")

MODE_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {"n_models": 150, "queries_per_sku": 2, "views_per_sku": 6, "epochs": 60, "distractors": 50_000, "max_latency_queries": 1000},
    "full": {"n_models": 500, "queries_per_sku": 3, "views_per_sku": 10, "epochs": 100, "distractors": 200_000, "max_latency_queries": 2000},
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sneaker Sniper: zero-tag visual search training & benchmark pipeline")
    p.add_argument("--mode", choices=sorted(MODE_PRESETS), default="quick", help="size preset (default: quick)")
    g = p.add_argument_group("data")
    g.add_argument("--catalog-dir", type=Path, default=None, help="user catalog <dir>/<sku_id>/*.jpg (default: synthetic)")
    g.add_argument("--queries-dir", type=Path, default=None, help="real query photos <dir>/<sku_id>/*.jpg (optional)")
    g.add_argument("--data-dir", type=Path, default=None, help="synthetic data cache (default: data/synthetic_<mode>)")
    g.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    g.add_argument("--n-models", type=int, default=None, help="synthetic sneaker models (each has 3-5 colourway SKUs)")
    g.add_argument("--colourways", type=int, nargs=2, default=(3, 5), metavar=("MIN", "MAX"))
    g.add_argument("--queries-per-sku", type=int, default=None)
    g.add_argument("--views-per-sku", type=int, default=None, help="distorted training views per train SKU")
    g.add_argument("--distortion-strength", type=float, default=1.0)
    g = p.add_argument_group("backbone")
    g.add_argument("--model", default=DEFAULT_MODEL, help="timm model name (e.g. resnet50)")
    g.add_argument("--image-size", type=int, default=224, help="input resolution (snapped to patch multiple for ViTs)")
    g.add_argument("--no-pretrained", action="store_true", help="random weights (offline smoke tests)")
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--device", default="auto")
    g = p.add_argument_group("metric learning")
    g.add_argument("--head", choices=HEAD_TYPES, default="residual_mlp")
    g.add_argument("--head-dim", type=int, default=None, help="output dim (linear / mlp heads)")
    g.add_argument("--loss", choices=LOSSES, default="infonce")
    g.add_argument("--epochs", type=int, default=None)
    g.add_argument("--temperature", type=float, default=0.07)
    g.add_argument("--margin", type=float, default=0.2)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--hard-negative-ratio", type=float, default=0.5)
    g.add_argument("--head-device", default="cpu", help="device for the projection head (cpu|cuda|auto); features are cached so cpu is enough")
    g.add_argument("--index-space", choices=["raw", "head"], default="head", help="embedding space indexed by FAISS")
    g = p.add_argument_group("indexes & benchmark")
    g.add_argument("--distractors", type=int, default=None, help="synthetic distractor vectors added to the index")
    g.add_argument("--distractor-mode", choices=DISTRACTOR_MODES, default="gaussian")
    g.add_argument("--nlist", type=int, default=None, help="IVF cells (default ~4*sqrt(N))")
    g.add_argument("--nprobe", type=int, default=64, help="IVF cells probed per query (default operating point)")
    g.add_argument("--hnsw-m", type=int, default=32)
    g.add_argument("--ef-construction", type=int, default=200)
    g.add_argument("--ef-search", type=int, default=64)
    g.add_argument("--max-latency-queries", type=int, default=None)
    g = p.add_argument_group("runtime")
    g.add_argument("--threads", type=int, default=4)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--no-mlflow", action="store_true")
    g.add_argument("--experiment", default="sneaker-sniper")
    args = p.parse_args(argv)
    for key, value in MODE_PRESETS[args.mode].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.data_dir is None:
        args.data_dir = REPO_ROOT / "data" / f"synthetic_{args.mode}"
    return args


# ------------------------------------------------------------------ helpers
def _image_ref(item: Any) -> str:
    return relative_to_repo(Path(item)) if isinstance(item, (str, Path)) else "<in-memory>"


def load_data(args: argparse.Namespace) -> tuple[Catalog, QuerySet, str]:
    if args.catalog_dir is not None:
        catalog = load_catalog_folder(args.catalog_dir)
        if args.queries_dir is not None:
            queries = load_query_folder(args.queries_dir, catalog)
            source = "user catalog + real queries"
        else:
            queries = make_synthetic_queries(catalog, args.queries_per_sku, args.distortion_strength, args.seed)
            source = "user catalog + simulated queries"
        return catalog, queries, source
    cfg = SyntheticDatasetConfig(
        n_models=args.n_models,
        colourways_per_model=tuple(args.colourways),
        queries_per_sku=args.queries_per_sku,
        distortion_strength=args.distortion_strength,
        seed=args.seed,
    )
    ds = bootstrap_synthetic_dataset(args.data_dir, cfg)
    return ds.catalog, ds.queries, "synthetic"


def embed_training_views(
    embedder: Embedder, catalog: Catalog, sku_indices: np.ndarray, views_per_sku: int, strength: float, seed: int
) -> np.ndarray:
    """``(S, V, d)`` raw features of simulated phone photos (rendered lazily, never saved)."""
    total = len(sku_indices) * views_per_sku
    flat = np.empty((total, embedder.dim), dtype=np.float32)
    pos = 0
    for i, (_, images) in enumerate(iter_distorted_views(catalog, sku_indices, views_per_sku, strength, seed, chunk_size=128)):
        flat[pos : pos + len(images)] = embedder.embed(images)
        pos += len(images)
        if (i + 1) % 10 == 0:
            logger.info("  training views rendered + embedded: %d / %d", pos, total)
    return flat.reshape(len(sku_indices), views_per_sku, embedder.dim)


def save_examples(
    catalog: Catalog, queries: QuerySet, q_mask: np.ndarray, catalog_emb: np.ndarray, query_emb: np.ndarray, path: Path, n_skus: int = 6
) -> None:
    """Contact sheet: catalog ref | phone photo | exact top-1 result, for a few test SKUs."""
    from src.indexer import build_flat_index

    first = catalog.first_image_index()
    index = build_flat_index(catalog_emb)
    images, captions = [], []
    test_q = np.flatnonzero(q_mask)
    seen: list[int] = []
    for qi in test_q:
        lab = int(queries.labels[qi])
        if lab in seen:
            continue
        seen.append(lab)
        _, ids = index.search(query_emb[qi : qi + 1], 1)
        top = int(ids[0, 0])
        hit = int(catalog.labels[top]) == lab
        images += [load_image(catalog.images[first[lab]]), load_image(queries.images[qi]), load_image(catalog.images[top])]
        captions += [f"catalog {catalog.sku_ids[lab]}", "phone photo (query)", f"top-1 {catalog.sku_ids[catalog.labels[top]]} {'OK' if hit else 'MISS'}"]
        if len(seen) >= n_skus:
            break
    if images:
        save_image_grid(images, captions, path, ncols=6)


# --------------------------------------------------------------------- main
def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    setup_logging()
    seed_everything(args.seed)
    configure_threads(args.threads)
    art = Path(args.artifacts_dir)
    for sub in ("embeddings", "indexes", "plots"):
        (art / sub).mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    # 1. data ------------------------------------------------------------------
    t0 = time.perf_counter()
    catalog, queries, source = load_data(args)
    timings["data_s"] = time.perf_counter() - t0
    splits = split_skus_by_group(catalog.groups, (0.6, 0.2, 0.2), seed=args.seed)
    sku_split = np.empty(catalog.n_skus, dtype=object)
    for name, idx in splits.items():
        sku_split[idx] = name
    q_split = sku_split[queries.labels]
    logger.info(
        "Data (%s): %d SKUs / %d catalog images / %d queries | split SKUs train=%d val=%d test=%d",
        source, catalog.n_skus, len(catalog), len(queries), len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )

    # 2. embeddings ------------------------------------------------------------
    embedder = Embedder(EmbedderConfig(args.model, not args.no_pretrained, args.image_size, args.batch_size, args.device))
    t0 = time.perf_counter()
    catalog_raw = embedder.embed(catalog.images, log_every=10)
    query_raw = embedder.embed(queries.images, log_every=10)
    timings["embed_s"] = time.perf_counter() - t0
    throughput = (len(catalog) + len(queries)) / max(timings["embed_s"], 1e-9)
    logger.info("Embedded %d images in %.1fs (%.0f img/s incl. JPEG decode + preprocessing)", len(catalog) + len(queries), timings["embed_s"], throughput)

    # 3-4. metric learning -----------------------------------------------------
    t0 = time.perf_counter()
    first = catalog.first_image_index()
    train_skus = splits["train"]
    views = embed_training_views(embedder, catalog, train_skus, args.views_per_sku, args.distortion_strength, args.seed + 1000)
    timings["train_views_s"] = time.perf_counter() - t0
    val_mask = q_split == "val"
    test_mask = q_split == "test"
    if not test_mask.any():
        test_mask = val_mask
    val = ValidationSet(catalog_raw, catalog.labels, query_raw[val_mask], queries.labels[val_mask]) if val_mask.any() else None
    head_cfg = HeadConfig(head_type=args.head, out_dim=args.head_dim)
    train_cfg = MetricTrainConfig(
        loss=args.loss, temperature=args.temperature, margin=args.margin, epochs=args.epochs, lr=args.lr,
        hard_negative_ratio=args.hard_negative_ratio, device=args.head_device, seed=args.seed,
    )
    t0 = time.perf_counter()
    result = train_projection_head(catalog_raw[first[train_skus]], views, val, head_cfg, train_cfg)
    timings["head_train_s"] = time.perf_counter() - t0
    head = result.head
    comparison = compare_raw_vs_head(catalog_raw, catalog.labels, query_raw[test_mask], queries.labels[test_mask], head, split="test")
    logger.info("Held-out test SKUs | raw: %s | head: %s", comparison.raw.as_dict(), comparison.head.as_dict())

    # 5. index space -----------------------------------------------------------
    if args.index_space == "head":
        catalog_emb, query_emb = apply_head(head, catalog_raw), apply_head(head, query_raw)
    else:
        catalog_emb, query_emb = catalog_raw, query_raw

    # 6a. catalog-only indexes (what the app serves) ---------------------------
    ivf_params = IVFParams(nlist=args.nlist, nprobe=args.nprobe, seed=args.seed)
    hnsw_params = HNSWParams(M=args.hnsw_m, ef_construction=args.ef_construction, ef_search=args.ef_search)
    suite = build_index_suite(catalog_emb, IVFParams(nlist=None, nprobe=args.nprobe, seed=args.seed), hnsw_params)
    catalog_rows = []
    _, exact_ids = suite["flat"].search(query_emb[test_mask], 10)
    for kind, index in suite.items():
        _, ids = index.search(query_emb[test_mask], 10)
        m = retrieval_metrics(ids_to_labels(ids, catalog.labels), queries.labels[test_mask])
        catalog_rows.append({"index": index.label, "kind": kind, **m.as_dict(), "overlap@10": overlap_at_k(exact_ids, ids, 10), "ntotal": index.ntotal, **{f"param_{k}": v for k, v in index.params.items()}})
        index.save(art / "indexes" / f"catalog_{kind}.faiss")
    catalog_eval = pd.DataFrame(catalog_rows)

    # 6b. benchmark with distractors -------------------------------------------
    bench_cfg = BenchmarkConfig(
        n_distractors=args.distractors, distractor_mode=args.distractor_mode, max_latency_queries=args.max_latency_queries,
        ivf=ivf_params, hnsw=hnsw_params, seed=args.seed,
    )
    t0 = time.perf_counter()
    bench = run_benchmark(catalog_emb, catalog.labels, query_emb[test_mask], queries.labels[test_mask], bench_cfg)
    timings["benchmark_s"] = time.perf_counter() - t0
    bench_df = bench.to_frame()

    # 7. artifacts ---------------------------------------------------------------
    np.save(art / "embeddings" / "catalog_raw.npy", catalog_raw)
    np.save(art / "embeddings" / "queries_raw.npy", query_raw)
    np.save(art / "embeddings" / "catalog_head.npy", apply_head(head, catalog_raw))
    np.save(art / "embeddings" / "queries_head.npy", apply_head(head, query_raw))
    save_head(head, art / "projection_head.pt")
    pd.DataFrame({
        "sku_id": [catalog.sku_ids[label] for label in catalog.labels],
        "model_id": [str(catalog.groups[label]) for label in catalog.labels],
        "label": catalog.labels,
        "image_path": [_image_ref(im) for im in catalog.images],
        "split": [sku_split[label] for label in catalog.labels],
    }).to_csv(art / "catalog.csv", index=False)
    pd.DataFrame({
        "image_path": [_image_ref(im) for im in queries.images],
        "sku_id": [catalog.sku_ids[label] for label in queries.labels],
        "label": queries.labels,
        "split": q_split,
    }).to_csv(art / "queries.csv", index=False)
    bench_df.to_csv(art / "benchmark.csv", index=False)
    catalog_eval.to_csv(art / "catalog_only_eval.csv", index=False)
    comparison.to_frame().to_csv(art / "head_comparison.csv", index=False)
    result.history.to_csv(art / "training_history.csv", index=False)

    plot_recall_latency(
        bench_df, art / "plots" / "recall_latency.png", k=bench.k,
        title=f"Recall vs latency: {bench.n_gallery} SKUs + {bench.n_distractors:,} {bench.distractor_mode} distractors, d={bench.dim}, {bench.n_queries} test queries",
    )
    plot_training_history(result.history, art / "plots" / "training_history.png", result.best_epoch)
    plot_head_comparison(comparison.to_frame(), art / "plots" / "head_comparison.png")
    save_examples(catalog, queries, test_mask, catalog_emb, query_emb, art / "plots" / "examples.png")

    defaults = bench.default_rows().set_index("kind")
    timings["total_s"] = time.perf_counter() - t_all
    summary: dict[str, Any] = {
        "mode": args.mode,
        "data_source": source,
        "n_skus": catalog.n_skus,
        "n_catalog_images": len(catalog),
        "n_queries": len(queries),
        "n_test_queries": int(test_mask.sum()),
        "n_train_skus": int(len(train_skus)),
        "backbone": args.model,
        "pretrained_loaded": embedder.pretrained_loaded,
        "embedding_dim": embedder.dim,
        "index_space": args.index_space,
        "head_best_epoch": result.best_epoch,
        "test_raw": comparison.raw.as_dict(),
        "test_head": comparison.head.as_dict(),
        "test_delta": comparison.delta(),
        "benchmark_ntotal": int(bench_df["ntotal"].max()),
        "benchmark_defaults": {
            kind: {c: float(defaults.loc[kind, c]) for c in ("param_value", "recall_at_1", "recall_at_10", "overlap_at_k", "p50_ms", "p95_ms", "memory_mb", "build_time_s")}
            for kind in defaults.index
        },
        "embed_throughput_img_s": throughput,
        "timings": timings,
    }
    with open(art / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    manifest = {
        "embedder": embedder.describe(),
        "index_space": args.index_space,
        "ivf": asdict(IVFParams(nlist=None, nprobe=args.nprobe, seed=args.seed)),
        "hnsw": asdict(hnsw_params),
        "data_source": source,
        "data_dir": relative_to_repo(args.data_dir) if args.catalog_dir is None else relative_to_repo(args.catalog_dir),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    with open(art / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    # MLflow (never fatal) ---------------------------------------------------------
    with MLflowTracker(REPO_ROOT, args.experiment, run_name=f"{args.mode}-{args.model.split('.')[0]}", enabled=not args.no_mlflow) as tracker:
        tracker.log_params({**manifest["args"], "pretrained_loaded": embedder.pretrained_loaded, "embedding_dim": embedder.dim,
                            "n_skus": catalog.n_skus, "n_queries": len(queries), "n_test_queries": int(test_mask.sum())})
        tracker.set_tags({"data_source": source, "backbone": args.model})
        for _, row in result.history.iterrows():
            tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=int(row["epoch"]))
        tracker.log_metrics({f"test_raw_{k}": v for k, v in comparison.raw.as_dict().items()})
        tracker.log_metrics({f"test_head_{k}": v for k, v in comparison.head.as_dict().items()})
        for _, row in bench_df.iterrows():
            step = int(row["param_value"])
            tracker.log_metrics({f"{row['kind']}_{c}": row[c] for c in ("recall_at_1", "recall_at_10", "overlap_at_k", "p50_ms", "p95_ms")}, step=step)
        for kind, vals in summary["benchmark_defaults"].items():
            tracker.log_metrics({f"default_{kind}_{k}": v for k, v in vals.items()})
        tracker.log_metrics({f"time_{k}": v for k, v in timings.items()})
        tracker.log_artifacts(art)

    # Console report (ASCII only) ---------------------------------------------------
    print("\n=== Sneaker Sniper - run summary ({}) ===".format(args.mode))
    print(f"catalog: {catalog.n_skus} SKUs | queries: {len(queries)} (test: {int(test_mask.sum())}) | backbone: {args.model} (pretrained={embedder.pretrained_loaded})")
    print("held-out test SKUs, exact search over full catalog:")
    print(comparison.to_frame().to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nbenchmark @ {int(bench_df['ntotal'].max()):,} vectors ({bench.n_distractors:,} {bench.distractor_mode} distractors), index space = {args.index_space}:")
    cols = ["index", "param_name", "param_value", "recall_at_1", "recall_at_10", "overlap_at_k", "p50_ms", "p95_ms", "memory_mb", "build_time_s"]
    print(bench_df[bench_df["is_default"]][cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nartifacts -> {art}   total time: {timings['total_s']:.1f}s")
    return summary


if __name__ == "__main__":
    main()
