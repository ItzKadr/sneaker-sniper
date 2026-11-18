"""Sneaker Sniper - interactive zero-tag visual search dashboard.

    streamlit run app.py

Works on a fresh clone: without ``artifacts/`` (i.e. before ``python train.py``)
a small synthetic catalog is rendered and embedded in memory. If pretrained
weights cannot be downloaded, the backbone falls back to random weights and a
warning is shown.

Environment overrides (used by the smoke test):
  SNEAKER_SNIPER_PRETRAINED=0         random-init backbone (no download)
  SNEAKER_SNIPER_IGNORE_ARTIFACTS=1   always use the in-memory demo catalog
  SNEAKER_SNIPER_DEMO_MODELS=30       demo catalog size (models x 3-4 colourways)
  SNEAKER_SNIPER_IMAGE_SIZE=224       backbone input resolution
  SNEAKER_SNIPER_DEMO_DISTRACTORS=20000  default distractors of the live benchmark
  SNEAKER_SNIPER_DEVICE=cpu           force a device
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

import src  # noqa: E402,F401  (environment defaults)
from src.distortion import DistortionParams, apply_distortion  # noqa: E402
from src.embedder import ALTERNATIVE_MODEL, DEFAULT_MODEL, Embedder, EmbedderConfig  # noqa: E402
from src.evaluator import BenchmarkConfig, overlap_at_k, pairwise_cosine_distance, run_benchmark, self_cosine  # noqa: E402
from src.indexer import INDEX_KINDS, INDEX_LABELS, HNSWParams, IVFParams  # noqa: E402
from src.plots import (  # noqa: E402
    fig_head_comparison,
    fig_recall_latency,
    fig_similarity_heatmap,
    fig_stability_sweep,
    fig_training_history,
)
from src.search import SearchEngine, build_demo_engine, load_engine_from_artifacts, stability_profile  # noqa: E402
from src.utils import REPO_ROOT, configure_threads  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


PRETRAINED = _env_flag("SNEAKER_SNIPER_PRETRAINED", True)
IGNORE_ARTIFACTS = _env_flag("SNEAKER_SNIPER_IGNORE_ARTIFACTS", False)
DEMO_MODELS = _env_int("SNEAKER_SNIPER_DEMO_MODELS", 30)
IMAGE_SIZE = _env_int("SNEAKER_SNIPER_IMAGE_SIZE", 224)
DEMO_DISTRACTORS = _env_int("SNEAKER_SNIPER_DEMO_DISTRACTORS", 20_000)

st.set_page_config(page_title="Sneaker Sniper", layout="wide")


# --------------------------------------------------------------------- caching
@st.cache_resource(show_spinner="Loading backbone (the first run downloads pretrained weights)...")
def get_embedder(model_name: str, pretrained: bool, image_size: int) -> Embedder:
    configure_threads(4)
    return Embedder(EmbedderConfig(model_name=model_name, pretrained=pretrained, image_size=image_size, batch_size=32))


def artifact_candidates(model_name: str) -> list[Path]:
    """``artifacts/`` (default run) and ``artifacts/<model>/`` (e.g. ``--artifacts-dir artifacts/resnet50``)."""
    return [ARTIFACTS, ARTIFACTS / model_name.split(".")[0]]


@st.cache_resource(show_spinner="Preparing the catalog index...")
def get_engine(model_name: str, pretrained: bool, image_size: int, demo_models: int, ignore_artifacts: bool) -> tuple[SearchEngine, str]:
    embedder = get_embedder(model_name, pretrained, image_size)
    note = "artifacts ignored (SNEAKER_SNIPER_IGNORE_ARTIFACTS=1)" if ignore_artifacts else ""
    if not ignore_artifacts:
        for candidate in artifact_candidates(model_name):
            engine, reason = load_engine_from_artifacts(candidate, embedder)
            if engine is not None:
                return engine, ""
            note = note or reason
    return build_demo_engine(embedder, n_models=demo_models, image_size=224), note


@st.cache_data(show_spinner="Measuring distortion stability...")
def get_stability(_engine: SearchEngine, engine_key: str, query_key: str, space: str, _image: Image.Image) -> pd.DataFrame:
    return stability_profile(_engine, _image, space)


@st.cache_data(show_spinner="Running live benchmark (building 3 indexes with distractors)...")
def get_live_benchmark(_engine: SearchEngine, engine_key: str, space: str, n_distractors: int, nprobe: int, ef_search: int) -> pd.DataFrame:
    gallery = _engine.gallery(space)
    queries, labels = _engine.eval_queries(space)
    cfg = BenchmarkConfig(
        n_distractors=n_distractors,
        max_latency_queries=300,
        nprobe_values=(1, 2, 4, 8, 16, 32, 64),
        ef_search_values=(8, 16, 32, 64, 128),
        ivf=IVFParams(nprobe=nprobe),
        hnsw=HNSWParams(M=32, ef_construction=100, ef_search=ef_search),
    )
    return run_benchmark(gallery, _engine.catalog.labels, queries, labels, cfg).to_frame()


@st.cache_data
def read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def artifact_csv(art_dir: Path | None, name: str) -> pd.DataFrame | None:
    """Read a CSV written by train.py from the artifacts directory the engine was loaded from."""
    if art_dir is None:
        return None
    path = art_dir / name
    return read_csv(str(path), path.stat().st_mtime) if path.is_file() else None


# ---------------------------------------------------------------------- sidebar
st.sidebar.title("Sneaker Sniper")
st.sidebar.caption("Zero-tag visual search: photo in, catalog SKU out.")
model_name = st.sidebar.selectbox("Backbone", [DEFAULT_MODEL, ALTERNATIVE_MODEL], help="DINOv2 ViT-S/14 (self-supervised) or ResNet-50 (supervised ImageNet)")
engine, engine_note = get_engine(model_name, PRETRAINED, IMAGE_SIZE, DEMO_MODELS, IGNORE_ARTIFACTS)
embedder = engine.embedder
engine_key = f"{model_name}|{PRETRAINED}|{IMAGE_SIZE}|{DEMO_MODELS}|{IGNORE_ARTIFACTS}|{engine.source}"
art_dir = Path(engine.info["artifacts_dir"]) if engine.source == "artifacts" else None

if embedder.warning:
    st.warning(embedder.warning)
elif not embedder.pretrained_loaded:
    st.warning("Backbone runs with RANDOM weights (SNEAKER_SNIPER_PRETRAINED=0) - results are not meaningful.")

space_labels = {"raw": "raw backbone", "head": "projection head (metric learning)"}
space = st.sidebar.radio("Embedding space", engine.spaces, format_func=lambda s: space_labels[s], index=len(engine.spaces) - 1)
k = st.sidebar.slider("Top-K", 1, 10, 5)
indexes = engine.indexes(space)
nlist = int(indexes["ivf"].params.get("nlist", 1))
nprobe = st.sidebar.slider("IVF nprobe (cells scanned)", 1, max(2, nlist), min(max(1, nlist // 4), nlist), help=f"nlist = {nlist} Voronoi cells")
ef_search = st.sidebar.select_slider("HNSW efSearch (beam width)", options=[4, 8, 16, 32, 64, 128, 256], value=32)

st.sidebar.divider()
st.sidebar.markdown(
    f"**Catalog:** {engine.catalog.n_skus} SKUs ({'train.py artifacts' if engine.source == 'artifacts' else 'in-memory demo'})  \n"
    f"**Backbone:** `{model_name}`  \n**Pretrained:** {embedder.pretrained_loaded} | **dim:** {embedder.dim} | "
    f"**input:** {embedder.data_config['input_size'][1]}px | **device:** {embedder.device}"
)
if engine_note and engine.source != "artifacts":
    st.sidebar.info(f"Demo catalog in use: {engine_note}. Run `python train.py` for the full pipeline.")

# ------------------------------------------------------------------- query input
st.title("Sneaker Sniper - find the SKU from a phone photo")
st.caption("No text tags: a frozen vision backbone embeds the photo, FAISS finds the nearest catalog images (Exact vs IVF vs HNSW).")

src_col, sim_col = st.columns([1, 2])
with src_col:
    mode = st.radio("Query source", ["Catalog SKU", "Upload a photo"], horizontal=True)
    true_label: int | None = None
    if mode == "Upload a photo":
        upload = st.file_uploader("Sneaker photo", type=["jpg", "jpeg", "png", "webp"])
        if upload is not None:
            raw_bytes = upload.getvalue()
            clean = Image.open(upload).convert("RGB")
            clean.thumbnail((512, 512))
            query_key = "upload-" + hashlib.md5(raw_bytes).hexdigest()
        else:
            st.info("No file uploaded yet - showing the first catalog SKU.")
            mode = "Catalog SKU"
    if mode == "Catalog SKU":
        sku = st.selectbox("Catalog SKU", engine.catalog.sku_ids, index=0)
        true_label = engine.catalog.sku_ids.index(sku)
        gid = int(engine.catalog.first_image_index()[true_label])
        clean = engine.gallery_image(gid)
        query_key = f"sku-{sku}"

with sim_col:
    st.markdown("**Smartphone camera simulator**")
    c1, c2, c3 = st.columns(3)
    rotation = c1.slider("Rotation (deg)", -45.0, 45.0, 8.0, 1.0)
    crop = c1.slider("Crop / zoom (frame kept)", 0.5, 1.0, 0.85, 0.01)
    exposure = c2.slider("Exposure (EV)", -2.5, 1.5, -0.5, 0.1)
    blur = c2.slider("Blur radius (px)", 0.0, 4.0, 0.8, 0.1)
    jpeg = c3.slider("JPEG quality", 5, 100, 60)
    temp = c3.slider("Colour temperature (cool -1 ... warm +1)", -1.0, 1.0, 0.3, 0.05)
    c4, c5, c6 = st.columns(3)
    clutter = c4.checkbox("Background clutter", value=True)
    perspective = c5.slider("Perspective tilt", 0.0, 0.25, 0.05, 0.01)
    noise = c6.slider("Sensor noise", 0.0, 15.0, 3.0, 0.5)

params = DistortionParams(
    rotation_deg=rotation, crop_scale=crop, exposure_ev=exposure, blur_radius=blur, jpeg_quality=int(jpeg),
    color_temp=temp, perspective=perspective, noise_std=noise, background="clutter" if clutter else "plain", seed=7,
)
distorted = apply_distortion(clean, params)

clean_vec = engine.encode(clean, space)
query_vec = engine.encode(distorted, space)
results = engine.search(query_vec, space, k=k, nprobe=nprobe, ef_search=ef_search)
exact_ids = results["flat"].ids[None, :]

img_a, img_b, metrics_col = st.columns([1, 1, 1.2])
img_a.image(clean, caption="clean image", width="stretch")
img_b.image(distorted, caption="simulated phone photo (query)", width="stretch")
with metrics_col:
    stability = self_cosine(clean_vec, query_vec)
    st.metric("Self-Cosine Stability", f"{stability:.3f}", help="cos(embedding(clean), embedding(distorted)) of the same image; 1.0 = perfectly invariant")
    m1, m2 = st.columns(2)
    m1.metric(f"Exact vs IVF overlap@{k}", f"{overlap_at_k(exact_ids, results['ivf'].ids[None, :], k):.2f}")
    m2.metric(f"Exact vs HNSW overlap@{k}", f"{overlap_at_k(exact_ids, results['hnsw'].ids[None, :], k):.2f}")
    if true_label is not None:
        rank_txt = []
        for kind in INDEX_KINDS:
            labels = [h.label for h in results[kind].hits]
            rank = labels.index(true_label) + 1 if true_label in labels else None
            rank_txt.append(f"{INDEX_LABELS[kind]}: {'rank ' + str(rank) if rank else 'not in top-' + str(k)}")
        st.markdown("**True SKU position**  \n" + "  \n".join(rank_txt))

tab_res, tab_heat, tab_trade, tab_stab, tab_ml, tab_cat = st.tabs(
    ["Results", "Similarity heatmap", "Recall vs latency", "Stability sweep", "Metric learning", "Catalog"]
)

with tab_res:
    cols = st.columns(3)
    for col, kind in zip(cols, INDEX_KINDS):
        resp = results[kind]
        with col:
            knob = {"flat": "exhaustive", "ivf": f"nprobe={indexes['ivf'].get_search_param()}/{nlist}", "hnsw": f"efSearch={ef_search}"}[kind]
            st.subheader(INDEX_LABELS[kind])
            st.caption(f"{knob} | search latency {resp.latency_ms:.3f} ms")
            for hit in resp.hits:
                mark = "  [MATCH]" if true_label is not None and hit.label == true_label else ""
                st.image(engine.gallery_image(hit.gallery_id), caption=f"#{hit.rank} {hit.sku_id}  cos={hit.score:.3f}{mark}", width="stretch")

with tab_heat:
    hits = results["flat"].hits
    vecs = np.vstack([query_vec] + [engine.gallery(space)[h.gallery_id] for h in hits])
    names = ["query"] + [f"#{h.rank} {h.sku_id}" for h in hits]
    st.markdown(f"Pairwise cosine distance (1 - cos) between the query and its {len(hits)} exact results ({len(names)}x{len(names)}).")
    st.plotly_chart(fig_similarity_heatmap(pairwise_cosine_distance(vecs), names), key="heatmap")
    st.caption("Light row/column for the query = results are tight around it; a dark block among results = they are dissimilar to each other (mixed shapes).")

with tab_trade:
    bench = artifact_csv(art_dir, "benchmark.csv")
    if bench is not None:
        st.markdown("**Benchmark from `train.py`** (held-out test queries, catalog + synthetic distractors - see README).")
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_recall_latency(bench, "p50_ms", "recall_at_1", "Recall@1"), key="bench_r1")
        c2.plotly_chart(fig_recall_latency(bench, "p95_ms", "overlap_at_k", "overlap@10 with exact"), key="bench_ov")
        st.dataframe(bench, hide_index=True)
    else:
        st.info("No `artifacts/benchmark.csv` yet - run `python train.py`, or run a live mini-benchmark below.")
    st.markdown("**Live mini-benchmark on the current catalog** (synthetic distractor vectors make latency differences visible).")
    n_distr = st.slider("Synthetic distractors", 0, 100_000, DEMO_DISTRACTORS, 1000, key="n_distr")
    if st.button("Run live benchmark", key="run_bench"):
        st.session_state["live_bench"] = get_live_benchmark(engine, engine_key, space, int(n_distr), int(nprobe), int(ef_search))
    live = st.session_state.get("live_bench")
    if live is not None:
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_recall_latency(live, "p50_ms", "recall_at_1", "Recall@1"), key="live_r1")
        c2.plotly_chart(fig_recall_latency(live, "p95_ms", "overlap_at_k", "overlap@10 with exact"), key="live_ov")
        st.dataframe(live, hide_index=True)

with tab_stab:
    st.markdown("Self-cosine stability of **this** image when one camera factor at a time gets worse.")
    st.plotly_chart(fig_stability_sweep(get_stability(engine, engine_key, query_key, space, clean)), key="stability")

with tab_ml:
    cmp_df = artifact_csv(art_dir, "head_comparison.csv")
    hist = artifact_csv(art_dir, "training_history.csv")
    if art_dir is not None and cmp_df is not None and hist is not None:
        st.markdown("Projection head trained with contrastive loss + hard negatives on frozen features; evaluated on **held-out models**.")
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_head_comparison(cmp_df), key="cmp")
        c2.plotly_chart(fig_training_history(hist), key="hist")
        metrics_path = art_dir / "metrics.json"
        if metrics_path.is_file():
            with open(metrics_path, encoding="utf-8") as fh:
                st.json(json.load(fh), expanded=False)
    else:
        st.info("Metric-learning results appear after `python train.py` (the cold-start demo uses the raw backbone only).")

with tab_cat:
    n_show = min(24, len(engine.catalog))
    st.markdown(f"First {n_show} of {len(engine.catalog)} catalog images.")
    grid = st.columns(6)
    for i in range(n_show):
        grid[i % 6].image(engine.gallery_image(i), caption=engine.catalog.sku_ids[int(engine.catalog.labels[i])], width="stretch")
