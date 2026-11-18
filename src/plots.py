"""Static (matplotlib, saved to artifacts/) and interactive (Plotly, used by the app) charts.

Colour roles are fixed per entity so an index keeps its colour everywhere:
Exact = blue, IVF = orange, HNSW = aqua (first three slots of a CVD-validated
categorical palette). Baselines are muted grey, the highlighted series blue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from PIL import Image  # noqa: E402

KIND_COLORS: dict[str, str] = {"flat": "#2a78d6", "ivf": "#eb6834", "hnsw": "#1baf7a"}
KIND_NAMES: dict[str, str] = {"flat": "Exact (FlatIP)", "ivf": "IVF-Flat (nprobe)", "hnsw": "HNSW (efSearch)"}
BASELINE = "#898781"
HIGHLIGHT = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def _nice_log_xaxis(ax: plt.Axes, values: Sequence[float]) -> None:
    """Log x-axis with 1-2-5 ticks printed as plain numbers (no overlapping minor labels)."""
    ax.set_xscale("log")
    vals = [float(v) for v in values if v and v > 0]
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    nice = [m * 10.0**e for e in range(-3, 4) for m in (1, 2, 5)]
    ticks = [t for t in nice if lo / 1.6 <= t <= hi * 1.6]
    if len(ticks) >= 2:
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ matplotlib
def plot_recall_latency(df: pd.DataFrame, path: str | Path, k: int = 10, title: str = "") -> Path:
    """Two panels: SKU Recall@1 vs p50 latency, overlap@K (vs exact) vs p95 latency."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panels = [("p50_ms", "recall_at_1", "p50 latency per query (ms, log)", "Recall@1 (correct SKU at rank 1)"),
              ("p95_ms", "overlap_at_k", "p95 latency per query (ms, log)", f"overlap@{k} with exact search")]
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, panels):
        for kind in ("flat", "ivf", "hnsw"):
            sub = df[df["kind"] == kind].sort_values("param_value")
            if sub.empty:
                continue
            ax.plot(sub[xcol], sub[ycol], marker="o", markersize=6, linewidth=2, color=KIND_COLORS[kind], label=KIND_NAMES[kind])
            if kind != "flat":
                for _, r in sub.iloc[[0, -1]].iterrows():
                    offset = (4, 6) if kind == "hnsw" else (5, -12)  # HNSW labels above, IVF below
                    ax.annotate(f"{int(r['param_value'])}", (r[xcol], r[ycol]), textcoords="offset points", xytext=offset, fontsize=8, color=INK_2)
            default = sub[sub["is_default"]]
            ax.scatter(default[xcol], default[ycol], s=140, facecolors="none", edgecolors=KIND_COLORS[kind], linewidths=1.5)
        _nice_log_xaxis(ax, df[xcol].tolist())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _style_axes(ax)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(title or "Recall vs latency trade-off (rings = default setting)", color=INK, fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def plot_training_history(history: pd.DataFrame, path: str | Path, best_epoch: int | None = None) -> Path:
    """Loss per epoch and validation Recall@1 per epoch (epoch 0 = before training)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    h = history[history["epoch"] > 0]
    axes[0].plot(h["epoch"], h["loss"], color=HIGHLIGHT, linewidth=2)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training loss")
    axes[0].set_title("Projection head - training loss")
    if "val_recall@1" in history:
        axes[1].plot(history["epoch"], history["val_recall@1"], color=HIGHLIGHT, linewidth=2, marker="o", markersize=4, label="val Recall@1")
        base = history.loc[history["epoch"] == 0, "val_recall@1"]
        if len(base):
            axes[1].axhline(float(base.iloc[0]), color=BASELINE, linewidth=1, label="epoch 0 (raw backbone)")
        if best_epoch is not None:
            best = history.loc[history["epoch"] == best_epoch, "val_recall@1"]
            if len(best):
                axes[1].scatter([best_epoch], [float(best.iloc[0])], s=140, facecolors="none", edgecolors=INK, linewidths=1.5, label=f"best epoch {best_epoch}")
        axes[1].legend(frameon=False, fontsize=9, loc="center right")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Recall@1 (validation SKUs)")
    axes[1].set_title("Validation Recall@1 (early stopping)")
    for ax in axes:
        _style_axes(ax)
    fig.tight_layout()
    return _save(fig, path)


def plot_head_comparison(comparison: pd.DataFrame, path: str | Path) -> Path:
    """Grouped bars: raw backbone (grey) vs projection head (blue) on held-out SKUs."""
    metrics = [c for c in comparison.columns if c.startswith("recall@") or c.startswith("mrr@")]
    raw = comparison[comparison["embedding"] == "raw backbone"].iloc[0]
    head = comparison[comparison["embedding"] == "projection head"].iloc[0]
    x = np.arange(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    b1 = ax.bar(x - w / 2 - 0.01, [raw[m] for m in metrics], w, color=BASELINE, label="raw backbone")
    b2 = ax.bar(x + w / 2 + 0.01, [head[m] for m in metrics], w, color=HIGHLIGHT, label="+ projection head")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title(f"Metric learning on held-out SKUs ({raw['split']} split)")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    _style_axes(ax)
    fig.tight_layout()
    return _save(fig, path)


def save_image_grid(images: Sequence[Image.Image], captions: Sequence[str], path: str | Path, ncols: int = 6, tile: int = 160) -> Path:
    """Contact sheet of images with short captions (catalog vs query examples)."""
    n = len(images)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * tile / 80, nrows * tile / 72))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, im, cap in zip(np.atleast_1d(axes).ravel(), images, captions):
        ax.imshow(im)
        ax.set_title(cap, fontsize=7, color=INK_2)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------- plotly
def fig_recall_latency(df: pd.DataFrame, x: str = "p50_ms", y: str = "recall_at_1", y_title: str | None = None) -> go.Figure:
    """Interactive recall/latency curve: one line per index family, hover shows the knob."""
    fig = go.Figure()
    for kind in ("flat", "ivf", "hnsw"):
        sub = df[df["kind"] == kind].sort_values("param_value")
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub[x], y=sub[y], mode="lines+markers", name=KIND_NAMES[kind],
                line={"color": KIND_COLORS[kind], "width": 2}, marker={"size": 9},
                customdata=np.stack([sub["param_name"], sub["param_value"], sub["p95_ms"], sub["memory_mb"]], axis=1),
                hovertemplate=(
                    "%{customdata[0]}=%{customdata[1]}<br>" + f"{y}=%{{y:.3f}}<br>{x}=%{{x:.3f}} ms"
                    + "<br>p95=%{customdata[2]:.3f} ms<br>memory=%{customdata[3]:.1f} MB<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        xaxis_title=f"{x.replace('_ms', '')} latency per query (ms, log scale)", yaxis_title=y_title or y,
        xaxis_type="log", height=420, margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def fig_similarity_heatmap(dist: np.ndarray, labels: Sequence[str]) -> go.Figure:
    """(K+1)x(K+1) pairwise cosine-distance heatmap (query + its K results)."""
    scale = [[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)]
    fig = go.Figure(
        go.Heatmap(
            z=dist, x=list(labels), y=list(labels), colorscale=scale, zmin=0.0, zmax=float(max(dist.max(), 1e-6)),
            colorbar={"title": "1 - cos"}, hovertemplate="%{y} vs %{x}<br>cosine distance=%{z:.3f}<extra></extra>",
            texttemplate="%{z:.2f}", textfont={"size": 9},
        )
    )
    fig.update_layout(height=520, margin={"l": 10, "r": 10, "t": 30, "b": 10}, yaxis={"autorange": "reversed"})
    return fig


def fig_head_comparison(comparison: pd.DataFrame) -> go.Figure:
    metrics = [c for c in comparison.columns if c.startswith("recall@") or c.startswith("mrr@")]
    fig = go.Figure()
    for name, color in (("raw backbone", BASELINE), ("projection head", HIGHLIGHT)):
        row = comparison[comparison["embedding"] == name]
        if row.empty:
            continue
        vals = [float(row.iloc[0][m]) for m in metrics]
        fig.add_trace(go.Bar(x=metrics, y=vals, name=name, marker_color=color, text=[f"{v:.3f}" for v in vals], textposition="outside",
                             hovertemplate="%{x}: %{y:.4f}<extra>" + name + "</extra>"))
    fig.update_layout(barmode="group", yaxis_range=[0, 1.1], height=380, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      legend={"orientation": "h", "y": -0.15})
    return fig


def fig_training_history(history: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("training loss", "validation Recall@1"))
    h = history[history["epoch"] > 0]
    fig.add_trace(go.Scatter(x=h["epoch"], y=h["loss"], mode="lines", line={"color": HIGHLIGHT, "width": 2}, name="loss"), row=1, col=1)
    if "val_recall@1" in history:
        fig.add_trace(go.Scatter(x=history["epoch"], y=history["val_recall@1"], mode="lines+markers", line={"color": HIGHLIGHT, "width": 2},
                                 marker={"size": 6}, name="val Recall@1"), row=1, col=2)
    fig.update_layout(height=340, showlegend=False, margin={"l": 10, "r": 10, "t": 40, "b": 10})
    fig.update_xaxes(title_text="epoch")
    return fig


def fig_stability_sweep(df: pd.DataFrame) -> go.Figure:
    """Small multiples: self-cosine stability vs severity, one panel per distortion factor."""
    factors = list(dict.fromkeys(df["factor"]))
    cols = 3
    rows = int(np.ceil(len(factors) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=factors, shared_yaxes=True)
    for i, f in enumerate(factors):
        sub = df[df["factor"] == f]
        fig.add_trace(
            go.Scatter(x=sub["level"], y=sub["cosine"], mode="lines+markers", line={"color": HIGHLIGHT, "width": 2}, marker={"size": 8},
                       name=f, hovertemplate=f + "=%{x}<br>cos(clean, distorted)=%{y:.3f}<extra></extra>"),
            row=i // cols + 1, col=i % cols + 1,
        )
    fig.update_layout(height=260 * rows, showlegend=False, margin={"l": 10, "r": 10, "t": 40, "b": 10})
    fig.update_yaxes(range=[min(0.0, float(df["cosine"].min()) - 0.05), 1.02])
    return fig
