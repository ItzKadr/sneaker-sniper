r"""Metric learning: a light projection head on top of *frozen* backbone features.

Why a head and not full fine-tuning? The backbone already gives good generic
features; what it lacks is **invariance to our specific nuisance factors** (white
balance, exposure, clutter, JPEG ...) and **sensitivity to fine colour differences**
between sibling colourways. A small head trained on cached features learns exactly
that in seconds, needs no GPU, and keeps the expensive backbone untouched.

Training pairs: anchor = a simulated phone photo of SKU :math:`i` (distortion
pipeline), positive = the clean catalog image of the same SKU. All other catalog
images in the batch are negatives.

Losses
------
* **InfoNCE / NT-Xent** (CLIP-style, symmetric) with temperature :math:`\tau`:

  .. math:: \mathcal L = -\frac{1}{B}\sum_i \log
            \frac{\exp(\langle a_i, p_i\rangle/\tau)}{\sum_j \exp(\langle a_i, p_j\rangle/\tau)}

* **Triplet loss with batch-hard mining** (Hermans et al., 2017):

  .. math:: \mathcal L = \frac{1}{|A|}\sum_a \big[\max_{p: y_p=y_a} d(a,p)
            - \min_{n: y_n \ne y_a} d(a,n) + m\big]_+,\quad d(x,y)=\lVert x-y\rVert_2

Hard negatives
--------------
Besides in-batch mining, batches are *composed* to contain hard negatives: every
epoch we find, for each training SKU, its nearest other SKUs in the current head
space (exact FAISS search) and put a seed SKU together with its neighbours into the
same batch (``hard_negative_ratio`` of the batch).
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .evaluator import RetrievalMetrics, evaluate_retrieval
from .indexer import build_flat_index
from .utils import l2_normalize, resolve_device

logger = logging.getLogger(__name__)

HEAD_TYPES: tuple[str, ...] = ("residual_mlp", "linear", "mlp")
LOSSES: tuple[str, ...] = ("infonce", "triplet")


# ---------------------------------------------------------------------- head
@dataclass
class HeadConfig:
    head_type: str = "residual_mlp"
    hidden_dim: int = 512
    out_dim: int | None = None  # None -> same as input dim (mandatory for residual_mlp)
    dropout: float = 0.1


class ProjectionHead(nn.Module):
    r"""Projection head with L2-normalised output.

    * ``residual_mlp``: :math:`h(x) = \mathrm{norm}(x + W_2\,\mathrm{GELU}(W_1\,\mathrm{LN}(x)))`
      with :math:`W_2 = 0` at init, so **epoch 0 == raw backbone** (training can only
      move away from a known-good starting point).
    * ``linear``: :math:`h(x) = \mathrm{norm}(W x)`; identity init when square, otherwise
      PCA init (see :func:`init_linear_with_pca`) - useful for dimension reduction.
    * ``mlp``: Linear-GELU-Dropout-Linear, random init.
    """

    def __init__(self, in_dim: int, config: HeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeadConfig()
        cfg = self.config
        if cfg.head_type not in HEAD_TYPES:
            raise ValueError(f"unknown head_type '{cfg.head_type}', expected {HEAD_TYPES}")
        self.in_dim = in_dim
        self.out_dim = cfg.out_dim or in_dim
        if cfg.head_type == "residual_mlp":
            if self.out_dim != in_dim:
                raise ValueError("residual_mlp requires out_dim == in_dim")
            self.norm = nn.LayerNorm(in_dim)
            self.fc1 = nn.Linear(in_dim, cfg.hidden_dim)
            self.fc2 = nn.Linear(cfg.hidden_dim, in_dim)
            self.drop = nn.Dropout(cfg.dropout)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        elif cfg.head_type == "linear":
            self.proj = nn.Linear(in_dim, self.out_dim, bias=False)
            if self.out_dim == in_dim:
                with torch.no_grad():
                    self.proj.weight.copy_(torch.eye(in_dim))
            else:
                nn.init.orthogonal_(self.proj.weight)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, self.out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.config.head_type
        if t == "residual_mlp":
            y = x + self.fc2(self.drop(F.gelu(self.fc1(self.norm(x)))))
        elif t == "linear":
            y = self.proj(x)
        else:
            y = self.mlp(x)
        return F.normalize(y, p=2, dim=-1)


def init_linear_with_pca(head: ProjectionHead, features: np.ndarray) -> None:
    """Initialise a (non-square) linear head with the top principal directions."""
    if head.config.head_type != "linear":
        raise ValueError("PCA init only applies to the linear head")
    x = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    k = min(head.out_dim, vt.shape[0])
    with torch.no_grad():
        w = head.proj.weight
        w.zero_()
        w[:k] = torch.from_numpy(vt[:k].astype(np.float32)).to(w.device)


# --------------------------------------------------------------------- losses
def info_nce_loss(
    anchors: torch.Tensor, positives: torch.Tensor, temperature: float = 0.07, symmetric: bool = True
) -> torch.Tensor:
    """InfoNCE / NT-Xent over a batch of aligned (anchor_i, positive_i) unit vectors."""
    logits = anchors @ positives.T / temperature
    targets = torch.arange(anchors.shape[0], device=anchors.device)
    loss = F.cross_entropy(logits, targets)
    if symmetric:
        loss = 0.5 * (loss + F.cross_entropy(logits.T, targets))
    return loss


def pairwise_euclidean(emb: torch.Tensor) -> torch.Tensor:
    r"""For unit vectors :math:`\lVert a-b\rVert_2 = \sqrt{2 - 2\langle a,b\rangle}`."""
    sim = emb @ emb.T
    return torch.sqrt(torch.clamp(2.0 - 2.0 * sim, min=1e-12))


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """Batch-hard triplet loss (hardest positive, hardest negative per anchor)."""
    dist = pairwise_euclidean(embeddings)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pos_mask = same & ~eye
    neg_mask = ~same
    hardest_pos = (dist * pos_mask).max(dim=1).values
    hardest_neg = dist.masked_fill(~neg_mask, float("inf")).min(dim=1).values
    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not bool(valid.any()):
        return embeddings.sum() * 0.0
    return F.relu(hardest_pos - hardest_neg + margin)[valid].mean()


# -------------------------------------------------------------- hard negatives
def mine_hard_negatives(embeddings: np.ndarray, n_neighbors: int) -> np.ndarray:
    """``(S, n_neighbors)`` indices of the most cosine-similar *other* items."""
    s = len(embeddings)
    n = int(max(0, min(n_neighbors, s - 1)))
    if n == 0:
        return np.zeros((s, 0), dtype=np.int64)
    _, ids = build_flat_index(embeddings).search(embeddings, n + 1)
    out = np.empty((s, n), dtype=np.int64)
    for i in range(s):
        row = ids[i][(ids[i] != i) & (ids[i] >= 0)]
        out[i] = row[:n] if len(row) >= n else np.pad(row, (0, n - len(row)), mode="edge")
    return out


class HardNegativeBatchSampler:
    """Compose batches of *unique* items, a fraction of which are mutual hard negatives."""

    def __init__(self, n_items: int, batch_size: int, hard_ratio: float = 0.5) -> None:
        self.n_items = n_items
        self.batch_size = min(batch_size, n_items)
        self.hard_ratio = float(np.clip(hard_ratio, 0.0, 1.0))
        self.neighbors: np.ndarray | None = None

    def update_neighbors(self, neighbors: np.ndarray) -> None:
        self.neighbors = neighbors

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        chosen: list[int] = []
        used: set[int] = set()
        target_hard = int(round(self.batch_size * self.hard_ratio)) if self.neighbors is not None else 0
        if target_hard > 0 and self.neighbors is not None:
            for seed in rng.permutation(self.n_items):
                if len(chosen) >= target_hard:
                    break
                if int(seed) in used:
                    continue
                group = [int(seed)] + [int(j) for j in self.neighbors[seed] if int(j) not in used]
                group = list(dict.fromkeys(group))[: target_hard - len(chosen)]
                chosen.extend(group)
                used.update(group)
        for j in rng.permutation(self.n_items):
            if len(chosen) >= self.batch_size:
                break
            if int(j) not in used:
                chosen.append(int(j))
                used.add(int(j))
        return np.asarray(chosen, dtype=np.int64)


# ------------------------------------------------------------------- training
@dataclass
class MetricTrainConfig:
    loss: str = "infonce"
    temperature: float = 0.07
    margin: float = 0.2
    epochs: int = 40
    batch_size: int = 64
    steps_per_epoch: int | None = None  # None -> 2 * ceil(n_train_skus / batch_size)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hard_negative_ratio: float = 0.5
    n_hard_neighbors: int = 3
    patience: int = 15
    eval_ks: tuple[int, ...] = (1, 5, 10)
    device: str = "cpu"  # tiny MLP on cached features: CPU is as fast, deterministic, and spares the GPU
    seed: int = 0


@dataclass
class ValidationSet:
    """Raw backbone features used for early stopping (gallery = full catalog)."""

    gallery: np.ndarray
    gallery_labels: np.ndarray
    queries: np.ndarray
    query_labels: np.ndarray


@dataclass
class TrainResult:
    head: ProjectionHead
    history: pd.DataFrame
    best_epoch: int
    best_val_recall1: float
    config: MetricTrainConfig = field(default_factory=MetricTrainConfig)


@torch.no_grad()
def apply_head(head: ProjectionHead, x: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Project raw features through the head -> C-contiguous float32 unit vectors."""
    head.eval()
    device = next(head.parameters()).device
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(x[i : i + batch_size], dtype=np.float32)).to(device)
        out.append(head(xb).cpu().numpy())
    if not out:
        return np.empty((0, head.out_dim), dtype=np.float32)
    return l2_normalize(np.concatenate(out, axis=0))


def _validate(head: ProjectionHead, val: ValidationSet, ks: Sequence[int]) -> RetrievalMetrics:
    return evaluate_retrieval(apply_head(head, val.gallery), val.gallery_labels, apply_head(head, val.queries), val.query_labels, ks)


def train_projection_head(
    ref_features: np.ndarray,
    view_features: np.ndarray,
    val: ValidationSet | None = None,
    head_config: HeadConfig | None = None,
    config: MetricTrainConfig | None = None,
) -> TrainResult:
    """Train a projection head on cached features.

    Args:
        ref_features: ``(S, d)`` raw features of the clean catalog image of each training SKU.
        view_features: ``(S, V, d)`` raw features of ``V`` simulated phone photos per SKU.
        val: optional validation set; enables early stopping on val Recall@1.
    """
    head_config = head_config or HeadConfig()
    cfg = config or MetricTrainConfig()
    if cfg.loss not in LOSSES:
        raise ValueError(f"unknown loss '{cfg.loss}', expected {LOSSES}")
    if view_features.ndim != 3 or view_features.shape[0] != ref_features.shape[0]:
        raise ValueError("view_features must have shape (S, V, d) matching ref_features (S, d)")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = resolve_device(cfg.device)
    n_skus, n_views, dim = view_features.shape

    head = ProjectionHead(dim, head_config).to(device)
    if head_config.head_type == "linear" and head.out_dim != dim:
        init_linear_with_pca(head, ref_features)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    refs_t = torch.from_numpy(np.ascontiguousarray(ref_features, dtype=np.float32)).to(device)
    views_t = torch.from_numpy(np.ascontiguousarray(view_features, dtype=np.float32)).to(device)
    sampler = HardNegativeBatchSampler(n_skus, cfg.batch_size, cfg.hard_negative_ratio)
    steps = cfg.steps_per_epoch or 2 * math.ceil(n_skus / sampler.batch_size)

    history: list[dict[str, Any]] = []
    best_state = copy.deepcopy(head.state_dict())
    best_epoch, best_key = 0, (-1.0, -1.0)
    if val is not None:
        m0 = _validate(head, val, cfg.eval_ks)
        history.append({"epoch": 0, "loss": float("nan"), **m0.as_dict("val_")})
        best_key = (m0.recall[min(cfg.eval_ks)], m0.mrr)
        logger.info("epoch   0 | val R@1=%.4f (init)", best_key[0])

    for epoch in range(1, cfg.epochs + 1):
        if cfg.hard_negative_ratio > 0 and cfg.n_hard_neighbors > 0:
            sampler.update_neighbors(mine_hard_negatives(apply_head(head, ref_features), cfg.n_hard_neighbors))
        head.train()
        losses = []
        for _ in range(steps):
            idx = sampler.sample(rng)
            vidx = rng.integers(0, n_views, size=len(idx))
            idx_t = torch.from_numpy(idx).to(device)
            anchors = head(views_t[idx_t, torch.from_numpy(vidx).to(device)])
            positives = head(refs_t[idx_t])
            if cfg.loss == "infonce":
                loss = info_nce_loss(anchors, positives, cfg.temperature)
            else:
                loss = batch_hard_triplet_loss(torch.cat([positives, anchors]), torch.cat([idx_t, idx_t]), cfg.margin)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        row: dict[str, Any] = {"epoch": epoch, "loss": float(np.mean(losses))}
        if val is not None:
            m = _validate(head, val, cfg.eval_ks)
            row.update(m.as_dict("val_"))
            key = (m.recall[min(cfg.eval_ks)], m.mrr)
            if key > best_key:
                best_key, best_epoch = key, epoch
                best_state = copy.deepcopy(head.state_dict())
            logger.info("epoch %3d | loss=%.4f | val R@1=%.4f MRR=%.4f", epoch, row["loss"], key[0], key[1])
        else:
            best_state, best_epoch = copy.deepcopy(head.state_dict()), epoch
            logger.info("epoch %3d | loss=%.4f", epoch, row["loss"])
        history.append(row)
        if val is not None and epoch - best_epoch >= cfg.patience:
            logger.info("early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
            break

    head.load_state_dict(best_state)
    head.eval()
    return TrainResult(head, pd.DataFrame(history), best_epoch, float(best_key[0]), cfg)


# ----------------------------------------------------------------- comparison
@dataclass
class HeadComparison:
    raw: RetrievalMetrics
    head: RetrievalMetrics
    split: str

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for name, m in (("raw backbone", self.raw), ("projection head", self.head)):
            rows.append({"embedding": name, "split": self.split, **m.as_dict()})
        return pd.DataFrame(rows)

    def delta(self) -> dict[str, float]:
        r, h = self.raw.as_dict(), self.head.as_dict()
        return {k: h[k] - r[k] for k in r}


def compare_raw_vs_head(
    gallery_raw: np.ndarray,
    gallery_labels: np.ndarray,
    queries_raw: np.ndarray,
    query_labels: np.ndarray,
    head: ProjectionHead,
    ks: Sequence[int] = (1, 5, 10),
    split: str = "test",
) -> HeadComparison:
    """Recall@K / MRR of exact search with raw features vs head-projected features."""
    raw = evaluate_retrieval(gallery_raw, gallery_labels, queries_raw, query_labels, ks)
    projected = evaluate_retrieval(apply_head(head, gallery_raw), gallery_labels, apply_head(head, queries_raw), query_labels, ks)
    return HeadComparison(raw, projected, split)


# ------------------------------------------------------------------ persist
def save_head(head: ProjectionHead, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": {k: v.cpu() for k, v in head.state_dict().items()}, "in_dim": head.in_dim, "config": asdict(head.config)},
        path,
    )
    return path


def load_head(path: str | Path, device: str = "cpu") -> ProjectionHead:
    blob = torch.load(Path(path), map_location=device, weights_only=True)
    head = ProjectionHead(int(blob["in_dim"]), HeadConfig(**blob["config"]))
    head.load_state_dict(blob["state_dict"])
    return head.to(device).eval()
