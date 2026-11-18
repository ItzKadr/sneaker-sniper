r"""timm feature extractor producing L2-normalised, FAISS-ready embeddings.

* Backbone: any timm model created with ``num_classes=0`` (pooled features).
  Default: **DINOv2 ViT-S/14** (``vit_small_patch14_dinov2.lvd142m``, self-supervised,
  384-D CLS token). Alternative: ``resnet50`` (supervised ImageNet, 2048-D GAP).
* Preprocessing comes from the model's own timm data config (mean/std,
  interpolation, crop_pct). Only the input resolution can be overridden: DINOv2's
  default config asks for 518x518 (37x37 patches) which is ~5x slower than 224;
  for ViTs we re-create the model with ``img_size`` snapped to a multiple of the
  patch size (14), and timm interpolates the position embeddings.
* Output: :math:`\hat z = z / \lVert z\rVert_2`, ``np.ascontiguousarray(float32)``
  - so FAISS inner product == cosine similarity.
* If pretrained weights cannot be downloaded (offline, HF hub error) the embedder
  falls back to random weights with a loud warning (``Embedder.warning``).
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import timm
import torch
import torch.nn.functional as F

from .dataset import ImageLike, load_image
from .utils import l2_normalize, resolve_device

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"
ALTERNATIVE_MODEL = "resnet50"


@dataclass
class EmbedderConfig:
    model_name: str = DEFAULT_MODEL
    pretrained: bool = True
    image_size: int | None = 224  # None -> use the data config's input size as-is
    batch_size: int = 64
    device: str = "auto"
    use_amp: bool = True  # fp16 autocast on CUDA (features re-cast to fp32 before normalising)
    allow_fallback: bool = True  # random weights instead of crashing when download fails
    num_threads: int = 4


def patch_size_from_name(model_name: str) -> int | None:
    """Parse the ViT patch size from a timm model name (``..._patch14_...`` -> 14)."""
    m = re.search(r"patch(\d+)", model_name)
    return int(m.group(1)) if m else None


def snap_to_multiple(size: int, multiple: int) -> int:
    """Round ``size`` to the nearest positive multiple of ``multiple`` (e.g. 224 for /14 -> 224, 230 -> 224)."""
    return max(multiple, int(round(size / multiple)) * multiple)


def _create_model(model_name: str, pretrained: bool, image_size: int | None) -> torch.nn.Module:
    kwargs: dict[str, Any] = {"num_classes": 0}
    if image_size is not None and patch_size_from_name(model_name) is not None:
        kwargs["img_size"] = image_size
    try:
        return timm.create_model(model_name, pretrained=pretrained, **kwargs)
    except TypeError as exc:  # CNNs do not accept img_size
        if "img_size" not in str(exc):
            raise
        kwargs.pop("img_size")
        return timm.create_model(model_name, pretrained=pretrained, **kwargs)


def create_backbone(
    model_name: str, pretrained: bool, image_size: int | None, allow_fallback: bool = True
) -> tuple[torch.nn.Module, bool, str | None]:
    """Create a timm backbone; returns ``(model, pretrained_loaded, warning_message)``."""
    if not pretrained:
        return _create_model(model_name, False, image_size), False, None
    try:
        return _create_model(model_name, True, image_size), True, None
    except Exception as exc:  # network / HF hub / corrupted cache: never crash the pipeline
        if not allow_fallback:
            raise
        msg = (
            f"Could not load pretrained weights for '{model_name}' "
            f"({type(exc).__name__}: {str(exc)[:200]}). Falling back to RANDOM weights - "
            "embeddings will be much less meaningful. Check the internet connection "
            "(weights are downloaded from the Hugging Face Hub on first use) and re-run."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        logger.warning(msg)
        return _create_model(model_name, False, image_size), False, msg


class Embedder:
    """Batched, normalised image -> vector encoder around a frozen timm backbone."""

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self.config = config or EmbedderConfig()
        cfg = self.config
        torch.set_num_threads(cfg.num_threads)
        self.device = resolve_device(cfg.device)
        image_size = cfg.image_size
        patch = patch_size_from_name(cfg.model_name)
        if image_size is not None and patch is not None:
            snapped = snap_to_multiple(image_size, patch)
            if snapped != image_size:
                logger.info("image_size %d is not a multiple of patch %d -> using %d", image_size, patch, snapped)
            image_size = snapped
        self.image_size = image_size
        model, self.pretrained_loaded, self.warning = create_backbone(
            cfg.model_name, cfg.pretrained, image_size, cfg.allow_fallback
        )
        self.model = model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.data_config: dict[str, Any] = timm.data.resolve_model_data_config(self.model)
        if image_size is not None:
            # timm keeps the *pretrained* input size in the config; align it with the model.
            self.data_config["input_size"] = (3, image_size, image_size)
        self.transform = timm.data.create_transform(**self.data_config, is_training=False)
        self.dim: int = int(self.model.num_features)
        self._batch_size = cfg.batch_size
        logger.info(
            "Embedder: %s (pretrained=%s) dim=%d input=%s device=%s",
            cfg.model_name, self.pretrained_loaded, self.dim, self.data_config["input_size"], self.device,
        )

    # ------------------------------------------------------------------ helpers
    def describe(self) -> dict[str, Any]:
        """Metadata stored next to embeddings (used to detect incompatible artifacts)."""
        return {
            "model_name": self.config.model_name,
            "pretrained_requested": self.config.pretrained,
            "pretrained_loaded": self.pretrained_loaded,
            "image_size": self.image_size,
            "input_size": list(self.data_config["input_size"]),
            "embedding_dim": self.dim,
            "data_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.data_config.items()},
            "config": asdict(self.config),
        }

    def preprocess(self, images: Sequence[ImageLike]) -> torch.Tensor:
        """PIL / paths -> normalised NCHW tensor using the model's data config."""
        return torch.stack([self.transform(load_image(im)) for im in images])

    @torch.inference_mode()
    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device, non_blocking=True)
        if self.device.startswith("cuda") and self.config.use_amp:
            with torch.autocast("cuda", dtype=torch.float16):
                feats = self.model(batch)
        else:
            feats = self.model(batch)
        return F.normalize(feats.float(), p=2, dim=1).cpu()

    # --------------------------------------------------------------------- API
    def embed(self, images: Sequence[ImageLike], batch_size: int | None = None, log_every: int = 0) -> np.ndarray:
        """Embed images -> ``(n, dim)`` C-contiguous float32 array with unit L2 rows.

        On CUDA OOM the batch size is halved and the batch retried.
        """
        n = len(images)
        out = np.empty((n, self.dim), dtype=np.float32)
        if n == 0:
            return out
        bs = max(1, batch_size or self._batch_size)
        start = 0
        n_batches = 0
        while start < n:
            stop = min(start + bs, n)
            try:
                feats = self._forward(self.preprocess(images[start:stop]))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                self._batch_size = bs
                logger.warning("CUDA OOM - retrying with batch_size=%d", bs)
                continue
            out[start:stop] = feats.numpy()
            n_batches += 1
            if log_every and n_batches % log_every == 0:
                logger.info("  embedded %d / %d", stop, n)
            start = stop
        # Hard re-normalisation in float32 (fp16 autocast + numpy copy safety).
        return l2_normalize(out)

    def embed_one(self, image: ImageLike) -> np.ndarray:
        """Embed a single image -> ``(dim,)`` unit vector."""
        return self.embed([image])[0]
