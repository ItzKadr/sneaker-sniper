r"""Reusable, parameterised "smartphone camera" distortion pipeline.

The same code produces (a) the synthetic *query photos* of the benchmark, (b) the
training views of the metric-learning head and (c) the live simulator in the app.

Pipeline (in the order a phone camera would roughly apply it)::

    product shot --(foreground mask)--> shoe layer (RGBA)
       -> perspective tilt -> in-plane rotation          (geometry, on the shoe layer)
       -> composite on a plain / cluttered background    (scene)
       -> random crop + zoom back to full resolution     (framing)
       -> white balance (colour temperature) -> exposure (in linear light)
       -> contrast -> saturation -> vignetting           (ISP / tone mapping)
       -> Gaussian blur -> sensor noise -> JPEG          (optics, sensor, codec)

Exposure is applied in (approximately) linear light,
:math:`x_{lin} = x^{2.2}`, :math:`x'_{lin} = 2^{EV} x_{lin}`, then re-encoded with
:math:`x' = x_{lin}'^{1/2.2}` - one EV doubles the photon count, not the pixel value.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

GAMMA = 2.2


@dataclass(frozen=True)
class DistortionParams:
    """All knobs of the phone-camera simulator. Defaults == identity transform."""

    rotation_deg: float = 0.0
    crop_scale: float = 1.0  # side fraction kept by the crop (1.0 = full frame, 0.6 = 1.67x zoom)
    crop_dx: float = 0.0  # crop centre offset in [-1, 1] of the available slack
    crop_dy: float = 0.0
    exposure_ev: float = 0.0  # exposure compensation in stops
    contrast: float = 1.0
    saturation: float = 1.0
    color_temp: float = 0.0  # -1 = cool / bluish ... +1 = warm / orange
    blur_radius: float = 0.0  # Gaussian blur radius in pixels (defocus / hand shake)
    noise_std: float = 0.0  # additive Gaussian sensor noise, 0-255 scale
    jpeg_quality: int = 100  # < 100 enables a JPEG re-encode at that quality
    perspective: float = 0.0  # max corner displacement as a fraction of the side
    background: str = "plain"  # "plain" | "clutter"
    vignette: float = 0.0  # 0..1 strength of radial darkening
    seed: int = 0  # drives clutter, perspective corners and noise

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_identity(self) -> bool:
        return self == DistortionParams(seed=self.seed)


IDENTITY = DistortionParams()


def sample_distortion_params(rng: np.random.Generator, strength: float = 1.0) -> DistortionParams:
    """Draw a random but realistic phone-photo distortion; ``strength`` in [0, 1.5] scales all ranges."""
    s = float(np.clip(strength, 0.0, 1.5))
    return DistortionParams(
        rotation_deg=float(np.clip(rng.normal(0.0, 8.0 * s), -25.0 * s, 25.0 * s)) if s > 0 else 0.0,
        crop_scale=float(rng.uniform(1.0 - 0.35 * s, 1.0)),
        crop_dx=float(rng.uniform(-1.0, 1.0)),
        crop_dy=float(rng.uniform(-1.0, 1.0)),
        exposure_ev=float(rng.uniform(-1.0 * s, 0.8 * s)),
        contrast=float(rng.uniform(1.0 - 0.3 * s, 1.0 + 0.25 * s)),
        saturation=float(rng.uniform(1.0 - 0.35 * s, 1.0 + 0.25 * s)),
        color_temp=float(rng.uniform(-0.8 * s, 0.8 * s)),
        blur_radius=float(rng.uniform(0.0, 1.8 * s)) if rng.random() < 0.6 else 0.0,
        noise_std=float(rng.uniform(0.0, 6.0 * s)),
        jpeg_quality=int(rng.integers(int(np.clip(85 - 50 * s, 10, 95)), 96)),
        perspective=float(rng.uniform(0.0, 0.12 * s)),
        background="clutter" if rng.random() < min(0.85, 0.7 * s) else "plain",
        vignette=float(rng.uniform(0.0, 0.35 * s)),
        seed=int(rng.integers(0, 2**31 - 1)),
    )


# ------------------------------------------------------------------- foreground
def estimate_foreground_mask(
    image: Image.Image,
    tol: float = 14.0,
    max_border_mad: float = 10.0,
) -> tuple[Image.Image, tuple[int, int, int], bool]:
    """Segment a product shot from its (near-uniform) studio background.

    The background colour is the median of the border pixels. Pixels whose RGB
    distance to it is below ``tol`` are background candidates; ``binary_fill_holes``
    re-includes white shoe parts enclosed by an outline. If the border is *not*
    uniform (median absolute deviation > ``max_border_mad``) the photo is treated as
    a real photo and the whole frame is foreground.

    Returns ``(mask L-image, background RGB, is_studio_shot)``.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0)
    bg = np.median(border, axis=0)
    mad = float(np.median(np.abs(border - bg)))
    bg_rgb = tuple(int(v) for v in bg)
    h, w = arr.shape[:2]
    if mad > max_border_mad:
        return Image.new("L", (w, h), 255), bg_rgb, False  # type: ignore[return-value]
    dist = np.sqrt(((arr - bg) ** 2).sum(axis=-1))
    fg = dist > tol
    fg = ndimage.binary_fill_holes(ndimage.binary_closing(fg, iterations=1))
    mask = Image.fromarray((fg * 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(0.8))
    return mask, bg_rgb, True  # type: ignore[return-value]


# ------------------------------------------------------------------ backgrounds
def make_clutter_background(size: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    """Procedural "real world" backdrop: floor/table texture + a few blurry objects."""
    w, h = size
    kind = int(rng.integers(3))
    base = np.array(
        [rng.uniform(90, 220), rng.uniform(80, 200), rng.uniform(60, 190)], dtype=np.float32
    )
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = 1.0 + rng.uniform(-0.25, 0.25) * (yy / h - 0.5) + rng.uniform(-0.2, 0.2) * (xx / w - 0.5)
    img = base[None, None, :] * grad[..., None]
    if kind == 0:  # wooden planks
        plank = max(8, int(h / rng.uniform(4, 9)))
        offsets = rng.uniform(-25, 25, size=h // plank + 2)
        img += offsets[(yy // plank).astype(int)][..., None]
        img += 8.0 * np.sin(xx / rng.uniform(3, 9) + rng.uniform(0, 6))[..., None] * rng.uniform(0.3, 1)
    elif kind == 1:  # tiles
        tile = max(10, int(w / rng.uniform(3, 7)))
        grout = ((xx % tile) < 2) | ((yy % tile) < 2)
        img[grout] *= 0.65
    else:  # carpet / concrete
        img += rng.normal(0, 14, size=(h, w, 1)).astype(np.float32)
    bg = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(bg)
    for _ in range(int(rng.integers(3, 9))):
        col = tuple(int(v) for v in rng.uniform(20, 240, size=3))
        x0, y0 = rng.uniform(-0.2 * w, w), rng.uniform(-0.2 * h, h)
        bw, bh = rng.uniform(0.1, 0.45) * w, rng.uniform(0.1, 0.45) * h
        if rng.random() < 0.5:
            draw.ellipse([x0, y0, x0 + bw, y0 + bh], fill=col)
        else:
            draw.rectangle([x0, y0, x0 + bw, y0 + bh], fill=col)
    return bg.filter(ImageFilter.GaussianBlur(float(rng.uniform(1.0, 3.0))))


# --------------------------------------------------------------------- geometry
def perspective_coefficients(dst: np.ndarray, src: np.ndarray) -> list[float]:
    r"""Solve the 8 homography coefficients used by ``Image.transform(PERSPECTIVE)``.

    PIL maps each *output* pixel :math:`(x, y)` to the *input* location
    :math:`\big(\tfrac{a x + b y + c}{g x + h y + 1}, \tfrac{d x + e y + f}{g x + h y + 1}\big)`,
    so we solve the 8x8 linear system given 4 correspondences ``dst -> src``.
    """
    rows = []
    rhs = []
    for (x, y), (u, v) in zip(dst, src):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.extend([u, v])
    coeffs = np.linalg.solve(np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64))
    return [float(c) for c in coeffs]


def geometry_matrix(
    size: tuple[int, int], rotation_deg: float, perspective: float, rng: np.random.Generator
) -> np.ndarray:
    r"""3x3 homography mapping *output* pixels to *input* pixels.

    Composition of a perspective tilt :math:`H_p` (image corners pulled inwards by a
    random fraction of up to ``perspective``) followed by an in-plane rotation about
    the centre :math:`A_r` (counter-clockwise for positive angles, like ``Image.rotate``):
    :math:`M = H_p A_r`. Fusing both into one matrix means a single resampling pass.
    """
    w, h = size
    hp = np.eye(3)
    if perspective > 0.0:
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        jitter = rng.uniform(0.0, perspective, size=(4, 2)) * np.array([w, h])
        inward = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float64)
        hp = np.append(perspective_coefficients(src + inward * jitter, src), 1.0).reshape(3, 3)
    a = -np.deg2rad(rotation_deg)
    cx, cy = w / 2.0, h / 2.0
    ar = np.array([[np.cos(a), np.sin(a), 0.0], [-np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    ar[0, 2] = cx - (ar[0, 0] * cx + ar[0, 1] * cy)
    ar[1, 2] = cy - (ar[1, 0] * cx + ar[1, 1] * cy)
    m = hp @ ar
    return m / m[2, 2]


def _apply_geometry(layer: Image.Image, matrix: np.ndarray) -> Image.Image:
    coeffs = [float(v) for v in matrix.flatten()[:8]]
    return layer.transform(layer.size, Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BILINEAR)


def _crop_box(w: int, h: int, scale: float, dx: float, dy: float) -> tuple[int, int, int, int]:
    scale = float(np.clip(scale, 0.2, 1.0))
    cw, ch = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    slack_x, slack_y = (w - cw) / 2.0, (h - ch) / 2.0
    cx = w / 2.0 + float(np.clip(dx, -1, 1)) * slack_x
    cy = h / 2.0 + float(np.clip(dy, -1, 1)) * slack_y
    left = int(round(cx - cw / 2.0))
    top = int(round(cy - ch / 2.0))
    return left, top, left + cw, top + ch


# ------------------------------------------------------------------- photometry
def white_balance_gains(color_temp: float) -> np.ndarray:
    """Per-channel RGB gains of a colour-temperature shift (luma-preserving).

    ``+1`` = warm (tungsten-like: red up, blue down); ``-1`` = cool (shade / LED).
    Gains are rescaled so the Rec.601 luma of a grey pixel is unchanged.
    """
    t = float(np.clip(color_temp, -1.5, 1.5))
    gains = np.array([1.0 + 0.28 * t, 1.0 + 0.04 * t, 1.0 - 0.32 * t], dtype=np.float32)
    luma = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return gains / float(gains @ luma)


_DECODE_LUT = ((np.arange(256, dtype=np.float64) / 255.0) ** GAMMA).astype(np.float32)
_ENCODE_LEVELS = 4096
_ENCODE_LUT = ((np.arange(_ENCODE_LEVELS, dtype=np.float64) / (_ENCODE_LEVELS - 1)) ** (1.0 / GAMMA)).astype(np.float32)


def _photometric(arr: np.ndarray, p: DistortionParams) -> np.ndarray:
    """Colour pipeline on a uint8 HxWx3 array; returns float32 in [0, 1].

    Gamma decode / encode use look-up tables (256 and 4096 entries) instead of
    ``np.power`` - same result to < 0.1 grey level, ~5x faster.
    """
    lin = _DECODE_LUT[arr]
    if p.color_temp != 0.0:
        lin = lin * white_balance_gains(p.color_temp)[None, None, :]
    if p.exposure_ev != 0.0:
        lin = lin * np.float32(2.0 ** p.exposure_ev)
    idx = (np.clip(lin, 0.0, 1.0) * (_ENCODE_LEVELS - 1) + 0.5).astype(np.int32)
    out = _ENCODE_LUT[idx]
    luma = out @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if p.contrast != 1.0:
        mean = float(luma.mean())
        out = (out - mean) * p.contrast + mean
        luma = out @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if p.saturation != 1.0:
        out = luma[..., None] + p.saturation * (out - luma[..., None])
    if p.vignette > 0.0:
        h, w = out.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        out = out * (1.0 - p.vignette * 0.5 * r2)[..., None]
    return np.clip(out, 0.0, 1.0)


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    """Encode + decode with JPEG at ``quality`` (1-95 meaningful)."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(np.clip(quality, 1, 95)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ----------------------------------------------------------------------- driver
def apply_distortion(
    image: Image.Image,
    params: DistortionParams,
    output_size: int | tuple[int, int] | None = None,
) -> Image.Image:
    """Turn a clean catalog image into a simulated smartphone photo.

    Deterministic: the same ``(image, params)`` always gives the same output, because
    every random choice (clutter, perspective corners, noise) is driven by ``params.seed``.
    """
    img = image.convert("RGB")
    w, h = img.size
    rng = np.random.default_rng(params.seed)

    needs_geometry = params.rotation_deg != 0.0 or params.perspective > 0.0
    if needs_geometry or params.background == "clutter":
        mask, bg_rgb, _ = estimate_foreground_mask(img)
        layer = img.convert("RGBA")
        layer.putalpha(mask)
        if needs_geometry:
            layer = _apply_geometry(layer, geometry_matrix((w, h), params.rotation_deg, params.perspective, rng))
        if params.background == "clutter":
            canvas = make_clutter_background((w, h), rng).convert("RGBA")
        else:
            canvas = Image.new("RGBA", (w, h), bg_rgb + (255,))
        canvas.alpha_composite(layer)
        img = canvas.convert("RGB")

    if params.crop_scale < 1.0:
        img = img.crop(_crop_box(w, h, params.crop_scale, params.crop_dx, params.crop_dy)).resize(
            (w, h), Image.Resampling.BICUBIC
        )

    photometric = (
        params.color_temp != 0.0
        or params.exposure_ev != 0.0
        or params.contrast != 1.0
        or params.saturation != 1.0
        or params.vignette > 0.0
    )
    if photometric:
        arr = np.asarray(img, dtype=np.uint8)
        img = Image.fromarray((_photometric(arr, params) * 255.0 + 0.5).astype(np.uint8), mode="RGB")

    if params.blur_radius > 0.0:
        img = img.filter(ImageFilter.GaussianBlur(params.blur_radius))
    if params.noise_std > 0.0:
        arr = np.asarray(img, dtype=np.float32)
        arr = arr + rng.normal(0.0, params.noise_std, size=arr.shape).astype(np.float32)
        img = Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8), mode="RGB")
    if params.jpeg_quality < 100:
        img = jpeg_roundtrip(img, params.jpeg_quality)

    if output_size is not None:
        size = (output_size, output_size) if isinstance(output_size, int) else output_size
        if img.size != tuple(size):
            img = img.resize(size, Image.Resampling.BICUBIC)
    return img


def single_factor_params(factor: str, level: float, seed: int = 0) -> DistortionParams:
    """Identity params with exactly one knob set to ``level`` (for stability sweeps)."""
    mapping: dict[str, str] = {
        "rotation": "rotation_deg",
        "crop": "crop_scale",
        "exposure": "exposure_ev",
        "blur": "blur_radius",
        "jpeg": "jpeg_quality",
        "color_temp": "color_temp",
        "noise": "noise_std",
        "perspective": "perspective",
    }
    if factor not in mapping:
        raise KeyError(f"unknown factor '{factor}', expected one of {sorted(mapping)}")
    value: float | int = int(level) if factor == "jpeg" else float(level)
    return replace(DistortionParams(seed=seed), **{mapping[factor]: value})


SWEEP_LEVELS: dict[str, list[float]] = {
    "rotation": [0, 5, 10, 20, 30, 45],
    "crop": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
    "exposure": [0.0, -0.5, -1.0, -1.5, -2.0, -2.5],
    "blur": [0.0, 0.5, 1.0, 2.0, 3.0, 4.0],
    "jpeg": [100, 70, 40, 20, 10, 5],
    "color_temp": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
}
