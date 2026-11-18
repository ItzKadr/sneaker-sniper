"""Procedural sneaker renderer (pure Pillow, no external assets).

Every SKU is a :class:`SneakerSpec` = *model geometry* (silhouette, sole, toe, logo,
closure, panels ...) + *colourway*. Several colourways share one geometry, exactly
like a real catalog ("Model X in 4 colourways"). Siblings of the same model are the
natural **hard negatives** for retrieval: identical shape, different colours.

Rendering is done at ``supersample`` x the target resolution and downsampled with a
Lanczos filter for anti-aliasing. The shoe is drawn in side view (toe pointing right)
on a plain light-grey "studio" background, like an e-commerce product shot.
"""

from __future__ import annotations

import colorsys
import logging
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)

RGB = tuple[int, int, int]
Point = tuple[float, float]

SILHOUETTES: tuple[str, ...] = ("low", "mid", "high")
TOE_SHAPES: tuple[str, ...] = ("round", "pointed", "square")
LOGOS: tuple[str, ...] = ("none", "stripes", "curve", "star", "circle", "chevron", "bolt", "diamond")
PATTERNS: tuple[str, ...] = ("solid", "perforated", "colorblock")
CLOSURES: tuple[str, ...] = ("laces", "strap", "slip_on")
TREADS: tuple[str, ...] = ("flat", "teeth")

STUDIO_BACKGROUND: RGB = (248, 248, 248)

NEUTRALS: dict[str, RGB] = {
    "white": (244, 244, 242),
    "off_white": (236, 230, 216),
    "cream": (238, 226, 196),
    "light_grey": (200, 202, 205),
    "grey": (140, 142, 146),
    "charcoal": (70, 72, 76),
    "black": (30, 30, 32),
    "tan": (196, 160, 112),
    "brown": (110, 72, 44),
    "navy": (32, 42, 84),
}
GUM: RGB = (176, 128, 74)


# --------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class SneakerGeometry:
    """Shape-level parameters shared by all colourways of one model.

    Lengths are fractions of the canvas side.
    """

    silhouette: str
    collar_height: float
    sole_thickness: float
    toe_shape: str
    toe_height: float
    length: float
    heel_lean: float
    tongue_rise: float
    throat_pos: float
    logo: str
    logo_scale: float
    logo_x: float
    n_stripes: int
    closure: str
    n_eyelets: int
    has_toe_cap: bool
    has_heel_tab: bool
    has_eyestay: bool
    has_mudguard: bool
    has_midsole_line: bool
    tread: str
    pattern: str


@dataclass(frozen=True)
class Colourway:
    """Colour assignment of the different shoe parts (RGB 0-255)."""

    name: str
    upper: RGB
    secondary: RGB
    accent: RGB
    sole: RGB
    outsole: RGB
    laces: RGB
    lining: RGB


@dataclass(frozen=True)
class SneakerSpec:
    """One catalog SKU = model geometry + colourway."""

    sku_id: str
    model_id: str
    geometry: SneakerGeometry
    colourway: Colourway

    def to_record(self) -> dict[str, object]:
        """Flatten into a dict suitable for a metadata CSV row."""
        rec: dict[str, object] = {"sku_id": self.sku_id, "model_id": self.model_id}
        rec.update({f"geo_{k}": v for k, v in asdict(self.geometry).items()})
        for k, v in asdict(self.colourway).items():
            rec[f"col_{k}"] = "#%02x%02x%02x" % tuple(v) if isinstance(v, (tuple, list)) else v
        return rec


# ------------------------------------------------------------------ colour helpers
def shade(c: RGB, f: float) -> RGB:
    """Darken (``f < 1``) a colour by scaling towards black."""
    return tuple(int(np.clip(v * f, 0, 255)) for v in c)  # type: ignore[return-value]


def tint(c: RGB, f: float) -> RGB:
    """Lighten a colour by mixing a fraction ``f`` of white into it."""
    return tuple(int(np.clip(v + (255 - v) * f, 0, 255)) for v in c)  # type: ignore[return-value]


def _hsv(rng: np.random.Generator, s: tuple[float, float], v: tuple[float, float]) -> RGB:
    h = float(rng.uniform(0.0, 1.0))
    r, g, b = colorsys.hsv_to_rgb(h, float(rng.uniform(*s)), float(rng.uniform(*v)))
    return (int(r * 255), int(g * 255), int(b * 255))


def _neutral(rng: np.random.Generator) -> RGB:
    names = list(NEUTRALS)
    return NEUTRALS[names[int(rng.integers(len(names)))]]


def _colour_distance(a: RGB, b: RGB) -> float:
    return float(np.linalg.norm(np.subtract(a, b, dtype=np.float64)))


def sample_colourway(rng: np.random.Generator, name: str) -> Colourway:
    """Sample a plausible sneaker colourway (neutral-heavy, like real catalogs)."""
    upper = _neutral(rng) if rng.random() < 0.4 else _hsv(rng, (0.45, 0.9), (0.35, 0.92))
    r = rng.random()
    if r < 0.4:
        secondary = _neutral(rng)
    elif r < 0.7:
        secondary = _hsv(rng, (0.4, 0.95), (0.3, 0.95))
    else:
        secondary = shade(upper, 0.65) if sum(upper) > 300 else tint(upper, 0.45)
    accent = _hsv(rng, (0.6, 1.0), (0.55, 1.0))
    if _colour_distance(accent, upper) < 90:
        accent = NEUTRALS["black"] if sum(upper) > 380 else NEUTRALS["white"]
    r = rng.random()
    if r < 0.5:
        sole = NEUTRALS["white"] if rng.random() < 0.7 else NEUTRALS["off_white"]
    elif r < 0.7:
        sole = NEUTRALS["black"]
    elif r < 0.85:
        sole = GUM
    else:
        sole = _hsv(rng, (0.3, 0.8), (0.4, 0.95))
    outsole = GUM if rng.random() < 0.15 else shade(sole, 0.6 if sum(sole) > 200 else 0.85)
    r = rng.random()
    laces = upper if r < 0.5 else (NEUTRALS["white"] if r < 0.8 else accent)
    lining = _neutral(rng) if rng.random() < 0.6 else accent
    return Colourway(name, upper, secondary, accent, sole, outsole, laces, lining)


def _colour_signature(cw: Colourway) -> np.ndarray:
    return np.array(cw.upper + cw.secondary + cw.accent + cw.sole, dtype=np.float64)


def sample_geometry(rng: np.random.Generator) -> SneakerGeometry:
    """Sample one sneaker *model* (shape, panels, logo, closure)."""
    silhouette = str(rng.choice(SILHOUETTES, p=[0.5, 0.25, 0.25]))
    collar_range = {"low": (0.19, 0.24), "mid": (0.27, 0.32), "high": (0.35, 0.42)}[silhouette]
    return SneakerGeometry(
        silhouette=silhouette,
        collar_height=float(rng.uniform(*collar_range)),
        sole_thickness=float(rng.uniform(0.045, 0.12)),
        toe_shape=str(rng.choice(TOE_SHAPES, p=[0.5, 0.25, 0.25])),
        toe_height=float(rng.uniform(0.095, 0.14)),
        length=float(rng.uniform(0.74, 0.86)),
        heel_lean=float(rng.uniform(-0.01, 0.035)),
        tongue_rise=float(rng.uniform(0.0, 0.045)),
        throat_pos=float(rng.uniform(0.33, 0.42)),
        logo=str(rng.choice(LOGOS)),
        logo_scale=float(rng.uniform(0.8, 1.25)),
        logo_x=float(rng.uniform(0.44, 0.58)),
        n_stripes=int(rng.integers(2, 5)),
        closure=str(rng.choice(CLOSURES, p=[0.7, 0.15, 0.15])),
        n_eyelets=int(rng.integers(4, 8)),
        has_toe_cap=bool(rng.random() < 0.5),
        has_heel_tab=bool(rng.random() < 0.6),
        has_eyestay=bool(rng.random() < 0.5),
        has_mudguard=bool(rng.random() < 0.35),
        has_midsole_line=bool(rng.random() < 0.4),
        tread=str(rng.choice(TREADS)),
        pattern=str(rng.choice(PATTERNS, p=[0.6, 0.2, 0.2])),
    )


def generate_catalog_specs(
    n_models: int,
    colourways_per_model: tuple[int, int] = (3, 5),
    seed: int = 0,
    min_colour_distance: float = 90.0,
) -> list[SneakerSpec]:
    """Sample a catalog of ``n_models`` models with several distinct colourways each.

    Colourways of the same model are re-sampled until their colour signature
    (upper + secondary + accent + sole, 12-D RGB) is at least ``min_colour_distance``
    away from the siblings already drawn, so no two SKUs are pixel-identical.
    """
    if n_models < 1:
        raise ValueError("n_models must be >= 1")
    lo, hi = colourways_per_model
    if lo < 1 or hi < lo:
        raise ValueError(f"invalid colourways_per_model={colourways_per_model}")
    rng = np.random.default_rng(seed)
    specs: list[SneakerSpec] = []
    for m in range(n_models):
        geometry = sample_geometry(rng)
        n_cw = int(rng.integers(lo, hi + 1))
        signatures: list[np.ndarray] = []
        for c in range(n_cw):
            cw = sample_colourway(rng, name=f"cw{c:02d}")
            for _ in range(50):
                sig = _colour_signature(cw)
                if all(np.linalg.norm(sig - s) > min_colour_distance for s in signatures):
                    break
                cw = sample_colourway(rng, name=f"cw{c:02d}")
            signatures.append(_colour_signature(cw))
            specs.append(SneakerSpec(f"M{m:04d}-C{c:02d}", f"M{m:04d}", geometry, cw))
    return specs


# ------------------------------------------------------------------ geometry utils
def _cubic(p0: Point, p1: Point, p2: Point, p3: Point, n: int = 20) -> list[Point]:
    """Sample a cubic Bezier curve ``B(t) = sum_i C(3,i) (1-t)^(3-i) t^i P_i``."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = (
        (1 - t) ** 3 * np.array(p0)
        + 3 * (1 - t) ** 2 * t * np.array(p1)
        + 3 * (1 - t) * t**2 * np.array(p2)
        + t**3 * np.array(p3)
    )
    return [(float(x), float(y)) for x, y in pts]


def _quad(p0: Point, p1: Point, p2: Point, n: int = 16) -> list[Point]:
    """Sample a quadratic Bezier curve."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t**2 * np.array(p2)
    return [(float(x), float(y)) for x, y in pts]


def _quad_point(p0: Point, p1: Point, p2: Point, t: float) -> Point:
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
    return (x, y)


def _star(cx: float, cy: float, r: float, inner: float = 0.45, n: int = 5) -> list[Point]:
    pts: list[Point] = []
    for i in range(2 * n):
        rad = r if i % 2 == 0 else r * inner
        ang = -np.pi / 2 + i * np.pi / n
        pts.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
    return pts


# ----------------------------------------------------------------------- renderer
def render_sneaker(
    spec: SneakerSpec,
    size: int = 256,
    supersample: int = 2,
    background: RGB = STUDIO_BACKGROUND,
) -> Image.Image:
    """Render one SKU as an RGB product shot of ``size x size`` pixels."""
    g, c = spec.geometry, spec.colourway
    S = float(size * supersample)
    W = int(S)
    px = max(1, int(round(0.004 * S)))  # base stroke width

    L = g.length * S
    collar_h = g.collar_height * S
    sole_h = g.sole_thickness * S
    toe_h = g.toe_height * S
    toe_len = 0.19 * L
    total_h = sole_h + collar_h + g.tongue_rise * S + 0.02 * S
    ground = 0.5 * S + 0.5 * total_h
    sole_top = ground - sole_h
    x_heel = 0.5 * S - L / 2
    x_toe = 0.5 * S + L / 2
    lean = g.heel_lean * S

    # ---- upper outline --------------------------------------------------------
    A = (x_heel, sole_top + 2 * px)
    B = (x_heel + 0.015 * S + lean, sole_top - collar_h)
    heel = _cubic(A, (x_heel - 0.03 * S, sole_top - 0.35 * collar_h), (x_heel - 0.01 * S + lean, sole_top - 0.85 * collar_h), B)
    throat_drop = {"low": 0.82, "mid": 0.9, "high": 0.95}[g.silhouette]
    C = (x_heel + g.throat_pos * L, sole_top - collar_h * throat_drop)
    dip = {"low": 0.25, "mid": 0.12, "high": 0.05}[g.silhouette] * collar_h
    collar = _cubic(B, (B[0] + 0.08 * L, B[1] + dip), (C[0] - 0.08 * L, C[1] + 0.5 * dip), C)
    T = (C[0] + 0.03 * L, C[1] - g.tongue_rise * S - 0.02 * S)
    D = (C[0] + 0.07 * L, C[1] + 0.01 * S)
    tongue = _quad(C, (C[0], T[1]), T, n=10) + [D]
    E = (x_toe - toe_len, sole_top - toe_h)
    lace_ctrl = ((D[0] + E[0]) / 2, (D[1] + E[1]) / 2 - 0.02 * S)
    vamp = _quad(D, lace_ctrl, E)
    if g.toe_shape == "round":
        x_front = x_toe
        toe = _cubic(E, (x_toe - 0.25 * toe_len, E[1]), (x_toe + 0.01 * S, E[1] + 0.35 * toe_h), (x_toe, sole_top))
    elif g.toe_shape == "pointed":
        x_front = x_toe + 0.02 * S
        toe = _cubic(E, (x_toe - 0.45 * toe_len, E[1] + 0.1 * toe_h), (x_front, sole_top - 0.25 * toe_h), (x_front, sole_top))
    else:  # square
        x_front = x_toe
        toe = _cubic(E, (x_toe - 0.05 * toe_len, E[1]), (x_toe, E[1] + 0.08 * toe_h), (x_toe, E[1] + 0.3 * toe_h))
        toe.append((x_toe, sole_top))
    upper_pts = heel + collar[1:] + tongue[1:] + vamp[1:] + toe[1:] + [(x_front, sole_top + 2 * px)]

    canvas = Image.new("RGBA", (W, W), background + (255,))

    # ---- soft contact shadow -------------------------------------------------------
    shadow = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse([x_heel, ground - 0.012 * S, x_front, ground + 0.018 * S], fill=(0, 0, 0, 40))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(0.01 * S)))

    draw = ImageDraw.Draw(canvas)
    # ---- sole ------------------------------------------------------------------------
    sole_box = [x_heel - 0.015 * S, sole_top - px, x_front + 0.012 * S, ground]
    radius = max(2.0, sole_h * 0.45)
    draw.rounded_rectangle(sole_box, radius=radius, fill=c.sole, outline=shade(c.sole, 0.7), width=px)
    out_h = sole_h * 0.32
    draw.rounded_rectangle([sole_box[0], ground - out_h, sole_box[2], ground], radius=min(radius, out_h / 2), fill=c.outsole)
    if g.tread == "teeth":
        step = 0.025 * S
        x = sole_box[0] + radius
        while x < sole_box[2] - radius:
            draw.rectangle([x, ground - 0.3 * out_h, x + 0.006 * S, ground], fill=shade(c.outsole, 0.65))
            x += step
    if g.has_midsole_line:
        y = sole_top + 0.42 * (sole_h - out_h)
        draw.line([(x_heel + 0.02 * S, y), (x_front - 0.02 * S, y)], fill=c.accent, width=max(1, int(0.006 * S)))

    # ---- upper fill + mask ---------------------------------------------------------------
    draw.polygon(upper_pts, fill=c.upper)
    upper_mask = Image.new("L", (W, W), 0)
    ImageDraw.Draw(upper_mask).polygon(upper_pts, fill=255)

    # ---- decorations clipped to the upper -------------------------------------------------
    deco = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    dd = ImageDraw.Draw(deco)
    if g.pattern == "colorblock":
        dd.polygon([(x_heel - 0.05 * S, 0), (x_heel + 0.42 * L, 0), (x_heel + 0.30 * L, S), (x_heel - 0.05 * S, S)], fill=c.secondary)
    if g.has_mudguard:
        dd.rectangle([x_heel - 0.05 * S, sole_top - 0.035 * S, x_front + 0.05 * S, sole_top + 3 * px], fill=c.secondary)
    if g.has_toe_cap:
        dd.ellipse([x_toe - 1.25 * toe_len, sole_top - 1.05 * toe_h, x_toe + 0.3 * toe_len, sole_top + 0.6 * toe_h], fill=c.secondary, outline=shade(c.secondary, 0.7), width=px)
    if g.has_heel_tab:
        dd.ellipse([x_heel - 0.08 * S, sole_top - 0.95 * collar_h, x_heel + 0.09 * S, sole_top + 0.05 * S], fill=c.secondary, outline=shade(c.secondary, 0.7), width=px)
    if g.pattern == "perforated":
        dot = 0.0045 * S
        y = E[1] + 0.02 * S
        while y < sole_top - 0.015 * S:
            x = x_toe - toe_len
            while x < x_toe - 0.02 * S:
                dd.ellipse([x - dot, y - dot, x + dot, y + dot], fill=shade(c.upper, 0.72))
                x += 0.018 * S
            y += 0.018 * S
    if g.has_eyestay:
        stay = [(x, y + 0.022 * S) for x, y in _quad(D, lace_ctrl, E)]
        dd.line(stay, fill=c.secondary, width=max(2, int(0.032 * S)), joint="curve")

    # Logo, centred on the side panel between sole and lace line.
    lx = x_heel + g.logo_x * L
    t_top = float(np.clip((lx - D[0]) / max(E[0] - D[0], 1.0), 0.0, 1.0))
    top_y = _quad_point(D, lace_ctrl, E, t_top)[1] if lx >= D[0] else C[1]
    panel_h = sole_top - top_y
    ly = sole_top - 0.5 * panel_h
    r = min(0.065 * S * g.logo_scale, 0.42 * panel_h)
    if g.logo == "stripes":
        w, gap = 0.022 * S, 0.02 * S
        slant = 0.35 * panel_h
        x0 = lx - (g.n_stripes * (w + gap)) / 2
        for i in range(g.n_stripes):
            xs = x0 + i * (w + gap)
            dd.polygon([(xs - slant / 2, sole_top + px), (xs - slant / 2 + w, sole_top + px), (xs + slant / 2 + w, top_y), (xs + slant / 2, top_y)], fill=c.accent)
    elif g.logo == "curve":
        dd.arc([lx - 2.0 * r, ly - 1.4 * r, lx + 1.6 * r, ly + 1.0 * r], start=20, end=165, fill=c.accent, width=max(2, int(0.35 * r)))
    elif g.logo == "star":
        dd.polygon(_star(lx, ly, r), fill=c.accent)
    elif g.logo == "circle":
        dd.ellipse([lx - r, ly - r, lx + r, ly + r], outline=c.accent, width=max(2, int(0.3 * r)))
    elif g.logo == "chevron":
        dd.line([(lx - 0.7 * r, ly - r), (lx + 0.5 * r, ly), (lx - 0.7 * r, ly + r)], fill=c.accent, width=max(2, int(0.32 * r)), joint="curve")
    elif g.logo == "bolt":
        dd.polygon([(lx - 0.2 * r, ly - r), (lx + 0.5 * r, ly - r), (lx + 0.1 * r, ly - 0.1 * r), (lx + 0.6 * r, ly - 0.1 * r), (lx - 0.4 * r, ly + r), (lx - 0.05 * r, ly + 0.15 * r), (lx - 0.55 * r, ly + 0.15 * r)], fill=c.accent)
    elif g.logo == "diamond":
        dd.polygon([(lx, ly - r), (lx + 0.75 * r, ly), (lx, ly + r), (lx - 0.75 * r, ly)], fill=c.accent)

    # Closure details living inside the upper.
    if g.closure == "strap":
        for t in (0.3, 0.62):
            p = _quad_point(D, lace_ctrl, E, t)
            dd.line([(p[0] - 0.03 * S, p[1] - 0.02 * S), (p[0] + 0.01 * S, p[1] + 0.1 * S)], fill=c.secondary, width=max(2, int(0.03 * S)))
    elif g.closure == "slip_on":
        dd.polygon([(C[0] - 0.01 * S, C[1]), (D[0] + 0.01 * S, D[1]), (C[0] + 0.035 * L, C[1] + 0.07 * S)], fill=shade(c.upper, 0.7))

    # Stitching: dashed line parallel to the sole.
    stitch_col = shade(c.upper, 0.75) if sum(c.upper) > 200 else tint(c.upper, 0.4)
    x, y = x_heel, sole_top - 0.014 * S
    while x < x_front:
        dd.line([(x, y), (x + 0.012 * S, y)], fill=stitch_col, width=max(1, px // 2 + 1))
        x += 0.022 * S

    alpha = ImageChops.multiply(deco.getchannel("A"), upper_mask)
    deco.putalpha(alpha)
    canvas.alpha_composite(deco)

    # ---- outline, collar lining, laces, pull tab ------------------------------------------
    draw = ImageDraw.Draw(canvas)
    draw.line(upper_pts + [upper_pts[0]], fill=shade(c.upper, 0.55), width=px + 1, joint="curve")
    draw.line(collar, fill=c.lining, width=max(2, int(0.012 * S)), joint="curve")
    if g.closure == "laces":
        eyelet_r = 0.006 * S
        for t in np.linspace(0.12, 0.85, g.n_eyelets):
            ex, ey = _quad_point(D, lace_ctrl, E, float(t))
            ey += 0.012 * S
            draw.line([(ex - 0.02 * S, ey + 0.004 * S), (ex + 0.012 * S, ey - 0.016 * S)], fill=c.laces, width=max(2, int(0.009 * S)))
            draw.ellipse([ex - eyelet_r, ey - eyelet_r, ex + eyelet_r, ey + eyelet_r], fill=shade(c.lining, 0.5))
        bx, by = D[0] + 0.01 * S, D[1] - 0.012 * S
        for dx in (-0.018 * S, 0.012 * S):
            draw.ellipse([bx + dx - 0.016 * S, by - 0.009 * S, bx + dx + 0.016 * S, by + 0.009 * S], outline=c.laces, width=max(2, int(0.006 * S)))
    if g.silhouette in ("mid", "high"):
        draw.rounded_rectangle([B[0] - 0.012 * S, B[1] - 0.03 * S, B[0] + 0.012 * S, B[1] + 0.015 * S], radius=0.006 * S, fill=c.accent)

    out = canvas.convert("RGB")
    if supersample != 1:
        out = out.resize((size, size), Image.Resampling.LANCZOS)
    return out


def render_catalog(specs: Sequence[SneakerSpec], size: int = 256, supersample: int = 2) -> list[Image.Image]:
    """Render a list of specs (convenience wrapper used by the in-memory demo)."""
    return [render_sneaker(s, size=size, supersample=supersample) for s in specs]
