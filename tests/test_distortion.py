from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.distortion import (
    GAMMA,
    DistortionParams,
    apply_distortion,
    estimate_foreground_mask,
    geometry_matrix,
    perspective_coefficients,
    sample_distortion_params,
    single_factor_params,
    white_balance_gains,
)
from src.render import generate_catalog_specs, render_sneaker


@pytest.fixture(scope="module")
def shoe() -> Image.Image:
    return render_sneaker(generate_catalog_specs(1, (1, 1), seed=2)[0], size=96)


def _arr(im: Image.Image) -> np.ndarray:
    return np.asarray(im).astype(np.float64)


def test_identity_is_exact(shoe: Image.Image) -> None:
    assert np.array_equal(_arr(apply_distortion(shoe, DistortionParams())), _arr(shoe))
    assert DistortionParams(seed=5).is_identity()


def test_size_preserved_and_resizable(shoe: Image.Image, rng: np.random.Generator) -> None:
    p = sample_distortion_params(rng, 1.0)
    assert apply_distortion(shoe, p).size == shoe.size
    assert apply_distortion(shoe, p, output_size=48).size == (48, 48)


def test_deterministic_given_seed(shoe: Image.Image) -> None:
    p = DistortionParams(rotation_deg=10, background="clutter", noise_std=5, jpeg_quality=50, seed=11)
    assert np.array_equal(_arr(apply_distortion(shoe, p)), _arr(apply_distortion(shoe, p)))
    q = DistortionParams(rotation_deg=10, background="clutter", noise_std=5, jpeg_quality=50, seed=12)
    assert not np.array_equal(_arr(apply_distortion(shoe, p)), _arr(apply_distortion(shoe, q)))


def test_exposure_is_applied_in_linear_light() -> None:
    grey = Image.new("RGB", (16, 16), (128, 128, 128))
    out = _arr(apply_distortion(grey, DistortionParams(exposure_ev=1.0)))
    # x' = (2 * x^2.2)^(1/2.2) = x * 2^(1/2.2): 128 -> 175.3 (NOT 256 like a naive gain)
    expected = 128 * 2 ** (1 / GAMMA)
    assert abs(out.mean() - expected) < 1.5


def test_colour_temperature_direction(shoe: Image.Image) -> None:
    def rb_ratio(im: Image.Image) -> float:
        a = _arr(im)
        return a[..., 0].mean() / a[..., 2].mean()

    base = rb_ratio(shoe)
    assert rb_ratio(apply_distortion(shoe, DistortionParams(color_temp=0.8))) > base
    assert rb_ratio(apply_distortion(shoe, DistortionParams(color_temp=-0.8))) < base


def test_white_balance_preserves_grey_luma() -> None:
    for t in (-1.0, -0.3, 0.0, 0.5, 1.0):
        g = white_balance_gains(t)
        assert float(g @ np.array([0.299, 0.587, 0.114])) == pytest.approx(1.0, abs=1e-6)


def test_lower_jpeg_quality_means_more_error(shoe: Image.Image) -> None:
    err = {q: np.abs(_arr(apply_distortion(shoe, DistortionParams(jpeg_quality=q))) - _arr(shoe)).mean() for q in (90, 10)}
    assert err[10] > err[90] > 0


def test_perspective_coefficients_map_dst_to_src() -> None:
    src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    dst = src + np.array([[1, 1], [-2, 1], [-1, -1], [1, -2]], dtype=float)
    a, b, c, d, e, f, g, h = perspective_coefficients(dst, src)
    for (x, y), (u, v) in zip(dst, src):
        den = g * x + h * y + 1
        assert (a * x + b * y + c) / den == pytest.approx(u, abs=1e-9)
        assert (d * x + e * y + f) / den == pytest.approx(v, abs=1e-9)


def test_rotation_matrix_keeps_centre_fixed(rng: np.random.Generator) -> None:
    m = geometry_matrix((100, 60), 33.0, 0.0, rng)
    centre = m @ np.array([50.0, 30.0, 1.0])
    assert centre[:2] / centre[2] == pytest.approx([50.0, 30.0])
    assert np.allclose(geometry_matrix((100, 60), 0.0, 0.0, rng), np.eye(3))


def test_foreground_mask_studio_vs_real_photo(shoe: Image.Image, rng: np.random.Generator) -> None:
    mask, bg, studio = estimate_foreground_mask(shoe)
    frac = (np.asarray(mask) > 127).mean()
    assert studio and 0.05 < frac < 0.9
    assert max(abs(v - 248) for v in bg) <= 2
    noisy = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    mask2, _, studio2 = estimate_foreground_mask(noisy)
    assert not studio2 and np.asarray(mask2).min() == 255


def test_sampler_ranges_and_zero_strength(rng: np.random.Generator) -> None:
    for _ in range(200):
        p = sample_distortion_params(rng, 1.0)
        assert -25 <= p.rotation_deg <= 25
        assert 0.65 <= p.crop_scale <= 1.0
        assert 35 <= p.jpeg_quality <= 95
        assert p.background in ("plain", "clutter")
    p0 = sample_distortion_params(rng, 0.0)
    assert p0.rotation_deg == 0 and p0.crop_scale == 1.0 and p0.exposure_ev == 0 and p0.perspective == 0


def test_single_factor_params() -> None:
    p = single_factor_params("jpeg", 20)
    assert p.jpeg_quality == 20 and p.rotation_deg == 0
    assert single_factor_params("rotation", 15).rotation_deg == 15
    with pytest.raises(KeyError):
        single_factor_params("fisheye", 1.0)
