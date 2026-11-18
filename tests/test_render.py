from __future__ import annotations

import numpy as np
import pytest

from src.render import generate_catalog_specs, render_sneaker, shade, tint


def test_catalog_specs_counts_ids_and_families() -> None:
    specs = generate_catalog_specs(5, (2, 3), seed=1)
    assert 10 <= len(specs) <= 15
    assert len({s.sku_id for s in specs}) == len(specs)
    for s in specs:
        assert s.sku_id.startswith(s.model_id)
    # colourways of one model share the geometry (=> natural hard negatives)
    by_model: dict[str, list] = {}
    for s in specs:
        by_model.setdefault(s.model_id, []).append(s)
    for siblings in by_model.values():
        assert all(sib.geometry == siblings[0].geometry for sib in siblings)


def test_specs_are_deterministic() -> None:
    assert generate_catalog_specs(3, seed=7) == generate_catalog_specs(3, seed=7)
    assert generate_catalog_specs(3, seed=7) != generate_catalog_specs(3, seed=8)


def test_render_size_mode_and_determinism() -> None:
    spec = generate_catalog_specs(1, (1, 1), seed=0)[0]
    a = render_sneaker(spec, size=96)
    b = render_sneaker(spec, size=96)
    assert a.size == (96, 96) and a.mode == "RGB"
    assert np.array_equal(np.asarray(a), np.asarray(b))
    # border = studio background, centre = shoe
    arr = np.asarray(a).astype(int)
    assert np.abs(arr[0, 0] - np.array([248, 248, 248])).max() <= 2
    assert np.abs(arr - 248).sum(axis=-1).astype(bool).mean() > 0.1


def test_sibling_colourways_render_differently() -> None:
    specs = generate_catalog_specs(1, (2, 2), seed=4)
    a, b = (np.asarray(render_sneaker(s, size=64)).astype(float) for s in specs)
    assert np.abs(a - b).mean() > 2.0


def test_colour_helpers() -> None:
    assert shade((100, 200, 50), 0.5) == (50, 100, 25)
    assert tint((0, 0, 0), 1.0) == (255, 255, 255)


@pytest.mark.parametrize("bad", [dict(n_models=0), dict(n_models=2, colourways_per_model=(3, 2))])
def test_invalid_spec_arguments(bad: dict) -> None:
    with pytest.raises(ValueError):
        generate_catalog_specs(**bad)
