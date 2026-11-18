"""Streamlit smoke tests via ``streamlit.testing.v1.AppTest`` (offline, random weights)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def offline_demo_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNEAKER_SNIPER_PRETRAINED", "0")
    monkeypatch.setenv("SNEAKER_SNIPER_IGNORE_ARTIFACTS", "1")
    monkeypatch.setenv("SNEAKER_SNIPER_DEMO_MODELS", "4")
    monkeypatch.setenv("SNEAKER_SNIPER_IMAGE_SIZE", "56")
    monkeypatch.setenv("SNEAKER_SNIPER_DEMO_DISTRACTORS", "1000")
    monkeypatch.setenv("SNEAKER_SNIPER_DEVICE", "cpu")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def test_app_cold_start_demo(offline_demo_env: None) -> None:
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    assert any("RANDOM weights" in w.value for w in at.warning)  # visible fallback warning
    labels = [m.label for m in at.metric]
    assert "Self-Cosine Stability" in labels

    at.slider[0].set_value(-20.0).run()  # rotate the simulated phone photo
    assert not at.exception, at.exception

    at.button(key="run_bench").click().run()  # live recall/latency benchmark
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 1


@pytest.mark.skipif(not (REPO / "artifacts" / "manifest.json").is_file(), reason="train.py artifacts not present")
def test_app_with_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uses artifacts/ if present; HF_HUB_OFFLINE keeps it network-free (cached weights or fallback)."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("SNEAKER_SNIPER_DEMO_MODELS", "4")
    monkeypatch.setenv("SNEAKER_SNIPER_DEVICE", "cpu")
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    at.sidebar.selectbox[0].set_value("resnet50").run()  # artifacts/resnet50 if present, else demo
    assert not at.exception, at.exception
