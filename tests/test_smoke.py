"""Smoke tests that run without any backend or network."""
from eval import run_eval
from src import config


def test_demo_approach_registered():
    assert "demo" in run_eval.APPROACHES


def test_demo_returns_metrics():
    row = run_eval.APPROACHES["demo"](0)
    assert row["approach"] == "demo"
    assert "score" in row


def test_config_has_backend():
    assert config.LLM_BASE_URL
    assert config.LLM_MODEL
