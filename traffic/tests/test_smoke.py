"""Smoke tests for Topic 2 — pure-Python sim, no API key needed."""
from traffic.environment import GridIntersection
from traffic.evaluate import evaluate
from traffic.simulator import run_episode
from traffic.agents.zeroshot_agent import choose_move


def test_episode_terminates():
    result = run_episode(GridIntersection(max_turns=40), choose_move)
    assert "success" in result and "collisions" in result and "steps" in result


def test_evaluate_aggregates():
    summary = evaluate("zeroshot", episodes=3)
    assert summary["episodes"] == 3
    assert 0.0 <= summary["success_rate"] <= 1.0
