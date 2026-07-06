"""Smoke tests for Topic 2. These stay pure-Python even when Ollama is installed."""
from src import llm
from traffic.agents.zeroshot_agent import choose_move
from traffic.environment import GridIntersection
from traffic.evaluate import evaluate
from traffic.simulator import run_episode


def test_episode_terminates(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda: False)
    result = run_episode(GridIntersection(max_turns=40), choose_move)
    assert "success" in result and "collisions" in result and "steps" in result


def test_evaluate_aggregates(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda: False)
    summary = evaluate("zeroshot", episodes=3)
    assert summary["episodes"] == 3
    assert 0.0 <= summary["success_rate"] <= 1.0


def test_light_cycle_reaches_ew_after_first_clearance():
    env = GridIntersection()
    phases = []
    for _ in range(9):
        phases.append(env.state.light_phase)
        env.step({})

    assert phases[:6] == ["NS"] * 6
    assert phases[6:8] == ["ALL_RED"] * 2
    assert phases[8] == "EW"
