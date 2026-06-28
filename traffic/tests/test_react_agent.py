from src import llm
from traffic.agents.react_agent import choose_move, explain_move
from traffic.environment import State, Vehicle


def test_react_fallback_returns_move_and_trace(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(llm, "complete", _raise)
    state = State(vehicles=[Vehicle(id="A", pos=(0, 5), goal=(9, 5))])
    result = explain_move(state, "A")
    assert result["move"] == (2, 0)
    assert result["action"] == "RIGHT2"
    assert result["thought"]
    assert result["observation"]
    assert result["backend"] == "fallback"


def test_react_choose_move_uses_explain_move(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(llm, "complete", _raise)
    state = State(vehicles=[Vehicle(id="B", pos=(5, 0), goal=(5, 9))])
    assert choose_move(state, "B") == (0, 2)


def test_react_does_not_mutate_state(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(llm, "complete", _raise)
    state = State(vehicles=[Vehicle(id="A", pos=(0, 5), goal=(9, 5))])
    explain_move(state, "A")
    assert state.vehicles[0].pos == (0, 5)
    assert state.turn == 0
