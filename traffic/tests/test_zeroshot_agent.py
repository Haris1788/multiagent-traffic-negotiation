from __future__ import annotations

import pytest

from src import llm
from traffic.agents.zeroshot_agent import _build_prompt, _parse_move, choose_move
from traffic.environment import GridIntersection, State, Vehicle

VALID_MOVES = {
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (2, 0), (-2, 0), (0, 2), (0, -2),
    (0, 0),
}
LLM_AVAILABLE = llm.is_available()


def _two_car_state() -> State:
    return State(vehicles=[
        Vehicle(id="A", pos=(2, 5), goal=(9, 5)),
        Vehicle(id="B", pos=(5, 2), goal=(5, 9)),
    ])


def test_parse_move_all_keywords():
    assert _parse_move("RIGHT") == (1, 0)
    assert _parse_move("LEFT") == (-1, 0)
    assert _parse_move("DOWN") == (0, 1)
    assert _parse_move("UP") == (0, -1)
    assert _parse_move("WAIT") == (0, 0)
    assert _parse_move("RIGHT2") == (2, 0)
    assert _parse_move("LEFT 2") == (-2, 0)
    assert _parse_move("DOWN_TWO") == (0, 2)


def test_parse_move_case_insensitive():
    assert _parse_move("right") == (1, 0)
    assert _parse_move("Right\n") == (1, 0)


def test_parse_move_embedded_in_text():
    assert _parse_move("I will move RIGHT now.") == (1, 0)


def test_parse_move_unknown_returns_wait():
    assert _parse_move("JUMP") == (0, 0)
    assert _parse_move("") == (0, 0)


def test_prompt_contains_car_id():
    assert "car A" in _build_prompt(_two_car_state(), "A")


def test_prompt_contains_position_and_goal():
    prompt = _build_prompt(_two_car_state(), "A")
    assert "(2, 5)" in prompt
    assert "(9, 5)" in prompt


def test_prompt_lists_other_active_cars():
    assert "car B" in _build_prompt(_two_car_state(), "A")


def test_prompt_omits_arrived_cars():
    state = State(vehicles=[
        Vehicle(id="A", pos=(2, 5), goal=(9, 5)),
        Vehicle(id="B", pos=(5, 9), goal=(5, 9), arrived=True),
    ])
    assert "No other active cars" in _build_prompt(state, "A")


def test_choose_move_fast_fallback_when_llm_unavailable(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(llm, "complete", _raise)
    state = State(vehicles=[Vehicle(id="A", pos=(0, 5), goal=(9, 5))])
    assert choose_move(state, "A") == (2, 0)


def test_choose_move_fallback_avoids_higher_priority_path(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(llm, "complete", _raise)
    state = State(vehicles=[
        Vehicle(id="A", pos=(4, 5), goal=(9, 5)),
        Vehicle(id="B", pos=(5, 4), goal=(5, 9)),
    ])
    assert choose_move(state, "B") == (0, 0)


def test_choose_move_fallback_does_not_mutate_state(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(llm, "complete", _raise)
    state = _two_car_state()
    original_pos = state.vehicles[0].pos
    original_turn = state.turn
    choose_move(state, "A")
    assert state.vehicles[0].pos == original_pos
    assert state.turn == original_turn


def test_arrived_car_returns_wait_without_llm(monkeypatch):
    def _raise(*args, **kwargs):
        raise AssertionError("LLM should not be called for arrived cars")

    monkeypatch.setattr(llm, "complete", _raise)
    state = State(vehicles=[Vehicle(id="A", pos=(9, 5), goal=(9, 5), arrived=True)])
    assert choose_move(state, "A") == (0, 0)


@pytest.mark.skipif(not LLM_AVAILABLE, reason="LLM backend is not running")
def test_llm_returns_legal_move_for_single_car():
    state = State(vehicles=[Vehicle(id="A", pos=(0, 5), goal=(9, 5))])
    assert choose_move(state, "A") in VALID_MOVES


@pytest.mark.skipif(not LLM_AVAILABLE, reason="LLM backend is not running")
def test_llm_returns_legal_move_with_two_cars():
    assert choose_move(_two_car_state(), "A") in VALID_MOVES


@pytest.mark.skipif(not LLM_AVAILABLE, reason="LLM backend is not running")
def test_full_episode_with_real_llm():
    from traffic.simulator import run_episode

    result = run_episode(GridIntersection(max_turns=50), choose_move)
    assert "success" in result
    assert "collisions" in result
    assert "steps" in result
    assert result["steps"] <= 50
