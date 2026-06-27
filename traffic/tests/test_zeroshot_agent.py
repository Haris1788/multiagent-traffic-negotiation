"""Tests for zeroshot_agent.  OWNER: R3 (Pratham).

All tests mock src.llm.complete so they run without any LLM server.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from traffic.environment import GridIntersection, State, Vehicle
from traffic.agents.zeroshot_agent import choose_move, _parse_move, _build_prompt


# ── _parse_move ──────────────────────────────────────────────────────────────

def test_parse_move_exact():
    assert _parse_move("RIGHT") == (1, 0)
    assert _parse_move("LEFT") == (-1, 0)
    assert _parse_move("UP") == (0, -1)
    assert _parse_move("DOWN") == (0, 1)
    assert _parse_move("WAIT") == (0, 0)


def test_parse_move_case_insensitive():
    assert _parse_move("right") == (1, 0)
    assert _parse_move("Right\n") == (1, 0)


def test_parse_move_embedded():
    # LLM might reply with surrounding text
    assert _parse_move("I will go RIGHT.") == (1, 0)


def test_parse_move_unknown_falls_back_to_wait():
    assert _parse_move("JUMP") == (0, 0)
    assert _parse_move("") == (0, 0)


# ── _build_prompt ────────────────────────────────────────────────────────────

def _simple_state() -> State:
    return State(
        vehicles=[
            Vehicle(id="A", pos=(2, 5), goal=(9, 5)),
            Vehicle(id="B", pos=(5, 2), goal=(5, 9)),
        ]
    )


def test_build_prompt_contains_car_id():
    prompt = _build_prompt(_simple_state(), "A")
    assert "car A" in prompt


def test_build_prompt_contains_positions():
    prompt = _build_prompt(_simple_state(), "A")
    assert "(2, 5)" in prompt
    assert "(9, 5)" in prompt  # goal


def test_build_prompt_mentions_other_car():
    prompt = _build_prompt(_simple_state(), "A")
    assert "car B" in prompt


def test_build_prompt_no_others_when_arrived():
    state = State(
        vehicles=[
            Vehicle(id="A", pos=(2, 5), goal=(9, 5)),
            Vehicle(id="B", pos=(5, 9), goal=(5, 9), arrived=True),
        ]
    )
    prompt = _build_prompt(state, "A")
    assert "No other active cars" in prompt


# ── choose_move ──────────────────────────────────────────────────────────────

def test_choose_move_uses_llm_reply():
    state = _simple_state()
    with patch("src.llm.complete", return_value="RIGHT") as mock_llm:
        move = choose_move(state, "A")
    assert move == (1, 0)
    mock_llm.assert_called_once()


def test_choose_move_does_not_mutate_state():
    state = _simple_state()
    original_pos = state.vehicles[0].pos
    with patch("src.llm.complete", return_value="RIGHT"):
        choose_move(state, "A")
    assert state.vehicles[0].pos == original_pos
    assert state.turn == 0


def test_choose_move_fallback_on_llm_error():
    """If the LLM is unreachable, agent falls back to greedy movement."""
    state = _simple_state()
    with patch("src.llm.complete", side_effect=ConnectionError("no server")):
        move = choose_move(state, "A")
    # Greedy: A is at x=2, goal x=9 → should move RIGHT = (1, 0)
    assert move == (1, 0)


def test_choose_move_arrived_car_waits():
    state = State(
        vehicles=[Vehicle(id="A", pos=(9, 5), goal=(9, 5), arrived=True)]
    )
    with patch("src.llm.complete", return_value="RIGHT"):
        move = choose_move(state, "A")
    assert move == (0, 0)


def test_choose_move_returns_legal_move_in_full_episode():
    """End-to-end: run a whole episode with mocked LLM returning greedy directions."""
    env = GridIntersection(max_turns=30)

    def _fake_llm(prompt: str, **_) -> str:
        # parse which car and direction it needs from the prompt, return RIGHT/DOWN
        if "car A" in prompt and "Your position" in prompt:
            return "RIGHT"
        return "DOWN"

    with patch("src.llm.complete", side_effect=_fake_llm):
        from traffic.simulator import run_episode
        result = run_episode(env, choose_move)

    assert "success" in result
    assert "collisions" in result
    assert result["steps"] <= 30
