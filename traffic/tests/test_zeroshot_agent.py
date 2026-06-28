"""Real integration tests for zeroshot_agent.  OWNER: R3 (Pratham).

All tests call the actual LLM (Ollama / llama3.1 running locally).
Start Ollama before running: ollama serve
Run: python -m pytest traffic/tests/test_zeroshot_agent.py -v
"""
from __future__ import annotations

import pytest

from traffic.environment import GridIntersection, State, Vehicle
from traffic.agents.zeroshot_agent import choose_move, _parse_move, _build_prompt

VALID_MOVES = {(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)}


# ── _parse_move (pure logic, no LLM) ────────────────────────────────────────

def test_parse_move_all_keywords():
    assert _parse_move("RIGHT") == (1, 0)
    assert _parse_move("LEFT")  == (-1, 0)
    assert _parse_move("DOWN")  == (0, 1)
    assert _parse_move("UP")    == (0, -1)
    assert _parse_move("WAIT")  == (0, 0)


def test_parse_move_case_insensitive():
    assert _parse_move("right") == (1, 0)
    assert _parse_move("Right\n") == (1, 0)


def test_parse_move_embedded_in_text():
    assert _parse_move("I will move RIGHT now.") == (1, 0)


def test_parse_move_unknown_returns_wait():
    assert _parse_move("JUMP") == (0, 0)
    assert _parse_move("") == (0, 0)


# ── _build_prompt (pure logic, no LLM) ──────────────────────────────────────

def _two_car_state() -> State:
    return State(vehicles=[
        Vehicle(id="A", pos=(2, 5), goal=(9, 5)),
        Vehicle(id="B", pos=(5, 2), goal=(5, 9)),
    ])


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


# ── Real LLM tests ───────────────────────────────────────────────────────────

def test_llm_returns_legal_move_for_single_car():
    """LLM must return one of the five legal moves for a simple straight-line scenario."""
    state = State(vehicles=[Vehicle(id="A", pos=(0, 5), goal=(9, 5))])
    move = choose_move(state, "A")
    assert move in VALID_MOVES, f"Illegal move returned: {move}"


def test_llm_returns_legal_move_with_two_cars():
    """LLM must return a legal move even with another car present (collision risk)."""
    state = _two_car_state()
    move = choose_move(state, "A")
    assert move in VALID_MOVES, f"Illegal move returned: {move}"


def test_choose_move_does_not_mutate_state():
    """The agent must never modify the state object."""
    state = _two_car_state()
    original_pos = state.vehicles[0].pos
    original_turn = state.turn
    choose_move(state, "A")
    assert state.vehicles[0].pos == original_pos
    assert state.turn == original_turn


def test_arrived_car_returns_wait():
    """An already-arrived car must get (0, 0) without touching the LLM."""
    state = State(vehicles=[Vehicle(id="A", pos=(9, 5), goal=(9, 5), arrived=True)])
    move = choose_move(state, "A")
    assert move == (0, 0)


def test_llm_moves_toward_goal_horizontal():
    """Car A needs to go right (goal is to the right). LLM should pick RIGHT."""
    state = State(vehicles=[Vehicle(id="A", pos=(0, 5), goal=(9, 5))])
    move = choose_move(state, "A")
    # LLM should choose RIGHT = (1,0) for a clear straight path
    assert move == (1, 0), f"Expected RIGHT=(1,0), got {move}"


def test_llm_moves_toward_goal_vertical():
    """Car B needs to go down (goal is below). LLM should pick DOWN."""
    state = State(vehicles=[Vehicle(id="B", pos=(5, 0), goal=(5, 9))])
    move = choose_move(state, "B")
    assert move == (0, 1), f"Expected DOWN=(0,1), got {move}"


def test_full_episode_with_real_llm():
    """Run a complete episode with real LLM driving both cars. Must terminate."""
    env = GridIntersection(max_turns=50)
    from traffic.simulator import run_episode
    result = run_episode(env, choose_move)
    assert "success" in result
    assert "collisions" in result
    assert "steps" in result
    assert result["steps"] <= 50
    print(f"\nEpisode result: success={result['success']}, "
          f"collisions={result['collisions']}, steps={result['steps']}")


def test_multiple_episodes_success_rate():
    """Run 3 episodes; expect at least 1 success (LLM isn't perfect but shouldn't fail all)."""
    from traffic.simulator import run_episode
    successes = 0
    for _ in range(3):
        env = GridIntersection(max_turns=40)
        result = run_episode(env, choose_move)
        if result["success"]:
            successes += 1
    print(f"\nSuccesses: {successes}/3")
    assert successes >= 1, "LLM agent failed all 3 episodes — check model or prompt"
