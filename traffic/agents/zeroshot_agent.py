"""Zero-shot driving agent. OWNER: R3.

Builds a plain-language prompt describing the grid state, calls the LLM once,
and parses the returned move. If the model backend is unavailable, it falls back
to a deterministic collision-aware policy so tests and demos stay responsive.
"""
from __future__ import annotations

import re

from src import llm
from traffic.environment import Move, State

_MOVE_MAP: dict[str, Move] = {
    "UP": (0, -1),
    "UP2": (0, -2),
    "DOWN": (0, 1),
    "DOWN2": (0, 2),
    "LEFT": (-1, 0),
    "LEFT2": (-2, 0),
    "RIGHT": (1, 0),
    "RIGHT2": (2, 0),
    "WAIT": (0, 0),
}

_ONE_STEP: dict[str, Move] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "WAIT": (0, 0),
}

_TWO_STEP: dict[str, Move] = {
    "UP": (0, -2),
    "DOWN": (0, 2),
    "LEFT": (-2, 0),
    "RIGHT": (2, 0),
}

_SYSTEM = (
    "You are a traffic-control agent for a 10x10 grid intersection. "
    "On every turn you receive the positions of all cars and must choose one move "
    "for the assigned car. Reply with exactly one token from: "
    "UP DOWN LEFT RIGHT UP2 DOWN2 LEFT2 RIGHT2 WAIT. "
    "Use x/y coordinates exactly as described. No punctuation, no explanation."
)


def _build_prompt(state: State, car_id: str) -> str:
    me = next(v for v in state.vehicles if v.id == car_id)
    others = [v for v in state.vehicles if v.id != car_id and not v.arrived]

    lines = [
        f"Turn {state.turn}.",
        f"You are car {car_id}.",
        f"Your position: {me.pos}. Your goal: {me.goal}. Your speed last turn: {me.speed}.",
    ]

    if others:
        other_desc = "; ".join(
            f"car {v.id} at {v.pos} going to {v.goal}" for v in others
        )
        lines.append(f"Other cars: {other_desc}.")
    else:
        lines.append("No other active cars.")

    lines += [
        "Grid is 10x10 (x=0..9, y=0..9). x increases RIGHT, y increases DOWN.",
        "Rules: no diagonals; max 2 cells/turn; if you moved at speed 2 you cannot stop instantly (brake to 1 first).",
        f"Good progress moves for car {car_id}, in priority order: {_format_candidate_moves(state, car_id)}.",
        "Choose the move that makes progress toward your goal while avoiding collisions.",
        "Reply with exactly one token: UP DOWN LEFT RIGHT UP2 DOWN2 LEFT2 RIGHT2 WAIT",
    ]
    return "\n".join(lines)


def _parse_move(text: str) -> Move:
    """Extract the first recognized legal move token from an LLM reply."""
    upper = text.upper()
    for direction, magnitude in re.findall(
        r"\b(UP|DOWN|LEFT|RIGHT|WAIT)(?:[\s_-]*(2|TWO))?\b",
        upper,
    ):
        if direction == "WAIT":
            return (0, 0)
        if magnitude:
            return _TWO_STEP[direction]
        return _ONE_STEP[direction]
    return (0, 0)


def _candidate_moves_toward_goal(state: State, car_id: str) -> list[Move]:
    me = next(v for v in state.vehicles if v.id == car_id)
    x, y = me.pos
    gx, gy = me.goal
    candidates: list[Move] = []

    if x != gx:
        step = 1 if gx > x else -1
        distance = abs(gx - x)
        if distance >= 2:
            candidates.append((2 * step, 0))
        candidates.append((step, 0))

    if y != gy:
        step = 1 if gy > y else -1
        distance = abs(gy - y)
        if distance >= 2:
            candidates.append((0, 2 * step))
        candidates.append((0, step))

    if (0, 0) not in candidates:
        candidates.append((0, 0))

    for move in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        if move not in candidates:
            candidates.append(move)
    return candidates


def _move_to_token(move: Move) -> str:
    return {
        (0, -1): "UP",
        (0, -2): "UP2",
        (0, 1): "DOWN",
        (0, 2): "DOWN2",
        (-1, 0): "LEFT",
        (-2, 0): "LEFT2",
        (1, 0): "RIGHT",
        (2, 0): "RIGHT2",
        (0, 0): "WAIT",
    }.get(move, "WAIT")


def _format_candidate_moves(state: State, car_id: str) -> str:
    return ", ".join(_move_to_token(move) for move in _candidate_moves_toward_goal(state, car_id)[:4])


def _effective_move_for_vehicle(state: State, car_id: str, move: Move) -> Move:
    vehicle = next(v for v in state.vehicles if v.id == car_id)
    if vehicle.speed != 2 or max(abs(move[0]), abs(move[1])) != 0:
        return move

    dx, dy = vehicle.last_move
    if dx > 0:
        return (1, 0)
    if dx < 0:
        return (-1, 0)
    if dy > 0:
        return (0, 1)
    if dy < 0:
        return (0, -1)
    return move


def _cells_crossed(pos: tuple[int, int], move: Move) -> list[tuple[int, int]]:
    dx, dy = move
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return [pos]
    sx = 0 if dx == 0 else dx // abs(dx)
    sy = 0 if dy == 0 else dy // abs(dy)
    return [(pos[0] + sx * i, pos[1] + sy * i) for i in range(1, steps + 1)]


def _fallback_move(state: State, car_id: str) -> Move:
    """Deterministic no-LLM policy used for tests, demos, and backend outages."""
    me = next((v for v in state.vehicles if v.id == car_id), None)
    if me is None or me.arrived:
        return (0, 0)

    active = [v for v in state.vehicles if not v.arrived]
    priority = sorted(v.id for v in active)
    my_rank = priority.index(car_id) if car_id in priority else len(priority)
    higher_priority = {v.id for v in active if priority.index(v.id) < my_rank}

    occupied = {v.pos for v in active if v.id != car_id}
    reserved: set[tuple[int, int]] = set()
    for other in active:
        if other.id not in higher_priority:
            continue
        other_move = _effective_move_for_vehicle(
            state,
            other.id,
            _candidate_moves_toward_goal(state, other.id)[0],
        )
        reserved.update(_cells_crossed(other.pos, other_move))

    for move in _candidate_moves_toward_goal(state, car_id):
        effective_move = _effective_move_for_vehicle(state, car_id, move)
        target = (me.pos[0] + effective_move[0], me.pos[1] + effective_move[1])
        if not (0 <= target[0] < 10 and 0 <= target[1] < 10):
            continue
        crossed = _cells_crossed(me.pos, effective_move)
        if any(cell in occupied for cell in crossed):
            continue
        if higher_priority and any(cell in reserved for cell in crossed):
            continue
        return move
    return (0, 0)


def _distance_to_goal(state: State, car_id: str, move: Move) -> int:
    me = next(v for v in state.vehicles if v.id == car_id)
    effective_move = _effective_move_for_vehicle(state, car_id, move)
    target = (me.pos[0] + effective_move[0], me.pos[1] + effective_move[1])
    goal = me.goal
    return abs(goal[0] - target[0]) + abs(goal[1] - target[1])


def _is_safe_move(state: State, car_id: str, move: Move) -> bool:
    me = next((v for v in state.vehicles if v.id == car_id), None)
    if me is None or me.arrived:
        return move == (0, 0)

    effective_move = _effective_move_for_vehicle(state, car_id, move)
    target = (me.pos[0] + effective_move[0], me.pos[1] + effective_move[1])
    if not (0 <= target[0] < 10 and 0 <= target[1] < 10):
        return False

    crossed = _cells_crossed(me.pos, effective_move)
    active = [v for v in state.vehicles if not v.arrived]
    occupied = {v.pos for v in active if v.id != car_id}
    if any(cell in occupied for cell in crossed):
        return False

    priority = sorted(v.id for v in active)
    my_rank = priority.index(car_id) if car_id in priority else len(priority)
    higher_priority = [v for v in active if priority.index(v.id) < my_rank]
    reserved: set[tuple[int, int]] = set()
    for other in higher_priority:
        other_move = _effective_move_for_vehicle(
            state,
            other.id,
            _candidate_moves_toward_goal(state, other.id)[0],
        )
        reserved.update(_cells_crossed(other.pos, other_move))
    return not any(cell in reserved for cell in crossed)


def _validate_or_correct_move(state: State, car_id: str, move: Move) -> Move:
    """Keep model choices inside the traffic rules before the environment step."""
    current_distance = _distance_to_goal(state, car_id, (0, 0))
    move_distance = _distance_to_goal(state, car_id, move)
    if _is_safe_move(state, car_id, move) and move_distance <= current_distance:
        return move
    return _fallback_move(state, car_id)


def choose_move(state: State, car_id: str) -> Move:
    """Return a legal ``(dx, dy)`` for ``car_id`` without mutating state."""
    me = next((v for v in state.vehicles if v.id == car_id), None)
    if me is None or me.arrived:
        return (0, 0)

    if not llm.is_available():
        return _fallback_move(state, car_id)

    prompt = _build_prompt(state, car_id)
    try:
        reply = llm.complete(prompt, system=_SYSTEM, max_tokens=16)
        return _validate_or_correct_move(state, car_id, _parse_move(reply))
    except Exception:
        return _fallback_move(state, car_id)
