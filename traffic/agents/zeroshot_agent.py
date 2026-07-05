"""Zero-shot driving agent. OWNER: R3.

Builds a plain-language prompt describing the grid state — including the
traffic-light phase, each car's turn intent, and right-of-way — calls the LLM
once, and parses the returned move. If the model backend is unavailable, it
falls back to a deterministic collision- and light-aware policy so tests and
demos stay responsive.
"""
from __future__ import annotations

import re

from src import llm
from traffic.environment import GridIntersection, Move, State, Vehicle

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
    "You are a traffic-control agent for a 10x10 grid intersection with a real "
    "traffic light and four-way right-of-way rules. On every turn you receive the "
    "light phase, the positions of all cars, and your intended maneuver, and must "
    "choose one move for the assigned car. Reply with exactly one token from: "
    "UP DOWN LEFT RIGHT UP2 DOWN2 LEFT2 RIGHT2 WAIT. "
    "Use x/y coordinates exactly as described. No punctuation, no explanation."
)


def _required_direction(pos: tuple[int, int], goal: tuple[int, int]) -> str:
    """The only move that makes progress toward *goal* from *pos* (or WAIT)."""
    x, y = pos
    gx, gy = goal
    if gx == x and gy != y:
        return "DOWN" if gy > y else "UP"
    if gy == y and gx != x:
        return "RIGHT" if gx > x else "LEFT"
    return "WAIT"


def _light_line(env_or_state, car_id: str) -> str:
    """Describe the light phase for this car, tolerating a bare State (tests)."""
    if isinstance(env_or_state, GridIntersection):
        me = next(v for v in env_or_state.state.vehicles if v.id == car_id)
        phase = env_or_state.light_for(me.direction)
        return f"Traffic light for your approach ({me.direction}): {phase}."
    return "Traffic light: GREEN (unsignaled test scenario)."


def _build_prompt(state: State, car_id: str, env: GridIntersection | None = None) -> str:
    me = next(v for v in state.vehicles if v.id == car_id)
    others = [v for v in state.vehicles if v.id != car_id and not v.arrived]

    x, y = me.pos
    gx, gy = me.goal

    hints = []
    if gx > x:
        hints.append(f"your goal is to the RIGHT (need to increase x from {x} to {gx})")
    elif gx < x:
        hints.append(f"your goal is to the LEFT (need to decrease x from {x} to {gx})")
    if gy > y:
        hints.append(f"your goal is DOWN (need to increase y from {y} to {gy})")
    elif gy < y:
        hints.append(f"your goal is UP (need to decrease y from {y} to {gy})")
    if not hints:
        hints.append("you are already at your goal, reply WAIT")

    required_word = _required_direction((x, y), (gx, gy))

    lines = [
        f"Turn {state.turn}. You are car {car_id}, approaching from the {me.direction}, "
        f"intending to go {me.turn}.",
        f"Your position: column={x}, row={y}. Your goal: column={gx}, row={gy}.",
        f"Direction to goal: {' and '.join(hints)}.",
        f"The single word that moves you toward the goal is: {required_word}.",
        f"Your speed last turn: {me.speed}.",
    ]

    if env is not None:
        lines.append(_light_line(env, car_id))
        lines.append(
            "Rules: on RED you must stop before the intersection box; on GREEN, "
            "straight and right-turning traffic has right-of-way over left turns, "
            "so a left-turning car must yield to oncoming straight traffic."
        )
    else:
        lines.append(_light_line(state, car_id))

    if others:
        other_desc = "; ".join(
            f"car {v.id} ({v.direction}, turning {v.turn}) is at column={v.pos[0]} row={v.pos[1]}"
            for v in others
        )
        lines.append(f"Other active cars: {other_desc}. Avoid moving into their position or path.")
    else:
        lines.append("No other active cars on the grid.")

    lines += [
        "Grid is 10x10 (x=0..9, y=0..9). x increases RIGHT, y increases DOWN.",
        "Rules: no diagonals; max 2 cells/turn; if you moved at speed 2 you cannot stop instantly (brake to 1 first).",
        f"Good progress moves for car {car_id}, in priority order: {_format_candidate_moves(state, car_id)}.",
        "Choose the move that makes progress toward your goal while respecting the light and avoiding collisions.",
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


_TURN_RANK = {"STRAIGHT": 0, "RIGHT": 0, "LEFT": 1}  # left turns yield


def _right_of_way_rank(v: Vehicle) -> int:
    """Lower rank = higher priority. Straight/right beat left turns."""
    return _TURN_RANK.get(v.turn, 0)


def _in_intersection_box(pos: tuple[int, int], size: int = 10) -> bool:
    cx = cy = size // 2
    return abs(pos[0] - cx) <= 1 and abs(pos[1] - cy) <= 1


def _light_blocks(env: GridIntersection | None, state: State, car_id: str) -> bool:
    """True if this car must stop for a red light (i.e. it's not already committed)."""
    if env is None:
        return False
    me = next(v for v in state.vehicles if v.id == car_id)
    if _in_intersection_box(me.pos, env.size):
        return False
    return env.light_for(me.direction) == "RED"


def _fallback_move(state: State, car_id: str, env: GridIntersection | None = None) -> Move:
    """Deterministic no-LLM policy used for tests, demos, and backend outages."""
    me = next((v for v in state.vehicles if v.id == car_id), None)
    if me is None or me.arrived:
        return (0, 0)

    if _light_blocks(env, state, car_id):
        return _effective_move_for_vehicle(state, car_id, (0, 0)) if me.speed == 2 else (0, 0)

    active = [v for v in state.vehicles if not v.arrived]
    # Right-of-way first (straight/right before left), then car id for stable tie-break.
    priority = sorted((v.id for v in active),
                      key=lambda vid: (_right_of_way_rank(next(v for v in active if v.id == vid)), vid))
    my_rank = priority.index(car_id) if car_id in priority else len(priority)
    higher_priority = {vid for vid in priority[:my_rank]}

    occupied = {v.pos for v in active if v.id != car_id}
    reserved: set[tuple[int, int]] = set()
    for other in active:
        if other.id not in higher_priority:
            continue
        if _light_blocks(env, state, other.id):
            continue
        other_move = _effective_move_for_vehicle(
            state, other.id, _candidate_moves_toward_goal(state, other.id)[0],
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


def _is_safe_move(state: State, car_id: str, move: Move, env: GridIntersection | None = None) -> bool:
    me = next((v for v in state.vehicles if v.id == car_id), None)
    if me is None or me.arrived:
        return move == (0, 0)

    effective_move = _effective_move_for_vehicle(state, car_id, move)
    target = (me.pos[0] + effective_move[0], me.pos[1] + effective_move[1])
    if not (0 <= target[0] < 10 and 0 <= target[1] < 10):
        return False

    if _light_blocks(env, state, car_id) and effective_move != (0, 0):
        return False

    crossed = _cells_crossed(me.pos, effective_move)
    active = [v for v in state.vehicles if not v.arrived]
    occupied = {v.pos for v in active if v.id != car_id}
    if any(cell in occupied for cell in crossed):
        return False

    my_v = next(v for v in active if v.id == car_id)
    priority = sorted((v.id for v in active),
                      key=lambda vid: (_right_of_way_rank(next(v for v in active if v.id == vid)), vid))
    my_rank = priority.index(car_id) if car_id in priority else len(priority)
    higher_priority = [v for v in active if v.id in priority[:my_rank]]
    reserved: set[tuple[int, int]] = set()
    for other in higher_priority:
        if _light_blocks(env, state, other.id):
            continue
        other_move = _effective_move_for_vehicle(
            state, other.id, _candidate_moves_toward_goal(state, other.id)[0],
        )
        reserved.update(_cells_crossed(other.pos, other_move))
    return not any(cell in reserved for cell in crossed)


def _validate_or_correct_move(state: State, car_id: str, move: Move,
                               env: GridIntersection | None = None) -> Move:
    """Keep model choices inside the traffic rules before the environment step."""
    current_distance = _distance_to_goal(state, car_id, (0, 0))
    move_distance = _distance_to_goal(state, car_id, move)
    if _is_safe_move(state, car_id, move, env) and move_distance <= current_distance:
        return move
    return _fallback_move(state, car_id, env)


def choose_move(state: State, car_id: str, env: GridIntersection | None = None) -> Move:
    """Return a legal ``(dx, dy)`` for ``car_id`` without mutating state."""
    me = next((v for v in state.vehicles if v.id == car_id), None)
    if me is None or me.arrived:
        return (0, 0)

    if not llm.is_available():
        return _fallback_move(state, car_id, env)

    prompt = _build_prompt(state, car_id, env)
    try:
        reply = llm.complete(prompt, system=_SYSTEM, max_tokens=16)
        return _validate_or_correct_move(state, car_id, _parse_move(reply), env)
    except Exception:
        return _fallback_move(state, car_id, env)
