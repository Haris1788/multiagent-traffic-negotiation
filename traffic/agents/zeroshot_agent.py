"""Zero-shot driving agent (BASELINE).  OWNER: R3 (Agent Dev A).

Feeds the LLM its position, goal, and other cars, and asks directly for a move.
"""
from __future__ import annotations

from traffic.environment import Move, State


def choose_move(state: State, car_id: str) -> Move:
    """Return (dx, dy) for this car.

    TODO(R3): build a prompt with the grid state and call src.llm.complete, then parse
    the move. Placeholder: greedily step one cell toward the goal.
    """
    me = next(v for v in state.vehicles if v.id == car_id)
    gx, gy = me.goal
    x, y = me.pos
    if x != gx:
        return (1 if gx > x else -1, 0)
    if y != gy:
        return (0, 1 if gy > y else -1)
    return (0, 0)
