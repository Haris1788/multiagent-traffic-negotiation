"""ReAct driving agent (COMPARISON).  OWNER: R4 (Agent Dev B).

Forces a Thought -> Action -> Observation trace so the model reasons about collisions
("Car B is at (4,5) moving north; if I move to (4,6) we collide; I must wait/brake")
before committing to a move.
"""
from __future__ import annotations

from traffic.agents.zeroshot_agent import choose_move as _greedy
from traffic.environment import Move, State


def choose_move(state: State, car_id: str) -> Move:
    """Return (dx, dy) chosen via an explicit reasoning loop.

    TODO(R4): implement the ReAct prompt (Thought/Action/Observation), call
    src.llm.complete, parse the final Action into a move, and keep the trace for
    explainability. Placeholder defers to the greedy baseline so the sim runs.
    """
    # TODO(R4): replace with real ReAct reasoning.
    return _greedy(state, car_id)
