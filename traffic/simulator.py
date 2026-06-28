"""Episode runner.  OWNER: R1 (Integration Lead).

Drives the environment with a given agent function and logs the outcome.
"""
from __future__ import annotations

from typing import Callable

from traffic.environment import GridIntersection, Move, State

AgentFn = Callable[[State, str], Move]


def run_episode(env: GridIntersection, agent_fn: AgentFn) -> dict:
    """Run one episode to completion and return outcome metrics."""
    env.reset()
    path: list[dict[str, tuple[int, int]]] = []
    while not env.is_done():
        moves = {v.id: agent_fn(env.state, v.id)
                 for v in env.state.vehicles if not v.arrived}
        env.step(moves)
        path.append({v.id: v.pos for v in env.state.vehicles})
    return {
        "success": env.all_arrived(),
        "collisions": env.collisions,
        "steps": env.state.turn,
        "path": path,
    }
