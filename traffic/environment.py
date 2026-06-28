"""Grid intersection environment.  OWNER: R2 (Environment Engineer).

A 10x10 grid with a 4-way intersection. Vehicles have a position, a goal exit, and
simple kinematics (move 1-2 cells/turn, no diagonals, 1 turn to brake).
This is pure Python + numpy — no LLM here, so it's fully testable on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Vehicle:
    id: str
    pos: tuple[int, int]
    goal: tuple[int, int]
    speed: int = 0          # cells moved last turn (for the braking rule)
    arrived: bool = False


@dataclass
class State:
    vehicles: list[Vehicle]
    turn: int = 0


Move = tuple[int, int]  # (dx, dy); magnitude 0,1,2 along a single axis


class GridIntersection:
    """TODO(R2): full kinematics + collision rules. Skeleton keeps it runnable."""

    def __init__(self, size: int = 10, vehicles: list[Vehicle] | None = None,
                 max_turns: int = 30) -> None:
        self.size = size
        self.max_turns = max_turns
        self._initial = vehicles or self._default_vehicles()
        self.state = State(vehicles=[Vehicle(**v.__dict__) for v in self._initial])
        self.collisions = 0

    @staticmethod
    def _default_vehicles() -> list[Vehicle]:
        return [
            Vehicle(id="A", pos=(0, 5), goal=(9, 5)),
            Vehicle(id="B", pos=(5, 0), goal=(5, 9)),
        ]

    def reset(self) -> State:
        self.state = State(vehicles=[Vehicle(**v.__dict__) for v in self._initial])
        self.collisions = 0
        return self.state

    def step(self, moves: dict[str, Move]) -> State:
        """Apply one move per car. TODO(R2): enforce speed/brake rules + detect crashes."""
        occupied: dict[tuple[int, int], str] = {}
        for v in self.state.vehicles:
            if v.arrived:
                continue
            dx, dy = moves.get(v.id, (0, 0))
            nx = max(0, min(self.size - 1, v.pos[0] + dx))
            ny = max(0, min(self.size - 1, v.pos[1] + dy))
            v.pos = (nx, ny)
            if v.pos in occupied:
                self.collisions += 1   # TODO(R2): refine collision semantics
            occupied[v.pos] = v.id
            if v.pos == v.goal:
                v.arrived = True
        self.state.turn += 1
        return self.state

    def is_done(self) -> bool:
        return all(v.arrived for v in self.state.vehicles) or self.state.turn >= self.max_turns

    def all_arrived(self) -> bool:
        return all(v.arrived for v in self.state.vehicles)
