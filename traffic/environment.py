"""Grid intersection environment.  OWNER: R2 (Environment Engineer).

A 10x10 grid with a 4-way signaled intersection. Four vehicles approach from
north, south, east and west, each with its own turn intent (straight / left /
right). Movement is simultaneous each turn subject to:

  * a traffic-light phase (NS green / EW green, with an all-red yellow buffer)
  * right-of-way rules (red light must stop; on green, straight and
    right-turning traffic has priority over left turns; left turns must yield)
  * variable speed with a one-turn braking/reaction delay (cannot jump from
    full speed to a dead stop in a single turn)

This is pure Python + numpy — no LLM here, so it's fully testable on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

Move = tuple[int, int]  # (dx, dy); magnitude 0,1,2 along a single axis
Turn = str              # "STRAIGHT" | "LEFT" | "RIGHT"

# Compass approach a vehicle starts from (i.e. the direction it is travelling).
Direction = str  # "N" | "S" | "E" | "W"   (N = travelling south->north etc. — see below)

_DIRECTION_VECTOR: dict[Direction, Move] = {
    "N": (0, -1),   # approaches from the south edge, heads north (up)
    "S": (0, 1),    # approaches from the north edge, heads south (down)
    "E": (1, 0),    # approaches from the west edge, heads east (right)
    "W": (-1, 0),   # approaches from the east edge, heads west (left)
}

# Which light axis controls a given approach direction.
_AXIS_OF: dict[Direction, str] = {"N": "NS", "S": "NS", "E": "EW", "W": "EW"}

_LEFT_OF: dict[Direction, Direction] = {"N": "W", "W": "S", "S": "E", "E": "N"}
_RIGHT_OF: dict[Direction, Direction] = {"N": "E", "E": "S", "S": "W", "W": "N"}


@dataclass
class Vehicle:
    id: str
    pos: tuple[int, int]
    goal: tuple[int, int]
    direction: Direction = "E"     # compass direction of travel on approach
    turn: Turn = "STRAIGHT"        # intended maneuver at the intersection
    speed: int = 0                 # cells moved last turn (for the braking rule)
    last_move: Move = (0, 0)
    arrived: bool = False
    waiting_turns: int = 0         # consecutive turns stuck at speed 0 (for fairness/logging)


@dataclass
class State:
    vehicles: list[Vehicle]
    turn: int = 0
    light_phase: str = "NS"        # "NS" green, "EW" green, or "ALL_RED" (yellow buffer)
    phase_timer: int = 0           # turns remaining in the current phase


# ── traffic light schedule ──────────────────────────────────────────────────
GREEN_DURATION = 6
ALL_RED_DURATION = 2

# 90-degree turn of a heading vector: left = counter-clockwise, right = clockwise
# (screen coords: +x is east/right, +y is south/down)
def _turn_heading(heading: Move, turn: Turn) -> Move:
    hx, hy = heading
    if turn == "LEFT":
        return (hy, -hx)
    if turn == "RIGHT":
        return (-hy, hx)
    return heading


def _goal_for(pos: tuple[int, int], direction: Direction, turn: Turn, size: int = 10) -> tuple[int, int]:
    """The exit cell reached by crossing the intersection and (optionally) turning.

    The car travels straight from *pos* until it reaches the lane it will
    occupy through the intersection box, then exits along the (possibly
    turned) heading to the far edge of the grid.
    """
    cx = cy = size // 2
    x, y = pos
    heading = _DIRECTION_VECTOR[direction]
    exit_heading = _turn_heading(heading, turn)

    # the lane coordinate the car ends up travelling along once it turns
    # (its own approach coordinate stays fixed until the turn point)
    if heading[0] == 0:      # approaching vertically (N/S) -> x is fixed pre-turn
        turn_point = (x, cy)
    else:                    # approaching horizontally (E/W) -> y is fixed pre-turn
        turn_point = (cx, y)

    ex, ey = exit_heading
    if ex != 0:
        return (0 if ex < 0 else size - 1, turn_point[1])
    return (turn_point[0], 0 if ey < 0 else size - 1)


def _assign_goals(vehicles: list["Vehicle"], size: int = 10) -> None:
    """Compute each vehicle's goal from its (pos, direction, turn), nudging
    any goal that collides with an already-assigned one onto the next free
    cell along the same outbound edge — so no two cars ever share a goal."""
    taken: set[tuple[int, int]] = set()
    for v in vehicles:
        goal = _goal_for(v.pos, v.direction, v.turn, size)
        gx, gy = goal
        # walk along whichever edge the goal sits on until a free cell is found
        if gx in (0, size - 1):
            step = 1 if gy < size // 2 else -1
            while goal in taken:
                gy += step
                goal = (gx, gy)
        else:
            step = 1 if gx < size // 2 else -1
            while goal in taken:
                gx += step
                goal = (gx, gy)
        taken.add(goal)
        v.goal = goal


def _urban_four_way() -> list["Vehicle"]:
    mid = 4
    vehicles = [
        Vehicle(id="N", pos=(mid, 9), goal=(0, 0), direction="N", turn="STRAIGHT"),
        Vehicle(id="S", pos=(mid + 1, 0), goal=(0, 0), direction="S", turn="STRAIGHT"),
        Vehicle(id="E", pos=(0, mid), goal=(0, 0), direction="E", turn="LEFT"),
        Vehicle(id="W", pos=(9, mid + 1), goal=(0, 0), direction="W", turn="RIGHT"),
    ]
    _assign_goals(vehicles)
    return vehicles


def _rush_hour_turns() -> list["Vehicle"]:
    """Every car turns (no one goes straight) — heavier negotiation load."""
    mid = 4
    vehicles = [
        Vehicle(id="N", pos=(mid, 9), goal=(0, 0), direction="N", turn="LEFT"),
        Vehicle(id="S", pos=(mid + 1, 0), goal=(0, 0), direction="S", turn="LEFT"),
        Vehicle(id="E", pos=(0, mid), goal=(0, 0), direction="E", turn="LEFT"),
        Vehicle(id="W", pos=(9, mid + 1), goal=(0, 0), direction="W", turn="LEFT"),
    ]
    _assign_goals(vehicles)
    return vehicles


def _all_straight() -> list["Vehicle"]:
    """Everyone crosses straight through — a simple baseline scenario."""
    mid = 4
    vehicles = [
        Vehicle(id="N", pos=(mid, 9), goal=(0, 0), direction="N", turn="STRAIGHT"),
        Vehicle(id="S", pos=(mid + 1, 0), goal=(0, 0), direction="S", turn="STRAIGHT"),
        Vehicle(id="E", pos=(0, mid), goal=(0, 0), direction="E", turn="STRAIGHT"),
        Vehicle(id="W", pos=(9, mid + 1), goal=(0, 0), direction="W", turn="STRAIGHT"),
    ]
    _assign_goals(vehicles)
    return vehicles


# Named starting configurations the GUI (and evaluation code) can pick from.
SCENARIOS: dict[str, "callable"] = {
    "Urban Four-Way": _urban_four_way,
    "Rush Hour (all turning)": _rush_hour_turns,
    "Straight Crossing": _all_straight,
}


class GridIntersection:
    """4-way signaled intersection with right-of-way and reaction-delay kinematics."""

    def __init__(self, size: int = 10, vehicles: list[Vehicle] | None = None,
                 max_turns: int = 40, scenario: str | None = None) -> None:
        self.size = size
        self.max_turns = max_turns
        if vehicles is not None:
            self._initial = vehicles
        elif scenario is not None:
            self._initial = SCENARIOS[scenario]()
        else:
            self._initial = self._default_vehicles()
        self.scenario = scenario or "Urban Four-Way"
        self.state = self._fresh_state()
        self.collisions = 0
        self.near_misses = 0
        self.red_light_violations = 0

    def _fresh_state(self) -> State:
        return State(
            vehicles=[Vehicle(**v.__dict__) for v in self._initial],
            turn=0,
            light_phase="NS",
            phase_timer=GREEN_DURATION,
        )

    @staticmethod
    def _default_vehicles() -> list[Vehicle]:
        return SCENARIOS["Urban Four-Way"]()

    def reset(self) -> State:
        self.state = self._fresh_state()
        self.collisions = 0
        self.near_misses = 0
        self.red_light_violations = 0
        return self.state

    # ── light phase ──────────────────────────────────────────────────────
    def _advance_light(self) -> None:
        self.state.phase_timer -= 1
        if self.state.phase_timer > 0:
            return
        if self.state.light_phase == "NS":
            self.state.light_phase = "ALL_RED"
            self.state.phase_timer = ALL_RED_DURATION
        elif self.state.light_phase == "EW":
            self.state.light_phase = "ALL_RED"
            self.state.phase_timer = ALL_RED_DURATION
        else:  # coming out of ALL_RED, alternate which axis gets green next
            self._next_green = "EW" if getattr(self, "_next_green", "EW") == "NS" else "NS"
            self.state.light_phase = self._next_green
            self.state.phase_timer = GREEN_DURATION

    def light_for(self, direction: Direction) -> str:
        """Return 'GREEN', 'RED' for the given approach direction's axis."""
        axis = _AXIS_OF[direction]
        if self.state.light_phase == "ALL_RED":
            return "RED"
        return "GREEN" if self.state.light_phase == axis else "RED"

    def _in_intersection_box(self, pos: tuple[int, int]) -> bool:
        cx = cy = self.size // 2
        return abs(pos[0] - cx) <= 1 and abs(pos[1] - cy) <= 1

    def _stop_line(self, v: Vehicle) -> tuple[int, int]:
        """The last cell before the intersection box on v's approach."""
        cx = cy = self.size // 2
        if v.direction == "N":
            return (v.pos[0], cy + 2)
        if v.direction == "S":
            return (v.pos[0], cy - 2)
        if v.direction == "E":
            return (cx - 2, v.pos[1])
        return (cx + 2, v.pos[1])

    def _is_valid_move(self, move: Move) -> bool:
        """Allow only 0, 1, or 2 cells along a single axis."""
        dx, dy = move
        if dx != 0 and dy != 0:
            return False
        if abs(dx) > 2 or abs(dy) > 2:
            return False
        return True

    def _apply_braking(self, vehicle: Vehicle, move: Move) -> Move:
        """A vehicle at speed 2 cannot stop instantly; it must brake to speed 1 first."""
        requested_speed = max(abs(move[0]), abs(move[1]))
        if vehicle.speed == 2 and requested_speed == 0:
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

    def _enforce_red_light(self, vehicle: Vehicle, move: Move) -> Move:
        """A vehicle facing red may not enter or cross the intersection box."""
        if self.light_for(vehicle.direction) == "GREEN":
            return move
        if self._in_intersection_box(vehicle.pos):
            # already committed to crossing; let it clear rather than stall mid-box
            return move
        dx, dy = move
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return move
        sx = 0 if dx == 0 else dx // abs(dx)
        sy = 0 if dy == 0 else dy // abs(dy)
        for i in range(1, steps + 1):
            cell = (vehicle.pos[0] + sx * i, vehicle.pos[1] + sy * i)
            if self._in_intersection_box(cell):
                self.red_light_violations += 1
                return self._apply_braking(vehicle, (0, 0))
        return move

    def step(self, moves: dict[str, Move]) -> State:
        """
        Simultaneous-move step:
        1. Validate + apply braking + enforce red lights
        2. Compute all proposed positions first (NO updates yet)
        3. Resolve collisions (including cell-swaps / path-crossing)
        4. Commit final state, advance the light schedule
        """
        proposed_positions: dict[str, tuple[int, int]] = {}
        proposed_moves: dict[str, Move] = {}
        proposed_paths: dict[str, list[tuple[int, int]]] = {}
        occupied_targets: dict[tuple[int, int], list[str]] = {}

        for v in self.state.vehicles:
            if v.arrived:
                continue

            move = moves.get(v.id, (0, 0))
            if not self._is_valid_move(move):
                move = (0, 0)
            move = self._apply_braking(v, move)
            move = self._enforce_red_light(v, move)

            dx, dy = move
            nx = max(0, min(self.size - 1, v.pos[0] + dx))
            ny = max(0, min(self.size - 1, v.pos[1] + dy))

            path = self._cells_crossed(v.pos, (nx - v.pos[0], ny - v.pos[1]))
            proposed_positions[v.id] = (nx, ny)
            proposed_moves[v.id] = move
            proposed_paths[v.id] = path
            occupied_targets.setdefault((nx, ny), []).append(v.id)

        # -----------------------------
        # 2. Resolve head-on / same-cell collisions
        # -----------------------------
        final_positions: dict[str, tuple[int, int]] = {}
        collided: set[str] = set()

        for pos, ids in occupied_targets.items():
            if len(ids) > 1:
                self.collisions += 1
                for vid in ids:
                    collided.add(vid)

        # -----------------------------
        # 2b. Resolve path-crossing collisions (two cars swap cells or cross
        #      through the same cell mid-move without landing on it together)
        # -----------------------------
        active_ids = [v.id for v in self.state.vehicles if not v.arrived]
        for i, id_a in enumerate(active_ids):
            for id_b in active_ids[i + 1:]:
                if id_a in collided and id_b in collided:
                    continue
                path_a = set(proposed_paths.get(id_a, []))
                path_b = set(proposed_paths.get(id_b, []))
                if path_a & path_b:
                    self.collisions += 1
                    collided.add(id_a)
                    collided.add(id_b)

        for v in self.state.vehicles:
            if v.arrived:
                continue
            if v.id in collided:
                final_positions[v.id] = v.pos
            else:
                final_positions[v.id] = proposed_positions[v.id]

        # -----------------------------
        # 2c. Near-miss tracking (adjacent cells inside the intersection box)
        # -----------------------------
        box_positions = [
            final_positions[v.id] for v in self.state.vehicles
            if not v.arrived and self._in_intersection_box(final_positions[v.id])
        ]
        for i, pa in enumerate(box_positions):
            for pb in box_positions[i + 1:]:
                if pa != pb and abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]) == 1:
                    self.near_misses += 1

        # -----------------------------
        # 3. Commit state update
        # -----------------------------
        for v in self.state.vehicles:
            if v.arrived:
                continue

            if v.id in collided:
                move = (0, 0)
            else:
                move = proposed_moves.get(v.id, (0, 0))

            v.pos = final_positions[v.id]
            dx, dy = move
            v.last_move = move
            v.speed = max(abs(dx), abs(dy))
            v.waiting_turns = 0 if v.speed > 0 else v.waiting_turns + 1

            if v.pos == v.goal:
                v.arrived = True

        self.state.turn += 1
        self._advance_light()
        return self.state

    @staticmethod
    def _cells_crossed(pos: tuple[int, int], move: Move) -> list[tuple[int, int]]:
        dx, dy = move
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return [pos]
        sx = 0 if dx == 0 else dx // abs(dx)
        sy = 0 if dy == 0 else dy // abs(dy)
        return [(pos[0] + sx * i, pos[1] + sy * i) for i in range(1, steps + 1)]

    def is_done(self) -> bool:
        return all(v.arrived for v in self.state.vehicles) or self.state.turn >= self.max_turns

    def all_arrived(self) -> bool:
        return all(v.arrived for v in self.state.vehicles)
