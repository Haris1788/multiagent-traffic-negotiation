# CONTRACT.md — the frozen interface

This is the **stable boundary** every role builds against. The signatures and shapes
below are **frozen**: R2/R3/R4/R5 can change *implementations* freely, but **not** these
types or function signatures without a team agreement (open an issue, ping R1).

If your code only touches the things in this file, your work and your teammates' work
compose without breaking each other. That's the whole point.

> Status: tests green (`python -m pytest` → 5 passed). Agents are still stubs; the
> contract holds regardless of whether the body is a stub or a real LLM call.

---

## The four frozen items

### 1. `Vehicle` — one car
```python
@dataclass
class Vehicle:
    id: str                  # "A", "B", ...
    pos: tuple[int, int]     # (x, y) on the grid, 0..size-1
    goal: tuple[int, int]    # (x, y) exit cell
    speed: int = 0           # cells moved last turn (for the braking rule)
    arrived: bool = False    # True once pos == goal
```

### 2. `State` — the whole world at one turn
```python
@dataclass
class State:
    vehicles: list[Vehicle]  # all cars, including arrived ones
    turn: int = 0            # turns elapsed
```

### 3. `Move` — what an agent decides
```python
Move = tuple[int, int]  # (dx, dy); magnitude 0, 1, or 2 along a SINGLE axis (no diagonals)
```
Valid examples: `(0, 0)` wait · `(1, 0)` east · `(-2, 0)` west fast · `(0, 1)` north.
Invalid: `(1, 1)` diagonal · `(3, 0)` too fast. The environment clamps to the grid; it is
the agent's job to return a legal `Move`.

### 4. `choose_move` — the agent entry point (the swappable unit)
```python
def choose_move(state: State, car_id: str) -> Move: ...
```
- **One function per approach**, same signature, in `traffic/agents/<name>_agent.py`.
- Pure decision: read `state`, return a `Move` for `car_id`. **Do not mutate `state`.**
- `zeroshot` and `react` are the two implementations being compared. A new variant =
  a new file exposing this exact signature, registered in `traffic/evaluate.py`.

These four — `Vehicle`, `State`, `Move`, `choose_move` — **never change shape.**

---

## Supporting interfaces (stable, owned)

These are the seams between roles. Treat them as stable; coordinate before changing.

| What | Signature | Owner |
|------|-----------|-------|
| Environment | `GridIntersection(size=10, vehicles=None, max_turns=30)` | R2 |
| Advance one turn | `env.step(moves: dict[str, Move]) -> State` | R2 |
| Reset | `env.reset() -> State` · `env.is_done()` · `env.all_arrived()` | R2 |
| Run one episode | `run_episode(env, agent_fn) -> dict` | R1 |
| Aggregate N runs | `evaluate(approach: str, episodes: int) -> dict` | R5 |

**Agent function type** (what `run_episode`/`evaluate` consume):
```python
AgentFn = Callable[[State, str], Move]   # exactly choose_move's signature
```

**`run_episode` returns** (per-episode row):
```python
{"success": bool, "collisions": int, "steps": int, "path": list[dict[str, tuple[int,int]]]}
```

**`evaluate` returns** (aggregate row — keep keys stable so R5 can plot):
```python
{"approach": str, "episodes": int, "success_rate": float,
 "avg_collisions": float, "avg_steps": float}
```

---

## Rules of the road (so implementations stay compatible)

1. **The environment is LLM-free and deterministic.** It's the experiment's control —
   no model calls in `environment.py`. (R2)
2. **Agents never mutate `state`.** Read it, return a `Move`. The env applies moves. (R3/R4)
3. **Every approach is reachable through `evaluate.py`** and logs the metric row above. (R5)
4. **One agent per file**, exposing `choose_move(state, car_id) -> Move`. Register it in
   `APPROACHES` in `traffic/evaluate.py`.
5. **Changing a frozen signature is a team decision.** Ping R1 first.

## Prove it runs (anyone, anytime)
```powershell
python -m pytest                                  # → green
python -m traffic.evaluate --list                 # lists approaches
python -m traffic.evaluate --approach react --episodes 50
```
