# Topic 2 — Multi-Agent Traffic Negotiation

**Goal:** Put 2–4 LLM-driven vehicles on a grid intersection and see whether they can
negotiate movement **without colliding**. Compare plain **zero-shot** prompting against
a structured **ReAct** (Thought → Action → Observation) loop.

## The research question

> Does explicit reasoning (ReAct) let LLMs handle spatial constraints they fail at
> when answering instinctively (zero-shot)?

## Loop

```text
for each turn:
   env state (positions, goals) ─▶ each car's agent picks a move
   moves ─▶ environment applies kinematics, detects collisions ─▶ new state
until all cars exit OR max turns reached
   ─▶ Evaluation (R5): success rate, collisions, time-steps
```

## Files & owners (work in parallel)

| File | Owner | Interface (contract) |
|------|-------|----------------------|
| `environment.py` | **R2** | `class GridIntersection`: `reset()`, `step(moves) -> State`, `is_done()`, `collisions`. |
| `agents/zeroshot_agent.py` | **R3** | `choose_move(state, car_id) -> Move`. |
| `agents/react_agent.py` | **R4** | `choose_move(state, car_id) -> Move` via Thought/Action/Observation. |
| `simulator.py` | **R1** | `run_episode(env, agent_fn) -> dict{success, collisions, steps, path}`. |
| `evaluate.py` | **R5** | approaches: `zeroshot`, `react`; aggregate over 50–100 episodes. |

## Approaches to compare (the experiment)

1. **zeroshot** — ask the LLM directly for the next move.
2. **react** — force a Thought→Action→Observation reasoning trace before moving.

## Metrics (R5)

- **Success rate** — all cars reach their exit.
- **Collision frequency** — collisions per episode.
- **Average time-steps** — efficiency.
- Plot a few chosen paths to visualize kinematic feasibility.

## Run

```powershell
python -m traffic.evaluate --list
python -m traffic.evaluate --approach react --episodes 50
```

> Optional: `pip install matplotlib pygame` (commented in requirements) for path plots
> / a live visualizer.
