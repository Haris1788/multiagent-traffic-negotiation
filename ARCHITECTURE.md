# Architecture

A single reference for the **whole system** — written to be presented to the team and
to the professor. It explains *what we are building, why it's shaped this way, and how
every piece fits together.*

**Project:** Multi-Agent Traffic Negotiation Simulator — **ReAct vs. Zero-Shot**.

---

## 1. The question and the design philosophy

**Research question:** does explicit step-by-step reasoning (**ReAct**) let an LLM
handle spatial constraints and avoid collisions that it fails at when answering
instinctively (**zero-shot**)?

To answer it credibly, the architecture makes **the reasoning strategy a variable**
and holds everything else fixed:

1. **One environment, swappable brains.** The grid, kinematics, and collision rules
   are identical for both approaches. Only the agent's `choose_move` differs.
2. **Everything is measured.** Both approaches run through the same evaluation harness
   and emit the same metric rows, so "ReAct is better" becomes a number, not a claim.
3. **The world has no LLM.** The environment is pure Python, so it's deterministic,
   fast, and fully testable offline — the LLM only ever decides a *move*.

This is exactly what the rubric rewards: a multi-agent system **plus** a comparison of
approaches with a real evaluation.

---

## 2. The components

```text
┌──────────────────────────── EVALUATION (R5) ────────────────────────────┐
│ traffic/evaluate.py — run N episodes per approach, aggregate metrics      │
│ success rate · collision frequency · average time-steps  → results/        │
└───────────────────────────────▲──────────────────────────────────────────┘
                                 │ run_episode(env, agent_fn) -> outcome
┌───────────────────────────── SIMULATOR (R1) ─────────────────────────────┐
│ traffic/simulator.py — turn loop: ask each car, apply moves, log the path  │
│ (upgrades to a LangGraph state graph once agents call the LLM)             │
└───────▲───────────────────────────────────────────────────▲──────────────┘
        │ choose_move(state, car_id) -> Move                 │ step(moves) -> State
┌───────┴──────────── AGENTS (R3 / R4) ──────────┐  ┌────────┴──── WORLD (R2) ────────┐
│ agents/zeroshot_agent.py  (baseline)           │  │ traffic/environment.py           │
│ agents/react_agent.py     (Thought/Action/Obs) │  │ GridIntersection: positions,     │
│ both call the LLM via src/llm.py               │  │ goals, kinematics, collisions    │
└────────────────────────────────────────────────┘  │ (pure Python + numpy, no LLM)    │
                                                      └──────────────────────────────────┘
```

| Component | File | Owner | Responsibility |
|-----------|------|-------|----------------|
| **World** | `traffic/environment.py` | R2 | `GridIntersection`: `reset()`, `step(moves) -> State`, `is_done()`, `all_arrived()`, `collisions`. Kinematics: move 1–2 cells/turn, no diagonals, 1 turn to brake. |
| **Baseline agent** | `traffic/agents/zeroshot_agent.py` | R3 | `choose_move(state, car_id) -> Move` from a single direct prompt. |
| **ReAct agent** | `traffic/agents/react_agent.py` | R4 | `choose_move(...)` via a Thought → Action → Observation trace before committing. |
| **Simulator** | `traffic/simulator.py` | R1 | `run_episode(env, agent_fn) -> {success, collisions, steps, path}`. |
| **Evaluation** | `traffic/evaluate.py` | R5 | `evaluate(approach, episodes)` → aggregated metrics; CLI to compare. |
| **LLM backend** | `src/llm.py`, `src/config.py` | shared | One `complete()` entry point to a free, vendor-neutral, OpenAI-compatible model (default local Ollama — no key, no cost). |

---

## 3. The agent contract (why parallel work is safe)

Both agents are interchangeable behind one signature:

```python
def choose_move(state: State, car_id: str) -> Move:   # Move = (dx, dy)
    ...
```

The simulator depends only on this signature, and the environment depends on nothing
above it. So R2 (world), R3 (baseline), R4 (ReAct), and R5 (evaluation) build
simultaneously; the stubs return valid moves today, so the simulator runs and the tests
pass before any LLM logic exists.

---

## 4. Data & control flow (one episode)

1. `simulator.run_episode` calls `env.reset()`.
2. Each turn: for every car not yet arrived, call `agent_fn(state, car_id)` to get a
   move. The agent (zero-shot or ReAct) builds a prompt from the state and calls the LLM
   via `src/llm.py`.
3. `env.step(moves)` applies the moves, enforces kinematics, and counts collisions.
4. Repeat until all cars exit or `max_turns` is hit.
5. The episode returns `{success, collisions, steps, path}`.
6. `evaluate.py` runs many episodes per approach and aggregates the metrics that go into
   the report and plots.

---

## 5. Technology choices & rationale

| Choice | Why |
|--------|-----|
| **Python + numpy** | Course standard; the grid/kinematics are simple array math. |
| **Pure-Python environment** | Deterministic, offline, instantly testable — the experiment's control. |
| **Free, vendor-neutral LLM backend** (OpenAI-compatible, e.g. local Ollama) | The model powering each car — one wrapper in `src/llm.py`, no key, no cost, no lock-in. |
| **LangGraph** | Clean state-graph orchestration for the ReAct loop once agents call the LLM. |
| **pandas + rich + pytest** | Metric aggregation, readable logs, fast tests. |

Optional extras (commented in `requirements.txt`): `matplotlib` for path/metric plots,
`pygame` for a live visualizer.

---

## 6. Repository map

```text
.
├── README.md            Overview + quickstart (audience: everyone)
├── ARCHITECTURE.md      This document (audience: team + professor)
├── TEAM_PLAN.md         Roles, weekly cadence, 1-month timeline
├── CONTRIBUTING.md      Setup + contribution workflow
├── src/                 Shared baseline (config, llm client)
├── traffic/             THE PROJECT (environment, agents, simulator, evaluation, tests)
├── eval/                Generic results helper
└── tests/               Root smoke tests
```

---

## 7. Design decisions & talking points

- **Stubs over emptiness.** Every agent ships as a working placeholder returning a valid
  move, so the simulator runs and tests pass before any LLM logic — progress is always
  demonstrable and integration never "big-bangs" at the end.
- **Baseline is a first-class, measured approach,** not an assumption — it's the control.
- **The environment never calls an LLM,** keeping the experiment deterministic and the
  cost at zero.
- **Linear simulator first, LangGraph later** — we add graph complexity only when the
  ReAct loop needs it; the `run_episode` interface stays the same.

---

## 8. Status

- ✅ Environment, both agents (stubs), simulator, evaluation, and tests build and pass.
- ✅ `python -m traffic.evaluate --approach zeroshot --episodes 50` produces metrics.
- ⏳ Next: real kinematics (R2), real LLM prompts for zero-shot (R3) and ReAct (R4), the
  50–100-episode comparison + plots (R5), and LangGraph orchestration (R1). See
  `TEAM_PLAN.md`.
