# 🚦 Multi-Agent Traffic Negotiation — ReAct vs. Zero-Shot

> **From Transformers to LLMs: Foundations of Generative AI** · TU Clausthal
> (Clausthal University of Technology) · Institute for Software and Systems
> Engineering (ISSE) — Machine Learning & Cognitive Software (CORE).

Can large language models **reason about space and avoid collisions**? We put 2–4
LLM-driven vehicles on a grid intersection and make them negotiate who goes when.
Then we test our core question:

> *Does forcing an LLM to reason step-by-step (**ReAct**: Thought → Action →
> Observation) let it handle spatial constraints it fails at when answering
> instinctively (**zero-shot**)?*

We answer it empirically — same environment, two driving strategies, measured over
50–100 simulated runs.

---

## 🎯 The experiment

| Approach | What the car's "brain" does | Role |
|----------|-----------------------------|------|
| **Zero-shot** (baseline) | Sees the grid + its goal, asked directly for its next move. | R3 |
| **ReAct** (comparison) | Reasons first — *"Car B is at (4,5) heading north; if I move to (4,6) we collide → I brake"* — then moves. | R4 |

**Metrics** (logged every run, compared at the end):

- **Success rate** — fraction of runs where all cars reach their exit.
- **Collision frequency** — collisions per run.
- **Average time-steps** — how efficiently they clear the intersection.
- Path plots to visualize a few representative runs.

---

## 🏗️ Architecture (short version)

```text
            ┌──────────────── simulator.py (R1) ────────────────┐
            │  loop: ask each car's agent → apply move → check  │
            │         collisions → repeat until done            │
            └───────▲───────────────────────────────▲──────────┘
                    │ choose_move(state, car_id)     │ step(moves)
            ┌───────┴────────┐               ┌───────┴───────────┐
            │  agents/ (R3/R4)│               │ environment.py (R2)│
            │ zeroshot · react│               │ grid + kinematics  │
            │   → call LLM    │               │ + collisions (pure │
            │  (src/llm.py)   │               │  Python, no LLM)   │
            └─────────────────┘               └────────────────────┘
                    │
            ┌───────▼──────────── evaluate.py (R5) ──────────────┐
            │  run N episodes per approach → aggregate metrics    │
            └─────────────────────────────────────────────────────┘
```

> 📐 Full design, data flow, and design trade-offs: **[`ARCHITECTURE.md`](ARCHITECTURE.md)**.

---

## 🚀 Quickstart

> Requires **Python 3.10+**. Windows commands shown; macOS/Linux notes inline.

```powershell
# 1. Create & activate the virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # macOS/Linux: source .venv/bin/activate

# 2. Install dependencies (all free, open-source)
pip install -r requirements.txt

# 3. Verify
python -m pytest                        # tests should pass
python -m traffic.evaluate --list       # -> zeroshot, react

# 4. Run a comparison
python -m traffic.evaluate --approach zeroshot --episodes 50
python -m traffic.evaluate --approach react    --episodes 50
```

**LLM backend (for the real agents):** free and local — no key, no cost. Install
[Ollama](https://ollama.com) once and pull a model (`ollama run llama3.1`); the agents
call it automatically. Override the endpoint/model via `.env` (see `.env.example`) if
you prefer a different one. The environment + tests run without any model.

---

## 📁 Repository layout

```text
.
├── README.md            ← you are here
├── ARCHITECTURE.md      ← full system architecture (team + professor)
├── TEAM_PLAN.md         ← 5 roles, weekly cadence, 1-month timeline
├── CONTRIBUTING.md      ← how to set up & contribute (read before coding)
├── requirements.txt     ← dependencies (all free)
├── .env.example         ← optional LLM-backend overrides
├── src/                 ← shared baseline (config, llm client)
├── traffic/             ← THE PROJECT (environment, agents, simulator, evaluation)
├── eval/                ← generic results helper
└── tests/               ← root smoke tests
```

---

## 👥 Team & workflow

Five people, five roles, one weekly meeting, a per-person weekly goal, a 1-month
timeline ending in the final presentation on **08.07**. Who owns what and the
week-by-week plan: **[`TEAM_PLAN.md`](TEAM_PLAN.md)**.

**New teammate?** Read **[`CONTRIBUTING.md`](CONTRIBUTING.md)**, run the Quickstart,
then find your role (R1–R5) in `TEAM_PLAN.md`.

---

## 🧰 Tech stack

Python · **LangGraph** (orchestration) · **free, vendor-neutral LLM backend**
(OpenAI-compatible, e.g. local Ollama — no key, no cost) · pandas + rich
(results & logging) · pytest. Everything is free and open-source.
