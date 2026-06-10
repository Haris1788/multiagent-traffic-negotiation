# Contributing

How to set up and work on this project. Read this before your first commit.

## Setup (once)

```powershell
git clone <repo-url>
cd <repo>
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                       # everything should pass
```

No API key or paid service is needed. To run the *real* LLM agents later, install
[Ollama](https://ollama.com) (free, local) — the environment and tests run without it.

## Find your job

Your role (R1–R5) and the files you own are in [`TEAM_PLAN.md`](TEAM_PLAN.md). The
system design is in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Workflow

1. **Branch** off `main`: `git checkout -b r3/zeroshot-prompt` (use your role + topic).
2. **Work against the contracts**, not other people's internals:
   - Agents implement `choose_move(state, car_id) -> Move`.
   - The environment exposes `reset()`, `step(moves)`, `is_done()`, `all_arrived()`.
   Keep these signatures stable so everyone integrates cleanly.
3. **Keep the environment LLM-free** — only agents call `src/llm.py`.
4. **Tests must pass** before you push: `python -m pytest`. Add a test for new behavior.
5. **Open a Pull Request** to `main`. R1 reviews/integrates. Small, frequent PRs.

## Conventions

- One agent per file under `traffic/agents/`; small, testable functions.
- Read the model/provider from `src.config` — never hardcode it.
- Log metrics as one dict per run so they aggregate into a table.
- Don't commit `.env`, `.venv/`, `results/`, or anything under `Misc/` (all gitignored).

## Commands cheat-sheet

```powershell
python -m pytest                                    # tests
python -m traffic.evaluate --list                   # list approaches
python -m traffic.evaluate --approach react --episodes 50
```
