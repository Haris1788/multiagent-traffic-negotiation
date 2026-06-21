# Team Plan — Roles, Cadence & 1-Month Timeline

We are **5 people**. Our assigned project is the **Multi-Agent Traffic Negotiation
Simulator (ReAct vs. Zero-Shot)**. The table maps each role onto the files they own so
everyone can build **in parallel** today.

## The 5 roles (mapped to the Traffic project)

| # | Role | Assigned to | Owns (files) | Builds, in parallel |
|---|------|-------------|--------------|---------------------|
| **R1** | **Integration Lead** | **Haris** | `traffic/simulator.py`, later the LangGraph graph | Builds the turn loop + the contracts R2–R5 plug into — owns a file like everyone else. Plus a **light** "does it all still run together?" check near deadlines. **Not** a daily reviewer; everyone merges their own work into `dev`, R1 promotes `dev → main`. |
| **R2** | **Environment Engineer** | **Azeel** | `traffic/environment.py` | The grid, kinematics (1–2 cells/turn, no diagonals, 1 turn to brake), and collision rules. Pure Python, no LLM. |
| **R3** | **Zero-Shot Agent (baseline)** | **Pratham** | `traffic/agents/zeroshot_agent.py` | The direct-prompt driver + move parsing — our control approach. |
| **R4** | **ReAct Agent (comparison)** | **Darshan** | `traffic/agents/react_agent.py` | The Thought→Action→Observation reasoning loop + its trace — our main hypothesis. |
| **R5** | **Evaluation & Results Lead** | **Kuldeep** | `traffic/evaluate.py`, metrics, plots, the final slide deck | The 50–100-episode sweep, the comparison table/plots, and the report. |

> Roles assigned (21.06): R1 Haris · R2 Azeel · R3 Pratham · R4 Darshan · R5 Kuldeep.
> R2's environment unblocks everyone, so prioritize it in W1.

## How we work (every week)

1. **One meeting per week** (~45 min). Everyone brings: what I did, what I learned,
   what's blocking me, and my idea for next week.
2. **Discuss → align → assign.** We agree the next week's goal for each person before
   leaving the meeting.
3. **Work in parallel** against the agreed interfaces. Small, frequent commits.
4. **Work on your own branch → merge into `dev`.** Pull `dev` first, run
   `python -m pytest` (green is the gate), then push your branch and merge into `dev`.
   CI re-runs the tests on `dev`. R1 promotes `dev → main` once green. `dev` is the
   default branch; `main` stays demo-able. No mandatory review.

## 1-month timeline (4 working weeks)

| Week | Dates | Goal | Per-role focus |
|------|-------|------|----------------|
| **W0** | now → 09.06 | Finalize topic; env ready (✅ done) | Everyone reads the course material + the chosen topic's README. |
| **W1** | 10.06 – 16.06 | **Thin slice runs end-to-end** (even if dumb). Kickoff meeting right after Session 1 (10.06). | R1 builds the turn loop + freezes contracts · R2 data layer · R3 first agent · R4 second agent · R5 metrics + baseline. |
| **W2** | 17.06 – 23.06 | **Baseline approach fully working + first eval numbers.** Register for coaching by **19.06**. | R1 turn loop solid on stubs · R2/R3/R4 harden their modules · R5 run baseline, log results. |
| **W3** | 24.06 – 30.06 | **Add the comparison approach** (RAG / ReAct / multi-agent) + full evaluation. Coaching session **24.06** — bring questions. | R4 leads the comparison approach · R5 runs the A/B eval + plots · R1 wires LangGraph once agents call the LLM. |
| **W4** | 01.07 – 07.07 | **Polish, ablations, write report & slides.** Submit presentation by **07.07**. | Everyone writes their section; R5 owns the deck; R1 a final "does it all still run?" check + architecture write-up. |
| **Present** | 08.07 | Final presentation (mandatory attendance). | Practice run the day before. |

## Reality check: low cadence → protect the critical path

The table above is the *ideal* (5 people every week). Realistically we may only get
**~one person working ~one day per week**. So we plan for the worst case and treat the
parallel work as a bonus, not a dependency.

**Critical path — the only thing that must happen, in order.** Each item is ≤1 focused
day and produces something demoable. If exactly one person shows up each week, they do
the next item on this list and nothing else:

1. **Real kinematics + collisions** (`environment.py`, R2) — the world must reject
   illegal moves and detect crashes. *Without this, no result means anything.*
2. **Real zero-shot prompt + move parsing** (`zeroshot_agent.py`, R3) — one LLM call,
   parse the move, fall back to a safe move if it's garbage.
3. **Run the baseline eval** (`evaluate.py`, R5) — 50 episodes, get real numbers in
   `results/`. **At this point we already have a gradeable artifact.**
4. **Real ReAct loop + trace** (`react_agent.py`, R4) — reuse R3's prompt, add
   Thought→Action→Observation.
5. **A/B comparison + one plot** (R5) — zero-shot vs ReAct table + a bar/path plot.
6. **Slides + write-up** (everyone, R5 owns deck).

Items 1→3 are the **minimum viable grade** (two approaches = current ReAct stub vs real
zero-shot, both measured). Items 4→5 are the **real research payoff**. Item 6 is mandatory.

**Rule of thumb:** never start an item before the one above it is merged and tests pass.
Whoever works that week picks up the lowest unfinished number. If two people show up,
the second takes parallel polish (tests, plots, README) — never a later critical item.

## Definition of "done" for the project (grading-aligned)

- A working (multi-)agent system or LLM workflow for the chosen topic.
- **At least two approaches compared** with a real evaluation (numbers + a plot).
- Clear write-up of the domain challenge and how agents address it.

## Milestones mapped to real dates (≈4 weeks, ~1 day/week each)

| By date | Critical-path item that MUST be green | Owner (default) |
|---------|----------------------------------------|-----------------|
| **16.06** (end W1) | #1 real kinematics + collisions merged | R2 |
| **23.06** (end W2) | #2 real zero-shot + #3 baseline numbers in `results/` — **MVP grade reached** | R3, R5 |
| **30.06** (end W3) | #4 real ReAct loop + #5 first A/B comparison | R4, R5 |
| **07.07** | #6 slides + write-up submitted | R5 + all |
| **08.07** | Final presentation (mandatory) | all |

If a week slips, the *whole tail shifts* — so finish #1–#3 early; they are the safety net.
