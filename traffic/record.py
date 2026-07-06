"""Episode recorder for the visual replay lab.  OWNER: R1.

Runs ONE real episode (LLM-driven whenever the backend is up) and writes the
complete decision trace — exact prompt, raw model reply, parsed vs. applied
move, validator corrections, backend and latency — to a JSON file plus a JS
data file that ``presentation/demos/5-llm-lab/index.html`` replays offline.

This is a visualization tool: it changes nothing about the experiment.
Aggregated metrics still come from ``traffic/evaluate.py``.

Usage:
    python -m traffic.record --approach react
    python -m traffic.record --approach zeroshot --model gemma2:2b --max-turns 20
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one real episode for the replay dashboard.")
    parser.add_argument("--approach", choices=["react", "zeroshot"], default="react")
    parser.add_argument("--model", default=None, help="override LLM_MODEL for this run (e.g. gemma2:2b)")
    parser.add_argument("--cars", type=int, choices=[2, 4, 6], default=4,
                        help="how many vehicles cross the intersection (default 4)")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--out", default=None, help="output basename (default: presentation/demos/5-llm-lab/data-<approach>)")
    args = parser.parse_args()

    if args.model:  # must happen before src.config is imported
        os.environ["LLM_MODEL"] = args.model

    from src import config, llm
    from traffic.agents import react_agent, zeroshot_agent
    from traffic.environment import GridIntersection, Vehicle
    from traffic.simulator import run_episode

    # crossing corridors: eastbound row 5, westbound row 4, southbound col 5, northbound col 4;
    # 6-car variant adds a second eastbound street (row 3) and southbound street (col 6).
    fleet = [
        Vehicle(id="A", pos=(0, 5), goal=(9, 5), direction="E"),
        Vehicle(id="B", pos=(5, 0), goal=(5, 9), direction="S"),
        Vehicle(id="C", pos=(9, 4), goal=(0, 4), direction="W"),
        Vehicle(id="D", pos=(4, 9), goal=(4, 0), direction="N"),
        Vehicle(id="E", pos=(0, 3), goal=(9, 3), direction="E"),
        Vehicle(id="F", pos=(6, 0), goal=(6, 9), direction="S"),
    ][: args.cars]

    decisions: list[dict] = []

    def record_zeroshot(state, car_id, env=None):
        me = next((v for v in state.vehicles if v.id == car_id), None)
        if me is None or me.arrived:
            return (0, 0)
        prompt = zeroshot_agent._build_prompt(state, car_id, env)
        rec = {
            "turn": state.turn, "car": car_id,
            "pos": list(me.pos), "goal": list(me.goal), "speed": me.speed,
            "prompt": prompt, "raw": "", "thought": "", "observation": "",
        }
        t0 = time.time()
        if llm.is_available():
            try:
                raw = llm.complete(prompt, system=zeroshot_agent._SYSTEM, max_tokens=16)
                parsed = zeroshot_agent._parse_move(raw)
                move = zeroshot_agent._validate_or_correct_move(state, car_id, parsed, env)
                rec.update(raw=raw, backend="llm",
                           parsed=list(parsed), corrected=move != parsed)
            except Exception as exc:
                move = zeroshot_agent._fallback_move(state, car_id, env)
                rec.update(raw=f"<error: {exc}>", backend="fallback",
                           parsed=list(move), corrected=False)
        else:
            move = zeroshot_agent._fallback_move(state, car_id, env)
            rec.update(backend="fallback", parsed=list(move), corrected=False)
        rec.update(latency=round(time.time() - t0, 2),
                   action=zeroshot_agent._move_to_token(move), move=list(move))
        decisions.append(rec)
        print(f"turn {rec['turn']:>2} | {args.approach:>8} | car {car_id} "
              f"{tuple(rec['pos'])} -> {rec['action']:<6} [{rec['backend']}] {rec['latency']}s", flush=True)
        return move

    def record_react(state, car_id, env=None):
        me = next((v for v in state.vehicles if v.id == car_id), None)
        if me is None or me.arrived:
            return (0, 0)
        prompt = "\n".join([
            zeroshot_agent._build_prompt(state, car_id, env),
            "",
            "Before acting, inspect possible conflicts with other cars.",
            "Return Thought, Action, and Observation only.",
        ])
        t0 = time.time()
        trace = react_agent.explain_move(state, car_id, env)
        parsed = react_agent._parse_move(
            react_agent._extract_section(trace["raw"], "Action") or trace["raw"]
        ) if trace["raw"] else trace["move"]
        rec = {
            "turn": state.turn, "car": car_id,
            "pos": list(me.pos), "goal": list(me.goal), "speed": me.speed,
            "prompt": prompt, "raw": trace["raw"],
            "thought": trace["thought"], "observation": trace["observation"],
            "backend": trace["backend"],
            "parsed": list(parsed), "move": list(trace["move"]),
            "corrected": tuple(parsed) != tuple(trace["move"]),
            "action": trace["action"],
            "latency": round(time.time() - t0, 2),
        }
        decisions.append(rec)
        print(f"turn {rec['turn']:>2} | {args.approach:>8} | car {car_id} "
              f"{tuple(rec['pos'])} -> {rec['action']:<6} [{rec['backend']}] {rec['latency']}s", flush=True)
        return trace["move"]

    agent = record_react if args.approach == "react" else record_zeroshot

    env = GridIntersection(vehicles=fleet, max_turns=args.max_turns)
    print(f"recording 1 episode | approach={args.approach} model={config.LLM_MODEL} "
          f"cars={args.cars} backend={config.LLM_BASE_URL}", flush=True)
    started = time.time()
    result = run_episode(env, agent)

    episode = {
        "meta": {
            "approach": args.approach,
            "model": config.LLM_MODEL,
            "cars": args.cars,
            "base_url": config.LLM_BASE_URL,
            "grid": getattr(env, "size", 10),
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_seconds": round(time.time() - started, 1),
        },
        "decisions": decisions,
        "path": [{k: list(v) for k, v in snap.items()} for snap in result["path"]],
        "result": {
            "success": bool(result["success"]),
            "collisions": int(result["collisions"]),
            "steps": int(result["steps"]),
        },
    }

    root = Path(__file__).resolve().parents[1]
    base = Path(args.out) if args.out else root / "presentation" / "demos" / "5-llm-lab" / f"data-{args.approach}"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(episode, indent=1), encoding="utf-8")
    var = "EP_REACT" if args.approach == "react" else "EP_ZS"
    base.with_suffix(".js").write_text(
        f"window.{var} = {json.dumps(episode)};", encoding="utf-8")
    print(f"\nwrote {base.with_suffix('.js')}")
    print(f"result: success={episode['result']['success']} "
          f"collisions={episode['result']['collisions']} steps={episode['result']['steps']} "
          f"decisions={len(decisions)} wall={episode['meta']['wall_seconds']}s")


if __name__ == "__main__":
    main()
