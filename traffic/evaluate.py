"""Evaluation for the traffic topic.  OWNER: R5 (Evaluation Lead).

Runs N episodes per approach and aggregates success / collisions / steps.
"""
from __future__ import annotations

import argparse

from traffic.agents.react_agent import choose_move as react_move
from traffic.agents.zeroshot_agent import choose_move as zeroshot_move
from traffic.environment import GridIntersection
from traffic.simulator import run_episode

APPROACHES = {
    "zeroshot": zeroshot_move,
    "react": react_move,
}


def evaluate(approach: str, episodes: int) -> dict:
    agent_fn = APPROACHES[approach]
    results = [run_episode(GridIntersection(), agent_fn) for _ in range(episodes)]
    n = len(results)
    return {
        "approach": approach,
        "episodes": n,
        "success_rate": sum(r["success"] for r in results) / n,
        "avg_collisions": sum(r["collisions"] for r in results) / n,
        "avg_steps": sum(r["steps"] for r in results) / n,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate traffic-negotiation approaches.")
    p.add_argument("--approach")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list or not args.approach:
        print("Approaches:", ", ".join(APPROACHES))
        return
    summary = evaluate(args.approach, args.episodes)
    print(summary)


if __name__ == "__main__":
    main()
