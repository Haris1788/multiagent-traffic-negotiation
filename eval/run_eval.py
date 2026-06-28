"""Evaluation harness: run an approach N times and log comparable results.

The grade depends on comparing approaches, so every approach registers here and
writes the same metric rows. Idea-specific approaches get added after the project
is assigned (09.06).

Usage:
    python -m eval.run_eval --list
    python -m eval.run_eval --approach baseline --runs 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

RESULTS_DIR = Path("results")

# Registry of approaches. Each maps a name -> callable(run_index) -> dict of metrics.
# Replace `demo` with the real approaches once the project is assigned.
APPROACHES: dict[str, "callable"] = {}


def register(name):
    def deco(fn):
        APPROACHES[name] = fn
        return fn
    return deco


@register("demo")
def _demo(run_index: int) -> dict:
    """Placeholder approach so the harness is runnable before real code exists."""
    return {"approach": "demo", "run": run_index, "score": 0.0, "note": "placeholder"}


def _write_results(approach: str, rows: list[dict], stamp: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    detail = RESULTS_DIR / f"{approach}_{stamp}.jsonl"
    with detail.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = RESULTS_DIR / "summary.csv"
    avg_score = sum(r.get("score", 0.0) for r in rows) / max(len(rows), 1)
    summary_row = {"approach": approach, "stamp": stamp, "runs": len(rows),
                   "avg_score": avg_score}
    write_header = not summary.exists()
    with summary.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(summary_row)
    print(f"Wrote {len(rows)} rows -> {detail}")
    print(f"Appended summary -> {summary} (avg_score={avg_score:.3f})")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare approaches and log results.")
    p.add_argument("--approach", help="Approach to run (see --list).")
    p.add_argument("--runs", type=int, default=1, help="Number of runs.")
    p.add_argument("--list", action="store_true", help="List registered approaches.")
    args = p.parse_args()

    if args.list or not args.approach:
        print("Registered approaches:", ", ".join(sorted(APPROACHES)) or "(none)")
        return

    if args.approach not in APPROACHES:
        raise SystemExit(f"Unknown approach '{args.approach}'. Try --list.")

    fn = APPROACHES[args.approach]
    rows = [fn(i) for i in range(args.runs)]
    # Deterministic stamp (no wall-clock dependency): use run count + env override.
    stamp = os.getenv("EVAL_STAMP", f"n{args.runs}")
    _write_results(args.approach, rows, stamp)


if __name__ == "__main__":
    main()
