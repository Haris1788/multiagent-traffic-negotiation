"""Evaluation for the traffic topic.  OWNER: R5 (Evaluation Lead).

Runs N episodes per approach and aggregates success / collisions / steps.
Saves per-episode results and produces simple comparison plots.
"""
from __future__ import annotations

import argparse
import json
import os
import csv
from pathlib import Path
from typing import Iterable

from traffic.agents.react_agent import choose_move as react_move
from traffic.agents.zeroshot_agent import choose_move as zeroshot_move
from traffic.environment import GridIntersection
from traffic.simulator import run_episode

APPROACHES = {
    "zeroshot": zeroshot_move,
    "react": react_move,
}


def evaluate(approach: str, episodes: int) -> dict:
    """Run `episodes` episodes for `approach` and return summary and per-episode rows.

    Returns a dict with keys: summary (dict) and rows (list of per-episode dicts).
    """
    if approach not in APPROACHES:
        raise ValueError(f"Unknown approach: {approach}")

    agent_fn = APPROACHES[approach]
    rows = [run_episode(GridIntersection(), agent_fn) for _ in range(episodes)]
    n = len(rows)
    summary = {
        "approach": approach,
        "episodes": n,
        "success_rate": sum(r["success"] for r in rows) / n,
        "avg_collisions": sum(r["collisions"] for r in rows) / n,
        "avg_steps": sum(r["steps"] for r in rows) / n,
    }
    return {"summary": summary, "rows": rows}


def save_results(rows: Iterable[dict], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # per-episode JSONL
    episodes_path = out_dir / f"{summary['approach']}_episodes.jsonl"
    with episodes_path.open("w", encoding="utf8") as fh:
        for r in rows:
            fh.write(json.dumps(r))
            fh.write("\n")
    # aggregated CSV (one row per approach)
    summary_path = out_dir / "summaries.csv"
    fieldnames = ["approach", "episodes", "success_rate", "avg_collisions", "avg_steps"]
    write_header = not summary_path.exists()
    # Append row safely using csv module
    with summary_path.open("a", newline="", encoding="utf8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: summary.get(k, "") for k in fieldnames})


def _write_svg_bar_chart(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    width = 640
    height = 480
    margin_left = 100
    margin_right = 40
    margin_top = 60
    margin_bottom = 80
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    max_value = max(values) if values else 1.0
    if max_value == 0:
        max_value = 1.0

    bar_count = len(values)
    bar_width = chart_width / max(bar_count, 1) * 0.6
    gap = chart_width / max(bar_count, 1) * 0.4

    bars = []
    for index, value in enumerate(values):
        x = margin_left + index * (bar_width + gap)
        bar_height = (value / max_value) * chart_height
        y = margin_top + chart_height - bar_height
        bars.append((x, y, bar_width, bar_height, labels[index]))

    def svg_text(x: float, y: float, text: str, size: int = 14, anchor: str = "start") -> str:
        return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="#222">{text}</text>'

    svg_lines = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>',
        svg_text(width / 2, 30, title, size=20, anchor="middle"),
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" stroke="#333"/>',
    ]

    # Y-axis labels and grid lines
    for i in range(5):
        value = max_value * i / 4
        y = margin_top + chart_height - (value / max_value) * chart_height
        svg_lines.append(f'<line x1="{margin_left - 5}" y1="{y}" x2="{margin_left + chart_width}" y2="{y}" stroke="#ddd"/>')
        svg_lines.append(svg_text(margin_left - 10, y + 5, f"{value:.2f}", size=12, anchor="end"))

    # Bars
    for x, y, w, h, label in bars:
        svg_lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#4c78a8"/>')
        svg_lines.append(svg_text(x + w / 2, margin_top + chart_height + 20, label, size=12, anchor="middle"))

    svg_lines.append(svg_text(width / 2, height - 10, "approach", size=14, anchor="middle"))
    svg_lines.append("</svg>")

    path.write_text("\n".join(svg_lines), encoding="utf8")


def plot_comparison(summaries_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # read CSV using csv module to avoid pandas dependency
    rows = []
    with summaries_csv.open("r", encoding="utf8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    if not rows:
        print("No summaries found to plot.")
        return

    approaches = [r["approach"] for r in rows]
    def to_float(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    metrics = ["success_rate", "avg_collisions", "avg_steps"]
    # Try matplotlib if available; otherwise produce SVGs.
    try:
        import matplotlib.pyplot as plt
        for m in metrics:
            values = [to_float(r.get(m, 0.0)) for r in rows]
            plt.figure(figsize=(6, 4))
            plt.title(m.replace("_", " ").title())
            plt.bar(approaches, values, color="C0")
            plt.ylabel(m)
            plt.xlabel("approach")
            plt.tight_layout()
            fp = out_dir / f"compare_{m}.png"
            plt.savefig(fp)
            plt.close()
    except Exception:
        print("matplotlib unavailable; writing SVG comparison plots instead.")
        for m in metrics:
            values = [to_float(r.get(m, 0.0)) for r in rows]
            fp = out_dir / f"compare_{m}.svg"
            _write_svg_bar_chart(fp, m.replace("_", " ").title(), approaches, values)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate traffic-negotiation approaches.")
    p.add_argument("--approaches", type=str, default="all", help="Comma separated approaches or 'all'")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    if args.approaches == "all":
        approaches = list(APPROACHES.keys())
    else:
        approaches = [a.strip() for a in args.approaches.split(",") if a.strip()]

    summaries = []
    for a in approaches:
        print(f"Running {args.episodes} episodes for: {a}")
        res = evaluate(a, args.episodes)
        save_results(res["rows"], res["summary"], out_dir)
        summaries.append(res["summary"])

    print("Summaries:")
    for s in summaries:
        print(s)

    # optionally create comparison plots
    if args.plot:
        summaries_csv = out_dir / "summaries.csv"
        if summaries_csv.exists():
            plot_comparison(summaries_csv, out_dir / "plots")
            print(f"Plots saved to {(out_dir / 'plots').resolve()}")
        else:
            print("No summaries found to plot; run evaluations first.")


if __name__ == "__main__":
    main()
