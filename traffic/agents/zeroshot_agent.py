"""Zero-shot driving agent.  OWNER: R3 (Pratham).

Builds a plain-language prompt describing the grid state, calls the LLM once
(no chain-of-thought), and parses the returned move.  Never mutates state.
"""
from __future__ import annotations

import re

from src import llm
from traffic.environment import Move, State

# ── legal moves the LLM may return ──────────────────────────────────────────
_MOVE_MAP: dict[str, Move] = {
    "UP":    (0, -1),
    "DOWN":  (0,  1),
    "LEFT":  (-1, 0),
    "RIGHT": (1,  0),
    "WAIT":  (0,  0),
}

_SYSTEM = (
    "You are a traffic-control agent for a 10×10 grid intersection. "
    "On every turn you receive the positions of all cars and must choose one move "
    "for the assigned car. Reply with exactly one word from: UP DOWN LEFT RIGHT WAIT. "
    "No punctuation, no explanation — one word only."
)


def _build_prompt(state: State, car_id: str) -> str:
    me = next(v for v in state.vehicles if v.id == car_id)
    others = [v for v in state.vehicles if v.id != car_id and not v.arrived]

    lines = [
        f"Turn {state.turn}.",
        f"You are car {car_id}.",
        f"Your position: {me.pos}. Your goal: {me.goal}. Your speed last turn: {me.speed}.",
    ]

    if others:
        other_desc = "; ".join(
            f"car {v.id} at {v.pos} going to {v.goal}" for v in others
        )
        lines.append(f"Other cars: {other_desc}.")
    else:
        lines.append("No other active cars.")

    lines += [
        "Grid is 10×10 (x=0..9, y=0..9). x increases RIGHT, y increases DOWN.",
        "Rules: no diagonals; max 2 cells/turn; if you moved at speed 2 you cannot stop instantly (brake to 1 first).",
        "Choose the move that makes progress toward your goal while avoiding collisions.",
        "Reply with exactly one word: UP DOWN LEFT RIGHT WAIT",
    ]
    return "\n".join(lines)


def _parse_move(text: str) -> Move:
    """Extract the first recognised move keyword from the LLM reply."""
    upper = text.upper()
    # prefer an exact word match so 'DOWNLOAD' doesn't match DOWN
    for word in re.findall(r"[A-Z]+", upper):
        if word in _MOVE_MAP:
            return _MOVE_MAP[word]
    return (0, 0)  # safe fallback: wait


def choose_move(state: State, car_id: str) -> Move:
    """Return a legal (dx, dy) for *car_id* based on a single LLM call.

    Does NOT mutate *state*.
    """
    me = next((v for v in state.vehicles if v.id == car_id), None)
    if me is None or me.arrived:
        return (0, 0)

    prompt = _build_prompt(state, car_id)
    try:
        reply = llm.complete(prompt, system=_SYSTEM, max_tokens=16)
        return _parse_move(reply)
    except Exception:
        # LLM unavailable — fall back to greedy step toward goal
        gx, gy = me.goal
        x, y = me.pos
        if x != gx:
            return (1 if gx > x else -1, 0)
        if y != gy:
            return (0, 1 if gy > y else -1)
        return (0, 0)
