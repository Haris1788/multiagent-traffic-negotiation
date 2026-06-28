from __future__ import annotations

import re
from typing import Any

from src import llm
from traffic.agents.zeroshot_agent import (
    _build_prompt as _build_zero_shot_prompt,
    _fallback_move,
    _parse_move,
    _validate_or_correct_move,
)
from traffic.environment import Move, State

_SYSTEM = (
    "You are a cautious traffic-control agent for a 10x10 grid intersection. "
    "Reason briefly, choose one legal action, then report a short observation. "
    "Use exactly this format:\n"
    "Thought: <one short sentence>\n"
    "Action: <UP|DOWN|LEFT|RIGHT|UP2|DOWN2|LEFT2|RIGHT2|WAIT>\n"
    "Observation: <one short sentence>"
)


def _extract_section(text: str, label: str) -> str:
    match = re.search(rf"{label}\s*:\s*(.+)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _move_to_token(move: Move) -> str:
    return {
        (0, -1): "UP",
        (0, -2): "UP2",
        (0, 1): "DOWN",
        (0, 2): "DOWN2",
        (-1, 0): "LEFT",
        (-2, 0): "LEFT2",
        (1, 0): "RIGHT",
        (2, 0): "RIGHT2",
        (0, 0): "WAIT",
    }.get(move, "WAIT")


def _fallback_trace(state: State, car_id: str) -> dict[str, Any]:
    move = _fallback_move(state, car_id)
    me = next((v for v in state.vehicles if v.id == car_id), None)
    if me is None:
        thought = f"Car {car_id} is not active, so it should wait."
    elif move == (0, 0):
        thought = f"Car {car_id} waits because a higher-priority path or occupied cell blocks progress."
    else:
        thought = f"Car {car_id} can move toward {me.goal} while respecting reserved paths."

    return {
        "move": move,
        "thought": thought,
        "action": _move_to_token(move),
        "observation": "The deterministic safety check kept the move legal for this turn.",
        "raw": "",
        "backend": "fallback",
    }

def explain_move(state: State, car_id: str) -> dict[str, Any]:
    """Return a chosen move plus a compact ReAct trace."""
    me = next((v for v in state.vehicles if v.id == car_id), None)
    if me is None or me.arrived:
        return {
            "move": (0, 0),
            "thought": f"Car {car_id} is already done or missing.",
            "action": "WAIT",
            "observation": "No action is needed.",
            "raw": "",
            "backend": "local",
        }

    if not llm.is_available():
        return _fallback_trace(state, car_id)

    prompt = "\n".join([
        _build_zero_shot_prompt(state, car_id),
        "",
        "Before acting, inspect possible conflicts with other cars.",
        "Return Thought, Action, and Observation only.",
    ])
    try:
        raw = llm.complete(prompt, system=_SYSTEM, max_tokens=96)
        action_text = _extract_section(raw, "Action") or raw
        parsed_move = _parse_move(action_text)
        move = _validate_or_correct_move(state, car_id, parsed_move)
        if move == (0, 0) and "WAIT" not in action_text.upper():
            return _fallback_trace(state, car_id)
        corrected = move != parsed_move
        return {
            "move": move,
            "thought": _extract_section(raw, "Thought") or "The model selected a legal traffic action.",
            "action": _move_to_token(move),
            "observation": (
                "The model action was corrected by the traffic safety validator."
                if corrected
                else _extract_section(raw, "Observation") or "The action was parsed into a legal move."
            ),
            "raw": raw,
            "backend": "llm",
        }
    except Exception:
        return _fallback_trace(state, car_id)


def choose_move(state: State, car_id: str) -> Move:
    """Return ``(dx, dy)`` chosen via a ReAct-style decision path."""
    return explain_move(state, car_id)["move"]
