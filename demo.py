"""Visual live demo: watch LLM agents negotiate the intersection."""
from __future__ import annotations
import os, time, sys
from traffic.environment import GridIntersection, State, Vehicle
from traffic.agents.zeroshot_agent import choose_move, _build_prompt, _parse_move
from src import llm

# ── ANSI colors ──────────────────────────────────────────────────────────────
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"
GRAY    = "\033[90m"
BOLD    = "\033[1m"
RESET   = "\033[0m"
BG_RED  = "\033[41m"
BG_GREEN= "\033[42m"

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def color_car(car_id: str) -> str:
    return (BOLD + BLUE + car_id + RESET) if car_id == "A" else (BOLD + RED + car_id + RESET)

def render_grid(state: State, size: int = 10, collision_pos: set = set()) -> None:
    grid = [["." for _ in range(size)] for _ in range(size)]

    # mark goals faintly
    for v in state.vehicles:
        gx, gy = v.goal
        grid[gy][gx] = GRAY + "x" + RESET

    # mark cars
    for v in state.vehicles:
        x, y = v.pos
        if v.arrived:
            grid[y][x] = GREEN + "✓" + RESET
        elif (x, y) in collision_pos:
            grid[y][x] = BG_RED + BOLD + v.id + RESET
        else:
            grid[y][x] = color_car(v.id)

    # header
    print(GRAY + "   " + "  ".join(str(i) for i in range(size)) + RESET)
    print(GRAY + "  ┌" + "───" * size + "┐" + RESET)
    for row_i, row in enumerate(grid):
        print(GRAY + f"{row_i} │" + RESET + "  ".join(row) + GRAY + " │" + RESET)
    print(GRAY + "  └" + "───" * size + "┘" + RESET)


def print_header(turn: int, max_turns: int, collisions: int):
    print(BOLD + CYAN + "╔══════════════════════════════════════════════════════════╗" + RESET)
    print(BOLD + CYAN + "║     MULTI-AGENT TRAFFIC NEGOTIATION  —  ZERO-SHOT LLM   ║" + RESET)
    print(BOLD + CYAN + "╚══════════════════════════════════════════════════════════╝" + RESET)
    print(f"  Turn: {BOLD}{turn}/{max_turns}{RESET}    "
          f"Collisions: {(RED+BOLD if collisions>0 else GREEN)+str(collisions)+RESET}    "
          f"Model: {YELLOW}llama3.1{RESET} via Ollama\n")


def print_legend():
    print(f"  Legend:  "
          f"{color_car('A')} = Car A (left→right)   "
          f"{color_car('B')} = Car B (top→bottom)   "
          f"{GRAY}x{RESET} = goal   "
          f"{GREEN}✓{RESET} = arrived\n")


def thinking_animation(car_id: str, duration: float = 0.0):
    frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    color = BLUE if car_id == "A" else RED
    end = time.time() + duration
    i = 0
    while time.time() < end:
        sys.stdout.write(f"\r  {color+BOLD}Car {car_id}{RESET} thinking {color+frames[i % len(frames)]+RESET}  ")
        sys.stdout.flush()
        time.sleep(0.08)
        i += 1
    sys.stdout.write("\r" + " " * 40 + "\r")


def print_car_decision(car_id: str, v: Vehicle, prompt_summary: str, raw_reply: str, move: tuple):
    color = BLUE if car_id == "A" else RED
    arrow_map = {(1,0):"-> RIGHT",(-1,0):"<- LEFT",(0,1):"v  DOWN",(0,-1):"^  UP",(0,0):"o  WAIT"}
    arrow = arrow_map.get(move, str(move))

    print(f"  {color+BOLD}Car {car_id}{RESET}  pos={YELLOW}{v.pos}{RESET} → goal={YELLOW}{v.goal}{RESET}  speed={v.speed}")
    print(f"  {GRAY}Prompt: {prompt_summary}{RESET}")
    print(f"  LLM replied: {MAGENTA+BOLD}'{raw_reply.strip()}'{RESET}  →  {GREEN+BOLD}{arrow}{RESET}")
    print()


def run_demo():
    MAX_TURNS = 30
    env = GridIntersection(max_turns=MAX_TURNS)
    state = env.reset()
    collision_cells: set = set()

    while not env.is_done():
        clear()
        print_header(state.turn, MAX_TURNS, env.collisions)
        print_legend()
        render_grid(state, collision_pos=collision_cells)
        print()

        moves = {}
        decisions = []

        for v in state.vehicles:
            if v.arrived:
                continue

            # show thinking spinner while LLM runs
            thinking_animation(v.id, 0)  # just clears line; real wait is in llm.complete

            prompt = _build_prompt(state, v.id)
            # summarise prompt for display
            others = [x for x in state.vehicles if x.id != v.id and not x.arrived]
            other_str = f"car {others[0].id} at {others[0].pos}" if others else "no others"
            prompt_summary = f"pos={v.pos} goal={v.goal} | {other_str}"

            sys.stdout.write(f"  {(BLUE if v.id=='A' else RED)+BOLD}Car {v.id}{RESET} asking LLM...  ")
            sys.stdout.flush()
            t0 = time.time()
            try:
                raw_reply = llm.complete(prompt, system=(
                    "You are a traffic-control agent for a 10×10 grid intersection. "
                    "Reply with exactly one word: UP DOWN LEFT RIGHT WAIT."
                ), max_tokens=16)
                move = _parse_move(raw_reply)
            except Exception:
                raw_reply = "WAIT (LLM error)"
                move = (0, 0)
            elapsed = time.time() - t0
            sys.stdout.write(f"done ({elapsed:.1f}s)\n")

            moves[v.id] = move
            decisions.append((v.id, v, prompt_summary, raw_reply, move))

        # re-render with decisions shown
        clear()
        print_header(state.turn, MAX_TURNS, env.collisions)
        print_legend()
        render_grid(state, collision_pos=collision_cells)
        print()
        print(BOLD + "  ── Decisions this turn ──" + RESET)
        for (car_id, v, ps, rr, mv) in decisions:
            print_car_decision(car_id, v, ps, rr, mv)

        # step
        prev_collisions = env.collisions
        state = env.step(moves)
        collision_cells = set()

        if env.collisions > prev_collisions:
            # find cars that didn't move (stayed = collision)
            for v in state.vehicles:
                if moves.get(v.id) != (0,0) and v.pos == next(
                    (x.pos for x in env._initial if x.id == v.id), None
                ):
                    collision_cells.add(v.pos)
            print(f"  {BG_RED+BOLD} ⚠ COLLISION! {RESET} Cars couldn't move.\n")
            time.sleep(2)
        else:
            time.sleep(1.2)

    # final screen
    clear()
    print_header(state.turn, MAX_TURNS, env.collisions)
    print_legend()
    render_grid(state)
    print()
    if env.all_arrived():
        print(BG_GREEN + BOLD + "  ✓  ALL CARS ARRIVED — SUCCESS!  " + RESET)
    else:
        print(RED + BOLD + "  ✗  Time limit reached — not all cars arrived." + RESET)
    print(f"\n  Steps: {BOLD}{state.turn}{RESET}   Collisions: {BOLD}{env.collisions}{RESET}\n")


if __name__ == "__main__":
    # enable ANSI on Windows
    os.system("color")
    run_demo()
