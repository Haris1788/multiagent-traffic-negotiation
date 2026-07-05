"""Episode runner.  OWNER: R1 (Integration Lead).

Drives the environment with a given agent function and logs the outcome.

Orchestrated as a LangGraph StateGraph (decide -> step -> loop until done).
"""
from __future__ import annotations

from typing import Callable, TypedDict

#Second Phase layer for Langraph integration
from langgraph.graph import StateGraph, START, END

from traffic.environment import GridIntersection, Move, State

AgentFn = Callable[..., Move]

# Graphstate for Langraph integration
class GraphState(TypedDict):
    env: GridIntersection
    agent_fn: AgentFn
    moves: dict[str, Move]
    path: list[dict[str, tuple[int, int]]]

#Function to decide moves for each non-arrived car (LangGraph)
def decide(state: GraphState) -> dict:
    """Each non-arrived car chooses a Move via the agent function."""
    env = state["env"]
    agent_fn = state["agent_fn"]
    moves = {
        v.id: agent_fn(env.state, v.id, env)
        for v in env.state.vehicles
        if not v.arrived
    }
    return {"moves": moves}

#Fucntion to step the environment with the chosen moves (LangGraph)
def step(state: GraphState) -> dict:
    """ Apply the chosen moves to the world and append the new positions."""
    env = state["env"]
    env.step(state["moves"])
    snapshot = {v.id: v.pos for v in env.state.vehicles}
    new_path = state["path"] + [snapshot]
    return {"path": new_path}

#Router function to determine if the episode is done (LangGraph)
def is_episode_done(state: GraphState) -> str:
    """Router: after a step, decide whether to loop again or finish."""
    if state["env"].is_done():
        return "end"
    return "continue"

#
def build_graph():
    """Wire the nodes and edges into a runnabble LangGraph."""
    g = StateGraph(GraphState)

    g.add_node("decide", decide)
    g.add_node("step", step)

    g.add_edge(START, "decide")
    g.add_edge("decide", "step")
    g.add_conditional_edges(
        "step",
        is_episode_done,
        {"continue": "decide", "end": END},
    )
    return g.compile()

# LangGraph Layer
# function to run a single episode of the environment with the given agent function
# That will help to test the agent function and see how it performs in the environment.
# Stop when all vehicles have arrived or when the maximum number of turns is reached.
def run_episode(env: GridIntersection, agent_fn: AgentFn) -> dict:
    """Run one episode to completion via the LangGraph and return outcome metrics."""
    env.reset()
    graph = build_graph()
    initial_state: GraphState = {
        "env": env,
        "agent_fn": agent_fn,
        "moves": {},
        "path": [],
    }
    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": 2 * env.max_turns + 10},
    )
    return {
        "success": env.all_arrived(),
        "collisions": env.collisions,
        "steps": env.state.turn,
        "path": final_state["path"],
    }

