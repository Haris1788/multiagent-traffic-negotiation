import csv
from traffic.simulator import run_episode
from traffic.environment import GridIntersection
from traffic.agents.zeroshot_agent import choose_move as zero_agent
from traffic.agents.react_agent import choose_move as react_agent

NUM_EPISODES = 100


def evaluate(agent_fn, agent_name):

    env = GridIntersection()

    rows = []

    successes = 0
    collisions = 0
    total_steps = 0

    for episode in range(1, NUM_EPISODES + 1):

        result = run_episode(env, agent_fn)

        rows.append({
            "Episode": episode,
            "Agent": agent_name,
            "Success": result["success"],
            "Collisions": result["collisions"],
            "Steps": result["steps"]
        })

        successes += int(result["success"])
        collisions += result["collisions"]
        total_steps += result["steps"]

    return rows, {
        "Agent": agent_name,
        "Success Rate": successes / NUM_EPISODES,
        "Collision Count": collisions,
        "Average Steps": total_steps / NUM_EPISODES,
    }


zero_rows, zero_summary = evaluate(zero_agent, "Zero-Shot")
react_rows, react_summary = evaluate(react_agent, "ReAct")

all_rows = zero_rows + react_rows

with open("results.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["Episode", "Agent", "Success", "Collisions", "Steps"],
    )
    writer.writeheader()
    writer.writerows(all_rows)

with open("summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Agent",
            "Success Rate",
            "Collision Count",
            "Average Steps",
        ],
    )
    writer.writeheader()
    writer.writerow(zero_summary)
    writer.writerow(react_summary)

print("Evaluation completed.")