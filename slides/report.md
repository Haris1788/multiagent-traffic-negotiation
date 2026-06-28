# Evaluation Report

## Title
Multi-Agent Traffic Negotiation — Zero-Shot vs ReAct

## Authors
Kuldeep (R5), Pratham (R3), Darshan (R4), Azeel (R2), Haris (R1)

## Abstract
This report compares two traffic negotiation agents — a zero-shot baseline and a ReAct-style comparison agent — on a grid intersection environment. The evaluation measures success rate, collisions, and episode length over 50 episodes per approach.

## Environment
The environment is a 10×10 grid intersection implemented in `traffic/environment.py`.
Vehicles move 0–2 cells per turn with no diagonal movement and must brake from speed 2 through speed 1 before stopping. Collisions are detected when multiple vehicles try to occupy the same cell in the same turn.

## Agents
- **Zero-shot baseline:** `traffic/agents/zeroshot_agent.py`
- **ReAct comparison:** `traffic/agents/react_agent.py`

Both agents currently use placeholder logic that greedily moves each vehicle toward its goal.

## Evaluation protocol
- Approaches evaluated: `zeroshot`, `react`
- Episodes per approach: `50`
- Output directory: `results/`
- Generated files: `results/summaries.csv`, `results/zeroshot_episodes.jsonl`, `results/react_episodes.jsonl`
- Plot files: `results/plots/compare_success_rate.svg`, `results/plots/compare_avg_collisions.svg`, `results/plots/compare_avg_steps.svg`

## Results
| Approach | Episodes | Success Rate | Avg Collisions | Avg Steps |
|---|---|---|---|---|
| zeroshot | 50 | 0.0 | 26.0 | 30.0 |
| react | 50 | 0.0 | 26.0 | 30.0 |

### Observations
- Both agents produced identical results because they currently use the same greedy move logic.
- No episodes succeeded within the 30-turn limit, so all runs ended with collisions or no completion.
- Average collisions were high, indicating the environment is currently very challenging for the placeholder behavior.

## Discussion
This evaluation pipeline is working end-to-end: the script runs both approaches, saves per-episode logs, appends summary CSV rows, and generates comparison plots.

To make the comparison meaningful, the next step is to implement real LLM-driven move selection in `traffic/agents/zeroshot_agent.py` and a reasoning loop in `traffic/agents/react_agent.py`.

## Appendix
### Reproducibility
Run the evaluation from the repository root:

```bash
py -m traffic.evaluate --approaches all --episodes 50 --out results --plot
```

### Notes
The current metrics reflect placeholder agent behavior, not a final research comparison. Improving agent logic will change the numbers and likely reduce collisions while increasing success rate.
