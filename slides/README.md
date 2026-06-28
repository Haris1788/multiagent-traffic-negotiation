# Slides / Deck (R5)

This folder holds a simple slide checklist and a minimal markdown deck you can expand.

Suggested slides:
- Title, authors, date
- Problem statement & domain
- Environment & kinematics
- Agents: Zero-shot (baseline) & ReAct (comparison)
- Evaluation protocol (episodes, metrics)
- Results: table + plots (success rate, collisions, steps)
- Short analysis & takeaways
- Appendix: reproducibility, commands to re-run evaluation

Quick re-run instructions (from repo root):

```bash
python -m pip install -r requirements.txt
python -m traffic.evaluate --approaches all --episodes 50 --out results --plot
```

Copy the generated images from `results/plots` into your slides.
