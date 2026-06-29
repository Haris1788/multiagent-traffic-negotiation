import csv
import os
import matplotlib.pyplot as plt

summary_file = "summary.csv"

agents = []
success_rates = []
collision_counts = []
average_steps = []

with open(summary_file, newline="") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        agents.append(row["Agent"])
        success_rates.append(float(row["Success Rate"]) * 100)  # Convert to %
        collision_counts.append(int(row["Collision Count"]))
        average_steps.append(float(row["Average Steps"]))

# Create output directory
os.makedirs("plots", exist_ok=True)

# -------------------------------
# Success Rate
# -------------------------------
plt.figure(figsize=(6, 4))
plt.bar(agents, success_rates)
plt.title("Success Rate")
plt.ylabel("Success (%)")
plt.ylim(0, 100)
plt.savefig("plots/success_rate.png")
plt.close()

# -------------------------------
# Collision Count
# -------------------------------
plt.figure(figsize=(6, 4))
plt.bar(agents, collision_counts)
plt.title("Collision Count")
plt.ylabel("Collisions")
plt.savefig("plots/collision_count.png")
plt.close()

# -------------------------------
# Average Steps
# -------------------------------
plt.figure(figsize=(6, 4))
plt.bar(agents, average_steps)
plt.title("Average Steps")
plt.ylabel("Steps")
plt.savefig("plots/average_steps.png")
plt.close()

print("Plots saved to the 'plots' folder.")