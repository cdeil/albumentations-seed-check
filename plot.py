"""Recreate the README figure: uv run --with matplotlib --no-project python plot.py."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = Path(__file__).parent
reports = [
    json.loads((base / f"results/{name}.json").read_text())["results"][0]
    for name in ("legacy", "current")
]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
for ax, data, title, color in zip(
    axes,
    reports,
    ["Albumentations 2.0.8", "AlbumentationsX 2.5.0"],
    ["#ef7044", "#269a83"],
):
    bars = ax.bar(
        ["Neither", "Horizontal\nonly", "Vertical\nonly", "Both"],
        data["counts"].values(),
        color=color,
    )
    ax.bar_label(bars, padding=3)
    ax.set_title(title, loc="left", weight="bold")
    ax.set_ylim(0, 240)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Calls out of 400")
fig.suptitle("Seed 0: two flips at p=0.5", fontsize=16, weight="bold")
fig.savefig(base / "results/seed-0.png", dpi=160)
plt.close(fig)
