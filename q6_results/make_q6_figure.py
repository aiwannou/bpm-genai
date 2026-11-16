"""Q6 figure: first-pass metrics per run vs the final value after the self-check.

    python make_q6_figure.py      -> q6_first_pass.png (from q6_metrics.csv)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
SHORT = {"GPT-OSS 120B (OpenAI)": "GPT-OSS", "Qwen 3.8 27B (Alibaba)": "Qwen"}
RUNS = ["GPT-OSS · A", "GPT-OSS · B", "Qwen · A", "Qwen · B"]

m = pd.read_csv(HERE / "q6_metrics.csv")
m["run"] = m.model.map(SHORT) + " · " + m.method
first = m[m.stage == "first pass"].set_index("run").loc[RUNS]

fig, axes = plt.subplots(1, 4, figsize=(10, 2.9), sharey=True)
for ax, metric in zip(axes, ["Precision", "Recall", "F1", "Correct repair rate"]):
    v = first[metric]
    bars = ax.bar(range(4), v, width=0.6, color=BLUE, edgecolor="white", linewidth=2)
    ax.axhline(1.0, color=ORANGE, linewidth=2, linestyle="--")
    for b, x in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, x - 0.02, f"{x:.2f}", ha="center", va="top",
                color="white", fontsize=9, fontweight="bold")
    ax.set_title(metric, color=INK, fontsize=11)
    ax.set_xticks(range(4), [r.replace(" · ", "\n") for r in RUNS], fontsize=9)
    ax.set_ylim(0.6, 1.04)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Score")
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE),
                    plt.Line2D([0], [0], color=ORANGE, linewidth=2, linestyle="--")],
           labels=["First pass (before self-check)", "Final, after self-check (all runs = 1.00)"],
           loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(HERE / "q6_first_pass.png", dpi=200)
print("saved q6_first_pass.png")
