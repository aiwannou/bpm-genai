"""Q7 figure: time (first pass + self-check) and tokens (input + output) per run.

    python make_q7_figure.py      -> q7_time_tokens.png (from q7_time_tokens.csv)
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

u = pd.read_csv(HERE / "q7_time_tokens.csv")
u["run"] = u.model.map(SHORT) + " · " + u.method
u["part"] = u.stage.where(u.stage == "first pass", "self-check")
t = u.groupby(["run", "part"])[["seconds", "input_tokens", "output_tokens"]].sum().unstack("part").fillna(0)
t = t.loc[RUNS]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2))
x = range(4)
fp, sc = t[("seconds", "first pass")], t[("seconds", "self-check")]
a1.bar(x, fp, width=0.6, color=BLUE, edgecolor="white", linewidth=2, label="First pass")
a1.bar(x, sc, width=0.6, bottom=fp, color=ORANGE, edgecolor="white", linewidth=2, label="Self-check")
for i, tot in enumerate(fp + sc):
    a1.text(i, tot + 5, f"{tot:.0f} s", ha="center", va="bottom", color=INK, fontsize=9)
a1.set_title("Wall-clock time per run (seconds)", color=INK, fontsize=11)
a1.set_ylim(0, 330)

inp = t[("input_tokens", "first pass")] + t[("input_tokens", "self-check")]
out = t[("output_tokens", "first pass")] + t[("output_tokens", "self-check")]
a2.bar(x, inp / 1000, width=0.6, color=BLUE, edgecolor="white", linewidth=2, label="Input tokens")
a2.bar(x, out / 1000, width=0.6, bottom=inp / 1000, color=ORANGE, edgecolor="white", linewidth=2,
       label="Output tokens")
for i, tot in enumerate((inp + out) / 1000):
    a2.text(i, tot + 1, f"{tot:.1f}k", ha="center", va="bottom", color=INK, fontsize=9)
a2.set_title("Tokens per run (thousands)", color=INK, fontsize=11)
a2.set_ylim(0, 78)
for ax in (a1, a2):
    ax.set_xticks(list(x), RUNS, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(HERE / "q7_time_tokens.png", dpi=200)
print("saved q7_time_tokens.png")
