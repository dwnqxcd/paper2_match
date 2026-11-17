# -*- coding: utf-8 -*-
"""Corner error of accepted frames, by scene: one scene carries a constant annotation offset."""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
P = os.path.join(PROJ, "results", "raw", "pipeline_ablation_full.csv")
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "stix"
rcParams["pdf.fonttype"] = 42

rows = [r for r in csv.DictReader(open(P, encoding="utf-8")) if r["accepted"] == "1"]
scenes = ["test01", "test02", "test03", "test04", "test05", "test06"]
data = [np.array([float(r["den"]) for r in rows if r["scene"] == s]) for s in scenes]

fig, ax = plt.subplots(figsize=(16.5 / 2.54, 6.6 / 2.54), dpi=300)
fig.patch.set_facecolor("white")
for i, (s, d) in enumerate(zip(scenes, data)):
    if len(d) == 0:
        continue
    hot = (s == "test06")
    col = "#a33" if hot else "#1f4e79"
    face = "#f2d9d9" if hot else "#dfe7f0"
    if len(d) < 5:                      # too few frames for a box: show the points
        ax.scatter(d, np.full(len(d), i), s=14, c=col, marker="D", zorder=4,
                   edgecolors="white", linewidths=0.5)
        continue
    ax.boxplot([d], positions=[i], widths=0.55, vert=False, showfliers=False,
               patch_artist=True, medianprops=dict(color=col, lw=1.2),
               boxprops=dict(facecolor=face, edgecolor=col, lw=0.7),
               whiskerprops=dict(color=col, lw=0.7), capprops=dict(color=col, lw=0.7))

ax.axvline(20, ls=":", lw=0.9, color="0.4")
ax.text(21, -0.42, "20 px", fontsize=7, color="0.35")
ax.set_xscale("log")
ax.set_xlim(5, 3e3)
ax.set_yticks(range(len(scenes)))
ax.set_yticklabels([s + " (%d)" % len(d) for s, d in zip(scenes, data)], fontsize=7)
ax.set_ylim(-0.6, 5.8)
ax.set_xlabel("corner error of accepted frames (px)", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(True, which="major", axis="x", ls=":", lw=0.4, color="0.88")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
out = os.path.join(HERE, "fig_scene_error.png")
fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
print("saved", out, [len(d) for d in data])
