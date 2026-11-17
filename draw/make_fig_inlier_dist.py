# -*- coding: utf-8 -*-
"""Inlier-count distributions on visible-to-satellite: why the thermal-visible gate does not transfer."""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
P = os.path.join(PROJ, "results", "raw", "map_vis_map_gsd1.csv")

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "stix"
rcParams["pdf.fonttype"] = 42

rows = list(csv.DictReader(open(P, encoding="utf-8")))
fig, ax = plt.subplots(figsize=(16.5 / 2.54, 6.8 / 2.54), dpi=300)
fig.patch.set_facecolor("white")
colors = {"sift": "#1f4e79", "loftr": "#c98a1b", "roma": "#2e7d5b"}
labels = {"sift": "SIFT", "loftr": "LoFTR", "roma": "RoMa"}

for m in ("sift", "loftr", "roma"):
    v = np.array([float(r["n_inliers"]) for r in rows
                  if r["method"] == m and r.get("n_inliers") not in ("", "nan")])
    v = np.sort(v)
    ax.plot(v, 100.0 * np.arange(1, len(v) + 1) / len(v), lw=1.4, color=colors[m],
            label="%s (median %d)" % (labels[m], int(np.median(v))))

ax.axvline(1000, ls="--", lw=0.9, color="0.3")
ax.text(1060, 42, "thermal-visible\nthreshold (1000)", fontsize=7, color="0.3")
ax.set_xscale("log")
ax.set_xlim(10, 5e3)
ax.set_ylim(0, 100)
ax.set_xlabel("inlier count per frame", fontsize=8)
ax.set_ylabel("cumulative fraction of frames (%)", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(True, which="major", ls=":", lw=0.4, color="0.88")
ax.legend(fontsize=7, frameon=False, loc="lower right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
out = os.path.join(HERE, "fig_inlier_dist.png")
fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
print("saved", out)
