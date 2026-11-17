# -*- coding: utf-8 -*-
"""Figure 9: oracle alignment on the converged frames.
(a) median corner error after allowing 0, 2 and 4 DoF per frame;
(b) median point displacement after similarity alignment, by distance from the principal point.
Reads results/raw/oracle_summary.json.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
S = json.load(open(os.path.join(PROJ, "results", "raw", "oracle_summary.json"), encoding="utf-8"))

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "stix"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

o = S["oracle"]
vals = [o["err_global"]["median"], o["err_trans"]["median"], o["err_sim"]["median"]]
labels = ["none", "translation\n(2 DoF)", "similarity\n(4 DoF)"]

fig, axes = plt.subplots(1, 2, figsize=(16.5 / 2.54, 7.0 / 2.54), dpi=300)
fig.patch.set_facecolor("white")

ax = axes[0]
b = ax.bar(range(3), vals, width=0.55, color="#1f4e79", alpha=0.85, edgecolor="black", lw=0.6)
for i, v in enumerate(vals):
    ax.text(i, v * 1.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("median corner error (px)", fontsize=8)
ax.set_title("(a)", fontsize=8, loc="left")
ax.tick_params(labelsize=7)
ax.set_ylim(0, max(vals) * 1.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax = axes[1]
rad = S["radial"]
ks = sorted(rad, key=lambda k: float(k.split("-")[0]))
xs = np.arange(len(ks))
yv = [rad[k] for k in ks]
ax.plot(xs, yv, "-o", color="#a33", ms=4, lw=1.1)
for x, y in zip(xs, yv):
    ax.text(x, y + max(yv) * 0.04, f"{y:.2f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels([k.replace("-", "-") for k in ks], fontsize=7)
ax.set_xlabel("distance from principal point (px)", fontsize=8)
ax.set_ylabel("median point displacement (px)", fontsize=8)
ax.set_title("(b)", fontsize=8, loc="left")
ax.tick_params(labelsize=7)
ax.set_ylim(0, max(yv) * 1.35)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout(pad=0.4)
out = os.path.join(HERE, "fig9_reference_floor.png")
fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.splitext(out)[0] + ".pdf", facecolor="white", bbox_inches="tight", pad_inches=0.02)
print("saved", out, "N_close =", S.get("N_close"))
