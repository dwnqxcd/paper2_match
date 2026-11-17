# -*- coding: utf-8 -*-
"""Gate mechanism: occupancy vs corner error, and the error CDF before/after the gate."""
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

rows = []
for r in csv.DictReader(open(P, encoding="utf-8")):
    try:
        rows.append((float(r["inl"]), float(r["cov"]), float(r["res"]), float(r["den"])))
    except Exception:
        continue
inl = np.array([a for a, _, _, _ in rows])
cov = np.array([b for _, b, _, _ in rows])
res = np.array([c for _, _, c, _ in rows])
err = np.array([d for _, _, _, d in rows])
N = len(rows)

# gate of Section 2.3
acc = (inl >= 1000) & (cov >= 0.5) & (res <= 2.5)
# frames the inlier count alone would accept but occupancy rejects
degen = (inl >= 1000) & (cov < 0.5)

fig, axes = plt.subplots(1, 2, figsize=(16.5 / 2.54, 7.4 / 2.54), dpi=300)
fig.patch.set_facecolor("white")
ax = axes[0]

ax.scatter(cov[acc], np.maximum(err[acc], 1), s=3, c="#4a7fb5", alpha=0.45,
           linewidths=0, label="accepted")
ax.scatter(cov[~acc], np.maximum(err[~acc], 1), s=4, c="#a33", alpha=0.7,
           linewidths=0, label="rejected")
ax.set_yscale("log")
ax.axvline(0.5, ls="--", lw=0.8, color="0.35")
ax.axhline(50, ls=":", lw=0.8, color="0.35")
ax.set_xlabel("inlier occupancy $\\omega$", fontsize=8)
ax.set_ylabel("corner error (px)", fontsize=8)
ax.tick_params(labelsize=7)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(1, 2e4)
ax.grid(True, which="major", ls=":", lw=0.4, color="0.88")
ax.legend(fontsize=7, frameon=False, loc="lower left", markerscale=2.2)
ax.annotate("occupancy $\\omega=0.5$", xy=(0.5, 1.6), xytext=(0.53, 2.6), fontsize=7, color="0.3")
ax.annotate("50 px", xy=(0.02, 50), xytext=(0.03, 68), fontsize=7, color="0.3")
ax.text(0.545, 4.2e3, "%d frames pass\nthe inlier count\nbut fail occupancy" % int(degen.sum()),
        fontsize=7, color="#a33", ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
ax.set_title("(a) gate mechanism", fontsize=8, pad=3)

ax = axes[1]
x = np.sort(err[err > 0])
ax.plot(x, 100.0 * np.arange(1, len(x) + 1) / len(x), "-", lw=1.2, color="0.45",
        label="all %d attempts" % N)
xa = np.sort(err[acc])
ax.plot(xa, 100.0 * np.arange(1, len(xa) + 1) / len(xa), "-", lw=1.3, color="#1f4e79",
        label="accepted (%d, %.1f%%)" % (int(acc.sum()), 100.0 * acc.sum() / N))
ax.set_xscale("log")
ax.set_xlim(1, 2e4)
ax.set_ylim(0, 100)
ax.axvline(10, ls=":", lw=0.8, color="0.35")
ax.axvline(50, ls=":", lw=0.8, color="0.35")
ax.set_xlabel("corner error (px)", fontsize=8)
ax.set_ylabel("cumulative fraction of frames (%)", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(True, which="major", ls=":", lw=0.4, color="0.88")
ax.legend(fontsize=7, frameon=False, loc="lower right")
ax.text(11.0, 5.0, "10 px", fontsize=7, color="0.3",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
ax.text(52, 5.0, "50 px", fontsize=7, color="0.3",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
ax.set_title("(b) error distribution", fontsize=8, pad=3)

for a in axes:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.tight_layout(pad=0.5, w_pad=2.6)
out = os.path.join(HERE, "fig_gate.png")
fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.06)
print("saved", out, "| accepted", int(acc.sum()), "| degen(inl>=1000,cov<0.5)", int(degen.sum()))
