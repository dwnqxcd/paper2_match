# -*- coding: utf-8 -*-
"""Figure 4: success at 10 px against the modality gap.

Reads the same all_method_stats.json that produces Table 2, so the figure cannot
drift away from the table.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
A = json.load(open(os.path.join(PROJ, "results", "raw", "all_method_stats.json"), encoding="utf-8"))

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "stix"
rcParams["pdf.fonttype"] = 42

tasks = ["vis_sat", "tir_vis", "tir_sat"]
labels = ["VIS-satellite", "TIR-VIS", "TIR-satellite"]
x = np.arange(len(tasks))

series = [("sift", "SIFT (detector-based)", "-o", "#404040"),
          ("loftr", "LoFTR (detector-free)", "-s", "#1c78bf"),
          ("roma", "RoMa (dense)", "-^", "#b83030")]
fig, ax = plt.subplots(figsize=(16.5 / 2.54, 9.0 / 2.54), dpi=300)
fig.patch.set_facecolor("white")
for key, name, mk, col in series:
    y = [A[t][key]["rate"] for t in tasks]
    ax.plot(x, y, mk, lw=1.6, ms=6.5, color=col, mfc=col, label=name, zorder=3)

cfog = [A[t]["cfog"]["rate"] if A[t]["cfog"].get("rate") is not None else np.nan for t in tasks]
ax.plot(x, cfog, "p", ls="None", ms=11, color="#6b3d99", mfc="#6b3d99",
        label="CFOG (cross-modal corr.)", zorder=4)

ax.annotate("classical wins", xy=(0, A["vis_sat"]["sift"]["rate"]),
            xytext=(0.16, A["vis_sat"]["sift"]["rate"] + 3.5), fontsize=7, color="#404040",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
ax.annotate("dense wins", xy=(1, A["tir_vis"]["roma"]["rate"]),
            xytext=(1, A["tir_vis"]["roma"]["rate"] + 4.5), ha="center", fontsize=7, color="#b83030")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.set_xlim(-0.35, 2.45)
ax.set_ylim(0, max(65, max(A[t]["sift"]["rate"] for t in tasks) + 10))
ax.set_ylabel("success rate @ 10 px (%)", fontsize=8)
ax.set_xlabel("modality gap", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, 1.09), handlelength=1.6, columnspacing=1.4)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.4)
out = os.path.join(HERE, "fig4_modality_gap.png")
fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
print("saved", out, "| SIFT vis", A["vis_sat"]["sift"]["rate"],
      "| RoMa tirvis", A["tir_vis"]["roma"]["rate"],
      "| CFOG tirsat", A["tir_sat"]["cfog"]["rate"])
