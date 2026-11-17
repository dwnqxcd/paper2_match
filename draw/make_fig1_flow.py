# -*- coding: utf-8 -*-
"""Figure 1: pipeline overview. Clean flow chart, white background, Times/STIX math."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "stix"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

W, H = 17.0, 13.2
fig, ax = plt.subplots(figsize=(16.5 / 2.54, H / 2.54), dpi=300)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 11); ax.set_ylim(0, 14); ax.axis("off")


def box(cx, cy, w, h, text, fs=8.0, lw=0.9, ls="-"):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.10",
                       fc="white", ec="black", lw=lw, ls=ls, zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.45)


def arrow(p0, p1, ls="-", label=None, lx=0.0, ly=0.0):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black",
                                ls=ls, shrinkA=1.5, shrinkB=1.5,
                                mutation_scale=9), zorder=1)
    if label:
        ax.text((p0[0] + p1[0]) / 2 + lx, (p0[1] + p1[1]) / 2 + ly, label,
                ha="center", va="center", fontsize=7.2,
                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"), zorder=4)


# ---- main column ----
box(4.2, 13.2, 7.2, 1.0,
    "Input: query $I_q$ with GSD $g_q$;  reference $I_r$ with GSD $g_r$\n"
    "(a low-resolution SIFT probe supplies $s$ when metadata are absent)", fs=7.8)

box(4.2, 11.3, 7.6, 2.0,
    "Stage 1 — scale normalization: one of two operations, by reference type\n"
    "georeferenced chip:  $s=g_q/g_r$ (or the SIFT probe);  if $|\\log_2 s|>0.35$:\n"
    "    $I_q\\leftarrow\\mathrm{resize}(I_q,s)$;  $H\\leftarrow H\\,\\mathrm{diag}(1/s,1/s,1)$\n"
    "camera image:  $I_r\\leftarrow\\mathrm{resize}(I_r,\\,w_q/w_r)$;  $H\\leftarrow\\mathrm{diag}(r,r,1)\\,H$", fs=7.0)

box(4.2, 9.3, 7.4, 1.5,
    "Stage 2 — adaptive correspondence sampling\n"
    "dense matcher $\\rightarrow$ warp $W$, certainty $C$;  "
    "$N=\\mathrm{clamp}(\\rho A,\\,2000,\\,8000)$\n"
    "stratified $G\\times G$ grid, $G=8$, per-cell quota $k=N/G^2$", fs=7.6)

box(4.2, 7.4, 7.4, 1.2,
    "Stage 3 — robust fit\n"
    "MAGSAC++, forward transfer error 3 px $\\rightarrow$ $H$ and inlier set", fs=7.8)

box(4.2, 5.7, 7.4, 1.0,
    "Verification (annotation-free):  $N_{inl}\\geq1000$,  "
    "$\\omega\\geq0.5$,  median residual $\\leq2.5$ px", fs=7.3)

# ---- decision diamond ----
cx, cy, hw, hh = 4.2, 4.1, 1.5, 0.95
ax.add_patch(Polygon([[cx, cy + hh], [cx + hw, cy], [cx, cy - hh], [cx - hw, cy]],
                     closed=True, fc="white", ec="black", lw=0.9, zorder=2))
ax.text(cx, cy, "all three\ncriteria met?", ha="center", va="center", fontsize=7.6, zorder=3)

box(4.2, 2.0, 4.4, 1.0, "Accept\noutput $H$ with the frame marked solved", fs=7.8)
box(8.5, 4.1, 3.6, 1.5,
    "Escalate\n(i) phase-congruency rematch\n(ii) similarity model (4 DoF)", fs=7.4)
box(8.5, 1.6, 3.6, 0.9, "Reject\nno transform is reported", fs=7.6)

# ---- arrows ----
arrow((4.2, 12.7), (4.2, 12.1))
arrow((4.2, 10.5), (4.2, 10.05))
arrow((4.2, 8.55), (4.2, 8.0))
arrow((4.2, 6.8), (4.2, 6.2))
arrow((4.2, 5.2), (4.2, 4.75))
arrow((4.2, 3.15), (4.2, 2.5), label="yes", lx=0.35)
arrow((5.7, 4.1), (6.7, 4.1), label="no", ly=0.28)
arrow((7.6, 3.33), (6.45, 2.15), ls="--", label="passes", lx=0.35, ly=-0.05)
arrow((8.5, 3.35), (8.5, 2.05), label="fails", lx=0.45)

ax.text(0.15, 0.25, "Thresholds are fixed per task pair; the pipeline uses no annotation at test time.",
        fontsize=7.0, ha="left", va="bottom", style="italic")

fig.tight_layout(pad=0.15)
out = os.path.join(HERE, "fig1_framework.png")
fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
fig.savefig(os.path.splitext(out)[0] + ".pdf", facecolor="white", bbox_inches="tight", pad_inches=0.03)
print("saved", out)
