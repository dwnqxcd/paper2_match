# -*- coding: utf-8 -*-
"""Figure 7: median (with interquartile band) of inlier count and corner error against the
sampling budget, over all 1,602 thermal-visible frames and three budgets."""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
P = os.path.join(PROJ, "results", "raw", "density_full.csv")

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

by = {}
for r in csv.DictReader(open(P, encoding="utf-8")):
    try:
        inl = int(r["inliers"])
    except Exception:
        continue
    err = None
    try:
        err = float(r["err_calib"]) if r.get("err_calib") not in ("", None) else None
    except Exception:
        pass
    by.setdefault(r["N"], {"inl": [], "err": [], "att": 0})
    by[r["N"]]["att"] += 1
    if inl >= 15:
        by[r["N"]]["inl"].append(inl)
        if err is not None:
            by[r["N"]]["err"].append(err)

ns = sorted(by, key=int)
x = np.array([int(n) for n in ns], float)


def stat(v):
    v = np.asarray(v, float)
    return np.median(v), np.percentile(v, 25), np.percentile(v, 75)


def num(v):
    """图上的数字：千位不加分隔（2000）；到万位才分组。"""
    n = int(round(float(v)))
    if n < 10000:
        return str(n)
    s = str(n); parts = []
    while len(s) > 4:
        parts.insert(0, s[-4:]); s = s[:-4]
    parts.insert(0, s)
    return ",".join(parts)


mi, lo_i, hi_i = zip(*[stat(by[n]["inl"]) for n in ns])
me, lo_e, hi_e = zip(*[stat(by[n]["err"]) for n in ns])
s10 = [100.0 * sum(1 for e in by[n]["err"] if e < 10) / max(1, by[n]["att"]) for n in ns]

fig, axes = plt.subplots(1, 2, figsize=(16.5 / 2.54, 7.4 / 2.54), dpi=300)
fig.patch.set_facecolor("white")

ax = axes[0]
ax.fill_between(x, lo_i, hi_i, color="#1f4e79", alpha=0.16, lw=0)
ax.plot(x, mi, "-o", color="#1f4e79", ms=4, lw=1.1)
ax.set_xscale("log")
ax.set_xticks(x); ax.set_xticklabels([num(v) for v in x])
yt = [0, 2000, 4000, 6000, 8000]
ax.set_yticks(yt); ax.set_yticklabels([num(v) for v in yt])
ax.set_ylim(0, 8000)
ax.set_xlabel("sampling budget N", fontsize=8)
ax.set_ylabel("inliers", fontsize=8)
ax.set_title("(a)", fontsize=8, loc="left")
ax.tick_params(labelsize=7)
r1 = mi[1] / mi[0]; r2 = mi[2] / mi[1]
ax.text(0.04, 0.96, f"median {num(mi[0])} to {num(mi[2])}\n(x{r1:.1f} and x{r2:.1f} per doubling)",
        transform=ax.transAxes, fontsize=7, va="top")

ax = axes[1]
ax.fill_between(x, lo_e, hi_e, color="#a33", alpha=0.16, lw=0)
ax.plot(x, me, "-o", color="#a33", ms=4, lw=1.1)
ax.set_xscale("log")
ax.set_xticks(x); ax.set_xticklabels([num(v) for v in x])
yt = [0, 5, 10, 15, 20, 25, 30]
ax.set_yticks(yt); ax.set_yticklabels([num(v) for v in yt])
ax.set_ylim(0, 30)
ax.set_xlabel("sampling budget N", fontsize=8)
ax.set_ylabel("corner error (px)", fontsize=8)
ax.set_title("(b)", fontsize=8, loc="left")
ax.tick_params(labelsize=7)
ax.text(0.04, 0.96, f"median {me[0]:.2f} to {me[2]:.2f} px\ns@10 " +
        ", ".join(f"{v:.1f}%" for v in s10),
        transform=ax.transAxes, fontsize=7, va="top")

for ax in axes:
    ax.grid(True, which="major", ls=":", lw=0.4, color="0.8")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", which="minor", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(pad=0.4)
out = os.path.join(HERE, "fig7_density.png")
fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.splitext(out)[0] + ".pdf", facecolor="white", bbox_inches="tight", pad_inches=0.02)
print("saved", out, "tiers", ns)
