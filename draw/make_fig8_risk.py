# -*- coding: utf-8 -*-
"""Figure 8: risk-coverage of the verification gate on the full thermal-visible set.

Sweeps the inlier-count threshold on pipeline_ablation_full.csv and plots, inside the
accepted set, the fraction above 50 px (gross) and above 20 px (reference-biased).
Annotation at T = 1,000 is computed from the same file, so it matches Table 6.
"""
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
rcParams["ps.fonttype"] = 42

rows = []
for r in csv.DictReader(open(P, encoding="utf-8")):
    try:
        inl = float(r["inl"]); e = float(r["den"]); cov = float(r["cov"]); res = float(r["res"])
    except Exception:
        continue
    rows.append((inl, e, cov, res))
N = len(rows)
inl = np.array([a for a, _, _, _ in rows])
err = np.array([b for _, b, _, _ in rows])
covv = np.array([c for _, _, c, _ in rows])
ress = np.array([d for _, _, _, d in rows])


def stats(T):
    # 三条准则同时生效：内点阈值 T、占用率 >= 0.5、残差 <= 2.5 px
    m = (inl >= T) & (covv >= 0.5) & (ress <= 2.5)
    n = int(m.sum())
    if n == 0:
        return 0.0, np.nan, np.nan
    return 100.0 * n / N, 100.0 * (err[m] > 50).mean(), 100.0 * (err[m] > 20).mean()


Ts = np.unique(np.concatenate([[0], inl]))
cov, g50, g20 = zip(*[stats(t) for t in Ts])

c1000, r50_1000, r20_1000 = stats(1000)

fig, ax = plt.subplots(figsize=(16.5 / 2.54, 9.0 / 2.54), dpi=300)
fig.patch.set_facecolor("white")

ax.plot(cov, g50, "-o", ms=4, lw=1.2, color="#1f4e79", label="gross failures (> 50 px)")
ax.plot(cov, g20, "-s", ms=4, lw=1.2, color="#a33", label="reference-biased (> 20 px)")
ax.set_xlabel("coverage: fraction of frames accepted (%)", fontsize=8)
ax.set_ylabel("error rate inside accepted set (%)", fontsize=8)
ax.tick_params(labelsize=7)
ax.set_xlim(max(cov), 50)
ax.set_ylim(-2, max(np.nanmax(g20), 40) + 5)
ax.grid(True, which="major", ls=":", lw=0.4, color="0.85")
ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2)

ax.annotate(f"T = 1,000 inliers\ncoverage {c1000:.1f}%, gross error {r50_1000:.1f}%",
            xy=(c1000, r50_1000), xytext=(c1000 - 6, 13.5), fontsize=7.5,
            arrowprops=dict(arrowstyle="-", lw=0.7, color="0.3"))
ax.text(max(cov) - 2, 31, "residual = annotation bias", fontsize=7.2, color="#a33")
ax.text(70, 4.6, "gate removes gross failures", fontsize=7.2, color="#1f4e79")

for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
out = os.path.join(HERE, "fig8_risk_coverage.png")
fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.splitext(out)[0] + ".pdf", facecolor="white", bbox_inches="tight", pad_inches=0.02)
print(f"saved {out}; T=1000 -> coverage {c1000:.1f}%, gross {r50_1000:.1f}%, ref-biased {r20_1000:.1f}%; N={N}")
