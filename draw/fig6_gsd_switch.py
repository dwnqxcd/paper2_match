# -*- coding: utf-8 -*-
"""fig6_gsd_switch.py 的 Python 回退实现（当无头 MATLAB 不可用时代替 .m）。
读取 results/raw/map_stats.json 的 fig6 字段，复刻 draw/fig6_gsd_switch.m 的样式：
6 组（VIS/TIR × SIFT/LoFTR/RoMa），每组 2 柱（before 灰 / after 蓝），Times New Roman，300 dpi + pdf。
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
JSON = os.path.join(PROJ, "results", "raw", "map_stats.json")

j = json.load(open(JSON, encoding="utf-8"))
after = j["fig6"]["after"]          # [vis_sift, vis_loftr, vis_roma, tir_sift, tir_loftr, tir_roma]
before = [0.0] * len(after)
labels = ["VIS-sat. SIFT", "VIS-sat. LoFTR", "VIS-sat. RoMa",
          "TIR-sat. SIFT", "TIR-sat. LoFTR", "TIR-sat. RoMa"]

# 尽量用 Times New Roman，缺失则回退
try:
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    FONT = "Times New Roman"
except Exception:
    FONT = "DejaVu Serif"
plt.rcParams["font.family"] = FONT

fig = plt.figure(figsize=(16.5 / 2.54, 9.5 / 2.54), dpi=300)
x = range(len(labels))
b = plt.bar(x, before, width=0.8, color=(0.80, 0.80, 0.80),
            edgecolor=(0.55, 0.55, 0.55), linewidth=0.5, label="without GSD normalization")
a = plt.bar(x, after, width=0.8, color=(0.11, 0.47, 0.75),
            edgecolor="none", label="with GSD normalization")
plt.xticks(list(x), labels, rotation=28, ha="right", fontsize=7)
plt.yticks(fontsize=7)
plt.ylabel("Success rate @ 10 px (%)", fontsize=8)
plt.ylim(0, 70)
for spine in ("top", "right"):
    plt.gca().spines[spine].set_visible(False)
plt.gca().tick_params(width=0.8, direction="out")
for k, v in enumerate(after):
    if v > 0:
        plt.text(k + 0.14, v + 2, "%.1f" % v, fontsize=8,
                 color=(0.11, 0.30, 0.55), ha="center")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=7)
plt.tight_layout()
fig.savefig(os.path.join(HERE, "fig6_gsd_switch.png"), dpi=300)
fig.savefig(os.path.join(HERE, "fig6_gsd_switch.pdf"))
print("fig6 saved (python fallback)")
