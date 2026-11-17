# -*- coding: utf-8 -*-
"""批量生成图 5（VIS-卫星 GSD 开/关）多方法的对应点 CSV，并写入 stats.json。

方法：SIFT（已有）/ LoFTR / XoFTR；条件：gsd0（未归一化）与 gsd1（GSD 归一化）。
同一帧 test02/76。仅用于示意；科学评测仍走 src/ 原生流程。
注意：不施加 max_side，保留 gsd0/gsd1 的真实尺度差异（这正是图 5 要展示的）。

用法：python draw/build_fig5_panels.py
"""
import os, sys, json, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
torch.set_num_threads(2)
import make_match_csv as M

MD = os.path.join(HERE, "match_data")
# 选帧依据：results/raw/map_vis_map_gsd{0,1}.csv 中 SIFT/RoMa/LoFTR 三者
# 在 gsd0 全部 >10 px（或无可解），gsd1 全部 <10 px —— 最能展示 GSD 归一化的作用。
SCENE, FRAME = "test02", 94
METHODS = ["sift", "roma", "loftr"]


def nrows(p):
    if not os.path.isfile(p):
        return -1
    with open(p, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


stats = {}
sp = os.path.join(MD, "stats.json")
if os.path.isfile(sp):
    stats = json.load(open(sp, encoding="utf-8"))

for m in METHODS:
    for gsd in (0, 1):
        key = f"vis_map_{m}_gsd{gsd}_f{FRAME}"
        out = os.path.join(MD, f"vis_map_{m}_g{gsd}_f{FRAME}.csv")
        if os.path.isfile(out) and nrows(out) > 0:
            print(f"[skip] {key} already has {nrows(out)} rows")
            continue
        st = M.extract("vis_map", SCENE, FRAME, m, gsd, out, dev="cpu")
        if st:
            stats[key] = st
        json.dump(stats, open(sp, "w", encoding="utf-8"), indent=2)

json.dump(stats, open(sp, "w", encoding="utf-8"), indent=2)
print("\nfig5 stats:", {k: v for k, v in stats.items() if k.startswith("vis_map")})
