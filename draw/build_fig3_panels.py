# -*- coding: utf-8 -*-
"""批量生成图 3（TIR-VIS）多方法 x 多场景的对应点 CSV，并写入 stats.json。

方法：XoFTR / RoMa / LoFTR / SIFT；场景：test02@33、test05@33。
全部在 CPU 上跑，各自长边缩到 <=1280（红外 1280x1024 保持原生，可见光 4032->1280），
只用于示意；科学评测仍走 src/ 下原生分辨率流程。

用法：python draw/build_fig3_panels.py
"""
import os, sys, json, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
torch.set_num_threads(2)
import make_match_csv as M

MD = os.path.join(HERE, "match_data")
SCENES = [("test01", 60), ("test02", 60), ("test03", 27), ("test05", 60)]  # 四类场景；每景取能体现差异的帧
METHODS = ["sift", "disk", "dedode", "loftr", "xoftr", "roma"]
MS = 0    # 原生分辨率；各匹配器内部自行缩放到 1024/1280，角点误差与评测同尺度

def nrows(p):
    if not os.path.isfile(p):
        return -1
    with open(p, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))

stats = {}
sp = os.path.join(MD, "stats.json")
if os.path.isfile(sp):
    stats = json.load(open(sp, encoding="utf-8"))

for sc, fr in SCENES:
    for m in METHODS:
        key = f"tir_vis_{m}_{sc}"
        out = os.path.join(MD, f"tir_vis_{m}_{sc}_f{fr}.csv")
        if os.path.isfile(out) and nrows(out) > 0:
            print(f"[skip] {key} already has {nrows(out)} rows")
            continue
        st = M.extract("tir_vis", sc, fr, m, 0, out, dev="cpu", max_side=MS)
        if st:
            stats[key] = st
        json.dump(stats, open(sp, "w", encoding="utf-8"), indent=2)

json.dump(stats, open(sp, "w", encoding="utf-8"), indent=2)
print("\nstats keys:", sorted(stats.keys()))
