# -*- coding: utf-8 -*-
"""图 5：VIS-卫星 3 场景 x 7 方法（含 CFOG）的对应点 CSV。

6 个学习/经典匹配器：run_map_baselines 同口径，gsd=1（查询图按 GSD 缩放到地图尺度）。
CFOG：本身带尺度搜索，不做 GSD；其输出是全局 H，这里在地图上采样网格点、
      用 H 投到地图块，得到示意用的对应（不是稀疏关键点，图注需说明）。

用法（建议 CUDA 关闭，纯 CPU）：python draw/build_fig5_panels2.py
"""
import os, sys, json, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)   # draw/ 优先，避免 src/ 下同名旧 make_match_csv 抢到 import
import numpy as np
import cv2
cv2.setNumThreads(0)
import torch
torch.set_num_threads(2)
import make_match_csv as M
import run_m3t_cfog as CF

MD = os.path.join(HERE, "match_data")
DATA = os.path.join(os.path.dirname(HERE), "data", "M3T", "test_dataset")
SCENES = [("test01", 60), ("test02", 60), ("test03", 27), ("test05", 60)]  # 四类场景；每景取能体现差异的帧
GA_METHODS = ["sift", "disk", "dedode", "loftr", "xoftr", "roma"]
BASE = 0.174


def nrows(p):
    if not os.path.isfile(p):
        return -1
    with open(p, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def cfog_panel(sc, fr, out):
    vs = json.load(open(os.path.join(DATA, "vis.json"), encoding="utf-8"))
    fi = fr - 1
    q = CF.load_gray(os.path.join(DATA, sc, "uav_vis", vs[sc]["uav_img_path"][fi]))
    mp = CF.load_gray(os.path.join(DATA, sc, "map", vs[sc]["map_path"][fi]))
    Hg = np.array(vs[sc]["H_matrix"][fi]).reshape(3, 3)
    H, extra = CF.estimate_cfog(q, mp, BASE, D=700, coarse_rot=15, nbins=8, sigma=4.0, refine=True)
    ha, wa = q.shape[:2]; hb, wb = mp.shape[:2]
    if H is None:
        rows = []
        ce = None
    else:
        ce = CF.corner_err(H, Hg, wa, ha)
        # 采样网格点，投到地图；落在图内的保留
        gx = np.linspace(0.08 * wa, 0.92 * wa, 9)
        gy = np.linspace(0.08 * ha, 0.92 * ha, 7)
        pts = np.array([[(x, y)] for y in gy for x in gx], np.float32)
        warped = cv2.perspectiveTransform(pts, H)
        rows = []
        for (x, y), (u, v) in zip(pts.reshape(-1, 2), warped.reshape(-1, 2)):
            if 0 <= u < wb and 0 <= v < hb:
                rows.append(dict(xa=round(float(x), 2), ya=round(float(y), 2),
                                 xb=round(float(u), 2), yb=round(float(v), 2),
                                 wa=wa, ha=ha, wb=wb, hb=hb))
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["xa", "ya", "xb", "yb", "wa", "ha", "wb", "hb"])
        wr.writeheader(); wr.writerows(rows)
    print(f"[vis_map/{sc}/{fr}/cfog] pts={len(rows)} corner={ce} -> {out}")
    return dict(candidates=len(rows), inliers=len(rows),
                corner=(round(float(ce), 2) if ce is not None else None))


stats = {}
sp = os.path.join(MD, "stats.json")
if os.path.isfile(sp):
    stats = json.load(open(sp, encoding="utf-8"))

for sc, fr in SCENES:
    for m in GA_METHODS:
        key = f"vis_map_{m}_{sc}"
        out = os.path.join(MD, f"vis_map_{m}_{sc}_f{fr}.csv")
        if os.path.isfile(out) and nrows(out) > 0:
            print(f"[skip] {key}")
            continue
        st = M.extract("vis_map", sc, fr, m, 1, out, dev="cpu", max_side=0)
        if st:
            stats[key] = st
        json.dump(stats, open(sp, "w", encoding="utf-8"), indent=2)

    key = f"vis_map_cfog_{sc}"
    out = os.path.join(MD, f"vis_map_cfog_{sc}_f{fr}.csv")
    if not (os.path.isfile(out) and nrows(out) > 0):
        st = cfog_panel(sc, fr, out)
        stats[key] = st
        json.dump(stats, open(sp, "w", encoding="utf-8"), indent=2)
    else:
        print(f"[skip] {key}")

print("\ndone")
