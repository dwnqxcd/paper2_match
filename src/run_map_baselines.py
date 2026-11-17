# -*- coding: utf-8 -*-
"""在 map 任务（红外↔卫星、可见光↔卫星）上评测 SIFT / LoFTR / RoMa。

两种预处理：
  --gsd 0  查询图用原生分辨率（与既有 SIFT/CFOG 结果一致）
  --gsd 1  查询图按 GSD 比例缩放到与地图块同尺度（公平对待 detector-free 方法）

用法：python src/run_map_baselines.py --task vis_map --methods sift,loftr,roma --gsd 1
"""
import os, sys, json, csv, argparse, time
import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_m3t_crossmodal as R
import run_baselines as BL

DATA = os.path.join(PROJ, "data", "M3T")
RAW = os.path.join(PROJ, "results", "raw")
GSD = {"ir": 0.245, "vis": 0.174}     # 查询图 1 px 对应多少地图 px
MIN_INLIERS = 15


def corner_err(He, Hg, W, Hn):
    return R.sym_err_px(He, Hg, W, Hn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="vis_map", choices=["ir_map", "vis_map"])
    ap.add_argument("--methods", default="sift,loftr,roma")
    ap.add_argument("--gsd", type=int, default=0)
    ap.add_argument("--per_scene", type=int, default=9)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dev = "cuda"
    methods = [x for x in a.methods.split(",") if x]
    for m in methods:
        if m in ("roma", "xoftr"):
            BL.build(m, dev)

    split = "test_dataset"
    ir = json.load(open(os.path.join(DATA, split, "ir.json"), encoding="utf-8"))
    vs = json.load(open(os.path.join(DATA, split, "vis.json"), encoding="utf-8"))

    rows = []
    for sc in ["test01", "test02", "test03", "test04", "test05", "test06"]:
        n = len(ir[sc]["uav_img_path"])
        for fi in range(0, n, max(1, n // a.per_scene)):
            Hir = np.array(ir[sc]["H_matrix"][fi]).reshape(3, 3)
            Hvs = np.array(vs[sc]["H_matrix"][fi]).reshape(3, 3)
            mp = os.path.join(DATA, split, sc, "map",
                              ir[sc]["map_path"][fi] if a.task == "ir_map" else vs[sc]["map_path"][fi])
            if not (os.path.isfile(mp) and os.path.getsize(mp) > 100000):
                continue                      # 地图块缺失/未下完
            if a.task == "ir_map":
                g0 = R.load_gray(os.path.join(DATA, split, sc, "uav_ir", ir[sc]["uav_img_path"][fi]))
                g1 = R.load_gray(os.path.join(DATA, split, sc, "map", ir[sc]["map_path"][fi]))
                Hg = Hir.copy(); ratio = GSD["ir"]
            else:
                g0 = R.load_gray(os.path.join(DATA, split, sc, "uav_vis", vs[sc]["uav_img_path"][fi]))
                g1 = R.load_gray(os.path.join(DATA, split, sc, "map", vs[sc]["map_path"][fi]))
                Hg = Hvs.copy(); ratio = GSD["vis"]

            if a.gsd:
                k = ratio                      # 查询图缩放系数（变为地图尺度）
                g0 = cv2.resize(g0, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
                S = np.diag([k, k, 1.0])
                Hg = Hg @ np.linalg.inv(S)     # 缩放后查询 -> 地图
            Wn, Hn = g0.shape[1], g0.shape[0]

            for mth in methods:
                t = time.time()
                try:
                    fn = {"sift": BL.m_sift, "loftr": BL.m_loftr,
                          "dedode": BL.m_dedode, "roma": BL.m_roma,
                          "disk": BL.m_disk, "xoftr": BL.m_xoftr}[mth]
                    p0, p1 = fn(g0, g1, dev)
                except Exception as e:
                    rows.append(dict(task=a.task, gsd=a.gsd, method=mth, scene=sc, frame=fi + 1,
                                     ok=0, note=str(e)[:40])); continue
                H = None
                if len(p0) >= 8:
                    H, _ = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 4.0,
                                              confidence=0.9999, maxIters=20000)
                if H is None:
                    rows.append(dict(task=a.task, gsd=a.gsd, method=mth, scene=sc, frame=fi + 1,
                                     ok=0, note="no H")); continue
                d = np.linalg.norm(cv2.perspectiveTransform(p0.reshape(-1, 1, 2), H).reshape(-1, 2) - p1, axis=1)
                inl = int((d < 4).sum())
                if inl < MIN_INLIERS:
                    rows.append(dict(task=a.task, gsd=a.gsd, method=mth, scene=sc, frame=fi + 1,
                                     ok=0, note=f"inl={inl}")); continue
                err = corner_err(H, Hg, Wn, Hn)
                rows.append(dict(task=a.task, gsd=a.gsd, method=mth, scene=sc, frame=fi + 1,
                                 ok=1, n_inliers=inl, corner_err=round(err, 2),
                                 rt=round(time.time() - t, 2)))
        print(f"  {sc} done", flush=True)

    out = a.out or os.path.join(RAW, f"map_{a.task}_gsd{a.gsd}.csv")
    keys = ["task", "gsd", "method", "scene", "frame", "ok", "n_inliers", "corner_err", "rt", "note"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in keys})
    print("->", out)

    print(f"\n{'(GSD 归一化)' if a.gsd else '(原生查询图)'}  {a.task}")
    print(f"{'method':8s} {'有效/尝试':>10s} {'中位':>8s} {'S@10':>7s}")
    for mth in methods:
        sub = [r for r in rows if r["method"] == mth]
        ok = [r for r in sub if r.get("ok")]
        if not ok:
            print(f"{mth:8s} {0:4d}/{len(sub):<4d}"); continue
        e = np.array([float(r["corner_err"]) for r in ok])
        print(f"{mth:8s} {len(ok):4d}/{len(sub):<4d} {np.median(e):8.1f} {np.mean(e < 10) * 100:6.1f}%")


if __name__ == "__main__":
    main()
