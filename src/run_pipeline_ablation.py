# -*- coding: utf-8 -*-
"""端到端消融：基线 → +GSD → +密度 → +验证门（同一批帧，逐级加组件）。

用法：python src/run_pipeline_ablation.py --per_scene 7 --out results/raw/pipeline_ablation.csv
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

GATE_INL = 1000          # 验证门：最小内点
GATE_COV = 0.50          # 验证门：内点网格覆盖率
GATE_RES = 2.5           # 验证门：内点残差


def grid_coverage(pts, w, h, g=8):
    gx = np.clip((pts[:, 0] / w * g).astype(int), 0, g - 1)
    gy = np.clip((pts[:, 1] / h * g).astype(int), 0, g - 1)
    return len(set(zip(gx, gy))) / float(g * g)


def fit_score(p0, p1, Hg, w, h):
    if len(p0) < 8:
        return None
    H, _ = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 3.0, confidence=0.9999, maxIters=20000)
    if H is None:
        return None
    d = np.linalg.norm(cv2.perspectiveTransform(p0.reshape(-1, 1, 2), H).reshape(-1, 2) - p1, axis=1)
    inl = int((d < 3).sum())
    if inl < 15:
        return None
    m = d < 3
    return dict(err=R.sym_err_px(H, Hg, w, h), n_inl=inl,
                res=float(np.median(d[m])), cov=grid_coverage(p0[m], w, h))


def sample_pts(model, g0, g1, num):
    from PIL import Image
    warp, cert = model.match(Image.fromarray(cv2.cvtColor(g0, cv2.COLOR_GRAY2RGB)),
                             Image.fromarray(cv2.cvtColor(g1, cv2.COLOR_GRAY2RGB)),
                             batched=True, device="cuda")
    mm, _ = model.sample(warp, cert, num=num)
    mm = mm.cpu().numpy()
    p0 = np.stack([g0.shape[1] * (mm[:, 0] + 1) / 2, g0.shape[0] * (mm[:, 1] + 1) / 2], 1).astype(np.float32)
    p1 = np.stack([g1.shape[1] * (mm[:, 2] + 1) / 2, g1.shape[0] * (mm[:, 3] + 1) / 2], 1).astype(np.float32)
    return p0, p1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_scene", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(RAW, "pipeline_ablation.csv"))
    a = ap.parse_args()

    BL.build("roma", "cuda")
    model = BL._CACHE["roma"]

    ir = json.load(open(os.path.join(DATA, "test_dataset", "ir.json"), encoding="utf-8"))
    vs = json.load(open(os.path.join(DATA, "test_dataset", "vis.json"), encoding="utf-8"))

    rows = []
    for sc in ["test01", "test02", "test03", "test04", "test05", "test06"]:
        n = len(ir[sc]["uav_img_path"])
        for fi in range(0, n, max(1, n // a.per_scene)):
            pv = os.path.join(DATA, "test_dataset", sc, "uav_vis", vs[sc]["uav_img_path"][fi])
            if not R._valid(pv):
                continue
            g0 = R.load_gray(os.path.join(DATA, "test_dataset", sc, "uav_ir", ir[sc]["uav_img_path"][fi]))
            g1n = R.load_gray(pv)
            k = g0.shape[1] / g1n.shape[1]
            g1 = cv2.resize(g1n, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            Hir = np.array(ir[sc]["H_matrix"][fi]).reshape(3, 3)
            Hvs = np.array(vs[sc]["H_matrix"][fi]).reshape(3, 3)
            Hg_gap = np.diag([k, k, 1.0]) @ (np.linalg.inv(Hvs) @ Hir)
            Hg_raw = np.linalg.inv(Hvs) @ Hir
            w, h = g0.shape[1], g0.shape[0]

            # 1) 基线：不做 GSD 对齐（VIS 原生分辨率）
            p0, p1 = sample_pts(model, g0, g1n, 2000)
            r_base = fit_score(p0, p1, Hg_raw, w, h)

            # 2) +GSD：VIS 缩放到 IR 尺度
            p0, p1 = sample_pts(model, g0, g1, 2000)
            r_gap = fit_score(p0, p1, Hg_gap, w, h)

            # 3) +密度：采样 8000
            p0d, p1d = sample_pts(model, g0, g1, 8000)
            r_den = fit_score(p0d, p1d, Hg_gap, w, h)

            # 4) +验证门
            acc = (r_den is not None and r_den["n_inl"] >= GATE_INL
                   and r_den["cov"] >= GATE_COV and r_den["res"] <= GATE_RES)
            rows.append(dict(scene=sc, frame=fi + 1,
                             base="" if r_base is None else round(r_base["err"], 2),
                             gap="" if r_gap is None else round(r_gap["err"], 2),
                             den="" if r_den is None else round(r_den["err"], 2),
                             inl="" if r_den is None else r_den["n_inl"],
                             cov="" if r_den is None else round(r_den["cov"], 2),
                             res="" if r_den is None else round(r_den["res"], 2),
                             accepted=int(acc),
                             err_final=(round(r_den["err"], 2) if acc else "")))
        print(f"  {sc} done", flush=True)

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print("->", a.out)

    N = len(rows)
    def stat(key):
        v = [float(r[key]) for r in rows if r[key] != ""]
        return (np.median(v), np.mean(np.array(v) < 10) * 100) if v else (float("nan"), 0.0)
    print(f"\nn={N} 帧")
    print(f"{'配置':22s} {'有效':>5s} {'中位':>7s} {'S@10':>6s} {'接受':>5s} {'接受集精确率':>10s}")
    for name, key in [("基线（无 GSD）", "base"), ("+ GSD 对齐", "gap"), ("+ 密度自适应", "den")]:
        m, s = stat(key)
        print(f"{name:22s} {len([r for r in rows if r[key] != '']):5d} {m:7.1f} {s:5.1f}% {'-':>5s} {'-':>10s}")
    acc = [r for r in rows if r["accepted"] == 1]
    if acc:
        e = np.array([float(r["err_final"]) for r in acc])
        print(f"{'+ 验证门':22s} {len(acc):5d} {np.median(e):7.1f} {np.mean(e < 10) * 100:5.1f}% "
              f"{len(acc):5d} {np.mean(e < 20) * 100:9.1f}%  (拒答 {N - len(acc)} 帧)")


if __name__ == "__main__":
    main()
