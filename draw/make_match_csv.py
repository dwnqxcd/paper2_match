# -*- coding: utf-8 -*-
"""为图 3（TIR-VIS）与图 5（VIS-sat）生成匹配示意所需的对应点 CSV。

输出格式与现有 draw/match_data/*.csv 完全一致（首行起每行列：xa,ya,xb,yb,wa,ha,wb,hb），
直接可被 draw/render_fig3_fig5.py 与旧的 .m 脚本读取。

约定（与现有 CSV 对齐）：
  TIR-VIS：A = 红外（原生），B = 可见光（原生，不做尺度缩放）。
  VIS-sat：A = 可见光（gsd1 时按 GSD 比例缩放到地图尺度后匹配，再反算回原生坐标），
           B = 地图块（原生 1000x1000）。

用法：
  python draw/make_match_csv.py --task tir_vis --scene test02 --frame 33 --method sift --out draw/match_data/tir_vis_sift.csv
  python draw/make_match_csv.py --task vis_map --scene test02 --frame 76 --method sift --gsd 0 --out draw/match_data/vis_map_sift_gsd0.csv
  python draw/make_match_csv.py --task vis_map --scene test02 --frame 76 --method roma --gsd 1 --out draw/match_data/vis_map_roma_gsd1.csv
"""
import os, sys, json, csv, argparse, time
import numpy as np
import cv2
cv2.setNumThreads(0)
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
import run_baselines as BL
import run_m3t_crossmodal as R

DATA = os.path.join(PROJ, "data", "M3T", "test_dataset")
GSD = {"ir": 0.245, "vis": 0.174}
DEV = "cuda"


def _cap(img, ms):
    """把单张图的较长边限制到 ms（不放大）；用于示意，坐标随返回图一致。"""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= ms:
        return img
    k = float(ms) / m
    return cv2.resize(img, (max(1, int(round(w * k))), max(1, int(round(h * k)))),
                      interpolation=cv2.INTER_AREA)


def grid_coverage(pts, w, h, g=8):
    if len(pts) < 1:
        return 0.0
    gx = np.clip((pts[:, 0] / w * g).astype(int), 0, g - 1)
    gy = np.clip((pts[:, 1] / h * g).astype(int), 0, g - 1)
    return len(set(zip(gx.tolist(), gy.tolist()))) / float(g * g)


def extract(task, scene, frame, method, gsd, out, dev="cuda", max_side=0):
    ir = json.load(open(os.path.join(DATA, "ir.json"), encoding="utf-8"))
    vs = json.load(open(os.path.join(DATA, "vis.json"), encoding="utf-8"))
    fi = frame - 1
    Hg = None
    if task == "tir_vis":
        irp = os.path.join(DATA, scene, "uav_ir", ir[scene]["uav_img_path"][fi])
        vip = os.path.join(DATA, scene, "uav_vis", vs[scene]["uav_img_path"][fi])
        A = R.load_gray(irp); B = R.load_gray(vip)
        wa, ha = A.shape[1], A.shape[0]
        wb, hb = B.shape[1], B.shape[0]
        k_back = 1.0
        Hg = np.array(ir[scene]["H_matrix"][fi]).reshape(3, 3)   # 红外 -> 可见光
        if max_side:                       # 仅用于示意：各自按长边缩到 <= max_side（红外保持原生）
            A2 = _cap(A, max_side); B2 = _cap(B, max_side)
            if A2 is not A or B2 is not B:
                A, B = A2, B2
                wa, ha = A.shape[1], A.shape[0]
                wb, hb = B.shape[1], B.shape[0]
                Hg = None                  # 缩放后原生 H 不再适用，corner 置空
        g0, g1 = A, B
    else:  # vis_map
        vip = os.path.join(DATA, scene, "uav_vis", vs[scene]["uav_img_path"][fi])
        mp = os.path.join(DATA, scene, "map", vs[scene]["map_path"][fi])
        A = R.load_gray(vip); B = R.load_gray(mp)
        wa, ha = A.shape[1], A.shape[0]
        wb, hb = B.shape[1], B.shape[0]
        k_back = 1.0
        if gsd == 1:
            k = GSD["vis"]                       # 可见光缩放到地图尺度
            A2 = cv2.resize(A, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            g0 = A2
            k_back = 1.0 / k                      # 反算回原生可见光坐标
        else:
            g0 = A
        g1 = B
        Hv = np.array(vs[scene]["H_matrix"][fi]).reshape(3, 3)   # 可见光 -> 地图
        Hg = Hv

    if method in ("roma", "xoftr", "loftr", "disk", "dedode"):
        BL.build(method, dev)
    t0 = time.time()
    p0, p1 = BL.__dict__["m_" + method](g0, g1, dev)
    rt = time.time() - t0
    p0 = np.asarray(p0, np.float32); p1 = np.asarray(p1, np.float32)
    p0_nat = p0 * k_back                       # 回到原生坐标，仅用于绘制/覆盖度

    candidates = len(p0)
    inliers = 0; resid = float("nan"); omega = 0.0; corner = float("nan")
    if candidates >= 8:
        # 在匹配器实际工作的尺度上拟合（4px 阈值才与评测一致）
        H, mask = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 4.0, confidence=0.9999, maxIters=20000)
        if H is not None:
            d = np.linalg.norm(cv2.perspectiveTransform(p0.reshape(-1, 1, 2), H).reshape(-1, 2) - p1, axis=1)
            m = d < 4
            inliers = int(m.sum()); omega = grid_coverage(p0_nat[m], wa, ha)
            if m.sum() > 0:
                resid = float(np.median(d[m]))
            if Hg is not None:
                # 与 run_map_baselines 完全一致：Hg 换算到匹配尺度，角点误差在该尺度下计量
                k_used = 1.0 / k_back if k_back else 1.0
                S = np.diag([k_used, k_used, 1.0])
                Hg_used = Hg @ np.linalg.inv(S)
                corner = float(R.sym_err_px(H, Hg_used, g0.shape[1], g0.shape[0]))

    rows = []
    for i in range(candidates):
        rows.append(dict(xa=round(float(p0_nat[i, 0]), 2), ya=round(float(p0_nat[i, 1]), 2),
                         xb=round(float(p1[i, 0]), 2), yb=round(float(p1[i, 1]), 2),
                         wa=wa, ha=ha, wb=wb, hb=hb))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["xa", "ya", "xb", "yb", "wa", "ha", "wb", "hb"])
        wr.writeheader(); wr.writerows(rows)
    print(f"[{task}/{scene}/{frame}/{method}/gsd{gsd}] cand={candidates} inl={inliers} "
          f"omega={omega:.3f} resid={resid:.2f} corner={corner:.2f} rt={rt:.2f}s -> {out}")
    return dict(candidates=candidates, inliers=inliers, omega=round(omega, 3),
                resid=round(resid, 2), corner=round(corner, 2) if corner==corner else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="tir_vis", choices=["tir_vis", "vis_map"])
    ap.add_argument("--scene", default="test02")
    ap.add_argument("--frame", type=int, default=33)
    ap.add_argument("--method", default="sift", choices=["sift", "disk", "loftr", "dedode", "roma", "xoftr"])
    ap.add_argument("--gsd", type=int, default=0)
    ap.add_argument("--dev", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_side", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.dev == "cpu":
        torch.set_num_threads(2)
    out = a.out or os.path.join(HERE, "match_data", f"{a.task}_{a.method}_g{a.gsd}.csv")
    extract(a.task, a.scene, a.frame, a.method, a.gsd, out,
            dev=a.dev, max_side=a.max_side)
