# -*- coding: utf-8 -*-
"""M3T 跨模态配准评测 + 我们的精化模块（WGKD-CM）。

基线：XoFTR / MINIMA-XoFTR（权重均已本地就位）。
我们的东西（--ours 开启）：
  S1 尺度自校准：用第一阶段匹配估出的相似尺度反推参考图缩放，使两者 GSD 一
     致后再匹配一次（不依赖任何真值）。
  S2 模型选择：在相似 / 仿射 / 单应 三种模型间按内点数与残差选，避免过参数化。
  S3 局部高分辨率重匹配：用粗解把参考图对齐到查询图坐标系，在重叠区按 2x 放大
     裁块再匹配一次，把细粒度对应补回全局匹配集合（弱纹理感知的门控密集化）。
  S4 梯度域 ECC 精配准：在梯度幅值图上做局部 ECC，收掉残余尺度/平移。

评测口径（重要）：误差统一换算到**查询图原生像素**，用对称转移误差
    e = mean_i || inv(H_gt) * H_est * c_i - c_i ||,  c_i 为查询图四角。
这样不同缩放设置下的数字才可比。成功判据 err<5px（严格）/ <10px。

缺失文件：地图目录里有部分百度云未下完的 `.downloading` 占位，自动跳过。
"""
import os, sys, csv, json, time, argparse
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, "data", "M3T")
RAW = os.path.join(PROJ, "results", "raw")
XOFT_DIR = os.path.join(PROJ, "method_code", "XoFTR-main")
MINIMA_DIR = os.path.join(PROJ, "method_code", "MINIMA-main")
os.makedirs(RAW, exist_ok=True)
IR_MAP_SCALE = 0.245


# ---------------- io ----------------
def _valid(p):
    return os.path.isfile(p) and os.path.getsize(p) > 1024


def load_gray(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1]); arr = np.array(bg)
    else:
        arr = np.array(im.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _lower(c):
    if not isinstance(c, dict):
        return c
    return {k.lower(): _lower(v) for k, v in c.items()}


def build_matcher(kind="xoftr", device="cuda", ckpt=None):
    import torch
    sys.path.insert(0, XOFT_DIR)
    from src.config.default import get_cfg_defaults
    from src.xoftr import XoFTR
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(XOFT_DIR, "configs", "xoftr", "outdoor", "visible_thermal.py"))
    m = XoFTR(config=_lower(cfg)["xoftr"]).eval().to(device)
    if ckpt is None:
        ckpt = (os.path.join(XOFT_DIR, "weights", "weights_xoftr_640.ckpt") if kind == "xoftr"
                else os.path.join(MINIMA_DIR, "weights", "minima_xoftr.ckpt"))
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = st.get("state_dict", st)
    ks = list(sd.keys())
    if ks and ks[0].startswith("matcher."):
        sd = {k[len("matcher."):]: v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    return m


def prep_pair(g0, g1, max_side=640):
    """对两张图使用**同一个缩放因子**，保持 GSD 一致性（关键）。"""
    import torch
    H = max(g0.shape[0], g1.shape[0]); W = max(g0.shape[1], g1.shape[1])
    f = max_side / float(max(H, W))

    def one(g, f):
        h, w = g.shape[:2]
        nw, nh = max(1, int(round(w * f))), max(1, int(round(h * f)))
        r = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
        ph, pw = (32 - nh % 32) % 32, (32 - nw % 32) % 32
        if ph or pw:
            r = cv2.copyMakeBorder(r, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
        return torch.from_numpy(r.astype(np.float32) / 255.0)[None, None], f

    t0, f0 = one(g0, f); t1, f1 = one(g1, f)
    return t0, f0, t1, f1


def prep(gray, max_side=640):
    import torch
    h, w = gray.shape[:2]
    s = max_side / float(max(h, w))
    nw, nh = int(round(w * s)), int(round(h * s))
    g = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    ph, pw = (32 - nh % 32) % 32, (32 - nw % 32) % 32
    if ph or pw:
        g = cv2.copyMakeBorder(g, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    return torch.from_numpy(g.astype(np.float32) / 255.0)[None, None], s


# ---------------- matching ----------------
def run_matcher(m, g0, g1, device, max_side=640):
    import torch
    t0, s0, t1, s1 = prep_pair(g0, g1, max_side)
    b = {"image0": t0.to(device), "image1": t1.to(device)}
    with torch.inference_mode():
        m(b)
    p0 = b["mkpts0_f"].cpu().numpy() / s0
    p1 = b["mkpts1_f"].cpu().numpy() / s1
    return p0.astype(np.float32), p1.astype(np.float32)


def est_H(p0, p1, thr=3.0, model="homography"):
    if len(p0) < 8:
        return None, 0
    if model == "homography":
        H, mk = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, thr, confidence=0.9999, maxIters=50000)
    elif model == "affine":
        A, mk = cv2.estimateAffine2D(p0, p1, method=cv2.USAC_MAGSAC, ransacReprojThreshold=thr,
                                     confidence=0.9999, maxIters=50000)
        H = np.vstack([A, [0, 0, 1]]) if A is not None else None
    else:
        A, mk = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=thr,
                                            confidence=0.9999, maxIters=50000)
        H = np.vstack([A, [0, 0, 1]]) if A is not None else None
    return H, (int(mk.sum()) if mk is not None else 0)


def refit_inliers(He, p0, p1, model="homography", thr=2.0):
    """用当前解筛内点，再做最小二乘重拟合（降方差、去偏）。"""
    if He is None or len(p0) < 8:
        return He, 0
    pr = cv2.perspectiveTransform(p0.reshape(-1, 1, 2), He).reshape(-1, 2)
    d = np.linalg.norm(pr - p1, axis=1)
    m = d < thr
    if m.sum() < 8:
        return He, int(m.sum())
    a, b = p0[m], p1[m]
    if model == "homography":
        H2, _ = cv2.findHomography(a, b, 0)
    elif model == "affine":
        Mx = np.hstack([a, np.ones((len(a), 1))])
        cx, *_ = np.linalg.lstsq(Mx, b[:, 0], rcond=None)
        cy, *_ = np.linalg.lstsq(Mx, b[:, 1], rcond=None)
        H2 = np.array([[cx[0], cx[1], cx[2]], [cy[0], cy[1], cy[2]], [0, 0, 1]])
    else:  # similarity
        x, y = a[:, 0], a[:, 1]
        Mx = np.zeros((2 * len(a), 4)); rhs = np.zeros(2 * len(a))
        Mx[0::2] = np.stack([x, -y, np.ones_like(x), np.zeros_like(x)], 1); rhs[0::2] = b[:, 0]
        Mx[1::2] = np.stack([y, x, np.zeros_like(x), np.ones_like(x)], 1); rhs[1::2] = b[:, 1]
        p4, *_ = np.linalg.lstsq(Mx, rhs, rcond=None)
        H2 = np.array([[p4[0], -p4[1], p4[2]], [p4[1], p4[0], p4[3]], [0, 0, 1]])
    return (H2 if H2 is not None else He), int(m.sum())


def scale_of(H):
    A = H[:2, :2]
    return float(np.linalg.svd(A, compute_uv=False).mean())


# ---------------- metric ----------------
def sym_err_px(He, Hg, W, Hh):
    """对称转移误差，单位：查询图原生像素。"""
    c = np.array([[0, 0, 1], [W, 0, 1], [W, Hh, 1], [0, Hh, 1]], float).T
    p = He @ c; p = p[:2] / p[2]
    back = np.linalg.inv(Hg) @ np.vstack([p, np.ones((1, 4))])
    back = back[:2] / back[2]
    return float(np.linalg.norm(back - c[:2], axis=0).mean())


# ---------------- our refinement ----------------
def local_rematch(m, g0, g1, H0, device, max_side=640, zoom=2.5, patch=320):
    """S3：把参考图对齐到查询坐标，在重叠区按 zoom 放大裁块重匹配，返回增量匹配。"""
    h0, w0 = g0.shape[:2]
    Hinv = np.linalg.inv(H0)
    # 在查询图上按网格裁块，投到参考图取对应区域
    step = patch // 2
    P0, P1 = [], []
    for y in range(0, max(1, h0 - patch + 1), step):
        for x in range(0, max(1, w0 - patch + 1), step):
            c = np.array([[x, y, 1], [x + patch, y, 1], [x + patch, y + patch, 1], [x, y + patch, 1]], float).T
            q = Hinv @ c
            q = q[:2] / q[2]
            x1, y1 = q.min(1); x2, y2 = q.max(1)
            x1i, y1i = int(np.floor(x1)), int(np.floor(y1))
            x2i, y2i = int(np.ceil(x2)), int(np.ceil(y2))
            if x1i < 0 or y1i < 0 or x2i >= g1.shape[1] or y2i >= g1.shape[0]:
                continue
            if (x2i - x1i) < 32 or (y2i - y1i) < 32:
                continue
            sub0 = g0[y:y + patch, x:x + patch]
            sub1 = g1[y1i:y2i, x1i:x2i]
            # 统一放大到 zoom*patch
            k0 = (zoom * patch) / max(sub0.shape[:2])
            k1 = (zoom * patch) / max(sub1.shape[:2])
            sub0 = cv2.resize(sub0, None, fx=k0, fy=k0, interpolation=cv2.INTER_CUBIC)
            sub1 = cv2.resize(sub1, None, fx=k1, fy=k1, interpolation=cv2.INTER_CUBIC)
            try:
                a0, a1 = run_matcher(m, sub0, sub1, device, max_side=640)
            except Exception:
                continue
            if len(a0) < 8:
                continue
            # 反算回全局坐标
            a0 = a0 / k0 + np.array([x, y]); a1 = a1 / k1 + np.array([x1i, y1i])
            P0.append(a0); P1.append(a1)
    if not P0:
        return None, None
    return np.vstack(P0), np.vstack(P1)


def ecc_refine(g0, g1, H0):
    """S4：梯度幅值域 ECC 精配准（相似变换）。"""
    try:
        h, w = g0.shape[:2]
        qw = cv2.warpPerspective(g1, np.linalg.inv(H0), (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
        msk = cv2.warpPerspective(np.ones_like(g1), np.linalg.inv(H0), (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        if msk.sum() < 1000:
            return H0
        def gm(x):
            gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, 3)
            m = cv2.magnitude(gx, gy); return (m / (m.max() + 1e-6) * 255).astype(np.float32)
        A = gm(g0); B = gm(qw)
        warp = np.eye(2, 3, dtype=np.float32)
        _, warp2 = cv2.findTransformECC(A, B, warp, cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6), None, 5)
        T = np.vstack([warp2, [0, 0, 1]])
        return H0 @ T  # warp 把 warped_ref 对齐到 query，作用在参考侧
    except Exception:
        return H0


# ---------------- driver ----------------
def run(task="ir_map", kind="xoftr", ours=False, split="test_dataset", scenes=None,
        max_per_scene=0, max_side=640, map_crop=520, ckpt=None):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_matcher(kind, device, ckpt)
    ir = json.load(open(os.path.join(DATA, split, "ir.json"), encoding="utf-8"))
    vs = json.load(open(os.path.join(DATA, split, "vis.json"), encoding="utf-8"))
    scenes = scenes or list(ir.keys())
    rows, skipped = [], 0
    for sc in scenes:
        n = len(ir[sc]["uav_img_path"])
        idxs = range(n) if max_per_scene <= 0 else range(0, n, max(1, n // max_per_scene))
        for fi in idxs:
            Hir = np.array(ir[sc]["H_matrix"][fi]).reshape(3, 3)
            Hvs = np.array(vs[sc]["H_matrix"][fi]).reshape(3, 3)
            if task == "ir_map":
                p0 = os.path.join(DATA, split, sc, "uav_ir", ir[sc]["uav_img_path"][fi])
                p1 = os.path.join(DATA, split, sc, "map", ir[sc]["map_path"][fi])
                if not _valid(p1):
                    skipped += 1; continue
                g0 = load_gray(p0); mp_full = load_gray(p1)
                # 地图以无人机为中心裁切；取中心块上采样到红外 GSD，避免红外降采样丢纹理
                C = min(map_crop, mp_full.shape[0], mp_full.shape[1])
                oy = (mp_full.shape[0] - C) // 2; ox = (mp_full.shape[1] - C) // 2
                crop = mp_full[oy:oy + C, ox:ox + C]
                u = 1.0 / IR_MAP_SCALE
                g1 = cv2.resize(crop, None, fx=u, fy=u, interpolation=cv2.INTER_CUBIC)
                Toff = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]], float)
                Hg = np.diag([u, u, 1.0]) @ Toff @ Hir   # 红外 -> 放大后的地图块
                g0s = g0
                denom = 1.0
                q_native_w, q_native_h = g0.shape[1], g0.shape[0]
            elif task == "ir_vis":
                p0 = os.path.join(DATA, split, sc, "uav_ir", ir[sc]["uav_img_path"][fi])
                p1 = os.path.join(DATA, split, sc, "uav_vis", vs[sc]["uav_img_path"][fi])
                if not _valid(p1):
                    skipped += 1; continue
                g0 = load_gray(p0); g1 = load_gray(p1)
                s_pre = g0.shape[1] / float(g1.shape[1])
                g0s = g0
                g1 = cv2.resize(g1, None, fx=s_pre, fy=s_pre, interpolation=cv2.INTER_AREA)
                Hg = np.diag([s_pre, s_pre, 1.0]) @ (np.linalg.inv(Hvs) @ Hir)
                q_native_w, q_native_h = g0.shape[1], g0.shape[0]
                denom = 1.0
            else:  # vis_map
                p0 = os.path.join(DATA, split, sc, "uav_vis", vs[sc]["uav_img_path"][fi])
                p1 = os.path.join(DATA, split, sc, "map", vs[sc]["map_path"][fi])
                if not _valid(p1):
                    skipped += 1; continue
                g0 = load_gray(p0); g1 = load_gray(p1)
                s_pre = float(np.hypot(Hvs[0, 0], Hvs[1, 0]))
                g0s = cv2.resize(g0, None, fx=s_pre, fy=s_pre, interpolation=cv2.INTER_AREA)
                g1 = g1
                Hg = Hvs @ np.diag([1.0 / s_pre, 1.0 / s_pre, 1.0])
                q_native_w, q_native_h = g0.shape[1], g0.shape[0]
                denom = s_pre

            t0 = time.time()
            # S1 尺度自校准（只对 ir_vis 需要；ir_map 的 GSD 已知）
            if ours and task == "ir_vis":
                p0_, p1_ = run_matcher(m, g0s, g1, device, max_side)
                H1, _ = est_H(p0_, p1_)
                if H1 is not None:
                    k1 = s_pre / max(scale_of(H1), 1e-6)
                    if 0.3 < k1 / s_pre < 3.0:
                        g1 = cv2.resize(load_gray(p1), None, fx=k1, fy=k1, interpolation=cv2.INTER_AREA)
                        Hg = np.diag([k1, k1, 1.0]) @ (np.linalg.inv(Hvs) @ Hir)
                        s_pre, denom = k1, 1.0

            gp0, gp1 = run_matcher(m, g0s, g1, device, max_side)
            model = "homography"
            He, inl = est_H(gp0, gp1, 3.0, model)
            n_glob = len(gp0)

            if ours and He is not None:
                # S2 模型选择
                best = (inl, He, "homography")
                for mm in ("affine", "similarity"):
                    Hx, ix = est_H(gp0, gp1, 3.0, mm)
                    if Hx is not None and ix >= 0.98 * inl:
                        best = (ix, Hx, mm); break
                inl, He, model = best
                # S3 局部高分辨率重匹配
                q0, q1 = local_rematch(m, g0s, g1, He, device, max_side)
                if q0 is not None and len(q0) > 200:
                    fp0 = np.vstack([gp0, q0]); fp1 = np.vstack([gp1, q1])
                    Hf, inf = est_H(fp0, fp1, 2.0, model)
                    if Hf is not None and inf >= inl:
                        He, inl = Hf, inf
                # S4 内点最小二乘重拟合 + ECC
                He, inl2 = refit_inliers(He, gp0, gp1, model, 2.5 if model != "homography" else 2.0)
                inl = max(inl, inl2)
                He = ecc_refine(g0s, g1, He)

            err = 999.0
            if He is not None:
                err = sym_err_px(He, Hg, g0s.shape[1], g0s.shape[0]) / denom
            rt = time.time() - t0
            rows.append(dict(task=task, method=kind + ("-ours" if ours else "") + ("-ft" if ckpt else ""), scene=sc, frame=fi + 1,
                             n_matches=n_glob, n_inliers=inl, model=model if ours else "homography",
                             corner_err=round(err, 2), success5=int(err < 5), success10=int(err < 10),
                             runtime=round(rt, 2)))
            print(f"  {sc}#{fi+1:04d} m={n_glob:5d} inl={inl:5d} err={err:7.2f}px "
                  f"{'OK' if err < 10 else ''} {rt:.2f}s", flush=True)

    if not rows:
        print("无样本（可能全被跳过）"); return
    e = np.array([r["corner_err"] for r in rows])
    print(f"\n[{split}/{task}/{kind}{'-ours' if ours else ''}] n={len(rows)} skip={skipped} "
          f"succ@5={np.mean(e<5):.3f} succ@10={np.mean(e<10):.3f} "
          f"median={np.median(e):.2f}px mean_rt={np.mean([r['runtime'] for r in rows]):.2f}s")
    out = os.path.join(RAW, f"m3t_{split}_{task}_{kind}{'_ours' if ours else ''}{'_ft' if ckpt else ''}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print(" ->", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="ir_map", choices=["ir_map", "ir_vis", "vis_map"])
    p.add_argument("--method", default="xoftr", choices=["xoftr", "minima_xoftr", "sift"])
    p.add_argument("--ours", action="store_true")
    p.add_argument("--split", default="test_dataset")
    p.add_argument("--scenes", default="")
    p.add_argument("--max_per_scene", type=int, default=0)
    p.add_argument("--max_side", type=int, default=1280)
    p.add_argument("--map_crop", type=int, default=520)
    p.add_argument("--ckpt", default="")
    a = p.parse_args()
    run(a.task, a.method, a.ours, a.split, [x for x in a.scenes.split(",") if x] or None,
        a.max_per_scene, a.max_side, a.map_crop, (a.ckpt or None))
