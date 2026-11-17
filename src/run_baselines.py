# -*- coding: utf-8 -*-
"""统一基线评测：SIFT / DISK / LoFTR / DeDoDe / XoFTR / RoMa 同一口径。

口径：角点误差（对称转移，原生像素），含训练集标定；同时报内点残差与覆盖率。
用法：python src/run_baselines.py --per_frame 6 --methods sift,disk,loftr,xoftr,roma
"""
import os, sys, json, csv, argparse, time
import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_m3t_crossmodal as R

W = os.path.join(PROJ, "method_code", "weight")
DD = os.path.join(PROJ, "method_code", "DeDoDe-main")
B = os.path.join(PROJ, "data", "M3T", "test_dataset")
CALIB = "0.98787,0.174,9.26,12.59"
MIN_INLIERS = 15      # 判为“收敛”的最小内点数


def _tensor(g0, g1, max_side=1280, ch=1):
    t0, s0, t1, s1 = R.prep_pair(g0, g1, max_side)
    if ch == 3:
        t0 = t0.repeat(1, 3, 1, 1); t1 = t1.repeat(1, 3, 1, 1)
    return t0, s0, t1, s1


# ---------------- 各方法：返回原生坐标匹配点 ----------------
def m_sift(g0, g1, dev):
    s = cv2.SIFT_create(nfeatures=4000)
    k0, d0 = s.detectAndCompute(g0, None)
    k1, d1 = s.detectAndCompute(g1, None)
    if d0 is None or d1 is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    mm = cv2.BFMatcher().knnMatch(d0, d1, k=2)
    good = [a for a, b in mm if a.distance < 0.85 * b.distance]
    p0 = np.float32([k0[g.queryIdx].pt for g in good])
    p1 = np.float32([k1[g.trainIdx].pt for g in good])
    return p0, p1


def _mutual(p0, p1, q0, q1):
    if len(p0) == 0 or len(q0) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    pr = cv2.perspectiveTransform(p0.reshape(-1, 1, 2), np.eye(3)).reshape(-1, 2)
    return p0, p1


_CACHE = {}


def _get(name, dev):
    if name in _CACHE:
        return _CACHE[name]
    import kornia.feature as K
    if name == "disk":
        m = K.DISK.from_pretrained("depth").to(dev).eval()
    elif name == "loftr":
        m = K.LoFTR(pretrained="outdoor").to(dev).eval()
    elif name == "dedode":
        sys.path.insert(0, DD)
        from DeDoDe import dedode_detector_L, dedode_descriptor_B
        det = dedode_detector_L(device=dev, weights=torch.load(
            os.path.join(DD, "weights", "dedode_detector_L_v2.pth"), map_location="cpu"))
        des = dedode_descriptor_B(device=dev, weights=torch.load(
            os.path.join(DD, "weights", "dedode_descriptor_B.pth"), map_location="cpu"))
        m = (det, des)
    _CACHE[name] = m
    return m


def m_disk(g0, g1, dev, num=4096):
    m = _get("disk", dev)
    t0, s0, t1, s1 = _tensor(g0, g1, 1024, ch=3)
    with torch.inference_mode():
        f0 = m(t0.to(dev), n=num, pad_if_not_divisible=True)
        f1 = m(t1.to(dev), n=num, pad_if_not_divisible=True)
        if isinstance(f0, (list, tuple)):
            f0 = f0[0]
        if isinstance(f1, (list, tuple)):
            f1 = f1[0]
    if f0.keypoints.shape[0] == 0 or f1.keypoints.shape[0] == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    d0 = torch.nn.functional.normalize(f0.descriptors, dim=1)
    d1 = torch.nn.functional.normalize(f1.descriptors, dim=1)
    sim = d0 @ d1.T
    v, i = sim.max(1)
    j = sim.max(0).indices
    idx = torch.arange(len(i), device=dev)
    keep = (j[i] == idx) & (v > 0.5)
    p0 = (f0.keypoints[keep] / s0).cpu().numpy().astype(np.float32)
    p1 = (f1.keypoints[i[keep]] / s1).cpu().numpy().astype(np.float32)
    return p0, p1


def m_loftr(g0, g1, dev):
    m = _get("loftr", dev)
    t0, s0, t1, s1 = _tensor(g0, g1, 1024, ch=1)
    with torch.inference_mode():
        out = m({"image0": t0.to(dev), "image1": t1.to(dev)})
    p0 = out["keypoints0"].cpu().numpy() / s0
    p1 = out["keypoints1"].cpu().numpy() / s1
    return p0.astype(np.float32), p1.astype(np.float32)


def m_dedode(g0, g1, dev, num=3000):
    det, des = _get("dedode", dev)
    t0, s0, t1, s1 = _tensor(g0, g1, 1024, ch=3)
    b0 = {"image": t0.to(dev)}; b1 = {"image": t1.to(dev)}
    with torch.inference_mode():
        k0 = det.detect(b0, num_keypoints=num)["keypoints"]
        k1 = det.detect(b1, num_keypoints=num)["keypoints"]
        d0 = des.describe_keypoints(b0, k0)["descriptions"][0]
        d1 = des.describe_keypoints(b1, k1)["descriptions"][0]
    k0 = k0[0].cpu().numpy(); k1 = k1[0].cpu().numpy()          # 归一化坐标
    d0 = torch.nn.functional.normalize(d0, dim=-1)
    d1 = torch.nn.functional.normalize(d1, dim=-1)
    sim = d0 @ d1.T
    v, i = sim.max(1)
    j = sim.max(0).indices
    idx = torch.arange(len(i), device=dev)
    keep = (j[i] == idx) & (v > 0.3)
    def to_px(k, w, h, s):
        x = w * (k[:, 0] + 1) / 2; y = h * (k[:, 1] + 1) / 2
        return np.stack([x, y], 1) / s
    p0 = to_px(k0[keep.cpu().numpy()], t0.shape[-1], t0.shape[-2], s0).astype(np.float32)
    p1 = to_px(k1[i[keep].cpu().numpy()], t1.shape[-1], t1.shape[-2], s1).astype(np.float32)
    return p0, p1


def m_xoftr(g0, g1, dev):
    m = _get("xoftr", dev)
    p0, p1 = R.run_matcher(m, g0, g1, dev, 1280)
    return p0, p1


def m_roma(g0, g1, dev, num=2000):
    from PIL import Image
    mdl = _get("roma", dev)
    A = Image.fromarray(cv2.cvtColor(g0, cv2.COLOR_GRAY2RGB))
    Bb = Image.fromarray(cv2.cvtColor(g1, cv2.COLOR_GRAY2RGB))
    warp, cert = mdl.match(A, Bb, batched=True, device=dev)
    mm, _ = mdl.sample(warp, cert, num=num)
    mm = mm.cpu().numpy()
    wA, hA = g0.shape[1], g0.shape[0]
    wB, hB = g1.shape[1], g1.shape[0]
    p0 = np.stack([wA * (mm[:, 0] + 1) / 2, hA * (mm[:, 1] + 1) / 2], 1).astype(np.float32)
    p1 = np.stack([wB * (mm[:, 2] + 1) / 2, hB * (mm[:, 3] + 1) / 2], 1).astype(np.float32)
    return p0, p1


def build(name, dev):
    if name == "roma":
        from romatch import roma_outdoor
        w = torch.load(os.path.join(W, "roma_outdoor.pth"), map_location="cpu", weights_only=False)
        dw = torch.load(os.path.join(W, "dinov2_vitl14_pretrain.pth"), map_location="cpu", weights_only=False)
        _CACHE["roma"] = roma_outdoor(device=dev, weights=w, dinov2_weights=dw)
    elif name == "xoftr":
        _CACHE["xoftr"] = R.build_matcher("xoftr", dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="sift,disk,loftr,xoftr,roma")
    ap.add_argument("--per_frame", type=int, default=6)
    ap.add_argument("--out", default="results/raw/baselines.csv")
    a = ap.parse_args()
    dev = "cuda"
    methods = [x for x in a.methods.split(",") if x]
    for m in methods:
        if m in ("roma", "xoftr"):
            build(m, dev)

    s, th, tx, ty = [float(x) for x in CALIB.split(",")]
    th = np.radians(th)
    Sinv = np.linalg.inv(np.array([[s * np.cos(th), -s * np.sin(th), tx],
                                   [s * np.sin(th), s * np.cos(th), ty], [0, 0, 1.0]]))

    IR = json.load(open(os.path.join(B, "ir.json"), encoding="utf-8"))
    VS = json.load(open(os.path.join(B, "vis.json"), encoding="utf-8"))

    rows = []
    for sc in ["test01", "test02", "test03", "test04", "test05", "test06"]:
        n = len(IR[sc]["uav_img_path"])
        for fi in range(0, n, max(1, n // a.per_frame)):
            pv = os.path.join(B, sc, "uav_vis", VS[sc]["uav_img_path"][fi])
            if not R._valid(pv):
                continue
            g0 = R.load_gray(os.path.join(B, sc, "uav_ir", IR[sc]["uav_img_path"][fi]))
            g1 = R.load_gray(pv)
            k = g0.shape[1] / g1.shape[1]
            g1 = cv2.resize(g1, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            Hir = np.array(IR[sc]["H_matrix"][fi]).reshape(3, 3)
            Hvs = np.array(VS[sc]["H_matrix"][fi]).reshape(3, 3)
            Hg = np.diag([k, k, 1.0]) @ (np.linalg.inv(Hvs) @ Hir)
            Wn, Hn = g0.shape[1], g0.shape[0]

            for mth in methods:
                t = time.time()
                try:
                    fn = {"sift": m_sift, "disk": m_disk, "loftr": m_loftr,
                          "dedode": m_dedode, "xoftr": m_xoftr, "roma": m_roma}[mth]
                    p0, p1 = fn(g0, g1, dev)
                except Exception as e:
                    rows.append(dict(method=mth, scene=sc, frame=fi + 1, ok=0, note=str(e)[:40]))
                    continue
                H = None
                if len(p0) >= 8:
                    H, _ = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 3.0,
                                              confidence=0.9999, maxIters=20000)
                if H is None:
                    rows.append(dict(method=mth, scene=sc, frame=fi + 1, ok=0, note="no H"))
                    continue
                d = np.linalg.norm(cv2.perspectiveTransform(p0.reshape(-1, 1, 2), H).reshape(-1, 2) - p1, axis=1)
                inl = int((d < 3).sum())
                if inl < MIN_INLIERS:      # 内点不足 -> 视为未收敛，避免退化解污染统计
                    rows.append(dict(method=mth, scene=sc, frame=fi + 1, ok=0, note=f"inl={inl}"))
                    continue
                res = float(np.median(d[d < 3]))
                e_raw = R.sym_err_px(H, Hg, Wn, Hn)
                e_cal = R.sym_err_px(H @ Sinv, Hg, Wn, Hn)
                rows.append(dict(method=mth, scene=sc, frame=fi + 1, ok=1, n_inliers=inl,
                                 resid=round(res, 3), err_raw=round(e_raw, 2),
                                 err_calib=round(e_cal, 2), rt=round(time.time() - t, 2)))
        print(f"  {sc} 完成", flush=True)

    os.makedirs(os.path.dirname(os.path.join(PROJ, a.out)), exist_ok=True)
    keys = ["method", "scene", "frame", "ok", "n_inliers", "resid", "err_raw", "err_calib", "rt", "note"]
    with open(os.path.join(PROJ, a.out), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in keys})
    print("->", a.out)

    print(f"\n{'method':8s} {'有效':>5s} {'中位':>8s} {'S@10':>6s} {'S@20':>6s} {'残差':>6s}")
    for mth in methods:
        sub = [r for r in rows if r["method"] == mth and r.get("ok")]
        if not sub:
            print(f"{mth:8s} 全部失败"); continue
        e = np.array([float(r["err_calib"]) for r in sub])
        res = np.array([float(r["resid"]) for r in sub if r["resid"] not in ("", None)])
        print(f"{mth:8s} {len(sub):5d} {np.median(e):8.1f} {np.mean(e<10)*100:5.0f}% "
              f"{np.mean(e<20)*100:5.0f}% {np.median(res) if len(res) else float('nan'):6.2f}")


if __name__ == "__main__":
    main()
