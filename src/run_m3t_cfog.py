# -*- coding: utf-8 -*-
"""M3T 跨模态配准：CFOG 稠密梯度方向描述子 + 中心约束的粗到精搜索。

任务：查询图（红外 uav_ir / 可见光 uav_vis，原生分辨率） -> 卫星地图 map（1000x1000）。
真值：data/M3T/{split}/ir.json|vis.json 的 H_matrix（3x3，查询图像素 -> 地图像素）。

关键先验（已核实的**数据集设计**，非逐帧真值）：每帧地图切片以无人机为中心，
即查询图足迹质心 ≈ 地图中心 (500,500)，IR 场景内 std≈5px。因此把平移约束在中心附近、
主要搜索旋转与尺度，可消除远距离重复纹理造成的误匹配。

流程：
  1) 地图降采样到 D x D；CFOG 通道；
  2) 粗搜：旋转(deg step) × 尺度(±) × 平移(中心±shift)；
  3) 精搜：旋转 1°、尺度 1%、平移 1px；
  4) 输出 H（原生查询坐标 -> 原生地图坐标）。

评测：四角投影平均误差，<10px 记为成功。
"""
import os, csv, json, time, argparse
import multiprocessing as mp
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, "data", "M3T")
RAW = os.path.join(PROJ, "results", "raw")
os.makedirs(RAW, exist_ok=True)


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


def load_meta(split, modality):
    fn = "ir.json" if modality == "ir" else "vis.json"
    return json.load(open(os.path.join(DATA, split, fn), encoding="utf-8"))


def cfog(gray, nbins=8, sigma=3.0):
    gf = gray.astype(np.float32)
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy); ang = np.arctan2(gy, gx)
    chans = []
    for k in range(nbins):
        c = k * 2 * np.pi / nbins
        d = np.angle(np.exp(1j * (ang - c)))
        w = np.maximum(0.0, np.cos(d)) ** 2
        chans.append(cv2.GaussianBlur(mag * w, (0, 0), sigma))
    C = np.stack(chans, -1).astype(np.float32)
    C /= (C.sum(2, keepdims=True) + 1e-6)
    return C


def build_template(query, s_ds, deg):
    """返回 (A, 模板灰度, 有效掩码, bx, by, cw, ch)。A 为 s*R(deg)（ds 尺度）。"""
    A = s_ds * np.array([[np.cos(np.radians(deg)), -np.sin(np.radians(deg))],
                         [np.sin(np.radians(deg)),  np.cos(np.radians(deg))]], np.float64)
    h, w = query.shape[:2]
    cs = A @ np.array([[0, 0], [w, 0], [w, h], [0, h]], float).T
    bx, by = np.floor(cs[0].min()), np.floor(cs[1].min())
    cw, ch = int(np.ceil(cs[0].max() - bx)) + 1, int(np.ceil(cs[1].max() - by)) + 1
    M = np.array([[A[0, 0], A[0, 1], -bx], [A[1, 0], A[1, 1], -by]], np.float64)
    Wt = cv2.warpAffine(query, M, (cw, ch), flags=cv2.INTER_LINEAR, borderValue=255)
    mask = cv2.warpAffine(np.ones((h, w), np.uint8), M, (cw, ch), flags=cv2.INTER_NEAREST, borderValue=0)
    return A, Wt, mask, bx, by, cw, ch


def _corr(cA, cB, valid):
    sc = 0.0
    for k in range(cA.shape[2]):
        a = cA[:, :, k][valid]; b = cB[:, :, k][valid]
        a = a - a.mean(); b = b - b.mean()
        sc += float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return sc / cA.shape[2]


def estimate_cfog(query, map_gray, base_scale, D=700, coarse_rot=10,
                  coarse_scales=(0.88, 1.0, 1.12), coarse_shifts=(-9, 0, 9),
                  nbins=8, sigma=4.0, fine_rot_r=7.0, fine_rot_step=1.0,
                  fine_scale_r=0.03, fine_scale_step=0.01, fine_shift=8,
                  refine=True):
    mh, mw = map_gray.shape[:2]
    mf = D / float(max(mh, mw))
    map_ds = cv2.resize(map_gray, (int(round(mw * mf)), int(round(mh * mf))), interpolation=cv2.INTER_AREA)
    sg = max(1.2, sigma * mf)
    cf_map = cfog(map_ds, nbins=nbins, sigma=sg)
    h, w = query.shape[:2]
    cen = np.array([cf_map.shape[1] / 2.0, cf_map.shape[0] / 2.0])

    def prep(deg, s):
        """构建一次模板与 CFOG，返回 (cf_t, er, A, bx, by, cw, ch, t)。"""
        A, Wt, mask, bx, by, cw, ch = build_template(query, s * mf, deg)
        cf_t = cfog(Wt, nbins=nbins, sigma=sg)
        er = cv2.erode(mask, np.ones((7, 7), np.uint8)) > 0
        t = cen - A @ np.array([w / 2.0, h / 2.0])
        return cf_t, er, A, bx, by, cw, ch, t

    def score_fixed(cf_t, er, bx, by, cw, ch, t, dx, dy):
        x0 = int(round(bx + t[0] + dx)); y0 = int(round(by + t[1] + dy))
        if er.sum() < 200 or x0 < 0 or y0 < 0 or x0 + cw > cf_map.shape[1] or y0 + ch > cf_map.shape[0]:
            return -9.0, None
        return _corr(cf_t, cf_map[y0:y0 + ch, x0:x0 + cw], er), (x0, y0, bx, by)

    # 粗搜：每个 (deg, s) 只建一次模板/CFOG，再遍历平移
    best = (-9.0, 0.0, base_scale, 0, 0)
    for deg in range(0, 360, coarse_rot):
        for sm in coarse_scales:
            s = base_scale * sm
            cf_t, er, A, bx, by, cw, ch, t = prep(deg, s)
            for dx in coarse_shifts:
                for dy in coarse_shifts:
                    v, _ = score_fixed(cf_t, er, bx, by, cw, ch, t, dx, dy)
                    if v > best[0]:
                        best = (v, float(deg), s, dx, dy)
    v, deg, s, dx, dy = best

    if refine:
        # 旋转 + 尺度细化：以粗解 (deg, s) 为基准，不原地漂移
        deg0, s0 = deg, s
        bF = (v, deg0, s0)
        for dd in np.arange(-fine_rot_r, fine_rot_r + 1e-6, fine_rot_step):
            for sm in np.arange(1 - fine_scale_r, 1 + fine_scale_r + 1e-6, fine_scale_step):
                cf_t, er, A, bx, by, cw, ch, t = prep(deg0 + dd, s0 * sm)
                vv, _ = score_fixed(cf_t, er, bx, by, cw, ch, t, dx, dy)
                if vv > bF[0]:
                    bF = (vv, deg0 + dd, s0 * sm)
        v, deg, s = bF
        # 平移细化（模板固定）
        cf_t, er, A, bx, by, cw, ch, t = prep(deg, s)
        for ddx in range(-fine_shift, fine_shift + 1):
            for ddy in range(-fine_shift, fine_shift + 1):
                vv, _ = score_fixed(cf_t, er, bx, by, cw, ch, t, dx + ddx, dy + ddy)
                if vv > v:
                    v, dx, dy = vv, dx + ddx, dy + ddy

    cf_t, er, A, bx, by, cw, ch, t = prep(deg, s)
    v, info = score_fixed(cf_t, er, bx, by, cw, ch, t, dx, dy)
    if info is None:
        return None, {"score": v}
    x0, y0, bx, by = info
    A = s * np.array([[np.cos(np.radians(deg)), -np.sin(np.radians(deg))],
                      [np.sin(np.radians(deg)),  np.cos(np.radians(deg))]])
    t = np.array([x0 - bx, y0 - by], np.float64) / mf
    H = np.array([[A[0, 0], A[0, 1], t[0]],
                  [A[1, 0], A[1, 1], t[1]], [0, 0, 1]], np.float64)
    return H, {"score": round(float(v), 4), "rot": round(deg, 1), "scale": round(s, 4)}


def estimate_sift(query, map_gray, base_scale, ratio=0.85, nfeatures=20000):
    qr = cv2.resize(query, None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    k1, d1 = sift.detectAndCompute(qr, None); k2, d2 = sift.detectAndCompute(map_gray, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None, {"n_matches": 0, "n_inliers": 0}
    m = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [x for x, y in m if x.distance < ratio * y.distance]
    if len(good) < 8:
        return None, {"n_matches": len(good), "n_inliers": 0}
    p1 = np.float32([k1[x.queryIdx].pt for x in good]); p2 = np.float32([k2[x.trainIdx].pt for x in good])
    He, mask = cv2.findHomography(p1, p2, cv2.USAC_MAGSAC, 4.0, confidence=0.9999, maxIters=20000)
    if He is None:
        return None, {"n_matches": len(good), "n_inliers": 0}
    return He @ np.diag([base_scale, base_scale, 1.0]), {"n_matches": len(good), "n_inliers": int(mask.sum())}


def corner_err(H_est, H_gt, W, Hh):
    c = np.array([[0, 0, 1], [W, 0, 1], [W, Hh, 1], [0, Hh, 1]], float).T

    def pr(H):
        q = H @ c; return (q[:2] / q[2]).T

    return float(np.linalg.norm(pr(H_est) - pr(H_gt), axis=1).mean())


# 全局：供 estimate_one 在子进程（spawn）重建路径，避免 pickle 大对象
_META = None
_SPLIT = "test_dataset"
_MODALITY = "ir"
_METHOD = "cfog"
_BASE = 0.245
_D = 700
_CROT = 15
_NBINS = 8
_SIGMA = 4.0
_REFINE = True
_DATA = None


def _task_list(meta, modality, split, scenes, max_per_scene, DATA):
    tasks = []
    for sc in scenes:
        s = meta[sc]; n = len(s["uav_img_path"])
        idxs = range(n) if max_per_scene <= 0 else range(0, n, max(1, n // max_per_scene))
        for fi in idxs:
            mp_path = os.path.join(DATA, split, sc, "map", s["map_path"][fi])
            if not (os.path.isfile(mp_path) and os.path.getsize(mp_path) > 100000):
                continue                      # 与地图基线一致：跳过缺失/未下完的地图块
            tasks.append((sc, fi))
    return tasks


def estimate_one(task):
    sc, fi = task
    sub = "uav_ir" if _MODALITY == "ir" else "uav_vis"
    s = _META[sc]
    mp_path = os.path.join(_DATA, _SPLIT, sc, "map", s["map_path"][fi])
    q = load_gray(os.path.join(_DATA, _SPLIT, sc, sub, s["uav_img_path"][fi]))
    mp = load_gray(mp_path)
    Hg = np.array(s["H_matrix"][fi]).reshape(3, 3)
    t0 = time.time()
    if _METHOD == "cfog":
        H_est, extra = estimate_cfog(q, mp, _BASE, D=_D, coarse_rot=_CROT,
                                     nbins=_NBINS, sigma=_SIGMA, refine=_REFINE)
    else:
        H_est, extra = estimate_sift(q, mp, _BASE)
    rt = time.time() - t0
    err = 999.0 if H_est is None else corner_err(H_est, Hg, q.shape[1], q.shape[0])
    return dict(split=_SPLIT, modality=_MODALITY, scene=sc, frame=fi + 1, method=_METHOD,
                rot=extra.get("rot", ""), scale=extra.get("scale", ""),
                score=extra.get("score", ""), n_matches=extra.get("n_matches", ""),
                n_inliers=extra.get("n_inliers", ""), corner_err=round(err, 2),
                success=int(err < 10), runtime=round(rt, 3))


def _init_worker(meta, split, modality, method, base, D, crot, nbins, sigma, refine, data):
    global _META, _SPLIT, _MODALITY, _METHOD, _BASE, _D, _CROT, _NBINS, _SIGMA, _REFINE, _DATA
    _META = meta; _SPLIT = split; _MODALITY = modality; _METHOD = method
    _BASE = base; _D = D; _CROT = crot; _NBINS = nbins; _SIGMA = sigma
    _REFINE = refine; _DATA = data
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


def run(split="test_dataset", modality="ir", method="cfog", scenes=None, max_per_scene=0,
        D=700, coarse_rot=15, nbins=8, sigma=4.0, refine=True, tag="", workers=1):
    global _META, _SPLIT, _MODALITY, _METHOD, _BASE, _D, _CROT, _NBINS, _SIGMA, _REFINE, _DATA
    _META = load_meta(split, modality)
    _SPLIT = split; _MODALITY = modality; _METHOD = method
    _BASE = 0.245 if modality == "ir" else 0.174
    _D = D; _CROT = coarse_rot; _NBINS = nbins; _SIGMA = sigma; _REFINE = refine
    _DATA = DATA
    scenes = scenes or list(_META.keys())
    out = os.path.join(RAW, f"m3t_{split}_{modality}_{method}{tag}.csv")
    fieldnames = ["split", "modality", "scene", "frame", "method", "rot", "scale",
                  "score", "n_matches", "n_inliers", "corner_err", "success", "runtime"]

    # 断点续跑：已完成 (scene,frame) 跳过
    done = set()
    write_header = True
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        with open(out, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["scene"], int(r["frame"])))
        write_header = False

    tasks = [t for t in _task_list(_META, modality, split, scenes, max_per_scene, DATA) if t not in done]
    if not tasks:
        print(f"[{split}/{modality}/{method}] 已全部完成（{len(done)} 帧），跳过", flush=True)
        return

    fout = open(out, "a", newline="", encoding="utf-8")
    wr = csv.DictWriter(fout, fieldnames=fieldnames)
    if write_header:
        wr.writeheader(); fout.flush()

    t0 = time.time()
    if workers > 1 and len(tasks) > 1:
        with mp.Pool(workers, initializer=_init_worker,
                     initargs=(_META, _SPLIT, _MODALITY, _METHOD, _BASE, _D,
                               _CROT, _NBINS, _SIGMA, _REFINE, _DATA)) as pool:
            for i, row in enumerate(pool.imap_unordered(estimate_one, tasks), 1):
                wr.writerow(row); fout.flush()
                if i % 50 == 0:
                    print(f"  [{modality}/{method}] {i}/{len(tasks)} 用时 {time.time()-t0:.0f}s", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            wr.writerow(estimate_one(t)); fout.flush()
            if i % 50 == 0:
                print(f"  [{modality}/{method}] {i}/{len(tasks)} 用时 {time.time()-t0:.0f}s", flush=True)
    fout.close()

    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    e = np.array([float(r["corner_err"]) for r in rows])
    rts = [float(r["runtime"]) for r in rows]
    print(f"\n[{split}/{modality}/{method}] n={len(rows)} success@10={np.mean(e<10):.3f} "
          f"success@20={np.mean(e<20):.3f} median={np.median(e):.1f} mean_rt={np.mean(rts):.2f}s", flush=True)
    print(" ->", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test_dataset")
    p.add_argument("--modality", default="ir", choices=["ir", "vis"])
    p.add_argument("--method", default="cfog", choices=["cfog", "sift"])
    p.add_argument("--scenes", default="")
    p.add_argument("--max_per_scene", type=int, default=0)
    p.add_argument("--D", type=int, default=700)
    p.add_argument("--coarse_rot", type=int, default=15)
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--no_refine", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    scenes = [x for x in a.scenes.split(",") if x]
    run(a.split, a.modality, a.method, scenes or None, a.max_per_scene,
        a.D, a.coarse_rot, a.bins, a.sigma, not a.no_refine, a.tag, a.workers)
