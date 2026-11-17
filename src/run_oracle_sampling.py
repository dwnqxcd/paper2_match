# -*- coding: utf-8 -*-
"""全量 oracle 对齐 + 等预算采样对照（热红外-可见光，1602 帧）。

每帧只匹配一次；从同一 warp 抽样两次：
  global : sample(num=2000)  —— 论文表 2/5 的口径
  strat  : sample(num=4000) 后按 8x8 网格每格取 31 个（共 2000），不足则由池内顺序补足
两者同预算（2000），比较空间分布的影响。

再对 global 的 H 做 oracle 对齐：
  none  : 原样
  trans : 只允许平移（2 DoF，最小二乘）
  sim   : 允许相似变换（4 DoF，Umeyama）

用法: python src/run_oracle_sampling.py
输出: results/raw/oracle_sampling_full.csv, results/raw/oracle_summary.json
"""
import os, sys, json, csv, time
import numpy as np
import cv2
cv2.setNumThreads(0)
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_baselines as BL
import run_m3t_crossmodal as R
from PIL import Image

B = os.path.join(PROJ, "data", "M3T", "test_dataset")
RAW = os.path.join(PROJ, "results", "raw")
OUT = os.path.join(RAW, "oracle_sampling_full.csv")
SUM = os.path.join(RAW, "oracle_summary.json")
DEV = "cuda"
NG = 2000; NPOOL = 4000; G = 8
LIMIT = int(os.environ.get("ORACLE_LIMIT", "0"))
SCENES = os.environ.get("ORACLE_SCENES", "test01,test02,test03,test04,test05,test06").split(",")
RB = [0, 200, 350, 480, 600, 800]

s, th, tx, ty = [float(x) for x in BL.CALIB.split(",")]
th = np.radians(th)
Sinv = np.linalg.inv(np.array([[s * np.cos(th), -s * np.sin(th), tx],
                               [s * np.sin(th), s * np.cos(th), ty], [0, 0, 1.0]]))
KEYS = ["scene", "frame", "n_global", "n_strat", "err_global", "err_strat", "err_trans", "err_sim"]


def pix(mm, w0, h0, w1, h1):
    """matches are normalized; map to query (w0,h0) and reference (w1,h1) pixels separately."""
    mm = mm.detach().cpu().numpy() if hasattr(mm, "detach") else np.asarray(mm)
    p0 = np.stack([w0 * (mm[:, 0] + 1) / 2, h0 * (mm[:, 1] + 1) / 2], 1).astype(np.float32)
    p1 = np.stack([w1 * (mm[:, 2] + 1) / 2, h1 * (mm[:, 3] + 1) / 2], 1).astype(np.float32)
    return p0, p1


def fit(p0, p1):
    if len(p0) < 8:
        return None, 0
    H, mk = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 3.0, confidence=0.9999, maxIters=20000)
    if H is None:
        return None, 0
    d = np.linalg.norm(cv2.perspectiveTransform(p0.reshape(-1, 1, 2), H).reshape(-1, 2) - p1, axis=1)
    return H, int((d < 3).sum())


def strat_idx(p0, w, h, quota, target):
    gx = np.clip((p0[:, 0] / w * G).astype(int), 0, G - 1)
    gy = np.clip((p0[:, 1] / h * G).astype(int), 0, G - 1)
    cells = {}
    order = np.arange(len(p0))
    for i in order:
        cells.setdefault((gx[i], gy[i]), []).append(i)
    picked = []
    for c, idxs in cells.items():
        picked.extend(idxs[:quota])
    if len(picked) < target:                      # 用池内其余点补足
        rest = [i for i in order if i not in set(picked)]
        picked.extend(rest[:target - len(picked)])
    return np.array(picked[:target], int)


def corners(W, Hh):
    return np.array([[0, 0], [W, 0], [W, Hh], [0, Hh]], np.float64)


def apply(Hm, c):
    q = Hm @ np.vstack([c.T, np.ones((1, 4))])
    return (q[:2] / q[2]).T


def oracle_trans(H, Hg, W, Hh):
    c = corners(W, Hh)
    p = apply(H, c); q = apply(Hg, c)
    t = (q - p).mean(0)
    T = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], float)
    return T @ H


def umeyama(src, dst):
    """least-squares similarity (4 DoF) mapping src->dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, D, Vt = np.linalg.svd(S)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1; Rm = U @ Vt
    var = ((src - mu_s) ** 2).sum() / len(src)
    sc = D.sum() / var
    t = mu_d - sc * Rm @ mu_s
    M = np.eye(3); M[:2, :2] = sc * Rm; M[:2, 2] = t
    return M


def oracle_sim(H, Hg, W, Hh):
    c = corners(W, Hh)
    p = apply(H, c); q = apply(Hg, c)
    return umeyama(p, q) @ H


def main():
    BL.build("roma", DEV); mdl = BL._CACHE["roma"]
    IR = json.load(open(os.path.join(B, "ir.json"), encoding="utf-8"))
    VS = json.load(open(os.path.join(B, "vis.json"), encoding="utf-8"))

    done = set()
    if os.path.isfile(OUT):
        for r in csv.DictReader(open(OUT, encoding="utf-8")):
            done.add((r["scene"], r["frame"]))
    new = not os.path.isfile(OUT)
    f = open(OUT, "a", newline="", encoding="utf-8"); wr = csv.DictWriter(f, fieldnames=KEYS)
    if new:
        wr.writeheader(); f.flush()

    radial = {i: [] for i in range(len(RB) - 1)}
    n_done = 0; stop = False; t0 = time.time()
    for sc in SCENES:
        if stop: break
        n = len(IR[sc]["uav_img_path"])
        for fi in range(int(os.environ.get("ORACLE_START", "0")), n):
            if LIMIT and n_done >= LIMIT:
                stop = True; break
            if (sc, str(fi + 1)) in done:
                continue
            pv = os.path.join(B, sc, "uav_vis", VS[sc]["uav_img_path"][fi])
            if not (os.path.isfile(pv) and R._valid(pv)):
                continue
            g0 = R.load_gray(os.path.join(B, sc, "uav_ir", IR[sc]["uav_img_path"][fi]))
            g1 = R.load_gray(pv)
            k = g0.shape[1] / g1.shape[1]
            g1 = cv2.resize(g1, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            Hg = np.diag([k, k, 1.0]) @ (np.linalg.inv(
                np.array(VS[sc]["H_matrix"][fi]).reshape(3, 3)) @
                np.array(IR[sc]["H_matrix"][fi]).reshape(3, 3))
            W, Hh = g0.shape[1], g0.shape[0]
            W1, H1 = g1.shape[1], g1.shape[0]

            A = Image.fromarray(cv2.cvtColor(g0, cv2.COLOR_GRAY2RGB))
            Bb = Image.fromarray(cv2.cvtColor(g1, cv2.COLOR_GRAY2RGB))
            with torch.inference_mode():
                warp, cert = mdl.match(A, Bb, batched=True, device=DEV)
                mmg, _ = mdl.sample(warp, cert, num=NG)
                mmp, _ = mdl.sample(warp, cert, num=NPOOL)
            p0g, p1g = pix(mmg, W, Hh, W1, H1)
            p0p, p1p = pix(mmp, W, Hh, W1, H1)

            rec = dict(scene=sc, frame=fi + 1, n_global=len(p0g), n_strat="",
                       err_global="", err_strat="", err_trans="", err_sim="")
            H0, _ = fit(p0g, p1g)
            if H0 is not None:
                Hc = H0 @ Sinv                      # 评分用的校正后变换
                e0 = R.sym_err_px(Hc, Hg, W, Hh)
                rec["err_global"] = round(e0, 2)
                if e0 < 50:            # oracle 与径向分析只在已收敛的帧上做
                    Ht = oracle_trans(Hc, Hg, W, Hh)
                    rec["err_trans"] = round(R.sym_err_px(Ht, Hg, W, Hh), 2)
                    Hs4 = oracle_sim(Hc, Hg, W, Hh)
                    rec["err_sim"] = round(R.sym_err_px(Hs4, Hg, W, Hh), 2)
                    r = np.linalg.norm(p0g - np.array([W / 2, Hh / 2]), axis=1)
                    disp = np.linalg.norm(
                        cv2.perspectiveTransform(p0g.reshape(-1, 1, 2), Hs4).reshape(-1, 2)
                        - cv2.perspectiveTransform(p0g.reshape(-1, 1, 2), Hg).reshape(-1, 2), axis=1)
                    for b in range(len(RB) - 1):
                        m = (r >= RB[b]) & (r < RB[b + 1])
                        if m.any():
                            radial[b].append(float(np.median(disp[m])))
            idx = strat_idx(p0p, W, Hh, max(1, NG // (G * G)), NG)
            rec["n_strat"] = len(idx)
            Hs, _ = fit(p0p[idx], p1p[idx])
            if Hs is not None:
                rec["err_strat"] = round(R.sym_err_px(Hs @ Sinv, Hg, W, Hh), 2)
            wr.writerow(rec); f.flush()
            n_done += 1
            if n_done % 50 == 0:
                print(f"  {n_done} frames, {(time.time()-t0)/60:.1f} min", flush=True)
        print(sc, "done", flush=True)
    f.close()

    rows = list(csv.DictReader(open(OUT, encoding="utf-8")))
    close = [r for r in rows if r.get("err_sim") not in ("", None)]

    def med(k, sub=None):
        sub = rows if sub is None else sub
        v = [float(r[k]) for r in sub if r.get(k) not in ("", None)]
        return (float(np.median(v)), len(v), 100.0 * sum(1 for e in v if e < 10) / max(1, len(rows))) if v else (None, 0, None)

    summ = {"N": len(rows), "N_close": len(close), "oracle": {}, "sampling": {}, "radial": {}}
    for k in ("err_global", "err_trans", "err_sim"):
        m, n, s10 = med(k, close)
        summ["oracle"][k] = dict(median=m, n=n, s10=s10)
    m, n, s10 = med("err_strat")
    summ["sampling"]["err_strat"] = dict(median=m, n=n, s10=s10)
    m, n, s10 = med("err_global")
    summ["sampling"]["err_global_all"] = dict(median=m, n=n, s10=s10)
    summ["sampling"]["equal_budget"] = dict(
        n=len(close),
        global_on_close=summ["oracle"]["err_global"]["median"],
        strat_on_close=med("err_strat", close)[0])
    summ["radial"] = {f"{RB[i]}-{RB[i+1]}": (float(np.median(v)) if v else None) for i, v in radial.items()}
    json.dump(summ, open(SUM, "w", encoding="utf-8"), indent=2)
    print(json.dumps(summ, indent=2))
    print("ALL DONE")


if __name__ == "__main__":
    main()
