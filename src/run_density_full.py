# -*- coding: utf-8 -*-
"""全量密度实验：热红外-可见光 1,602 帧，N=2000 与 N=8000 两档。

每帧只跑一次 RoMa，得到 warp/certainty；再从同一结果分别抽 N 个对应点，
因此两档之间只差采样数，是受控对比。拟合与误差口径完全复用 run_baselines：
  - IR 原生、VIS 缩放到 IR 宽度；Hg = diag(k,k,1) @ (Hvs^-1 @ Hir)
  - USAC_MAGSAC 3.0 px；内点 = 重投影 <3 px；内点 <15 视为未收敛
  - err_calib = sym_err_px(H @ Sinv, Hg, Wn, Hn)
逐帧追加写入，可断点续跑。

用法：python src/run_density_full.py
输出：results/raw/density_full.csv
      scene,frame,N,cand,inliers,resid,err_raw,err_calib,rt
      （rt 为 sample+fit 的耗时，不含匹配）
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
OUT = os.path.join(RAW, "density_full.csv")
DEV = "cuda"
NS = tuple(int(x) for x in os.environ.get("DENSITY_NS", "2000,8000").split(",") if x.strip())
LIMIT = int(os.environ.get("DENSITY_LIMIT", "0"))   # >0 时只跑前 K 帧，用于冒烟测速
MIN_INL = 15

s, th, tx, ty = [float(x) for x in BL.CALIB.split(",")]
th = np.radians(th)
Sinv = np.linalg.inv(np.array([[s * np.cos(th), -s * np.sin(th), tx],
                               [s * np.sin(th), s * np.cos(th), ty], [0, 0, 1.0]]))

KEYS = ["scene", "frame", "N", "cand", "inliers", "resid", "err_raw", "err_calib", "rt"]


def done_set():
    d = set()
    if os.path.isfile(OUT):
        with open(OUT, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d.add((r["scene"], r["frame"], r["N"]))
    return d


def main():
    BL.build("roma", DEV)
    mdl = BL._CACHE["roma"]
    IR = json.load(open(os.path.join(B, "ir.json"), encoding="utf-8"))
    VS = json.load(open(os.path.join(B, "vis.json"), encoding="utf-8"))

    done = done_set()
    new = not os.path.isfile(OUT)
    fout = open(OUT, "a", newline="", encoding="utf-8")
    wr = csv.DictWriter(fout, fieldnames=KEYS)
    if new:
        wr.writeheader(); fout.flush()

    t_start = time.time()
    n_done = 0
    stop = False
    for sc in ["test01", "test02", "test03", "test04", "test05", "test06"]:
        if stop:
            break
        n = len(IR[sc]["uav_img_path"])
        for fi in range(n):
            if LIMIT and n_done >= LIMIT:
                stop = True
                break
            keys = [(sc, str(fi + 1), str(N)) for N in NS]
            if all(k in done for k in keys):
                continue
            pv = os.path.join(B, sc, "uav_vis", VS[sc]["uav_img_path"][fi])
            if not (os.path.isfile(pv) and R._valid(pv)):
                continue
            g0 = R.load_gray(os.path.join(B, sc, "uav_ir", IR[sc]["uav_img_path"][fi]))
            g1 = R.load_gray(pv)
            k = g0.shape[1] / g1.shape[1]
            g1 = cv2.resize(g1, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            Hir = np.array(IR[sc]["H_matrix"][fi]).reshape(3, 3)
            Hvs = np.array(VS[sc]["H_matrix"][fi]).reshape(3, 3)
            Hg = np.diag([k, k, 1.0]) @ (np.linalg.inv(Hvs) @ Hir)
            Wn, Hn = g0.shape[1], g0.shape[0]

            A = Image.fromarray(cv2.cvtColor(g0, cv2.COLOR_GRAY2RGB))
            Bb = Image.fromarray(cv2.cvtColor(g1, cv2.COLOR_GRAY2RGB))
            try:
                with torch.inference_mode():
                    warp, cert = mdl.match(A, Bb, batched=True, device=DEV)
            except Exception as e:
                for N in NS:
                    wr.writerow(dict(scene=sc, frame=fi + 1, N=N, cand="", inliers="",
                                     resid="", err_raw="", err_calib="", rt=""))
                fout.flush()
                print(f"  {sc}/{fi+1} match failed: {e}", flush=True)
                continue

            wA, hA = g0.shape[1], g0.shape[0]
            wB, hB = g1.shape[1], g1.shape[0]
            for N in NS:
                if (sc, str(fi + 1), str(N)) in done:
                    continue
                t0 = time.time()
                mm, _ = mdl.sample(warp, cert, num=N)
                mm = mm.detach().cpu().numpy() if hasattr(mm, "detach") else np.asarray(mm)
                p0 = np.stack([wA * (mm[:, 0] + 1) / 2, hA * (mm[:, 1] + 1) / 2], 1).astype(np.float32)
                p1 = np.stack([wB * (mm[:, 2] + 1) / 2, hB * (mm[:, 3] + 1) / 2], 1).astype(np.float32)
                rec = dict(scene=sc, frame=fi + 1, N=N, cand=len(p0), inliers="",
                           resid="", err_raw="", err_calib="", rt="")
                if len(p0) >= 8:
                    H, _ = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 3.0,
                                              confidence=0.9999, maxIters=20000)
                    if H is not None:
                        d = np.linalg.norm(cv2.perspectiveTransform(p0.reshape(-1, 1, 2), H).reshape(-1, 2) - p1, axis=1)
                        inl = int((d < 3).sum())
                        rec["inliers"] = inl
                        if inl >= MIN_INL:
                            rec["resid"] = round(float(np.median(d[d < 3])), 3)
                            rec["err_raw"] = round(R.sym_err_px(H, Hg, Wn, Hn), 2)
                            rec["err_calib"] = round(R.sym_err_px(H @ Sinv, Hg, Wn, Hn), 2)
                rec["rt"] = round(time.time() - t0, 3)
                wr.writerow(rec); fout.flush()
            n_done += 1
            if n_done % 25 == 0:
                el = time.time() - t_start
                print(f"  {n_done} frames, {el/60:.1f} min elapsed", flush=True)
        print(f"{sc} done", flush=True)

    fout.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
