# -*- coding: utf-8 -*-
"""Repeat the correspondence draw several times per frame to measure draw-to-draw
variability of the density study. One matcher pass per frame; the same warp field
is sampled repeatedly, so only the sampler's randomness varies."""
import os, sys, json, csv, time
import numpy as np
import cv2
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_baselines as BL
import run_m3t_crossmodal as R

B = os.path.join(PROJ, "data", "M3T", "test_dataset")
RAW = os.path.join(PROJ, "results", "raw")
OUT = os.path.join(RAW, "density_repeats.csv")
DEV = "cuda"
REPS = {"2000": 5, "4000": 5}
PER_SCENE = int(os.environ.get("REP_PER_SCENE", "40"))
MIN_INL = 15

s, th, tx, ty = [float(x) for x in BL.CALIB.split(",")]
th = np.radians(th)
Sinv = np.linalg.inv(np.array([[s * np.cos(th), -s * np.sin(th), tx],
                               [s * np.sin(th), s * np.cos(th), ty], [0, 0, 1.0]]))
KEYS = ["scene", "frame", "N", "rep", "cand", "inliers", "resid", "err_calib", "rt"]


def main():
    BL.build("roma", DEV)
    mdl = BL._CACHE["roma"]
    IR = json.load(open(os.path.join(B, "ir.json"), encoding="utf-8"))
    VS = json.load(open(os.path.join(B, "vis.json"), encoding="utf-8"))
    done = set()
    if os.path.isfile(OUT):
        for r in csv.DictReader(open(OUT, encoding="utf-8")):
            done.add((r["scene"], r["frame"], r["N"], r["rep"]))
    new = not os.path.isfile(OUT)
    fout = open(OUT, "a", newline="", encoding="utf-8")
    wr = csv.DictWriter(fout, fieldnames=KEYS)
    if new:
        wr.writeheader(); fout.flush()

    t0 = time.time(); n_done = 0
    for sc in ["test01", "test02", "test03", "test04", "test05", "test06"]:
        n = len(IR[sc]["uav_img_path"])
        step = max(1, n // PER_SCENE)
        for fi in range(0, n, step):
            todo = [(N, rep) for N, R_ in REPS.items() for rep in range(R_)
                    if (sc, str(fi + 1), N, str(rep)) not in done]
            if not todo:
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
                print(f"  {sc}/{fi+1} match failed: {e}", flush=True)
                continue
            wA, hA = g0.shape[1], g0.shape[0]
            wB, hB = g1.shape[1], g1.shape[0]
            for N, rep in todo:
                t = time.time()
                mm, _ = mdl.sample(warp, cert, num=int(N))
                mm = mm.detach().cpu().numpy() if hasattr(mm, "detach") else np.asarray(mm)
                p0 = np.stack([wA * (mm[:, 0] + 1) / 2, hA * (mm[:, 1] + 1) / 2], 1).astype(np.float32)
                p1 = np.stack([wB * (mm[:, 2] + 1) / 2, hB * (mm[:, 3] + 1) / 2], 1).astype(np.float32)
                rec = dict(scene=sc, frame=fi + 1, N=N, rep=rep, cand=len(p0),
                           inliers="", resid="", err_calib="", rt="")
                if len(p0) >= 8:
                    H, _ = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 3.0,
                                              confidence=0.9999, maxIters=20000)
                    if H is not None:
                        d = np.linalg.norm(cv2.perspectiveTransform(p0.reshape(-1, 1, 2), H).reshape(-1, 2) - p1, axis=1)
                        inl = int((d < 3).sum())
                        rec["inliers"] = inl
                        if inl >= MIN_INL:
                            rec["resid"] = round(float(np.median(d[d < 3])), 3)
                            rec["err_calib"] = round(R.sym_err_px(H @ Sinv, Hg, Wn, Hn), 2)
                rec["rt"] = round(time.time() - t, 3)
                wr.writerow(rec); fout.flush()
            n_done += 1
            if n_done % 20 == 0:
                print(f"  {n_done} frames, {(time.time()-t0)/60:.1f} min", flush=True)
    print("done", n_done, "frames")


if __name__ == "__main__":
    main()
