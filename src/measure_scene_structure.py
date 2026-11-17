# -*- coding: utf-8 -*-
"""Quantify cross-modal edge structure per scene, independent of any matcher.

For each scene we sample frames, bring the query to map scale with the platform
GSD ratio, warp its edge map into the map chip with the reference homography, and
measure how much of the two edge sets agree. The per-scene means are then
correlated with the per-scene success rate reported in Table 4.
"""
import os, sys, json, csv, argparse
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_m3t_crossmodal as R

DATA = os.path.join(PROJ, "data", "M3T", "test_dataset")
RAW = os.path.join(PROJ, "results", "raw")
GSD = {"vis": 0.174}
SCENES = ["test01", "test02", "test03", "test04", "test05", "test06"]


def edge_agreement(q_edges_warp, r_edges, tol=2.0):
    """Symmetric agreement between two edge maps in the same frame."""
    q = (q_edges_warp > 0).astype(np.uint8)      # Canny writes 0/255; binarize first
    r = (r_edges > 0).astype(np.uint8)
    if q.sum() == 0 or r.sum() == 0:
        return np.nan, np.nan, np.nan
    d_ref = cv2.distanceTransform(1 - r, cv2.DIST_L2, 3)
    d_q = cv2.distanceTransform(1 - q, cv2.DIST_L2, 3)
    prec = float((d_ref[q > 0] <= tol).mean())   # warped query edges near a reference edge
    rec = float((d_q[r > 0] <= tol).mean())      # reference edges near a warped query edge
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_scene", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(RAW, "scene_structure.csv"))
    a = ap.parse_args()

    vs = json.load(open(os.path.join(DATA, "vis.json"), encoding="utf-8"))
    rows = []
    for sc in SCENES:
        n = len(vs[sc]["uav_img_path"])
        for fi in range(0, n, max(1, n // a.per_scene)):
            mp = os.path.join(DATA, sc, "map", vs[sc]["map_path"][fi])
            vp = os.path.join(DATA, sc, "uav_vis", vs[sc]["uav_img_path"][fi])
            if not (os.path.isfile(mp) and os.path.getsize(mp) > 100000 and os.path.isfile(vp)):
                continue
            g0 = R.load_gray(vp)
            g1 = R.load_gray(mp)
            k = GSD["vis"]
            g0 = cv2.resize(g0, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            S = np.diag([k, k, 1.0])
            Hg = np.array(vs[sc]["H_matrix"][fi]).reshape(3, 3) @ np.linalg.inv(S)
            W, H = g1.shape[1], g1.shape[0]

            qe = cv2.Canny(g0, 60, 160)
            re = cv2.Canny(g1, 60, 160)
            qw = cv2.warpPerspective(qe, Hg, (W, H), flags=cv2.INTER_NEAREST)
            prec, rec, f1 = edge_agreement(qw, re)
            rows.append(dict(scene=sc, frame=fi + 1,
                             edge_density=float(re.mean() / 255.0),
                             precision=prec, recall=rec, f1=f1))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", a.out, len(rows), "frames")


if __name__ == "__main__":
    main()
