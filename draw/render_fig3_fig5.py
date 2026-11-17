# -*- coding: utf-8 -*-
"""匹配示意矩阵渲染（图 3 TIR-VIS 6x3；图 5 卫星对 7x3）。

约定
----
* 每个面板 = 左图 A + 间隔 + 右图 B 的拼接画布，对应点用连线 + 圆点表示。
* 画布尺寸按 CSV 的 wa/ha/wb/hb 缩放，源图分辨率不同也坐标对齐。
* 行 = 方法，列 = 典型场景；只在每行第一列写方法名，列头写 (a)(b)(c)。
* 不标注内点/候选数（示意口径与正文全量口径不同，标了会误导）。
* 连线颜色只表示几何是否正确：角点误差 < 10 px 记绿，否则红。
* 连线做确定性下采样（max_lines），避免稠密匹配器糊成一团。

运行
----
  python draw/render_fig3_fig5.py --fig 3
  python draw/render_fig3_fig5.py --fig 5
"""
import os, csv, json, argparse
import numpy as np
import cv2
cv2.setNumThreads(0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, "data", "M3T", "test_dataset")
MD = os.path.join(HERE, "match_data")

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

OK_C = (0.10, 0.65, 0.30)
BAD_C = (0.85, 0.20, 0.20)
LINE_C = (1.0, 0.35, 0.05)   # 单一橙色，不携带“对/错”含义

# ---- 图 3：TIR-VIS，6 方法（行，按 Table 2 顺序） x 3 场景（列） ----
TIRV_SCENES = [("test01", 60), ("test02", 60), ("test03", 27), ("test05", 60)]
TIRV_METHODS = [("sift", "SIFT"), ("disk", "DISK"), ("dedode", "DeDoDe"),
                ("loftr", "LoFTR"), ("xoftr", "XoFTR"), ("roma", "RoMa")]


def tirvis_rows():
    rows = []
    for m, lab in TIRV_METHODS:
        row = []
        for sc, fr in TIRV_SCENES:
            row.append(dict(
                a=f"{DATA}/{sc}/uav_ir/{fr:04d}_T.JPG",
                b=f"{DATA}/{sc}/uav_vis/{fr:04d}_W.JPG",
                csv=f"{MD}/tir_vis_{m}_{sc}_f{fr}.csv",
                key=f"tir_vis_{m}_{sc}", method=lab, scene=sc, frame=fr))
        rows.append(row)
    return rows


# ---- 图 5：VIS-卫星（GSD 归一化），7 方法（行） x 3 场景（列） ----
FIG5_SCENES = [("test01", 60), ("test02", 60), ("test03", 27), ("test05", 60)]
FIG5_METHODS = [("sift", "SIFT"), ("disk", "DISK"), ("dedode", "DeDoDe"),
                ("loftr", "LoFTR"), ("xoftr", "XoFTR"), ("roma", "RoMa"), ("cfog", "CFOG")]


def fig5_rows():
    rows = []
    for m, lab in FIG5_METHODS:
        row = []
        for sc, fr in FIG5_SCENES:
            row.append(dict(
                a=f"{DATA}/{sc}/uav_vis/{fr:04d}_W.JPG",
                b=f"{DATA}/{sc}/map/{fr:04d}.tif",
                csv=f"{MD}/vis_map_{m}_{sc}_f{fr}.csv",
                key=f"vis_map_{m}_{sc}", method=lab, scene=sc, frame=fr))
        rows.append(row)
    return rows


def load_rgb(p):
    buf = np.fromfile(p, dtype=np.uint8)
    if buf.size == 0:
        raise FileNotFoundError(p)
    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def read_csv(p):
    xa = []; ya = []; xb = []; yb = []; geo = None
    if not os.path.isfile(p):
        return (np.array([]),) * 4 + (None, None, None, None)
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if geo is None:
                geo = (float(r["wa"]), float(r["ha"]), float(r["wb"]), float(r["hb"]))
            xa.append(float(r["xa"])); ya.append(float(r["ya"]))
            xb.append(float(r["xb"])); yb.append(float(r["yb"]))
    if geo is None:                      # 文件存在但只有表头（0 候选）
        geo = (None, None, None, None)
    return tuple(np.array(v, np.float32) for v in (xa, ya, xb, yb)) + geo


def make_canvas(pa, pb, csvp, HD, GAP):
    xa, ya, xb, yb, wa, ha, wb, hb = read_csv(csvp)
    A = load_rgb(pa); B = load_rgb(pb)
    if wa is None:
        ha, wa = A.shape[0], A.shape[1]
        hb, wb = B.shape[0], B.shape[1]
    n = len(xa)
    # 逐条线判定内点/外点：在原始坐标上做一次 MAGSAC 拟合，4px 阈值
    mask = np.zeros(n, bool)
    if n >= 8:
        p0 = np.stack([xa, ya], 1).astype(np.float32)
        p1 = np.stack([xb, yb], 1).astype(np.float32)
        H, mk = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 4.0,
                                   confidence=0.9999, maxIters=20000)
        if mk is not None:
            mask = (mk.ravel() > 0)
    sa = HD / float(ha); sb = HD / float(hb)
    A2 = cv2.resize(A, (max(1, int(round(wa * sa))), max(1, int(round(ha * sa)))),
                    interpolation=cv2.INTER_AREA)
    B2 = cv2.resize(B, (max(1, int(round(wb * sb))), max(1, int(round(hb * sb)))),
                    interpolation=cv2.INTER_AREA)
    wd = A2.shape[1] + GAP + B2.shape[1]
    ht = max(A2.shape[0], B2.shape[0])
    canvas = np.full((ht, wd, 3), 255, np.uint8)
    canvas[:A2.shape[0], :A2.shape[1]] = A2
    off = A2.shape[1] + GAP
    canvas[:B2.shape[0], off:off + B2.shape[1]] = B2
    return (canvas, xa * sa, ya * sa, off + xb * sb, yb * sb, n, mask)


def draw_panel(ax, canvas, xa, ya, xb, yb, mask=None, max_lines=70, **kw):
    ax.imshow(canvas)
    ax.set_axis_off()
    n = len(xa)
    if n:
        if n > max_lines:
            idx = np.random.default_rng(0).choice(n, size=max_lines, replace=False); idx.sort()
        else:
            idx = np.arange(n)
        for i in idx:
            ax.plot([xa[i], xb[i]], [ya[i], yb[i]], color=LINE_C, alpha=0.75, lw=0.55, zorder=2)
        ax.scatter(xa[idx], ya[idx], s=4, c=[LINE_C], edgecolors="w", linewidths=0.25, zorder=3)
        ax.scatter(xb[idx], yb[idx], s=4, c=[LINE_C], edgecolors="w", linewidths=0.25, zorder=3)
    ax.set_xlim(-1, canvas.shape[1]); ax.set_ylim(canvas.shape[0], -1)


RAW = os.path.join(PROJ, "results", "raw")
_LBL2M = {"SIFT": "sift", "DISK": "disk", "DeDoDe": "dedode", "LoFTR": "loftr",
          "XoFTR": "xoftr", "RoMa": "roma", "CFOG": "cfog"}


def _eval_map(kind):
    """从权威评测 CSV 取 (method, scene, frame)->误差；用于决定面板绿/红。"""
    E = {}

    def add(path, method, col="err_calib"):
        if not os.path.isfile(path):
            return
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                mc = r.get("method")
                if method and mc is not None and mc != method:
                    continue
                v = r.get(col)
                try:
                    e = float(v) if v not in (None, "") else None
                except ValueError:
                    e = None
                E[(method or mc, r["scene"], str(r["frame"]))] = e

    if kind == "tir":
        add(os.path.join(RAW, "m3t_tirvis_sift_full.csv"), "sift")
        add(os.path.join(RAW, "disk_dedode_full.csv"), "disk")
        add(os.path.join(RAW, "disk_dedode_full.csv"), "dedode")
        add(os.path.join(RAW, "loftr_full.csv"), "loftr")
        add(os.path.join(RAW, "m3t_xoftr_full.csv"), "xoftr")
        add(os.path.join(RAW, "m3t_roma_full.csv"), "roma")
    else:
        add(os.path.join(RAW, "map_vis_map_gsd1.csv"), None, col="corner_err")
        add(os.path.join(RAW, "m3t_test_dataset_vis_cfog_full.csv"), "cfog", col="corner_err")
    return E


def _panel_ok(evald, p):
    mk = _LBL2M.get(p["method"], p["method"].lower())
    e = evald.get((mk, p["scene"], str(p["frame"])))
    return e is not None and e < 10


def render(rows, out_png, col_titles=None, width_cm=17.0, HD=300, GAP=12,
           hgap_cm=0.22, vgap_cm=0.20, top_cm=0.20, bottom_cm=0.12, max_lines=70,
           eval_kind="tir"):
    stats = {}
    sp = os.path.join(MD, "stats.json")
    if os.path.isfile(sp):
        stats = json.load(open(sp, encoding="utf-8"))
    evald = _eval_map(eval_kind)

    nrow = len(rows); ncol = max(len(r) for r in rows)
    probe = None
    for r in rows:
        for p in r:
            c, *_ = make_canvas(p["a"], p["b"], p["csv"], HD, GAP)
            probe = c.shape; break
        if probe:
            break
    asp = probe[1] / probe[0]
    pw = (width_cm - (ncol - 1) * hgap_cm) / ncol
    ph = pw / asp
    head = 0.55 if col_titles else 0.0
    height_cm = top_cm + head + nrow * ph + (nrow - 1) * vgap_cm + bottom_cm

    fig = plt.figure(figsize=(width_cm / 2.54, height_cm / 2.54), dpi=300)
    fig.patch.set_facecolor("white")

    for ri, r in enumerate(rows):
        for ci, p in enumerate(r):
            canvas, xa, ya, xb, yb, n, mask = make_canvas(p["a"], p["b"], p["csv"], HD, GAP)
            left = (ci * (pw + hgap_cm)) / width_cm
            bottom = (bottom_cm + (nrow - 1 - ri) * (ph + vgap_cm)) / height_cm
            ax = fig.add_axes([left, bottom, pw / width_cm, ph / height_cm])
            draw_panel(ax, canvas, xa, ya, xb, yb, mask, max_lines)
            if ci == 0:
                ax.text(0.008, 0.988, p["method"], transform=ax.transAxes, fontsize=7.2,
                        color="k", va="top", ha="left", zorder=5,
                        bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none", alpha=0.82))

    if col_titles:
        for ci, t in enumerate(col_titles):
            cx = (ci * (pw + hgap_cm) + pw / 2) / width_cm
            cy = 1 - (top_cm + 0.10) / height_cm
            fig.text(cx, cy, t, ha="center", va="center", fontsize=8)

    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    pdf = os.path.splitext(out_png)[0] + ".pdf"
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved: {out_png} ({width_cm:.1f} x {height_cm:.1f} cm), {nrow}x{ncol} panels")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", type=int, default=3, choices=[3, 5])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.fig == 3:
        render(tirvis_rows(), a.out or os.path.join(HERE, "fig3_tirvis_matches.png"),
               col_titles=["(a)", "(b)", "(c)", "(d)"], width_cm=17.0, HD=300, GAP=10, max_lines=70)
    else:
        render(fig5_rows(), a.out or os.path.join(HERE, "fig5_map_gsd_matches.png"),
               col_titles=["(a)", "(b)", "(c)", "(d)"], width_cm=17.0, HD=280, GAP=10, max_lines=70,
               eval_kind="vis")
