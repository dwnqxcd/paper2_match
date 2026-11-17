# -*- coding: utf-8 -*-
"""汇总所有方法 x 任务对的 attempted-based s@10（角点误差 < 10 px / 全部尝试帧）。

读取 results/raw 下已有的真实 CSV，而不是依赖手稿里写死的百分比。
缺失的格子直接标 MISSING，并给出分母（尝试帧数）以便发现不一致。

成功口径（与 compute_map_stats.py 完全一致）：
  - 地图任务 CSV：ok=='1' 且 corner_err 非空且 < 10
  - run_baselines 格式（disk/dedode/loftr 等）：ok=='1' 且 err_calib 非空且 < 10
  - roma/xoftr full：err_calib 非空且 < 10（每行即一帧尝试）
分母 = 该 (task,method) 出现的唯一 (scene,frame) 数。
"""
import os, sys, csv, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RAW = os.path.join(PROJ, "results", "raw")
METHODS = ["sift", "disk", "dedode", "loftr", "xoftr", "roma", "cfog"]

# 每个 (task, method) -> 源 CSV 与在 CSV 里的判定方式
# task 键：vis_sat / tir_sat 用 gsd1 归一化结果；tir_vis 用直接结果
SOURCES = {
    # ---------- TIR-VIS（红外-可见光，近单位尺度，无 GSD 归一化）----------
    ("tir_vis", "disk"):   ("disk_dedode_full.csv", "baselines"),
    ("tir_vis", "dedode"): ("disk_dedode_full.csv", "baselines"),
    ("tir_vis", "loftr"):  ("loftr_full.csv", "baselines"),
    ("tir_vis", "roma"):   ("m3t_roma_full.csv", "roma_xoftr"),
    ("tir_vis", "xoftr"):  ("m3t_xoftr_full.csv", "roma_xoftr"),
    ("tir_vis", "sift"):   ("m3t_tirvis_sift_full.csv", "baselines"),  # 全量 1602 帧
    ("tir_vis", "cfog"):   (None, None),  # CFOG 是卫星对专用相关法，不在 TIR-VIS 评估（见正文说明）
    # ---------- VIS-sat（可见光-卫星，gsd1 归一化）----------
    ("vis_sat", "sift"):   ("map_vis_map_gsd1.csv", "map"),
    ("vis_sat", "loftr"):  ("map_vis_map_gsd1.csv", "map"),
    ("vis_sat", "roma"):   ("map_vis_map_gsd1.csv", "map"),
    ("vis_sat", "disk"):   ("map_vis_map_gsd1_extra.csv", "map"),
    ("vis_sat", "dedode"): ("map_vis_map_gsd1_extra.csv", "map"),
    ("vis_sat", "xoftr"):  ("map_vis_map_gsd1_extra.csv", "map"),
    ("vis_sat", "cfog"):   ("m3t_test_dataset_vis_cfog_full.csv", "cfog"),
    # ---------- TIR-sat（红外-卫星，gsd1 归一化）----------
    ("tir_sat", "sift"):   ("map_ir_map_gsd1.csv", "map"),
    ("tir_sat", "loftr"):  ("map_ir_map_gsd1.csv", "map"),
    ("tir_sat", "roma"):   ("map_ir_map_gsd1.csv", "map"),
    ("tir_sat", "disk"):   ("map_ir_map_gsd1_extra.csv", "map"),
    ("tir_sat", "dedode"): ("map_ir_map_gsd1_extra.csv", "map"),
    ("tir_sat", "xoftr"):  ("map_ir_map_gsd1_extra.csv", "map"),
    ("tir_sat", "cfog"):   ("m3t_test_dataset_ir_cfog_full.csv", "cfog"),
}

TASK_LABEL = {"vis_sat": "VIS-satellite", "tir_sat": "TIR-satellite", "tir_vis": "TIR-VIS"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rate_map(rows, method):
    """地图 CSV：ok=='1' 且 corner_err<10。"""
    sub = [r for r in rows if r["method"] == method]
    att = set((r["scene"], r["frame"]) for r in sub)
    ok_err = []
    for r in sub:
        if r.get("ok") == "1" and r.get("corner_err") not in ("", None):
            ok_err.append(float(r["corner_err"]))
    return _finish(att, ok_err)


def rate_baselines(rows, method):
    """run_baselines/disk_dedode 格式：ok=='1' 且 err_calib<10。"""
    sub = [r for r in rows if r["method"] == method]
    att = set((r["scene"], r["frame"]) for r in sub)
    ok_err = []
    for r in sub:
        if r.get("ok") == "1" and r.get("err_calib") not in ("", None):
            ok_err.append(float(r["err_calib"]))
    return _finish(att, ok_err)


def rate_roma_xoftr(rows, method=None):
    """roma/xoftr full：每行一帧，err_calib<10 即成功（无 ok 列）。"""
    att = set((r["scene"], r["frame"]) for r in rows)
    ok_err = []
    for r in rows:
        if r.get("err_calib") not in ("", None):
            try:
                ok_err.append(float(r["err_calib"]))
            except ValueError:
                pass
    return _finish(att, ok_err)


def rate_cfog(rows, method=None):
    """CFOG 输出：每行一帧，corner_err<10 即成功。"""
    att = set((r["scene"], r["frame"]) for r in rows)
    ok_err = []
    for r in rows:
        if r.get("corner_err") not in ("", None):
            try:
                ok_err.append(float(r["corner_err"]))
            except ValueError:
                pass
    return _finish(att, ok_err)


def _finish(att, ok_err):
    n = len(att)
    succ = sum(1 for e in ok_err if e < 10.0)
    med = float(np.median(ok_err)) if ok_err else float("nan")
    return dict(attempted=n, success=succ, rate=round(succ / n * 100, 1) if n else None,
                median_ok=round(med, 1) if ok_err else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(RAW, "all_method_stats.json"))
    a = ap.parse_args()

    cache = {}
    stats = {}
    for task in TASK_LABEL:
        stats[task] = {}
        for m in METHODS:
            src, kind = SOURCES[(task, m)]
            if src is None:
                stats[task][m] = dict(attempted=None, success=None, rate=None, median_ok=None, note="MISSING")
                continue
            p = os.path.join(RAW, src)
            if not os.path.isfile(p):
                stats[task][m] = dict(attempted=None, success=None, rate=None, median_ok=None, note="FILE_MISSING")
                continue
            if src not in cache:
                cache[src] = load(p)
            rows = cache[src]
            if kind == "map":
                res = rate_map(rows, m)
            elif kind == "baselines":
                res = rate_baselines(rows, m)
            elif kind == "cfog":
                res = rate_cfog(rows)
            else:
                res = rate_roma_xoftr(rows)
            res["note"] = f"{src} ({kind})"
            stats[task][m] = res

    # ---- 打印矩阵 ----
    print("=" * 92)
    print("All-method attempted-based s@10  (corner err < 10 px over ALL attempted frames)")
    print("=" * 92)
    hdr = "method".ljust(8) + "".join(TASK_LABEL[t].rjust(26) for t in TASK_LABEL)
    print(hdr)
    for m in METHODS:
        line = m.ljust(8)
        for t in TASK_LABEL:
            c = stats[t][m]
            if c["rate"] is None:
                line += "MISSING".rjust(26)
            else:
                line += f"{c['rate']:.1f}% ({c['success']}/{c['attempted']})".rjust(26)
        print(line)

    print("\n分母（尝试帧数）对照：")
    for m in METHODS:
        parts = []
        for t in TASK_LABEL:
            c = stats[t][m]
            parts.append(f"{TASK_LABEL[t]}={c['attempted']}")
        print("  " + m.ljust(8) + "  " + "  ".join(parts))

    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n-> wrote {a.json}")


if __name__ == "__main__":
    main()
