# -*- coding: utf-8 -*-
"""从地图基线 CSV 计算稿件所需的成功率（口径：分母 = 全部尝试帧，含 ok=0 的失败帧）。

用法：python src/compute_map_stats.py [--csv-dir results/raw] [--json results/raw/map_stats.json]
输出：
  - Table 4（VIS-sat / TIR-sat 列）与 Table 5（GSD 效应）的数字
  - Fig6 数组（attempted-based，全量）
  - 每个场景（test01..test06）的成功率（评测红线：补充 per-scene 鲁棒性证据）
  - 把以上全部写入 --json 指定的文件，供 fig6_gsd_switch.m 与手稿生成脚本直接读取
"""
import os, sys, csv, argparse, json
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RAW = os.path.join(PROJ, "results", "raw")

TASKS = ["vis_map", "ir_map"]
METHODS = ["sift", "loftr", "roma"]
SCENES = ["test01", "test02", "test03", "test04", "test05", "test06"]


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _rate(rows, key):
    """对给定 (task,method) 集合：返回 (attempted, success@10, median_err_ok)。"""
    attempted = set()
    ok_err = []
    for r in rows:
        if (r["task"], r["method"]) != key:
            continue
        attempted.add((r["scene"], r["frame"]))
        if r.get("ok") == "1" and r.get("corner_err") not in ("", None):
            ok_err.append(float(r["corner_err"]))
    n_att = len(attempted)
    succ = sum(1 for e in ok_err if e < 10.0)
    med = float(np.median(ok_err)) if ok_err else float("nan")
    return n_att, succ, med


def _rate_per_scene(rows, task, method):
    """返回 {scene: (attempted, success@10, median_err_ok)}。"""
    out = {}
    for sc in SCENES:
        sub = [r for r in rows if r["task"] == task and r["method"] == method and r["scene"] == sc]
        attempted = set((r["scene"], r["frame"]) for r in sub)
        ok_err = [float(r["corner_err"]) for r in sub
                  if r.get("ok") == "1" and r.get("corner_err") not in ("", None)]
        n_att = len(attempted)
        succ = sum(1 for e in ok_err if e < 10.0)
        med = float(np.median(ok_err)) if ok_err else float("nan")
        out[sc] = (n_att, succ, med)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=RAW)
    ap.add_argument("--json", default=os.path.join(RAW, "map_stats.json"))
    a = ap.parse_args()

    stats = {}
    per_scene = {}
    for t in TASKS:
        for g in (0, 1):
            rows = load(os.path.join(a.csv_dir, f"map_{t}_gsd{g}.csv"))
            s = {m: _rate(rows, (t, m)) for m in METHODS}
            stats[(t, g)] = s
            if g == 1:
                per_scene[t] = {m: _rate_per_scene(rows, t, m) for m in METHODS}

    # ---- Table 4 / Table 5 ----
    print("=" * 72)
    print("Table 4 / Table 5 数字（分母 = 全部尝试帧，含失败帧；全量 1,598）")
    print("=" * 72)
    for t in TASKS:
        label = "VIS-satellite" if t == "vis_map" else "TIR-satellite"
        print(f"\n[{label}]  (gsd0=无归一化, gsd1=有归一化)")
        for m in METHODS:
            n0, s0, _ = stats[(t, 0)].get(m, (0, 0, float("nan")))
            n1, s1, e1 = stats[(t, 1)].get(m, (0, 0, float("nan")))
            r0 = f"{s0 / n0 * 100:.1f}%" if n0 else "0.0%"
            r1 = f"{s1 / n1 * 100:.1f}%" if n1 else "0.0%"
            print(f"  {m:6s}  gsd0={r0:>6s}({s0}/{n0})   gsd1={r1:>6s}({s1}/{n1})  median_ok={e1:.1f}px")

    # ---- Fig6 数组 ----
    after_vis = [stats[("vis_map", 1)][m][1] / max(1, stats[("vis_map", 1)][m][0]) * 100 for m in METHODS]
    after_ir = [stats[("ir_map", 1)][m][1] / max(1, stats[("ir_map", 1)][m][0]) * 100 for m in METHODS]
    before = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    after = after_vis + after_ir
    print("\n" + "=" * 72)
    print("Fig6 数组（before 全 0；after = gsd1 全量 attempted-based）")
    print("=" * 72)
    print("after = [", " ".join(f"{x:.1f}" for x in after), "]")
    print("       VIS-sat:", " ".join(f"{x:.1f}" for x in after_vis))
    print("       TIR-sat:", " ".join(f"{x:.1f}" for x in after_ir))

    # ---- per-scene 鲁棒性证据 ----
    print("\n" + "=" * 72)
    print("Per-scene 成功率（gsd1，attempted-based；补充鲁棒性证据）")
    print("=" * 72)
    for t in TASKS:
        label = "VIS-satellite" if t == "vis_map" else "TIR-satellite"
        print(f"\n[{label}]")
        header = "  scene   " + " ".join(f"{m:>8s}" for m in METHODS) + "   att"
        print(header)
        for sc in SCENES:
            cells, atts = [], []
            for m in METHODS:
                n_att, succ, _ = per_scene[t][m][sc]
                atts.append(n_att)
                cells.append(f"{succ / n_att * 100:7.1f}%" if n_att else "   0.0%")
            print("  " + sc + "  " + " ".join(cells) + f"   {max(atts)}")

    # ---- autoscale 触发率（若存在）----
    autoscale = {}
    for t in TASKS:
        p = os.path.join(a.csv_dir, f"autoscale_{t}.csv")
        if os.path.isfile(p):
            rows = load(p)
            trig = float(np.mean([int(r["trigger"]) for r in rows]) * 100) if rows else 0.0
            rts = [float(r["rt"]) for r in rows if r.get("rt") not in ("", None)]
            autoscale[t] = {"trigger_rate": trig, "n": len(rows),
                            "mean_rt_per_pair": (float(np.mean(rts)) if rts else None)}
            print(f"\nautoscale_{t}.csv: 触发归一化帧占比 = {trig:.1f}%  (n={len(rows)})"
                  + (f"  平均耗时 = {np.mean(rts):.2f}s/对" if rts else ""))

    # ---- 写出 JSON ----
    def cell(t, m, g):
        n, s, e = stats[(t, g)][m]
        return {"attempted": n, "success": s,
                "rate": round(s / n * 100, 1) if n else 0.0,
                "median_ok": round(float(e), 1) if not np.isnan(e) else None}

    out_json = {
        "methods": METHODS,
        "scenes": SCENES,
        "vis_sat": {m: {"gsd0": cell("vis_map", m, 0), "gsd1": cell("vis_map", m, 1)} for m in METHODS},
        "tir_sat": {m: {"gsd0": cell("ir_map", m, 0), "gsd1": cell("ir_map", m, 1)} for m in METHODS},
        "fig6": {"before": [round(x, 1) for x in before], "after": [round(x, 1) for x in after]},
        "per_scene_vis": {m: {sc: cell_from_tuple(per_scene["vis_map"][m][sc]) for sc in SCENES} for m in METHODS},
        "per_scene_tir": {m: {sc: cell_from_tuple(per_scene["ir_map"][m][sc]) for sc in SCENES} for m in METHODS},
        "autoscale": autoscale,
    }
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print(f"\n-> wrote {a.json}")


def cell_from_tuple(tup):
    n_att, succ, med = tup
    return {"attempted": n_att, "success": succ,
            "rate": round(succ / n_att * 100, 1) if n_att else 0.0,
            "median_ok": round(float(med), 1) if not np.isnan(med) else None}


if __name__ == "__main__":
    main()
