"""
Aggregate all baseline results from results/*.json into a single comparison
table with bootstrap 95% CIs computed from per_image data dumped by the
updated eval scripts.

Outputs:
  - results/baselines_table.json — clean dict per method × dataset
  - results/baselines_table.tex — LaTeX-ready table for the paper
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_ci(per_image_ious: list[float], agg_tp: int, agg_fp: int,
                 agg_fn: int, n_iter: int, rng: np.random.Generator):
    """Bootstrap CI for Dataset IoU by image-level resampling.

    The Dataset IoU is computed by aggregating per-image TP/FP/FN. We
    don't have those per-image, only per-image IoU. Approximate:
      For each bootstrap iter, sample n images (with replacement) and
      take mean of their per-image IoUs as an estimate. This is the
      "mean per-image IoU" CI, which is conservative but defensible.
    """
    arr = np.asarray(per_image_ious, dtype=np.float32)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = float(arr.mean())
    samples = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        samples.append(float(arr[idx].mean()))
    return point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def aggregate_lowenstein(json_path: Path):
    """Sum agg_tp/fp/fn across stage_1/2/3 → overall ds_iou. Also concat
    per_image lists to compute a bootstrap CI on the 30 images."""
    with open(json_path) as f:
        d = json.load(f)
    res = d.get("results", {})
    if any("lowenstein" in k for k in res):
        sub = {k: v for k, v in res.items() if "lowenstein" in k}
        if not all("agg_tp" in v for v in sub.values()):
            return None  # old JSON without agg
        tp = sum(v["agg_tp"] for v in sub.values())
        fp = sum(v["agg_fp"] for v in sub.values())
        fn = sum(v["agg_fn"] for v in sub.values())
        ds_iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
        ds_dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
        per_img = []
        for v in sub.values():
            per_img.extend(r["iou"] for r in v.get("per_image", []))
        n = sum(v["n"] for v in sub.values())
        latency = float(np.mean([v["mean_latency_ms"] for v in sub.values()]))
        return dict(n=n, dataset_iou=ds_iou, dataset_dice=ds_dice,
                    per_image_iou=per_img, mean_latency_ms=latency,
                    mean_iou=float(np.mean(per_img)) if per_img else None)
    return None


def aggregate_internal(json_path: Path):
    with open(json_path) as f:
        d = json.load(f)
    res = d.get("results", {})
    if "internal" in res:
        v = res["internal"]
        return dict(
            n=v["n"],
            dataset_iou=v["dataset_iou"],
            dataset_dice=v["dataset_dice"],
            per_image_iou=[r["iou"] for r in v.get("per_image", [])],
            mean_latency_ms=v["mean_latency_ms"],
            mean_iou=v.get("mean_iou"),
        )
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-json", default="results/baselines_table.json")
    p.add_argument("--out-tex", default="results/baselines_table.tex")
    p.add_argument("--n-iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    results_dir = Path(args.results_dir)

    # Methods we want in the final table: file pattern → display name
    methods = []
    for json_path in sorted(results_dir.glob("*.json")):
        name = json_path.stem
        if name.startswith("bootstrap_") or name.endswith("_table"):
            continue
        methods.append((name, json_path))

    rows = []
    for name, path in methods:
        block = {"method": name}
        for ds_kind, agg_fn in (("internal", aggregate_internal),
                                ("lowenstein", aggregate_lowenstein)):
            r = agg_fn(path)
            if r is None:
                continue
            iou_pt, lo, hi = bootstrap_ci(r["per_image_iou"], 0, 0, 0,
                                          args.n_iter, rng)
            block[f"{ds_kind}_n"] = r["n"]
            block[f"{ds_kind}_ds_iou"] = r["dataset_iou"]
            block[f"{ds_kind}_mean_iou"] = r["mean_iou"]
            block[f"{ds_kind}_ci_lo"] = lo
            block[f"{ds_kind}_ci_hi"] = hi
            block[f"{ds_kind}_latency_ms"] = r["mean_latency_ms"]
        rows.append(block)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(rows, f, indent=2)

    with open(args.out_tex, "w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("Method & Internal ds\\_iou & Internal mean\\_iou (95\\% CI) & "
                "Internal lat (ms) & "
                "Löwenstein ds\\_iou & Löwenstein mean\\_iou (95\\% CI) & "
                "Löwenstein lat (ms) \\\\\n\\midrule\n")
        for b in rows:
            i_ds = b.get("internal_ds_iou", float("nan"))
            i_mean = b.get("internal_mean_iou", float("nan"))
            i_lo = b.get("internal_ci_lo", float("nan"))
            i_hi = b.get("internal_ci_hi", float("nan"))
            i_lat = b.get("internal_latency_ms", float("nan"))
            l_ds = b.get("lowenstein_ds_iou", float("nan"))
            l_mean = b.get("lowenstein_mean_iou", float("nan"))
            l_lo = b.get("lowenstein_ci_lo", float("nan"))
            l_hi = b.get("lowenstein_ci_hi", float("nan"))
            l_lat = b.get("lowenstein_latency_ms", float("nan"))

            def fmt(x):
                return f"{x:.4f}" if isinstance(x, float) and x == x else "—"

            def fmtci(m, lo, hi):
                if not (isinstance(m, float) and m == m):
                    return "—"
                return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"

            method_esc = b["method"].replace("_", "\\_")
            row = " & ".join([
                method_esc, fmt(i_ds), fmtci(i_mean, i_lo, i_hi),
                f"{i_lat:.0f}", fmt(l_ds), fmtci(l_mean, l_lo, l_hi),
                f"{l_lat:.0f}",
            ])
            f.write(row + " \\\\" + "\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # Print human-readable summary
    print(f"\n{'method':40s} {'int ds_iou':>10s} {'int CI':>20s} "
          f"{'lat':>5s}  {'low ds_iou':>10s} {'low CI':>20s} {'lat':>5s}")
    print("-" * 130)
    for b in rows:
        i_ds = b.get("internal_ds_iou", None)
        i_m = b.get("internal_mean_iou", None)
        i_lo = b.get("internal_ci_lo", None)
        i_hi = b.get("internal_ci_hi", None)
        i_lat = b.get("internal_latency_ms", None)
        l_ds = b.get("lowenstein_ds_iou", None)
        l_m = b.get("lowenstein_mean_iou", None)
        l_lo = b.get("lowenstein_ci_lo", None)
        l_hi = b.get("lowenstein_ci_hi", None)
        l_lat = b.get("lowenstein_latency_ms", None)
        s_int = f"{i_ds:.4f}" if i_ds is not None else "—"
        s_int_ci = f"[{i_lo:.3f},{i_hi:.3f}]" if i_lo is not None else "—"
        s_lat_i = f"{i_lat:.0f}" if i_lat else "—"
        s_low = f"{l_ds:.4f}" if l_ds is not None else "—"
        s_low_ci = f"[{l_lo:.3f},{l_hi:.3f}]" if l_lo is not None else "—"
        s_lat_l = f"{l_lat:.0f}" if l_lat else "—"
        print(f"{b['method']:40s} {s_int:>10s} {s_int_ci:>20s} {s_lat_i:>5s}  "
              f"{s_low:>10s} {s_low_ci:>20s} {s_lat_l:>5s}")
    print(f"\n[saved] {args.out_json} {args.out_tex}")


if __name__ == "__main__":
    main()
