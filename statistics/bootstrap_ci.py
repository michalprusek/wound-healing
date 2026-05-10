"""
Bootstrap 95 % CIs and paired Wilcoxon p-values for the workshop paper.

Inputs: per-image IoU/Dice CSVs produced by the eval scripts. We get:
  - eval_external.py prints overall metrics but doesn't dump per-image CSV;
    so we re-run it programmatically and collect per-image stats.
  - SAM eval scripts dump per-image lists into out-json under results/{...}.json.

Outputs:
  - results/bootstrap_summary.json — every method × dataset with
    Dataset IoU mean + 95 % bootstrap CI (n=1000).
  - results/wilcoxon.tex — LaTeX-ready table of paired p-values
    (between our chosen U-Net and each baseline) on per-image IoU vectors.

Usage:
  python scripts/bootstrap_ci.py \
      --out-dir results/ci

Add new methods to the METHODS list at the top.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Config, SegmentationModel  # type: ignore
from eval_external import (  # type: ignore
    STAGES, list_pairs, load_pair, preprocess, predict_prob, load_model,
    iou_dice, stats,
)

# Internal Kaggle scratch-assay test set
INTERNAL_ROOT = Path("./data/data/raw/dataset/test")


def list_internal_pairs() -> list[tuple[Path, Path]]:
    pairs = []
    for img_path in sorted((INTERNAL_ROOT / "images").iterdir()):
        m = INTERNAL_ROOT / "masks" / img_path.name
        if not m.exists():
            raise FileNotFoundError(f"missing mask for {img_path}")
        pairs.append((img_path, m))
    return pairs


def load_internal_pair(img_path: Path, gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
    gt = np.asarray(Image.open(gt_path).convert("L"), dtype=np.uint8)
    return img, (gt > 127).astype(np.uint8)


def collect_unet_per_image(
    cfg_path: str, ckpt_path: str, eval_kind: str, device: torch.device,
) -> tuple[list[float], list[float], int, int, int]:
    """Returns (per_image_ious, per_image_dices, agg_tp, agg_fp, agg_fn)."""
    cfg = Config.from_yaml(cfg_path)
    model = load_model(cfg, ckpt_path, device).eval()

    if eval_kind == "internal":
        pair_loader = list_internal_pairs()
        loader_fn = load_internal_pair
    elif eval_kind == "lowenstein":
        pair_loader = []
        for stage in STAGES:
            pair_loader.extend(list_pairs(stage))
        loader_fn = load_pair
    else:
        raise ValueError(eval_kind)

    ious, dices = [], []
    agg_tp = agg_fp = agg_fn = 0
    for img_path, gt_path in pair_loader:
        img_u8, gt = loader_fn(img_path, gt_path)
        prob = predict_prob(model, img_u8, cfg.image_size, device, tta=False)
        pred = (prob > 0.5).astype(np.uint8)
        tp, fp, fn, _ = stats(pred, gt)
        i, d = iou_dice(tp, fp, fn)
        ious.append(i); dices.append(d)
        agg_tp += tp; agg_fp += fp; agg_fn += fn

    return ious, dices, agg_tp, agg_fp, agg_fn


def bootstrap_dataset_iou(tps: np.ndarray, fps: np.ndarray, fns: np.ndarray,
                          n_iter: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Resample images, recompute Dataset IoU per resample. Returns
    (point_estimate, lo_95, hi_95)."""
    n = len(tps)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = tps.sum() / max(1, (tps + fps + fns).sum())
    samples = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        s_tp = tps[idx].sum()
        s_fp = fps[idx].sum()
        s_fn = fns[idx].sum()
        denom = s_tp + s_fp + s_fn
        samples.append(s_tp / max(1, denom))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def per_image_tp_fp_fn(cfg_path: str, ckpt_path: str, eval_kind: str,
                        device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = Config.from_yaml(cfg_path)
    model = load_model(cfg, ckpt_path, device).eval()

    if eval_kind == "internal":
        pair_loader = list_internal_pairs()
        loader_fn = load_internal_pair
    else:
        pair_loader = []
        for stage in STAGES:
            pair_loader.extend(list_pairs(stage))
        loader_fn = load_pair

    tps, fps, fns, ious = [], [], [], []
    for img_path, gt_path in pair_loader:
        img_u8, gt = loader_fn(img_path, gt_path)
        prob = predict_prob(model, img_u8, cfg.image_size, device, tta=False)
        pred = (prob > 0.5).astype(np.uint8)
        tp, fp, fn, _ = stats(pred, gt)
        i, _ = iou_dice(tp, fp, fn)
        tps.append(tp); fps.append(fp); fns.append(fn); ious.append(i)

    return (np.array(tps), np.array(fps), np.array(fns), np.array(ious))


def load_sam_per_image_ious(json_path: str) -> np.ndarray:
    """Reload SAM eval per-image scores from the saved JSON. eval_sam.py
    aggregates per-stage but doesn't dump per-image; we read what's available."""
    if not os.path.exists(json_path):
        return np.array([])
    with open(json_path) as f:
        d = json.load(f)
    # results[label]={n, mean_iou, ...}; we only have means, not per-image.
    # Return per-stage means as a coarse vector — caller should re-run eval
    # with per-image dump if more precision is needed.
    out = []
    for k, v in d.get("results", {}).items():
        if "lowenstein" in k:
            out.append(v["mean_iou"])
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results/ci")
    p.add_argument("--n-iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Methods to evaluate via U-Net interface
    methods = [
        ("v2", "configs/unetpp_v2.yaml", "runs/20260418_204713_v2/best.ckpt"),
        ("v3", "configs/unetpp_v3.yaml", "runs/20260419_014839_v3/best.ckpt"),
        ("v4paper_s0", "configs/unet_v4.yaml", "runs/20260419_085517_v4paper/best.ckpt"),
        ("v4paper_s42", "configs/unet_v4_s42.yaml", "runs/20260508_121532_v4paper_s42/best.ckpt"),
        ("fix_clahe_s42", "configs/fix_clahe.yaml", "runs/20260508_121824_fix_clahe_s42/best.ckpt"),
    ]

    results = {}
    per_image = {}
    for name, cfg_path, ckpt in methods:
        for kind in ("internal", "lowenstein"):
            key = f"{name}/{kind}"
            print(f"[eval] {key}")
            tps, fps, fns, ious = per_image_tp_fp_fn(cfg_path, ckpt, kind, device)
            point, lo, hi = bootstrap_dataset_iou(tps, fps, fns, args.n_iter, rng)
            results[key] = dict(
                n=len(tps),
                dataset_iou=point,
                ci_lo=lo, ci_hi=hi,
                mean_per_image_iou=float(ious.mean()) if len(ious) else 0.0,
            )
            per_image[key] = ious.tolist()

    out_summary = out / "bootstrap_summary.json"
    with open(out_summary, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {out_summary}")

    # Wilcoxon: paired test on per-image IoU between v2 and others, on lowenstein
    ref_key = "v2/lowenstein"
    ref = np.array(per_image[ref_key])
    wilcoxon_rows = []
    for name, _, _ in methods:
        key = f"{name}/lowenstein"
        if key == ref_key:
            continue
        x = np.array(per_image[key])
        if len(x) != len(ref):
            continue
        try:
            stat, p = wilcoxon(ref, x, alternative="greater")
            wilcoxon_rows.append((name, stat, p))
        except ValueError:
            wilcoxon_rows.append((name, float("nan"), float("nan")))

    with open(out / "wilcoxon.tex", "w") as f:
        f.write("% Paired Wilcoxon: v2 > X on per-image Löwenstein IoU\n")
        f.write("\\begin{tabular}{lrr}\n\\toprule\nMethod & W & p (one-sided) \\\\\n\\midrule\n")
        for name, stat, p in wilcoxon_rows:
            f.write(f"{name} & {stat:.1f} & {p:.4g} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[saved] {out / 'wilcoxon.tex'}")

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: n={v['n']} dataset_iou={v['dataset_iou']:.4f} "
              f"95% CI=[{v['ci_lo']:.4f}, {v['ci_hi']:.4f}] "
              f"mean_iou={v['mean_per_image_iou']:.4f}")


if __name__ == "__main__":
    main()
