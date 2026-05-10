"""
Cellpose 2 baseline for wound-healing segmentation.

Cellpose detects individual cell instances in microscopy images (the 'cyto3'
model is the latest pre-trained generalist). Strategy:
  foreground = union of detected cell instances
  wound = NOT foreground (post-processed to keep largest connected component)

Usage:
  python scripts/eval_cellpose.py \
      --model cyto3 \
      --eval both \
      --out-json results/cellpose_cyto3.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_external import STAGES, list_pairs, load_pair  # type: ignore

INTERNAL_ROOT = Path("/data/AUTHOR/wounds/data/raw/dataset/test")


def list_internal_pairs():
    pairs = []
    for img_path in sorted((INTERNAL_ROOT / "images").iterdir()):
        m = INTERNAL_ROOT / "masks" / img_path.name
        if not m.exists():
            raise FileNotFoundError(f"missing mask for {img_path}")
        pairs.append((img_path, m))
    return pairs


def load_internal_pair(img_path, gt_path):
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
    gt = np.asarray(Image.open(gt_path).convert("L"), dtype=np.uint8)
    return img, (gt > 127).astype(np.uint8)


def keep_largest_component(mask):
    if mask.sum() == 0:
        return mask
    lab, _ = ndi.label(mask, structure=np.ones((3, 3), np.uint8))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    if not sizes.size:
        return mask
    return (lab == sizes.argmax()).astype(np.uint8)


def stats(pred, gt):
    pb = pred.astype(bool); gb = gt.astype(bool)
    return (int((pb & gb).sum()), int((pb & ~gb).sum()), int((~pb & gb).sum()))


def iou_dice(tp, fp, fn):
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return iou, dice


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cpsam",
                   help="Cellpose model name (cellpose 4: only 'cpsam' built-in)")
    p.add_argument("--eval", choices=["internal", "lowenstein", "both"],
                   default="both")
    p.add_argument("--diameter", type=float, default=None,
                   help="cell diameter prior; None = auto-estimate")
    p.add_argument("--out-json", default=None)
    p.add_argument("--invert", action="store_true",
                   help="invert image intensity before Cellpose (wound assays "
                        "have bright background; Cellpose expects dark bg)")
    args = p.parse_args()

    from cellpose import models
    use_gpu = torch.cuda.is_available()
    print(f"[load] Cellpose {args.model} gpu={use_gpu}")
    model = models.CellposeModel(model_type=args.model, gpu=use_gpu)

    sets = []
    if args.eval in ("internal", "both"):
        sets.append(("internal", list_internal_pairs(), load_internal_pair))
    if args.eval in ("lowenstein", "both"):
        for stage in STAGES:
            sets.append((f"lowenstein/{stage}", list_pairs(stage), load_pair))

    summary = dict(model=args.model, diameter=args.diameter,
                   inverted=args.invert, results={})
    for label, pairs, loader in sets:
        per_image = []
        agg_tp = agg_fp = agg_fn = 0
        latencies = []
        for img_path, gt_path in pairs:
            img_u8, gt = loader(img_path, gt_path)
            inp = (255 - img_u8) if args.invert else img_u8
            t0 = time.perf_counter()
            out = model.eval(inp, diameter=args.diameter)
            # Cellpose 4 returns (masks, flows, styles); some versions add diams
            masks_inst = out[0]
            latencies.append((time.perf_counter() - t0) * 1000.0)

            foreground = (masks_inst > 0).astype(np.uint8)
            wound = keep_largest_component((1 - foreground).astype(np.uint8))

            tp, fp, fn = stats(wound, gt)
            i, d = iou_dice(tp, fp, fn)
            per_image.append(dict(image=img_path.name, iou=i, dice=d,
                                  n_cells=int(masks_inst.max())))
            agg_tp += tp; agg_fp += fp; agg_fn += fn

        ds_iou, ds_dice = iou_dice(agg_tp, agg_fp, agg_fn)
        block = dict(
            n=len(pairs),
            mean_iou=float(np.mean([r["iou"] for r in per_image])),
            mean_dice=float(np.mean([r["dice"] for r in per_image])),
            dataset_iou=ds_iou,
            dataset_dice=ds_dice,
            mean_latency_ms=float(np.mean(latencies)),
            mean_n_cells=float(np.mean([r["n_cells"] for r in per_image])),
            agg_tp=agg_tp, agg_fp=agg_fp, agg_fn=agg_fn,
            per_image=per_image,
        )
        summary["results"][label] = block
        print(f"[{label}] n={block['n']} "
              f"mean_iou={block['mean_iou']:.4f} "
              f"ds_iou={block['dataset_iou']:.4f} "
              f"latency={block['mean_latency_ms']:.0f}ms "
              f"avg_cells={block['mean_n_cells']:.0f}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
