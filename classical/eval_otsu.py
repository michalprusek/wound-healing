"""
Naive Otsu thresholding baseline. Wound regions in scratch assays are bright;
Otsu picks the optimal global threshold to separate bright/dark.

Strategy:
  1. Grayscale → Gaussian blur → Otsu binary
  2. Foreground = bright pixels (assumed wound)
  3. Optionally apply morphological closing + keep largest connected component
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_external import STAGES, list_pairs, load_pair  # type: ignore

INTERNAL_ROOT = Path("./data/data/raw/dataset/test")


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


def keep_largest(mask):
    if mask.sum() == 0:
        return mask
    lab, _ = ndi.label(mask, structure=np.ones((3, 3), np.uint8))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    if not sizes.size:
        return mask
    return (lab == sizes.argmax()).astype(np.uint8)


def otsu_wound(img_u8, blur_sigma=2, close_k=21, post=True):
    """Otsu thresholding with paper-style hole handling.

    Pipeline:
      1. Gaussian blur, threshold (bright = wound).
      2. Morphological close to bridge boundary discontinuities.
      3. Find contours with RETR_CCOMP (2-level hierarchy).
      4. Take the largest external polygon, fill it solid.
      5. Subtract its internal polygons (cell-holes inside wound).
      6. Return uint8 wound mask.
    """
    blur = cv2.GaussianBlur(img_u8, (0, 0), blur_sigma) if blur_sigma > 0 else img_u8
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not post:
        return (binary > 0).astype(np.uint8)
    if close_k > 1:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern)

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours or hierarchy is None:
        return np.zeros_like(img_u8, dtype=np.uint8)
    hierarchy = hierarchy[0]  # (N, 4) [next, prev, first_child, parent]

    external_idxs = [i for i, h in enumerate(hierarchy) if h[3] == -1]
    if not external_idxs:
        return np.zeros_like(img_u8, dtype=np.uint8)
    largest_ext = max(external_idxs, key=lambda i: cv2.contourArea(contours[i]))

    out = np.zeros_like(img_u8, dtype=np.uint8)
    cv2.drawContours(out, contours, largest_ext, color=1, thickness=cv2.FILLED)
    # Subtract internal polygons whose parent is the largest external
    for i, h in enumerate(hierarchy):
        if h[3] == largest_ext:
            cv2.drawContours(out, contours, i, color=0, thickness=cv2.FILLED)
    return out


def stats(pred, gt):
    pb = pred.astype(bool); gb = gt.astype(bool)
    return (int((pb & gb).sum()), int((pb & ~gb).sum()), int((~pb & gb).sum()))


def iou_dice(tp, fp, fn):
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return iou, dice


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", choices=["internal", "lowenstein", "both"], default="both")
    p.add_argument("--blur-sigma", type=float, default=2.0)
    p.add_argument("--close-k", type=int, default=21)
    p.add_argument("--no-post", action="store_true",
                   help="disable morphology + largest-component post-process")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    sets = []
    if args.eval in ("internal", "both"):
        sets.append(("internal", list_internal_pairs(), load_internal_pair))
    if args.eval in ("lowenstein", "both"):
        for stage in STAGES:
            sets.append((f"lowenstein/{stage}", list_pairs(stage), load_pair))

    summary = dict(method="otsu", blur_sigma=args.blur_sigma,
                   close_k=args.close_k, post=not args.no_post, results={})
    for label, pairs, loader in sets:
        per_image = []
        agg_tp = agg_fp = agg_fn = 0
        latencies = []
        for img_path, gt_path in pairs:
            img_u8, gt = loader(img_path, gt_path)
            t0 = time.perf_counter()
            wound = otsu_wound(img_u8, args.blur_sigma, args.close_k,
                               post=not args.no_post)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            tp, fp, fn = stats(wound, gt)
            i, d = iou_dice(tp, fp, fn)
            per_image.append(dict(image=img_path.name, iou=i, dice=d))
            agg_tp += tp; agg_fp += fp; agg_fn += fn

        ds_iou, ds_dice = iou_dice(agg_tp, agg_fp, agg_fn)
        block = dict(
            n=len(pairs),
            mean_iou=float(np.mean([r["iou"] for r in per_image])),
            mean_dice=float(np.mean([r["dice"] for r in per_image])),
            dataset_iou=ds_iou,
            dataset_dice=ds_dice,
            mean_latency_ms=float(np.mean(latencies)),
            agg_tp=agg_tp, agg_fp=agg_fp, agg_fn=agg_fn,
            per_image=per_image,
        )
        summary["results"][label] = block
        print(f"[{label}] n={block['n']} "
              f"mean_iou={block['mean_iou']:.4f} "
              f"ds_iou={block['dataset_iou']:.4f} "
              f"latency={block['mean_latency_ms']:.1f}ms")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
