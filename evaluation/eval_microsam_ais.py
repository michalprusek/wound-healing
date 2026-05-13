"""µSAM (vit_b_lm) eval using PROPER AIS workflow with trained decoder.

Loads the InstanceSegmentationWithDecoder via micro_sam.automatic_segmentation.
Per Archit et al. Nat. Methods 2025, AIS is the recommended workflow for `_lm` models.
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

warnings.filterwarnings("ignore")

INTERNAL_ROOT = Path("/data/wounds/wounds/data/raw/dataset/test")
EXTERNAL_ROOT = Path("/data/wounds/wounds/data/external_lowenstein/woundhealing_v3")
STAGES = ("stage_1", "stage_2", "stage_3")


def list_internal():
    pairs = []
    for p in sorted((INTERNAL_ROOT / "images").iterdir()):
        gt = INTERNAL_ROOT / "masks" / p.name
        if gt.exists():
            pairs.append((p, gt))
    return pairs


def list_external(stage):
    out = []
    for p in sorted((EXTERNAL_ROOT / "original" / stage).iterdir()):
        gt = EXTERNAL_ROOT / "groundtruth" / stage / f"{p.stem}.png"
        if gt.exists():
            out.append((p, gt))
    return out


def load_pair(p, g):
    img = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
    gt = np.asarray(Image.open(g).convert("L"), dtype=np.uint8)
    return img, (gt > 127).astype(np.uint8)


def keep_largest(m):
    if m.sum() == 0:
        return m
    lab, _ = ndi.label(m, structure=np.ones((3, 3), np.uint8))
    sizes = np.bincount(lab.ravel())
    if sizes.size <= 1:
        return m
    sizes[0] = 0
    return (lab == sizes.argmax()).astype(np.uint8)


def stats(p, g):
    pb = p.astype(bool)
    gb = g.astype(bool)
    tp = int((pb & gb).sum())
    fp = int((pb & ~gb).sum())
    fn = int((~pb & gb).sum())
    return tp, fp, fn


def iou_dice(tp, fp, fn):
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return iou, dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--model-type", default="vit_b_lm")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--halo", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional: process only first N images per dataset (debug).")
    args = ap.parse_args()

    from micro_sam.automatic_segmentation import (
        get_predictor_and_segmenter,
        automatic_instance_segmentation,
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] model={args.model_type} mode=AIS device={dev}", flush=True)
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=args.model_type,
        device=dev,
        segmentation_mode="ais",
        is_tiled=True,
    )
    print(f"[load] predictor={type(predictor).__name__} segmenter={type(segmenter).__name__}",
          flush=True)

    sets = [("internal", list_internal())]
    for s in STAGES:
        sets.append((f"lowenstein/{s}", list_external(s)))

    summary = {
        "method": "microsam_ais",
        "model_type": args.model_type,
        "workflow": f"AIS ({type(segmenter).__name__})",
        "tile_shape": [args.tile, args.tile],
        "halo": [args.halo, args.halo],
        "results": {},
    }

    for label, pairs in sets:
        if args.limit:
            pairs = pairs[: args.limit]
        per_image = []
        latencies = []
        agg_tp = agg_fp = agg_fn = 0
        agg_pred = agg_gt = agg_total = 0
        for ip, gp in pairs:
            img, gt = load_pair(ip, gp)
            t0 = time.perf_counter()
            try:
                instances = automatic_instance_segmentation(
                    predictor=predictor,
                    segmenter=segmenter,
                    input_path=img,
                    ndim=2,
                    tile_shape=(args.tile, args.tile),
                    halo=(args.halo, args.halo),
                    verbose=False,
                )
            except Exception as e:
                print(f"  ERROR {ip.name}: {e}", flush=True)
                continue
            cells = (np.asarray(instances) > 0).astype(np.uint8)
            wound = keep_largest((1 - cells).astype(np.uint8))
            dt_ms = (time.perf_counter() - t0) * 1000
            latencies.append(dt_ms)
            tp, fp, fn = stats(wound, gt)
            i, d = iou_dice(tp, fp, fn)
            n_inst = int(np.asarray(instances).max())
            pred_pix = int(wound.sum())
            gt_pix = int(gt.sum())
            tot_pix = int(gt.size)
            per_image.append({
                "image": ip.name, "iou": i, "dice": d,
                "n_instances": n_inst, "latency_ms": dt_ms,
                "pred_pixels": pred_pix, "gt_pixels": gt_pix,
            })
            agg_tp += tp; agg_fp += fp; agg_fn += fn
            agg_pred += pred_pix; agg_gt += gt_pix; agg_total += tot_pix
            print(f"  [{label}] {ip.name}: iou={i:.4f} cells={n_inst} "
                  f"lat={dt_ms:.0f}ms", flush=True)

        ds_iou, ds_dice = iou_dice(agg_tp, agg_fp, agg_fn)
        wra_err = abs(agg_pred - agg_gt) / agg_total * 100.0 if agg_total else 0.0
        block = {
            "n": len(per_image),
            "mean_iou": float(np.mean([r["iou"] for r in per_image])) if per_image else 0.0,
            "mean_dice": float(np.mean([r["dice"] for r in per_image])) if per_image else 0.0,
            "dataset_iou": float(ds_iou),
            "dataset_dice": float(ds_dice),
            "wra_err_pct": float(wra_err),
            "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "agg_tp": agg_tp, "agg_fp": agg_fp, "agg_fn": agg_fn,
            "pred_total": agg_pred, "gt_total": agg_gt, "img_total": agg_total,
            "per_image": per_image,
        }
        summary["results"][label] = block
        print(f"==> {label}: ds_iou={ds_iou:.4f} mean_iou={block['mean_iou']:.4f} "
              f"WRA_err={wra_err:.2f}% n={block['n']}", flush=True)

    # Aggregate Löwenstein across stages
    low_keys = [k for k in summary["results"] if k.startswith("lowenstein/")]
    if low_keys:
        tp = sum(summary["results"][k]["agg_tp"] for k in low_keys)
        fp = sum(summary["results"][k]["agg_fp"] for k in low_keys)
        fn = sum(summary["results"][k]["agg_fn"] for k in low_keys)
        pp = sum(summary["results"][k]["pred_total"] for k in low_keys)
        gg = sum(summary["results"][k]["gt_total"] for k in low_keys)
        tt = sum(summary["results"][k]["img_total"] for k in low_keys)
        ds_iou, ds_dice = iou_dice(tp, fp, fn)
        wra = abs(pp - gg) / tt * 100.0 if tt else 0.0
        per_imgs = []
        for k in low_keys:
            per_imgs += summary["results"][k]["per_image"]
        summary["results"]["lowenstein_agg"] = {
            "n": len(per_imgs),
            "mean_iou": float(np.mean([r["iou"] for r in per_imgs])) if per_imgs else 0.0,
            "dataset_iou": float(ds_iou),
            "dataset_dice": float(ds_dice),
            "wra_err_pct": float(wra),
            "agg_tp": tp, "agg_fp": fp, "agg_fn": fn,
            "pred_total": pp, "gt_total": gg, "img_total": tt,
        }
        print(f"==> lowenstein_agg: ds_iou={ds_iou:.4f} WRA_err={wra:.2f}% n={len(per_imgs)}",
              flush=True)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] saved {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
