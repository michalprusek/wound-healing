"""
Evaluate a trained SMP checkpoint on internal + Löwenstein and dump results in
the same JSON format as eval_sam.py / eval_otsu.py (for the aggregator).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Config, SegmentationModel  # type: ignore
from eval_external import (  # type: ignore
    STAGES, list_pairs, load_pair, predict_prob, load_model, stats, iou_dice,
)
from eval_otsu import list_internal_pairs, load_internal_pair  # type: ignore


def eval_set(model, cfg, pairs, loader_fn, device):
    per_image = []
    agg_tp = agg_fp = agg_fn = 0
    latencies = []
    for img_path, gt_path in pairs:
        img_u8, gt = loader_fn(img_path, gt_path)
        t0 = time.perf_counter()
        prob = predict_prob(model, img_u8, cfg.image_size, device, tta=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        pred = (prob > 0.5).astype(np.uint8)
        tp, fp, fn, _ = stats(pred, gt)
        i, d = iou_dice(tp, fp, fn)
        per_image.append(dict(image=img_path.name, iou=i, dice=d))
        agg_tp += tp; agg_fp += fp; agg_fn += fn
    ds_iou, ds_dice = iou_dice(agg_tp, agg_fp, agg_fn)
    return dict(
        n=len(pairs),
        mean_iou=float(np.mean([r["iou"] for r in per_image])),
        mean_dice=float(np.mean([r["dice"] for r in per_image])),
        dataset_iou=ds_iou,
        dataset_dice=ds_dice,
        mean_latency_ms=float(np.mean(latencies)),
        agg_tp=agg_tp, agg_fp=agg_fp, agg_fn=agg_fn,
        per_image=per_image,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.ckpt, device).eval()

    summary = dict(method=args.name, results={})

    print(f"[{args.name}] internal")
    res = eval_set(model, cfg, list_internal_pairs(), load_internal_pair, device)
    summary["results"]["internal"] = res
    print(f"  ds_iou={res['dataset_iou']:.4f} mean_iou={res['mean_iou']:.4f} "
          f"latency={res['mean_latency_ms']:.0f}ms")

    for stage in STAGES:
        res = eval_set(model, cfg, list_pairs(stage), load_pair, device)
        summary["results"][f"lowenstein/{stage}"] = res
        print(f"[{args.name}] lowenstein/{stage}: "
              f"ds_iou={res['dataset_iou']:.4f} mean_iou={res['mean_iou']:.4f}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
