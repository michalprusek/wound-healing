"""
Evaluate trained nnU-Net checkpoint on internal test + Löwenstein.
Uses nnU-Net's inference API to predict and computes IoU/Dice in same
format as eval_smp_to_json.py.

Usage:
  python scripts/eval_nnunet.py \
      --dataset 1 --config 2d --fold 0 \
      --trainer nnUNetTrainer_250epochs \
      --out-json results/nnunet_250ep.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_external import STAGES, list_pairs, load_pair  # type: ignore
from eval_otsu import list_internal_pairs, load_internal_pair  # type: ignore


def stats(pred, gt):
    pb = pred.astype(bool); gb = gt.astype(bool)
    return (int((pb & gb).sum()), int((pb & ~gb).sum()), int((~pb & gb).sum()))


def iou_dice(tp, fp, fn):
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return iou, dice


def stage_input(pairs, tmp_in: Path, loader_fn, prefix: str):
    """Copy/convert a list of (img,gt) pairs into nnU-Net inference layout.
    Returns mapping {file_stem: gt_array} for later eval."""
    tmp_in.mkdir(parents=True, exist_ok=True)
    name_to_gt = {}
    for img_path, gt_path in pairs:
        img_u8, gt = loader_fn(img_path, gt_path)
        # encode safely as filename
        stem = f"{prefix}_{img_path.stem}".replace(" ", "_").replace(",", "_")
        Image.fromarray(img_u8, mode="L").save(tmp_in / f"{stem}_0000.png")
        name_to_gt[stem] = gt
    return name_to_gt


def run_predict(in_dir, out_dir, dataset, cfg, fold, trainer, plans):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.file_path_utilities import get_output_folder
    import torch

    model_folder = get_output_folder(dataset, trainer, plans, cfg)
    print(f"[load] {model_folder}")
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda"), verbose=False,
        verbose_preprocessing=False, allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder, use_folds=(fold,), checkpoint_name="checkpoint_final.pth"
    )
    predictor.predict_from_files(
        list_of_lists_or_source_folder=str(in_dir),
        output_folder_or_list_of_truncated_output_files=str(out_dir),
        save_probabilities=False, overwrite=True,
        num_processes_preprocessing=4, num_processes_segmentation_export=4,
    )


def collect_preds(out_dir: Path, name_to_gt: dict):
    rows = []
    agg_tp = agg_fp = agg_fn = 0
    for stem, gt in name_to_gt.items():
        pred_path = out_dir / f"{stem}.png"
        if not pred_path.exists():
            print(f"WARN: missing prediction for {stem}")
            continue
        pred = np.asarray(Image.open(pred_path).convert("L"), dtype=np.uint8)
        pred_bin = (pred > 0).astype(np.uint8)
        if pred_bin.shape != gt.shape:
            from PIL import Image as PI
            pred_bin = np.asarray(
                PI.fromarray(pred_bin * 255, mode="L").resize(
                    (gt.shape[1], gt.shape[0]), PI.NEAREST), dtype=np.uint8) > 0
            pred_bin = pred_bin.astype(np.uint8)
        tp, fp, fn = stats(pred_bin, gt)
        i, d = iou_dice(tp, fp, fn)
        rows.append(dict(image=stem, iou=i, dice=d))
        agg_tp += tp; agg_fp += fp; agg_fn += fn
    return rows, agg_tp, agg_fp, agg_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=1, type=int)
    p.add_argument("--config", default="2d")
    p.add_argument("--fold", default=0, type=int)
    p.add_argument("--trainer", default="nnUNetTrainer_250epochs")
    p.add_argument("--plans", default="nnUNetPlans")
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    summary = dict(model=f"nnunet_{args.trainer}_fold{args.fold}", results={})
    tmp = Path(tempfile.mkdtemp(prefix="nnunet_eval_"))
    print(f"[tmp] {tmp}")
    try:
        # Internal
        in_int = tmp / "internal_in"; out_int = tmp / "internal_out"
        name_to_gt = stage_input(list_internal_pairs(), in_int,
                                 load_internal_pair, "internal")
        t0 = time.perf_counter()
        run_predict(in_int, out_int, args.dataset, args.config, args.fold,
                    args.trainer, args.plans)
        lat_int = (time.perf_counter() - t0) * 1000.0 / max(1, len(name_to_gt))
        rows, tp, fp, fn = collect_preds(out_int, name_to_gt)
        ds_iou, ds_dice = iou_dice(tp, fp, fn)
        summary["results"]["internal"] = dict(
            n=len(rows),
            mean_iou=float(np.mean([r["iou"] for r in rows])),
            mean_dice=float(np.mean([r["dice"] for r in rows])),
            dataset_iou=ds_iou, dataset_dice=ds_dice,
            mean_latency_ms=lat_int,
            agg_tp=tp, agg_fp=fp, agg_fn=fn, per_image=rows,
        )
        print(f"[internal] ds_iou={ds_iou:.4f} mean_iou="
              f"{summary['results']['internal']['mean_iou']:.4f} "
              f"latency={lat_int:.0f}ms")

        # Löwenstein per-stage
        for stage in STAGES:
            sd_in = tmp / f"low_{stage}_in"; sd_out = tmp / f"low_{stage}_out"
            n2g = stage_input(list_pairs(stage), sd_in, load_pair, stage)
            t0 = time.perf_counter()
            run_predict(sd_in, sd_out, args.dataset, args.config, args.fold,
                        args.trainer, args.plans)
            lat = (time.perf_counter() - t0) * 1000.0 / max(1, len(n2g))
            rows, tp, fp, fn = collect_preds(sd_out, n2g)
            ds_iou, ds_dice = iou_dice(tp, fp, fn)
            summary["results"][f"lowenstein/{stage}"] = dict(
                n=len(rows),
                mean_iou=float(np.mean([r["iou"] for r in rows])),
                mean_dice=float(np.mean([r["dice"] for r in rows])),
                dataset_iou=ds_iou, dataset_dice=ds_dice,
                mean_latency_ms=lat,
                agg_tp=tp, agg_fp=fp, agg_fn=fn, per_image=rows,
            )
            print(f"[lowenstein/{stage}] ds_iou={ds_iou:.4f}")

        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[saved] {args.out_json}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
