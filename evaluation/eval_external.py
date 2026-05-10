"""
Evaluate a wound-healing checkpoint on the external Löwenstein dataset
(woundhealing_v3 from Kaggle: katjalwenstein/woundhealing).

The external set has a different layout (stages, not train/test), 30 images,
RGB inputs, vertical-wound orientations — none of which the model saw in
training. This is the canonical generalization test used in the original-paper paper.

Reports:
  - Dataset IoU  (micro: one IoU over all pixels aggregated across images)
  - Dataset Dice (micro)
  - Per-image IoU / Dice (mean across images)
  - Per-stage breakdown (stage_1 / stage_2 / stage_3)

The "Dataset IoU" number is what the original-paper paper (Table 2) reports — use it
for apples-to-apples comparison with their 89.50 %.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Config, SegmentationModel, tta_predict  # type: ignore


EXTERNAL_ROOT = Path("/data/AUTHOR/wounds/data/external_lowenstein/woundhealing_v3")
STAGES = ("stage_1", "stage_2", "stage_3")


def list_pairs(stage: str) -> list[tuple[Path, Path]]:
    """Pair image files with ground-truth masks by stem name. Masks are always
    .png; images are a mix of .jpg and .png."""
    img_dir = EXTERNAL_ROOT / "original" / stage
    gt_dir = EXTERNAL_ROOT / "groundtruth" / stage
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        stem = img_path.stem
        gt_path = gt_dir / f"{stem}.png"
        if not gt_path.exists():
            raise FileNotFoundError(f"no mask for {img_path}")
        pairs.append((img_path, gt_path))
    return pairs


def load_pair(img_path: Path, gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
    gt = np.asarray(Image.open(gt_path).convert("L"), dtype=np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)
    return img, gt_bin


def preprocess(img_u8: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    pil = Image.fromarray(img_u8, mode="L").resize((size, size), Image.BILINEAR)
    arr = np.array(pil, dtype=np.uint8, copy=True)
    x = torch.from_numpy(arr).to(device).unsqueeze(0).unsqueeze(0)
    return x.float()  # SegmentationModel.forward will do /255-0.5


def predict_prob(model, img_u8: np.ndarray, size: int,
                 device: torch.device, tta: bool) -> np.ndarray:
    x = preprocess(img_u8, size, device)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        if tta:
            logits = tta_predict(model, x)
        else:
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
    prob = torch.sigmoid(logits)[0, 0]
    prob_up = torch.nn.functional.interpolate(
        prob[None, None], size=img_u8.shape,
        mode="bilinear", align_corners=False,
    )[0, 0]
    return prob_up.cpu().numpy()


def best_threshold_iou(prob: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    best = (0.0, 0.5)
    for thr in np.arange(0.05, 0.99, 0.05):
        pb = prob > thr
        gb = gt > 0
        inter = (pb & gb).sum()
        union = (pb | gb).sum()
        iou = inter / union if union else 0.0
        if iou > best[0]:
            best = (float(iou), float(thr))
    return best


def stats(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    tn = int(np.logical_and(~pred_b, ~gt_b).sum())
    return tp, fp, fn, tn


def iou_dice(tp: int, fp: int, fn: int) -> tuple[float, float]:
    denom_iou = tp + fp + fn
    iou = tp / denom_iou if denom_iou else 1.0
    denom_dice = 2 * tp + fp + fn
    dice = 2 * tp / denom_dice if denom_dice else 1.0
    return iou, dice


def load_model(cfg: Config, ckpt_path: str, device: torch.device):
    m = SegmentationModel(
        arch=cfg.arch, encoder=cfg.encoder, lr=cfg.lr, T_max=1,
        loss_name=cfg.loss,
        aux_params=cfg.aux_params, aux_weight=cfg.aux_weight,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(state["state_dict"])
    return m.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--name", required=True, help="label for output row")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="binarization threshold")
    p.add_argument("--best-thresh-per-image", action="store_true",
                   help="oracle: use the best threshold per image (ceiling)")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda")
    model = load_model(cfg, args.ckpt, device)

    per_stage: dict[str, dict[str, float]] = {}
    all_pair_ious: list[float] = []
    all_pair_dices: list[float] = []
    agg_tp = agg_fp = agg_fn = 0

    for stage in STAGES:
        pair_ious: list[float] = []
        pair_dices: list[float] = []
        stage_tp = stage_fp = stage_fn = 0
        for img_path, gt_path in list_pairs(stage):
            img, gt = load_pair(img_path, gt_path)
            prob = predict_prob(model, img, cfg.image_size, device, args.tta)
            if args.best_thresh_per_image:
                _, thr = best_threshold_iou(prob, gt)
            else:
                thr = args.threshold
            pred = (prob > thr).astype(np.uint8)
            tp, fp, fn, _ = stats(pred, gt)
            i, d = iou_dice(tp, fp, fn)
            pair_ious.append(i)
            pair_dices.append(d)
            stage_tp += tp; stage_fp += fp; stage_fn += fn
        ds_iou, ds_dice = iou_dice(stage_tp, stage_fp, stage_fn)
        per_stage[stage] = dict(
            n=len(pair_ious),
            mean_iou=float(np.mean(pair_ious)),
            mean_dice=float(np.mean(pair_dices)),
            dataset_iou=ds_iou,
            dataset_dice=ds_dice,
        )
        all_pair_ious.extend(pair_ious)
        all_pair_dices.extend(pair_dices)
        agg_tp += stage_tp; agg_fp += stage_fp; agg_fn += stage_fn

    total_ds_iou, total_ds_dice = iou_dice(agg_tp, agg_fp, agg_fn)
    tta_tag = "+tta" if args.tta else ""
    thr_tag = "[oracle-thr]" if args.best_thresh_per_image else f"[thr={args.threshold}]"
    print(f"\n=== {args.name}{tta_tag} {thr_tag} — external Löwenstein (30 images) ===")
    for stage in STAGES:
        s = per_stage[stage]
        print(f"  {stage}: n={s['n']} mean_iou={s['mean_iou']:.4f} "
              f"mean_dice={s['mean_dice']:.4f} "
              f"ds_iou={s['dataset_iou']:.4f} ds_dice={s['dataset_dice']:.4f}")
    print(f"  OVERALL: n={len(all_pair_ious)} "
          f"mean_iou={np.mean(all_pair_ious):.4f} "
          f"mean_dice={np.mean(all_pair_dices):.4f} "
          f"dataset_iou={total_ds_iou:.4f} dataset_dice={total_ds_dice:.4f}")


if __name__ == "__main__":
    main()
