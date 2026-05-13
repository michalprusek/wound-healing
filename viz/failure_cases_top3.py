"""
Failure-case figure for top 3 architectures (mit_b5, manet, nnU-Net).
Picks N worst Löwenstein images by mean IoU across the 3 models, renders
side-by-side panels.

Usage:
  python scripts/failure_cases_top3.py \
      --manet-ckpt runs/<>_arch_manet_s42/best.ckpt \
      --mitb5-ckpt runs/<>_segformer_mitb5/best.ckpt \
      --nnunet-dir nnunet_results/Dataset001_Wounds/.../fold_0 \
      --n 5 --out paper/figures/failure_cases_top3.pdf
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Config, SegmentationModel  # type: ignore
from train_segformer_mit import MitSegmentor  # type: ignore
from eval_external import (  # type: ignore
    STAGES, list_pairs, load_pair, predict_prob, load_model, stats, iou_dice,
)


def predict_smp(model, cfg, img_u8, device):
    return predict_prob(model, img_u8, cfg.image_size, device, tta=False)


def predict_mit(model, img_u8, device, size=256):
    pil = Image.fromarray(img_u8, mode="L").resize((size, size), Image.BILINEAR)
    a = np.array(pil, dtype=np.uint8)
    x = torch.from_numpy(a).to(device).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        logits = model.forward(x)
    p = torch.sigmoid(logits)[0, 0]
    return F.interpolate(p[None, None], size=img_u8.shape, mode="bilinear",
                         align_corners=False)[0, 0].cpu().numpy()


def predict_nnunet_batch(images_dict, nnunet_dir, tmp_root, device):
    """Run nnU-Net inference on dict {stem: img_u8}. Returns dict {stem: pred_mask}."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    in_dir = tmp_root / "nn_in"; out_dir = tmp_root / "nn_out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stems = {}
    for stem, img_u8 in images_dict.items():
        safe = stem.replace(" ", "_").replace(",", "_")
        safe_stems[safe] = stem
        Image.fromarray(img_u8, mode="L").save(in_dir / f"{safe}_0000.png")

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=True, device=device, verbose=False,
        verbose_preprocessing=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(nnunet_dir), use_folds=(0,), checkpoint_name="checkpoint_final.pth"
    )
    predictor.predict_from_files(
        list_of_lists_or_source_folder=str(in_dir),
        output_folder_or_list_of_truncated_output_files=str(out_dir),
        save_probabilities=False, overwrite=True,
        num_processes_preprocessing=4, num_processes_segmentation_export=4,
    )
    out = {}
    for safe, stem in safe_stems.items():
        p = out_dir / f"{safe}.png"
        if p.exists():
            arr = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
            out[stem] = (arr > 0).astype(np.uint8)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manet-ckpt", required=True)
    p.add_argument("--manet-cfg", default="/data/wounds/wounds/configs/arch_manet.yaml")
    p.add_argument("--mitb5-ckpt", required=True)
    p.add_argument("--nnunet-dir", required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[load] manet")
    cfg_manet = Config.from_yaml(args.manet_cfg)
    manet = load_model(cfg_manet, args.manet_ckpt, device).eval()

    print("[load] mit_b5")
    mit = MitSegmentor.load_from_checkpoint(args.mitb5_ckpt, map_location=device).to(device).eval()

    # First: gather all Löwenstein images
    all_pairs = []
    for stage in STAGES:
        for ip, gp in list_pairs(stage):
            all_pairs.append((stage, ip, gp))
    print(f"[data] {len(all_pairs)} Löwenstein images")

    # Predict with manet + mit_b5
    records = []
    for stage, ip, gp in all_pairs:
        img, gt = load_pair(ip, gp)
        prob_manet = predict_smp(manet, cfg_manet, img, device)
        prob_mit = predict_mit(mit, img, device)
        pred_manet = (prob_manet > 0.5).astype(np.uint8)
        pred_mit = (prob_mit > 0.5).astype(np.uint8)
        tp, fp, fn, _ = stats(pred_manet, gt)
        iou_manet, _ = iou_dice(tp, fp, fn)
        tp, fp, fn, _ = stats(pred_mit, gt)
        iou_mit, _ = iou_dice(tp, fp, fn)
        records.append(dict(stage=stage, stem=ip.stem, img=img, gt=gt,
                            pred_manet=pred_manet, pred_mit=pred_mit,
                            iou_manet=iou_manet, iou_mit=iou_mit))

    # Predict with nnU-Net (batch)
    print("[load] nnU-Net + predict batch")
    tmp = Path(tempfile.mkdtemp(prefix="nn_failure_"))
    try:
        images_dict = {f"{r['stage']}_{r['stem']}": r["img"] for r in records}
        nn_preds = predict_nnunet_batch(images_dict, Path(args.nnunet_dir), tmp, device)
        for r in records:
            key = f"{r['stage']}_{r['stem']}".replace(" ", "_").replace(",", "_")
            pred = nn_preds.get(key)
            if pred is None or pred.shape != r["gt"].shape:
                # resize if shape mismatch
                if pred is not None:
                    pi = Image.fromarray(pred * 255, mode="L").resize(
                        (r["gt"].shape[1], r["gt"].shape[0]), Image.NEAREST)
                    pred = (np.asarray(pi) > 0).astype(np.uint8)
                else:
                    pred = np.zeros_like(r["gt"], dtype=np.uint8)
            tp, fp, fn, _ = stats(pred, r["gt"])
            iou_nn, _ = iou_dice(tp, fp, fn)
            r["pred_nn"] = pred
            r["iou_nn"] = iou_nn
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Pick N worst by mean IoU
    for r in records:
        r["mean_iou"] = (r["iou_manet"] + r["iou_mit"] + r["iou_nn"]) / 3.0
    records.sort(key=lambda r: r["mean_iou"])
    pick = records[: args.n]

    # Render: rows = images, cols = original | GT | manet | mit_b5 | nnU-Net
    fig, axes = plt.subplots(args.n, 5, figsize=(15, 3 * args.n), squeeze=False)
    col_titles = ["Original", "Ground truth", "MAnet", "MiT-B5", "nnU-Net"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=10)

    for i, r in enumerate(pick):
        axes[i, 0].imshow(r["img"], cmap="gray")
        axes[i, 0].set_ylabel(f"{r['stage']}/{r['stem']}\n"
                              f"MAnet:{r['iou_manet']:.2f} | "
                              f"MiT:{r['iou_mit']:.2f} | "
                              f"nnU:{r['iou_nn']:.2f}", fontsize=7)
        # GT overlay: red
        gt_o = np.zeros((*r["gt"].shape, 4), dtype=np.float32)
        gt_o[..., 0] = 1.0; gt_o[..., 3] = r["gt"] * 0.55
        axes[i, 1].imshow(r["img"], cmap="gray")
        axes[i, 1].imshow(gt_o)
        # Pred overlays per model: green
        for col, key in [(2, "pred_manet"), (3, "pred_mit"), (4, "pred_nn")]:
            po = np.zeros((*r[key].shape, 4), dtype=np.float32)
            po[..., 1] = 1.0; po[..., 3] = r[key] * 0.55
            axes[i, col].imshow(r["img"], cmap="gray")
            axes[i, col].imshow(po)
        for c in range(5):
            axes[i, c].set_xticks([]); axes[i, c].set_yticks([])

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(str(Path(args.out).with_suffix(".png")), dpi=150, bbox_inches="tight")
    print(f"[saved] {args.out}")
    print("Worst cases (mean IoU):")
    for r in pick:
        print(f"  {r['stage']}/{r['stem']}: mean={r['mean_iou']:.3f} "
              f"(MAnet {r['iou_manet']:.2f} / MiT {r['iou_mit']:.2f} / nnU {r['iou_nn']:.2f})")


if __name__ == "__main__":
    main()
