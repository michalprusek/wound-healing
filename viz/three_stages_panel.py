"""3x7 qualitative panel: rows = (open / partially closed / nearly closed) wound,
cols = Orig, GT, TScratch, SAM 2 Hiera-L, Classical (ours), nnU-Net 250ep, MAnet.

One run per dataset (internal or lowenstein). No legend / no y-axis labels.
Equal vertical and horizontal gaps between panels.

Usage (on tulen/kajman):
    source /data/wounds/wounds/.venv/bin/activate
    python /data/wounds/wounds/wound-healing/viz/three_stages_panel.py \
        --dataset internal \
        --manet-ckpt /data/wounds/wounds/runs/<RUN>_arch_manet_s42/best.ckpt \
        --manet-cfg  /data/wounds/wounds/wound-healing/training/configs/arch_manet.yaml \
        --nnunet-dir /data/wounds/wounds/nnunet_results/Dataset001_Wounds/nnUNetTrainer_250epochs__nnUNetPlans__2d/fold_0 \
        --out-pdf /data/wounds/wounds/paper/img/three_stages_internal.pdf

    python /data/wounds/wounds/wound-healing/viz/three_stages_panel.py \
        --dataset lowenstein --manet-ckpt ... --nnunet-dir ... \
        --out-pdf /data/wounds/wounds/paper/img/three_stages_lowenstein.pdf
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parent.parent
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "classical"))
sys.path.insert(0, str(REPO_ROOT / "training"))

# evaluation utilities
from eval_external import (  # type: ignore
    STAGES, list_pairs, load_pair, predict_prob, load_model, stats, iou_dice,
)
from eval_tscratch import tscratch_predict  # type: ignore
import classical_pipeline as cls  # type: ignore
from train import Config  # type: ignore
from train_segformer_mit import MitSegmentor  # type: ignore
import torch.nn.functional as F

# Mirror the absolute cluster paths used in eval_sam2_proper.py so this script
# works regardless of cwd. Override via --internal-root / --lowenstein-root.
INTERNAL_ROOT_DEFAULT = Path("/data/wounds/wounds/data/raw/dataset/test")


def _list_internal(root: Path):
    pairs = []
    for ip in sorted((root / "images").iterdir()):
        gp = root / "masks" / ip.name
        if gp.exists():
            pairs.append((ip, gp))
    return pairs


# -------------------- selectors --------------------

def gather_pairs(dataset: str, internal_root: Path):
    """Return list[(label, img_path, gt_path)] for the chosen dataset."""
    out = []
    if dataset == "internal":
        for ip, gp in _list_internal(internal_root):
            out.append((ip.stem, ip, gp))
    elif dataset == "lowenstein":
        for stage in STAGES:
            for ip, gp in list_pairs(stage):
                out.append((f"{stage}/{ip.stem}", ip, gp))
    else:
        raise ValueError(dataset)
    return out


def pick_three_by_coverage(pairs):
    """Pick (open, partial, closed) representative images by GT mask fraction.

    open  -> high GT fraction (large wound area)
    mid   -> middle quantile
    closed-> low GT fraction (mostly merged)
    """
    rows = []
    for label, ip, gp in pairs:
        img = np.asarray(Image.open(ip).convert("L"), dtype=np.uint8)
        gt = np.asarray(Image.open(gp).convert("L"), dtype=np.uint8)
        gt_bin = (gt > 127).astype(np.uint8)
        rows.append(dict(label=label, ip=ip, gp=gp, img=img, gt=gt_bin,
                          frac=float(gt_bin.mean())))
    rows.sort(key=lambda r: r["frac"])
    n = len(rows)
    # avoid degenerate extremes -> use 0.15 / 0.50 / 0.85 quantiles
    closed = rows[max(0, int(0.15 * n))]
    partial = rows[int(0.50 * n)]
    opn = rows[min(n - 1, int(0.85 * n))]
    return [opn, partial, closed]  # render order = open at top


# -------------------- helpers --------------------

def keep_largest(m: np.ndarray) -> np.ndarray:
    if m.sum() == 0:
        return m
    lab, _ = ndi.label(m, structure=np.ones((3, 3), np.uint8))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return (lab == sizes.argmax()).astype(np.uint8) if sizes.size else m


def iou_of(pred: np.ndarray, gt: np.ndarray) -> float:
    pb, gb = pred.astype(bool), gt.astype(bool)
    tp = int((pb & gb).sum())
    fp = int((pb & ~gb).sum())
    fn = int((~pb & gb).sum())
    return tp / (tp + fp + fn) if (tp + fp + fn) else 1.0


# -------------------- predictors --------------------

def predict_tscratch(img_u8: np.ndarray) -> np.ndarray:
    """Paper-default TScratch parameters; returns wound mask (0/1)."""
    return tscratch_predict(img_u8).astype(np.uint8)


def predict_classical(img_u8: np.ndarray) -> np.ndarray:
    """Classical pipeline -> wound mask (0/1)."""
    cnts = cls.wound_contours(img=img_u8)
    if not cnts:
        return np.zeros_like(img_u8, dtype=np.uint8)
    mask = np.zeros(img_u8.shape, dtype=np.uint8)
    # contours are Nx2 arrays of (x, y) — drawContours needs Nx1x2 int32
    cnts_cv = [c.reshape(-1, 1, 2).astype(np.int32) for c in cnts]
    cv.drawContours(mask, cnts_cv, -1, color=1, thickness=-1)
    return mask


def predict_manet(img_u8: np.ndarray, model, cfg, device) -> np.ndarray:
    """MAnet (SMP) -> wound mask (0/1) at original resolution, threshold 0.5."""
    prob = predict_prob(model, img_u8, cfg.image_size, device, tta=False)
    return (prob > 0.5).astype(np.uint8)


def predict_sam2_batch(images: dict[str, np.ndarray],
                       ckpt: str, cfg_yaml: str,
                       pred_iou=0.75, stability=0.85,
                       crop_n_layers=1, crop_downscale=2,
                       min_area_frac=0.01):
    """Run SAM 2 Hiera-L AMG with PROPER strategy (union-bright + largest_cc)
    matching eval_sam2_proper.py. Returns dict {key: wound_mask 0/1}."""
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] SAM2 Hiera-L (proper) ckpt={ckpt}", flush=True)
    sam2 = build_sam2(cfg_yaml, ckpt, device=dev)
    amg = SAM2AutomaticMaskGenerator(
        sam2,
        pred_iou_thresh=pred_iou,
        stability_score_thresh=stability,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_downscale,
    )
    out = {}
    for key, img_u8 in images.items():
        rgb = np.stack([img_u8] * 3, axis=-1)
        t0 = time.perf_counter()
        masks = amg.generate(rgb)
        H, W = img_u8.shape
        img_median = float(np.median(img_u8))
        min_area = int(min_area_frac * H * W)
        union = np.zeros((H, W), dtype=bool)
        kept = 0
        for m in masks:
            seg = np.asarray(m["segmentation"], dtype=bool)
            if seg.sum() < min_area:
                continue
            if float(img_u8[seg].mean()) > img_median:
                union |= seg
                kept += 1
        wound = (keep_largest(union.astype(np.uint8))
                 if union.any() else np.zeros((H, W), dtype=np.uint8))
        print(f"  sam2 {key}: n_masks={len(masks)} kept={kept} "
              f"t={time.perf_counter() - t0:.1f}s", flush=True)
        out[key] = wound
    del amg, sam2
    torch.cuda.empty_cache()
    return out


def predict_nnunet_batch(images: dict[str, np.ndarray], nnunet_dir: Path,
                         tmp_root: Path) -> dict[str, np.ndarray]:
    """Run nnU-Net inference on given images dict. Returns dict {key: mask 0/1}."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    in_dir = tmp_root / "nn_in"; out_dir = tmp_root / "nn_out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = {}
    for key, img_u8 in images.items():
        s = key.replace(" ", "_").replace(",", "_").replace("/", "_")
        safe[s] = (key, img_u8.shape)
        Image.fromarray(img_u8, mode="L").save(in_dir / f"{s}_0000.png")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=True, device=dev,
        verbose=False, verbose_preprocessing=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(nnunet_dir), use_folds=(0,), checkpoint_name="checkpoint_final.pth"
    )
    predictor.predict_from_files(
        list_of_lists_or_source_folder=str(in_dir),
        output_folder_or_list_of_truncated_output_files=str(out_dir),
        save_probabilities=False, overwrite=True,
        num_processes_preprocessing=2, num_processes_segmentation_export=2,
    )
    out = {}
    for s, (key, shape) in safe.items():
        p = out_dir / f"{s}.png"
        if not p.exists():
            out[key] = np.zeros(shape, dtype=np.uint8)
            continue
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        if arr.shape != shape:
            arr = np.asarray(
                Image.fromarray((arr > 0).astype(np.uint8) * 255, mode="L")
                .resize((shape[1], shape[0]), Image.NEAREST), dtype=np.uint8)
        out[key] = (arr > 0).astype(np.uint8)
    return out


# -------------------- rendering --------------------

def render_panel(picks, preds, out_pdf: Path, model_titles: list[str]):
    """picks: list of 3 dicts with keys label/img/gt.
       preds: dict {row_label: {model_name: mask}}.
       Writes PDF + PNG side-by-side, equal row/col spacing, no ylabel."""
    n_rows = len(picks)
    col_titles = ["Orig.", "GT"] + model_titles
    n_cols = len(col_titles)

    # square cells -> wspace == hspace produces visually equal gaps
    cell_in = 1.7
    gap_frac = 0.05  # fraction of cell width
    fig_w = cell_in * n_cols + gap_frac * cell_in * (n_cols - 1)
    fig_h = cell_in * n_rows + gap_frac * cell_in * (n_rows - 1)
    # extra height for a top-row title strip
    fig_h += 0.30

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)
    plt.subplots_adjust(wspace=gap_frac, hspace=gap_frac,
                        left=0.005, right=0.995, top=0.93, bottom=0.005)

    red_cmap = ListedColormap([(0, 0, 0, 0), (1, 0, 0, 0.55)])
    green_cmap = ListedColormap([(0, 0, 0, 0), (0, 1, 0, 0.55)])

    for i, row in enumerate(picks):
        img = row["img"]; gt = row["gt"]
        for j in range(n_cols):
            ax = axes[i][j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=10, pad=4)
            if j == 0:
                continue  # plain original
            if j == 1:
                ax.imshow(gt, cmap=red_cmap, vmin=0, vmax=1,
                          interpolation="nearest")
                continue
            mname = col_titles[j]
            mask = preds[row["label"]].get(mname)
            if mask is None:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", color="red", fontsize=12)
            else:
                ax.imshow(mask, cmap=green_cmap, vmin=0, vmax=1,
                          interpolation="nearest")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[saved] {out_pdf}\n[saved] {out_png}", flush=True)


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["internal", "lowenstein"], required=True)
    ap.add_argument("--manet-ckpt", required=True)
    ap.add_argument("--manet-cfg", required=True)
    ap.add_argument("--nnunet-dir", required=True,
                    help="trained-model folder containing fold_0/checkpoint_final.pth")
    ap.add_argument("--sam2-ckpt",
                    default="/home/AUTHOR/checkpoints/sam2/sam2.1_hiera_large.pt")
    ap.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--internal-root", type=Path, default=INTERNAL_ROOT_DEFAULT)
    ap.add_argument("--out-pdf", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[scan] {args.dataset}", flush=True)
    all_pairs = gather_pairs(args.dataset, args.internal_root)
    print(f"[scan] {len(all_pairs)} pairs", flush=True)
    picks = pick_three_by_coverage(all_pairs)
    for tag, p in zip(("open", "partial", "closed"), picks):
        print(f"[pick] {tag}: {p['label']}  GT_frac={p['frac']:.3f}", flush=True)

    images_dict = {p["label"]: p["img"] for p in picks}

    # ---------- CPU predictors ----------
    preds_per_row: dict[str, dict[str, np.ndarray]] = {p["label"]: {} for p in picks}

    for p in picks:
        print(f"[tscratch] {p['label']}", flush=True)
        m = predict_tscratch(p["img"])
        preds_per_row[p["label"]]["TScratch"] = m
        print(f"    IoU={iou_of(m, p['gt']):.3f}", flush=True)

    for p in picks:
        print(f"[classical] {p['label']}", flush=True)
        m = predict_classical(p["img"])
        preds_per_row[p["label"]]["Classical (ours)"] = m
        print(f"    IoU={iou_of(m, p['gt']):.3f}", flush=True)

    # ---------- MAnet ----------
    print("[load] MAnet", flush=True)
    cfg = Config.from_yaml(args.manet_cfg)
    manet = load_model(cfg, args.manet_ckpt, device)
    for p in picks:
        m = predict_manet(p["img"], manet, cfg, device)
        preds_per_row[p["label"]]["MAnet"] = m
        print(f"[manet] {p['label']}: IoU={iou_of(m, p['gt']):.3f}", flush=True)
    del manet
    torch.cuda.empty_cache()

    # ---------- SAM 2 ----------
    sam2_out = predict_sam2_batch(images_dict, args.sam2_ckpt, args.sam2_cfg)
    for p in picks:
        m = sam2_out[p["label"]]
        preds_per_row[p["label"]]["SAM 2 Hiera-L"] = m
        print(f"[sam2] {p['label']}: IoU={iou_of(m, p['gt']):.3f}", flush=True)

    # ---------- nnU-Net ----------
    tmp = Path(tempfile.mkdtemp(prefix="nn_three_"))
    try:
        nn_out = predict_nnunet_batch(images_dict, Path(args.nnunet_dir), tmp)
        for p in picks:
            m = nn_out[p["label"]]
            preds_per_row[p["label"]]["nnU-Net 250ep"] = m
            print(f"[nnunet] {p['label']}: IoU={iou_of(m, p['gt']):.3f}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---------- render ----------
    model_titles = ["TScratch", "SAM 2 Hiera-L", "Classical (ours)",
                    "nnU-Net 250ep", "MAnet"]
    render_panel(picks, preds_per_row, Path(args.out_pdf), model_titles)


if __name__ == "__main__":
    main()
