"""Qualitative visualization panel: 6 prompt-free foundation models on 5 representative
wound-healing test images.

Layout: rows=images, cols=[Original, GT(red overlay), Cellpose, SAM ViT-B, SAM 2,
SAM 3, MedSAM, µSAM]. Predictions shown as semitransparent green overlay on
grayscale original. Each panel titled with model name + per-image IoU.

SAM 3 uses text='scratch' + biggest-mask (matches paper Tab. 1, scratch_biggest
strategy from eval_sam3_promptsweep.py). The earlier text='cell' + cells_brightest
methodology was reviewer-flagged as semantically wrong (segments cells, not wound).

Models are loaded sequentially to keep GPU memory in check; predictions are cached
into a dict before plotting.

Usage:
    source /data/AUTHOR/wounds/.venv/bin/activate
    python /data/AUTHOR/wounds/scripts/viz_foundation_qualitative.py \
        --out-pdf /data/AUTHOR/wounds/viz/foundation_qualitative.pdf \
        --out-png /data/AUTHOR/wounds/viz/foundation_qualitative.png
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# HF token discovery for SAM 3 (it's a gated repo)
_TOK = Path.home() / ".cache/huggingface/token"
if "HF_TOKEN" not in os.environ and _TOK.exists():
    os.environ["HF_TOKEN"] = _TOK.read_text().strip()

INTERNAL_ROOT = Path("/data/AUTHOR/wounds/data/raw/dataset/test")
EXTERNAL_ROOT = Path("/data/AUTHOR/wounds/data/external_lowenstein/woundhealing_v3")

SAM_VITB_CKPT = "/home/AUTHOR/checkpoints/sam/sam_vit_b_01ec64.pth"
SAM2_CKPT = "/home/AUTHOR/checkpoints/sam2/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
MICROSAM_CKPT = "/data/AUTHOR/wounds/checkpoints/vit_b_lm.pt"


# --- shared utils ---
def keep_largest(m: np.ndarray) -> np.ndarray:
    if m.sum() == 0:
        return m
    lab, _ = ndi.label(m, structure=np.ones((3, 3), np.uint8))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return (lab == sizes.argmax()).astype(np.uint8) if sizes.size else m


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pb, gb = pred.astype(bool), gt.astype(bool)
    tp = int((pb & gb).sum())
    fp = int((pb & ~gb).sum())
    fn = int((~pb & gb).sum())
    return tp / (tp + fp + fn) if (tp + fp + fn) else 1.0


def load_pair(img_path: Path, gt_path: Path):
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
    gt = np.asarray(Image.open(gt_path).convert("L"), dtype=np.uint8)
    return img, (gt > 127).astype(np.uint8)


# --- per-model predictors. Each takes (img u8 grayscale HxW, name) and returns wound mask u8(0/255). ---

def predict_cellpose(images, ext_flags):
    """Cellpose 4 cpsam, flow_threshold=0.7, cellprob_threshold=0.0. cells->complement->keep_largest."""
    from cellpose import models as cp_models
    print("[load] Cellpose cpsam", flush=True)
    use_gpu = torch.cuda.is_available()
    model = cp_models.CellposeModel(model_type="cpsam", gpu=use_gpu)
    out = []
    for img, _is_ext in zip(images, ext_flags):
        t0 = time.perf_counter()
        # cellpose 4: best cfg per CLAUDE.md / paper
        try:
            res = model.eval(img, diameter=None,
                             flow_threshold=0.7, cellprob_threshold=0.0)
        except TypeError:
            # fallback if API differs
            res = model.eval(img, diameter=None)
        masks_inst = res[0]
        fg = (masks_inst > 0).astype(np.uint8)
        wound = keep_largest((1 - fg).astype(np.uint8))
        print(f"  cellpose img n_cells={int(masks_inst.max())} "
              f"t={time.perf_counter() - t0:.1f}s", flush=True)
        out.append((wound * 255).astype(np.uint8))
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return out


def _amg_brightest(img, segs):
    if not segs:
        return np.zeros_like(img, dtype=np.uint8)
    best, best_score = None, -1.0
    for s in segs:
        if s.sum() == 0:
            continue
        score = float(img[s].mean())
        if score > best_score:
            best_score, best = score, s
    if best is None:
        return np.zeros_like(img, dtype=np.uint8)
    return keep_largest(best.astype(np.uint8))


def predict_sam_vitb(images, ext_flags):
    """SAM ViT-B AMG default thresholds, brightest strategy."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[load] SAM ViT-B", flush=True)
    sam = sam_model_registry["vit_b"](checkpoint=SAM_VITB_CKPT)
    sam.to(dev).eval()
    amg = SamAutomaticMaskGenerator(sam)
    out = []
    for img, _is_ext in zip(images, ext_flags):
        rgb = np.stack([img] * 3, axis=-1)
        t0 = time.perf_counter()
        cell_masks = amg.generate(rgb)
        segs = [np.array(m["segmentation"], dtype=bool, copy=True) for m in cell_masks]
        w = _amg_brightest(img, segs)
        print(f"  sam_vitb img n={len(segs)} t={time.perf_counter() - t0:.1f}s",
              flush=True)
        out.append((w * 255).astype(np.uint8))
    del amg, sam
    torch.cuda.empty_cache()
    gc.collect()
    return out


def predict_sam2(images, ext_flags):
    """SAM 2 Hiera-L AMG default, brightest strategy."""
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[load] SAM 2 hiera_l", flush=True)
    sam2 = build_sam2(SAM2_CFG, SAM2_CKPT, device=dev)
    amg = SAM2AutomaticMaskGenerator(sam2)
    out = []
    for img, _is_ext in zip(images, ext_flags):
        rgb = np.stack([img] * 3, axis=-1)
        t0 = time.perf_counter()
        cell_masks = amg.generate(rgb)
        segs = [np.array(m["segmentation"], dtype=bool, copy=True) for m in cell_masks]
        w = _amg_brightest(img, segs)
        print(f"  sam2 img n={len(segs)} t={time.perf_counter() - t0:.1f}s",
              flush=True)
        out.append((w * 255).astype(np.uint8))
    del amg, sam2
    torch.cuda.empty_cache()
    gc.collect()
    return out


def predict_sam3(images, ext_flags):
    """SAM 3 with text='scratch', biggest-mask strategy: among instances returned for
    prompt 'scratch', pick the one with largest area, then keep_largest CC.
    This matches the corrected paper Tab. 1 methodology (scratch_biggest).
    """
    from transformers import Sam3Processor, Sam3Model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[load] SAM 3 (facebook/sam3) — text='scratch', biggest-mask", flush=True)
    model = Sam3Model.from_pretrained("facebook/sam3").to(dev).eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    if dev == "cuda":
        try:
            model = model.half()
        except Exception:
            pass
    threshold, mask_threshold = 0.3, 0.0  # matches eval_sam3_promptsweep.py defaults

    def _run(rgb):
        pil = Image.fromarray(rgb)
        inputs = processor(images=pil, text="scratch", return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = inputs.get("original_sizes")
        if hasattr(target_sizes, "tolist"):
            target_sizes = target_sizes.tolist()
        res = processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=mask_threshold,
            target_sizes=target_sizes)[0]
        masks = res.get("masks", None)
        out_l = []
        if masks is not None and len(masks) > 0:
            for m in masks:
                arr = m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
                if arr.dtype != bool:
                    arr = arr > 0.5
                out_l.append(arr.astype(bool))
        return out_l

    out = []
    for img, _is_ext in zip(images, ext_flags):
        rgb = np.stack([img] * 3, axis=-1)
        t0 = time.perf_counter()
        try:
            scratch_masks = _run(rgb)
        except Exception as e:
            print(f"  sam3 ERROR: {e}", flush=True)
            scratch_masks = []
        # biggest_mask: pick largest-area instance, then keep_largest CC
        if scratch_masks:
            best, best_area = None, 0
            for s in scratch_masks:
                a = int(s.sum())
                if a > best_area:
                    best_area, best = a, s
            if best is None or best_area == 0:
                w = np.zeros_like(img, dtype=np.uint8)
            else:
                w = keep_largest(best.astype(np.uint8))
        else:
            w = np.zeros_like(img, dtype=np.uint8)
        print(f"  sam3 img n_scratch_masks={len(scratch_masks)} "
              f"t={time.perf_counter() - t0:.1f}s", flush=True)
        out.append((w * 255).astype(np.uint8))
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()
    return out


def predict_medsam(images, ext_flags):
    """MedSAM (wanglab/medsam-vit-base): bbox=full-image internal, bbox=brightspot external."""
    from transformers import SamProcessor, SamModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[load] MedSAM (wanglab/medsam-vit-base)", flush=True)
    proc = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
    # Force the safetensors revision (a0a0e4bf...) — the default snapshot ships
    # only pytorch_model.bin which transformers >= 5 refuses to load with torch < 2.6.
    try:
        model = SamModel.from_pretrained(
            "wanglab/medsam-vit-base", use_safetensors=True
        ).to(dev).eval()
    except Exception as e1:
        print(f"  medsam: safetensors load failed ({e1}); trying explicit revision",
              flush=True)
        model = SamModel.from_pretrained(
            "wanglab/medsam-vit-base",
            revision="a0a0e4bf38278213a0df876d2378f5736731a15e",
            use_safetensors=True,
        ).to(dev).eval()

    def _brightspot_bbox(img):
        thr = max(int(np.percentile(img, 70)), 100)
        fg = (img > thr).astype(np.uint8)
        fg = keep_largest(fg)
        if fg.sum() == 0:
            H, W = img.shape
            return [0, 0, W - 1, H - 1]
        ys, xs = np.where(fg > 0)
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    def _predict_bbox(rgb, box):
        inputs = proc(rgb, input_boxes=[[box]], return_tensors="pt").to(dev)
        with torch.inference_mode():
            o = model(**inputs, multimask_output=False)
        masks = proc.image_processor.post_process_masks(
            o.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        return masks[0][0, 0].numpy().astype(np.uint8)

    out = []
    for img, is_ext in zip(images, ext_flags):
        rgb = np.stack([img] * 3, axis=-1)
        H, W = img.shape
        if is_ext:
            box = _brightspot_bbox(img)
        else:
            box = [0, 0, W - 1, H - 1]
        t0 = time.perf_counter()
        try:
            m = _predict_bbox(rgb, box)
        except Exception as e:
            print(f"  medsam ERROR: {e}", flush=True)
            m = np.zeros_like(img, dtype=np.uint8)
        w = keep_largest(m)
        print(f"  medsam img bbox_ext={is_ext} t={time.perf_counter() - t0:.1f}s",
              flush=True)
        out.append((w * 255).astype(np.uint8))
    del model, proc
    torch.cuda.empty_cache()
    gc.collect()
    return out


def predict_microsam(images, ext_flags):
    """µSAM via vanilla SAM with vit_b_lm checkpoint, AMG relaxed (0.5 / 0.7),
    cells_complement strategy."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[load] µSAM (vit_b_lm)", flush=True)
    sam = sam_model_registry["vit_b"](checkpoint=None)
    sd = torch.load(MICROSAM_CKPT, map_location="cpu", weights_only=False)
    sam.load_state_dict(sd)
    sam.to(dev).eval()
    amg = SamAutomaticMaskGenerator(
        sam, pred_iou_thresh=0.5, stability_score_thresh=0.7,
        points_per_side=32)
    out = []
    for img, _is_ext in zip(images, ext_flags):
        rgb = np.stack([img] * 3, axis=-1)
        t0 = time.perf_counter()
        cell_masks = amg.generate(rgb)
        segs = [np.array(m["segmentation"], dtype=bool, copy=True) for m in cell_masks]
        fg = np.zeros(img.shape, dtype=bool)
        for s in segs:
            fg |= s
        w = keep_largest((~fg).astype(np.uint8))
        print(f"  microsam img n={len(segs)} t={time.perf_counter() - t0:.1f}s",
              flush=True)
        out.append((w * 255).astype(np.uint8))
    del amg, sam
    torch.cuda.empty_cache()
    gc.collect()
    return out


# --- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-pdf", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--skip-sam3", action="store_true",
                    help="skip SAM3 (e.g. if HF gated access issue)")
    args = ap.parse_args()

    # 5 chosen images: 4 internal (different cell lines + experiments) + 1 Lowenstein.
    chosen = [
        # (label, img_path, gt_path, is_external)
        ("Internal: 5min/5%FBS/Ex01_002",
         INTERNAL_ROOT / "images" / "5min, low glucose, 5% FBS, Ex 01_002.jpg",
         INTERNAL_ROOT / "masks" / "5min, low glucose, 5% FBS, Ex 01_002.jpg",
         False),
        ("Internal: MiaPaca-2_E01_D5_040",
         INTERNAL_ROOT / "images" / "MiaPaca-2_E01_D5_040.jpg",
         INTERNAL_ROOT / "masks" / "MiaPaca-2_E01_D5_040.jpg",
         False),
        ("Internal: MRC-5_E02_C6_027",
         INTERNAL_ROOT / "images" / "MRC-5_E02_C6_027.jpg",
         INTERNAL_ROOT / "masks" / "MRC-5_E02_C6_027.jpg",
         False),
        ("Internal: UFH-001_C2_P23_044",
         INTERNAL_ROOT / "images" / "UFH-001_Control_2_Pozice_23_044.jpg",
         INTERNAL_ROOT / "masks" / "UFH-001_Control_2_Pozice_23_044.jpg",
         False),
        ("Lowenstein/stage_2: 16h-8-50-5x",
         EXTERNAL_ROOT / "original" / "stage_2" / "16h-8-50-5x.jpg",
         EXTERNAL_ROOT / "groundtruth" / "stage_2" / "16h-8-50-5x.png",
         True),
    ]

    # Verify existence
    for label, ip, gp, _ in chosen:
        if not ip.exists():
            raise FileNotFoundError(f"image missing: {ip}")
        if not gp.exists():
            raise FileNotFoundError(f"gt missing: {gp}")
        print(f"[ok] {label}", flush=True)

    # Load images
    images = []
    gts = []
    ext_flags = []
    labels = []
    for label, ip, gp, is_ext in chosen:
        img, gt = load_pair(ip, gp)
        print(f"[loaded] {label}: img shape={img.shape} gt frac={gt.mean():.3f}",
              flush=True)
        images.append(img)
        gts.append(gt)
        ext_flags.append(is_ext)
        labels.append(label)

    # Run each model. Cache predictions: model_name -> [pred_u8 per image].
    preds = {}
    failed = {}

    def _safe_run(name, fn):
        try:
            t0 = time.perf_counter()
            preds[name] = fn(images, ext_flags)
            print(f"[done] {name} in {time.perf_counter() - t0:.1f}s", flush=True)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[FAIL] {name}: {e}\n{tb}", flush=True)
            failed[name] = str(e)
            preds[name] = [None] * len(images)

    _safe_run("Cellpose", predict_cellpose)
    _safe_run("SAM ViT-B", predict_sam_vitb)
    _safe_run("SAM 2", predict_sam2)
    if not args.skip_sam3:
        _safe_run("SAM 3", predict_sam3)
    else:
        preds["SAM 3"] = [None] * len(images)
        failed["SAM 3"] = "skipped"
    _safe_run("MedSAM", predict_medsam)
    _safe_run("µSAM", predict_microsam)

    # Compute per-image IoUs
    model_names = ["Cellpose", "SAM ViT-B", "SAM 2", "SAM 3", "MedSAM", "µSAM"]
    iou_table = {m: [] for m in model_names}
    for i in range(len(images)):
        for m in model_names:
            p = preds[m][i] if preds.get(m) else None
            if p is None:
                iou_table[m].append(None)
            else:
                iou_table[m].append(iou((p > 127).astype(np.uint8), gts[i]))

    # Print numeric table
    print("\n=== Per-image IoU ===", flush=True)
    print("image\t" + "\t".join(model_names), flush=True)
    for i, lbl in enumerate(labels):
        row = [lbl]
        for m in model_names:
            v = iou_table[m][i]
            row.append("FAIL" if v is None else f"{v:.3f}")
        print("\t".join(row), flush=True)

    # --- render panel ---
    n_rows = len(images)
    col_titles = ["Original", "Ground truth"] + model_names
    n_cols = len(col_titles)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows),
                             squeeze=False)

    # Build colormaps for overlays
    red_cmap = ListedColormap([(0, 0, 0, 0), (1, 0, 0, 0.5)])
    green_cmap = ListedColormap([(0, 0, 0, 0), (0, 1, 0, 0.5)])

    for i in range(n_rows):
        img = images[i]
        gt = gts[i]
        for j, ctitle in enumerate(col_titles):
            ax = axes[i][j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                title = ctitle if i == 0 else ""
                if i == 0:
                    ax.set_title(title, fontsize=9)
                # Y-axis label = image short name
                short = labels[i].replace("Internal: ", "").replace(
                    "Lowenstein/stage_2: ", "Löwenstein s2: ")
                ax.set_ylabel(short, fontsize=7, rotation=90,
                              labelpad=4, va="center")
            elif j == 1:
                ax.imshow(gt, cmap=red_cmap, vmin=0, vmax=1, interpolation="nearest")
                if i == 0:
                    ax.set_title("Ground truth", fontsize=9)
            else:
                model_name = ctitle
                p = preds[model_name][i] if preds.get(model_name) else None
                v = iou_table[model_name][i]
                if p is None or v is None:
                    # FAILED: black panel
                    ax.imshow(np.zeros_like(img), cmap="gray", vmin=0, vmax=255)
                    ax.text(0.5, 0.5, "FAILED",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=14, color="red", weight="bold")
                    if i == 0:
                        ax.set_title(f"{model_name}\n(FAILED)", fontsize=8)
                else:
                    pmask = (p > 127).astype(np.uint8)
                    ax.imshow(pmask, cmap=green_cmap, vmin=0, vmax=1,
                              interpolation="nearest")
                    title_str = (f"{model_name}\nIoU={v:.3f}" if i == 0
                                 else f"IoU={v:.3f}")
                    ax.set_title(title_str, fontsize=8)

    plt.tight_layout()

    # Save
    out_pdf = Path(args.out_pdf)
    out_png = Path(args.out_png)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[saved] {out_pdf}", flush=True)
    print(f"[saved] {out_png}", flush=True)

    # Save numeric data alongside
    summary = {
        "images": labels,
        "iou_per_model": {m: iou_table[m] for m in model_names},
        "failed_models": failed,
    }
    import json
    json_path = out_pdf.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {json_path}", flush=True)


if __name__ == "__main__":
    main()
