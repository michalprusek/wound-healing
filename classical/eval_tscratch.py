"""
TScratch baseline (Geback, Schulz, Koumoutsakos & Detmar, BioTechniques 46:265-274, 2009).

Faithful Python re-implementation of the TScratch algorithm following the
original MATLAB source in github.com/cselab/TScratch (file:
scratcharea_one.m + scratch_applythresh.m + scratch_fillholes.m +
scratch_removeislands.m + scratch_getdefaultopts.m).

Algorithm (per scratcharea_one.m):
  1. Mirror-extended fast discrete curvelet transform (FDCT) of the image.
  2. Sum magnitudes of curvelet coefficients on the mid-frequency levels
     (default `levels = 4:5` in MATLAB 1-based indexing, i.e. the two finest
     scales just below the very fine details).
  3. Optional background-intensity compensation by rescaling using the
     coarsest curvelet level (skipped here — see CAVEAT below).
  4. Morphological opening with a disk structuring element of radius 7.
  5. Threshold at `default_thresh * max(opentotmag)` with `default_thresh=0.25`
     (TScratch supports auto-threshold finding from derivative of the
     occupied-area-vs-threshold curve; here we use the fixed default which
     is the TScratch fall-back when auto-finding fails — keeping the script
     parameter-free and reproducible).
  6. Erosion (disk size 2), hole filling (holes < 7% of largest open area),
     island removal (islands < 7% of largest closed area).
  7. The thresholded mask is the *cell* mask (`sheets`); wound = NOT(sheets).

CAVEATS
-------
* CurveLab's FDCT is **not** open-source for non-academic use, so this
  re-implementation uses the **Uniform Discrete Curvelet Transform (UDCT)**
  from the `curvelets` PyPI package (Nguyen & Chauris 2010), a license-clean
  alternative. UDCT has different decimation/orientation but produces a
  similar multi-scale, multi-directional decomposition with magnitude
  envelopes that capture the same "where is texture" cue TScratch exploits.
* Background-compensation in step 3 was skipped (the MATLAB code uses the
  coarsest level FFT2 in a way specific to FDCT's `C{1}{1}` storage).
* Auto-threshold finder is not re-implemented — fixed default 0.25 is used.
* Images are resized to 512x512 for the curvelet decomposition (UDCT requires
  power-of-2 dims; original 2048x2048 would be 16x slower and most TScratch
  experiments were run on <=1024 px images). Output mask is upsampled back
  to original size with nearest-neighbour interpolation.
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import morphology

INTERNAL_ROOT = Path("/data/AUTHOR/wounds/data/raw/dataset/test")
EXTERNAL_ROOT = Path("/data/AUTHOR/wounds/data/external_lowenstein/woundhealing_v3")
STAGES = ("stage_1", "stage_2", "stage_3")


def list_lowenstein_pairs(stage):
    img_dir = EXTERNAL_ROOT / "original" / stage
    gt_dir = EXTERNAL_ROOT / "groundtruth" / stage
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        gt_path = gt_dir / f"{img_path.stem}.png"
        if not gt_path.exists():
            raise FileNotFoundError(f"no mask for {img_path}")
        pairs.append((img_path, gt_path))
    return pairs


def load_lowenstein_pair(img_path, gt_path):
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
    gt = np.asarray(Image.open(gt_path).convert("L"), dtype=np.uint8)
    return img, (gt > 127).astype(np.uint8)


# -------------------- dataset loaders --------------------

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


# -------------------- TScratch core --------------------

# Cache UDCT transform per shape (initialization is expensive)
_UDCT_CACHE = {}

def get_udct(shape, num_scales=5, wedges_per_direction=6):
    from curvelets.numpy import UDCT
    key = (shape, num_scales, wedges_per_direction)
    if key not in _UDCT_CACHE:
        _UDCT_CACHE[key] = UDCT(
            shape=shape,
            num_scales=num_scales,
            wedges_per_direction=wedges_per_direction,
        )
    return _UDCT_CACHE[key]


def curvelet_magnitude(img_f64, levels, num_scales=5, wedges_per_direction=6):
    """Compute summed magnitude over selected curvelet levels.

    UDCT level structure (after .forward(img)):
      c[0] = lowpass: nested list of complex arrays (1 sublist of 1 array)
      c[1..nscales-1] = bandpass: list of [direction_block(s)] of [wedge arrays]
                       (2 direction blocks (real,imag mirror) of W wedge arrays)

    Magnitude of a level = sum over wedges of |coef| upsampled to image size.
    """
    H, W = img_f64.shape
    udct = get_udct((H, W), num_scales, wedges_per_direction)
    coeffs = udct.forward(img_f64)
    mag = np.zeros((H, W), dtype=np.float64)
    for lvl in levels:
        if lvl < 0 or lvl >= len(coeffs):
            continue
        level_data = coeffs[lvl]
        # Sum |coef| over all direction blocks and wedges
        for dir_block in level_data:
            wedges = dir_block if isinstance(dir_block, list) else [dir_block]
            for wedge in wedges:
                if not hasattr(wedge, 'shape'):
                    continue
                a = np.abs(wedge).astype(np.float64)
                # Upsample to image size by nearest-neighbour (fast & avoids
                # ringing); this mirrors TScratch's per-pixel magnitude map.
                if a.shape != (H, W):
                    # Use scipy zoom (order=0) for nearest-neighbour
                    fy = H / a.shape[0]
                    fx = W / a.shape[1]
                    a = ndi.zoom(a, (fy, fx), order=0)
                mag[:a.shape[0], :a.shape[1]] += a[:H, :W]
    return mag


def fill_small_holes(sheets, fraction=0.07):
    """Fill open-area connected components smaller than `fraction` of the
    largest open-area component (matches scratch_fillholes.m)."""
    out = sheets.copy()
    lbl, n = ndi.label(~sheets.astype(bool))
    if n == 0:
        return out
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0  # background of label 0 (which is the foreground here)
    if sizes.size == 0 or sizes.max() == 0:
        return out
    max_sz = sizes.max()
    for k in range(1, n + 1):
        if sizes[k] < fraction * max_sz:
            out[lbl == k] = 1
    return out


def remove_small_islands(sheets, fraction=0.07):
    """Remove closed-area CC smaller than `fraction` of the largest closed
    component (matches scratch_removeislands.m)."""
    out = sheets.copy()
    lbl, n = ndi.label(sheets.astype(bool))
    if n == 0:
        return out
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    if sizes.size == 0 or sizes.max() == 0:
        return out
    max_sz = sizes.max()
    for k in range(1, n + 1):
        if sizes[k] < fraction * max_sz:
            out[lbl == k] = 0
    return out


def tscratch_predict(img_u8,
                     proc_size=512,
                     levels=(2, 3),
                     num_scales=5,
                     wedges_per_direction=6,
                     disk_radius=7,
                     thresh=0.25,
                     erode_size=2,
                     hole_area=0.07,
                     island_area=0.07):
    """Return wound mask (uint8, 1=wound, 0=cells) at original image size."""
    H0, W0 = img_u8.shape

    # Resize for curvelet processing (UDCT requires fixed shape; processing
    # 2048x2048 with num_scales=5/wedges=6 is very slow).
    pil = Image.fromarray(img_u8, mode="L").resize((proc_size, proc_size), Image.BILINEAR)
    img_f = np.array(pil, dtype=np.float64)

    # Step 1-2: curvelet magnitude on selected mid-frequency levels.
    totmag = curvelet_magnitude(img_f, levels=levels,
                                num_scales=num_scales,
                                wedges_per_direction=wedges_per_direction)

    # Step 4: morphological opening with disk(disk_radius).
    selem = morphology.disk(disk_radius)
    opentotmag = morphology.opening(totmag, selem)

    # Step 5: threshold at 25% of max.
    M = opentotmag.max()
    if M <= 0:
        sheets = np.zeros_like(opentotmag, dtype=bool)
    else:
        sheets = opentotmag > (thresh * M)

    # Step 6a: erosion with margin padding (matches scratch_applythresh.m).
    if erode_size > 0:
        es = erode_size
        padded = np.ones((sheets.shape[0] + 2*es, sheets.shape[1] + 2*es), dtype=bool)
        padded[es:-es, es:-es] = sheets
        eroded = ndi.binary_erosion(padded, iterations=es)
        sheets = eroded[es:-es, es:-es]

    # Step 6b/c: fill small holes, remove small islands (TScratch order).
    sheets = fill_small_holes(sheets.astype(np.uint8), hole_area)
    sheets = remove_small_islands(sheets.astype(np.uint8), island_area)

    # Wound = NOT(cells). Convert to uint8.
    wound = (1 - sheets.astype(np.uint8)).astype(np.uint8)

    # Resize back to original.
    wound_pil = Image.fromarray((wound * 255).astype(np.uint8), mode="L").resize(
        (W0, H0), Image.NEAREST
    )
    return (np.array(wound_pil) > 127).astype(np.uint8)


# -------------------- evaluation --------------------

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
    p.add_argument("--proc-size", type=int, default=512)
    p.add_argument("--levels", type=str, default="2,3")
    p.add_argument("--num-scales", type=int, default=5)
    p.add_argument("--wedges", type=int, default=6)
    p.add_argument("--disk", type=int, default=7)
    p.add_argument("--thresh", type=float, default=0.25)
    p.add_argument("--erode", type=int, default=2)
    p.add_argument("--hole-area", type=float, default=0.07)
    p.add_argument("--island-area", type=float, default=0.07)
    p.add_argument("--out-json", default="/data/AUTHOR/wounds/results/tscratch.json")
    p.add_argument("--limit", type=int, default=0, help="limit images per set (debug)")
    args = p.parse_args()

    levels = tuple(int(x) for x in args.levels.split(","))

    sets = []
    if args.eval in ("internal", "both"):
        sets.append(("internal", list_internal_pairs(), load_internal_pair))
    if args.eval in ("lowenstein", "both"):
        for stage in STAGES:
            sets.append((f"lowenstein/{stage}",
                         list_lowenstein_pairs(stage), load_lowenstein_pair))

    summary = dict(
        method="tscratch",
        algorithm=("Faithful re-implementation of TScratch (Geback et al. 2009) "
                   "using the open Uniform Discrete Curvelet Transform (UDCT, "
                   "Nguyen & Chauris 2010) in place of CurveLab FDCT. Pipeline: "
                   "UDCT magnitude over mid-frequency levels -> disk opening -> "
                   "fixed threshold (0.25 of max) -> erode -> hole-fill -> "
                   "island-remove -> wound = NOT(cells)."),
        parameters=dict(
            proc_size=args.proc_size, levels=list(levels),
            num_scales=args.num_scales, wedges_per_direction=args.wedges,
            disk_radius=args.disk, threshold=args.thresh, erode_size=args.erode,
            hole_area=args.hole_area, island_area=args.island_area,
        ),
        results={},
    )
    for label, pairs, loader in sets:
        if args.limit:
            pairs = pairs[:args.limit]
        per_image = []
        agg_tp = agg_fp = agg_fn = 0
        agg_pred_pix = 0
        agg_gt_pix = 0
        agg_total = 0
        latencies = []
        for img_path, gt_path in pairs:
            img_u8, gt = loader(img_path, gt_path)
            t0 = time.perf_counter()
            wound = tscratch_predict(
                img_u8,
                proc_size=args.proc_size,
                levels=levels,
                num_scales=args.num_scales,
                wedges_per_direction=args.wedges,
                disk_radius=args.disk,
                thresh=args.thresh,
                erode_size=args.erode,
                hole_area=args.hole_area,
                island_area=args.island_area,
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
            tp, fp, fn = stats(wound, gt)
            i, d = iou_dice(tp, fp, fn)
            per_image.append(dict(image=img_path.name, iou=i, dice=d,
                                  pred_px=int(wound.sum()), gt_px=int(gt.sum())))
            agg_tp += tp; agg_fp += fp; agg_fn += fn
            agg_pred_pix += int(wound.sum())
            agg_gt_pix += int(gt.sum())
            agg_total += int(wound.size)
            print(f"  [{label}] {img_path.name}: IoU={i:.3f} ({latencies[-1]:.0f}ms)",
                  flush=True)

        ds_iou, ds_dice = iou_dice(agg_tp, agg_fp, agg_fn)
        wra_err = abs(agg_pred_pix - agg_gt_pix) / agg_total * 100.0 if agg_total else 0.0
        block = dict(
            n=len(pairs),
            mean_iou=float(np.mean([r["iou"] for r in per_image])) if per_image else 0.0,
            mean_dice=float(np.mean([r["dice"] for r in per_image])) if per_image else 0.0,
            dataset_iou=float(ds_iou),
            dataset_dice=float(ds_dice),
            wra_error_pct=float(wra_err),
            agg_tp=agg_tp, agg_fp=agg_fp, agg_fn=agg_fn,
            agg_pred_pixels=agg_pred_pix, agg_gt_pixels=agg_gt_pix,
            agg_total_pixels=agg_total,
            mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            per_image=per_image,
        )
        summary["results"][label] = block
        print(f"==> {label}: dataset_iou={ds_iou:.4f} mean_iou={block['mean_iou']:.4f} "
              f"WRA_err={wra_err:.2f}% n={len(pairs)}", flush=True)

    # Aggregate Löwenstein across stages
    low_keys = [k for k in summary["results"] if k.startswith("lowenstein/")]
    if low_keys:
        tp = sum(summary["results"][k]["agg_tp"] for k in low_keys)
        fp = sum(summary["results"][k]["agg_fp"] for k in low_keys)
        fn = sum(summary["results"][k]["agg_fn"] for k in low_keys)
        pp = sum(summary["results"][k]["agg_pred_pixels"] for k in low_keys)
        gp = sum(summary["results"][k]["agg_gt_pixels"] for k in low_keys)
        tot = sum(summary["results"][k]["agg_total_pixels"] for k in low_keys)
        n = sum(summary["results"][k]["n"] for k in low_keys)
        ds_iou, ds_dice = iou_dice(tp, fp, fn)
        wra = abs(pp - gp) / tot * 100.0 if tot else 0.0
        all_per_img = []
        for k in low_keys:
            all_per_img += summary["results"][k]["per_image"]
        summary["results"]["lowenstein/all"] = dict(
            n=n,
            mean_iou=float(np.mean([r["iou"] for r in all_per_img])) if all_per_img else 0.0,
            dataset_iou=float(ds_iou),
            dataset_dice=float(ds_dice),
            wra_error_pct=float(wra),
            agg_tp=tp, agg_fp=fp, agg_fn=fn,
            agg_pred_pixels=pp, agg_gt_pixels=gp, agg_total_pixels=tot,
        )
        print(f"==> lowenstein/all: dataset_iou={ds_iou:.4f} WRA_err={wra:.2f}% n={n}",
              flush=True)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {args.out_json}")


if __name__ == "__main__":
    main()
