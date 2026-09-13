"""SAM ViT-B with PROPER methodology for OOD bright-field wound segmentation.

Fixes reviewer-flagged issues:
 - Tuned AMG thresholds (pred_iou=0.75, stability=0.85) — defaults too strict for OOD.
 - Multi-scale crops (crop_n_layers=1, downscale=2) — better recall on large wounds.
 - "union-bright" mask selection (replaces hacky single-best "brightest"):
     union of all AMG masks where mean_intensity_in_mask > image_median AND
     area > 1% of image; then keep_largest_connected_component.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

INTERNAL_ROOT = Path('./data/data/raw/dataset/test')
EXTERNAL_ROOT = Path('./data/data/external_lowenstein/woundhealing_v3')
STAGES = ('stage_1', 'stage_2', 'stage_3')
CKPT = './checkpoints/sam/sam_vit_b_01ec64.pth'


def list_internal():
    return [(p, INTERNAL_ROOT/'masks'/p.name)
            for p in sorted((INTERNAL_ROOT/'images').iterdir())
            if (INTERNAL_ROOT/'masks'/p.name).exists()]


def list_external(s):
    out = []
    for p in sorted((EXTERNAL_ROOT/'original'/s).iterdir()):
        gt = EXTERNAL_ROOT/'groundtruth'/s/f'{p.stem}.png'
        if gt.exists():
            out.append((p, gt))
    return out


def load_pair(p, g):
    img = np.asarray(Image.open(p).convert('L'), dtype=np.uint8)
    gt = np.asarray(Image.open(g).convert('L'), dtype=np.uint8)
    return img, (gt > 127).astype(np.uint8)


def keep_largest(m):
    if m.sum() == 0:
        return m
    lab, _ = ndi.label(m, structure=np.ones((3, 3), np.uint8))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return (lab == sizes.argmax()).astype(np.uint8) if sizes.size else m


def stats(p, g):
    pb, gb = p.astype(bool), g.astype(bool)
    return int((pb & gb).sum()), int((pb & ~gb).sum()), int((~pb & gb).sum())


def iou_dice(tp, fp, fn):
    iou = tp / (tp+fp+fn) if (tp+fp+fn) else 1.0
    dice = 2*tp / (2*tp+fp+fn) if (2*tp+fp+fn) else 1.0
    return iou, dice


def union_bright_mask(masks, gray, H, W, min_area_frac=0.01):
    """Union of AMG masks satisfying:
       mean_intensity_in_mask > image_median  AND  area > 1% of image.
    Then keep largest connected component."""
    if not masks:
        return np.zeros((H, W), dtype=np.uint8), 0
    img_median = float(np.median(gray))
    min_area = int(min_area_frac * H * W)
    union = np.zeros((H, W), dtype=bool)
    n_kept = 0
    for m in masks:
        seg = np.asarray(m['segmentation'], dtype=bool)
        if seg.sum() < min_area:
            continue
        mi = float(gray[seg].mean())
        if mi > img_median:
            union |= seg
            n_kept += 1
    if not union.any():
        return np.zeros((H, W), dtype=np.uint8), 0
    return keep_largest(union.astype(np.uint8)), n_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--pred-iou', type=float, default=0.75)
    ap.add_argument('--stability', type=float, default=0.85)
    ap.add_argument('--crop-n-layers', type=int, default=1)
    ap.add_argument('--crop-downscale', type=int, default=2)
    ap.add_argument('--min-area-frac', type=float, default=0.01)
    args = ap.parse_args()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[load] SAM ViT-B proper: pred_iou={args.pred_iou} stab={args.stability} '
          f'crop_n_layers={args.crop_n_layers} ds={args.crop_downscale} '
          f'min_area={args.min_area_frac}', flush=True)
    sam = sam_model_registry['vit_b'](checkpoint=CKPT).to(dev).eval()
    amg = SamAutomaticMaskGenerator(
        sam,
        pred_iou_thresh=args.pred_iou,
        stability_score_thresh=args.stability,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_downscale,
    )

    sets = [('internal', list_internal())]
    for s in STAGES:
        sets.append((f'lowenstein/{s}', list_external(s)))

    summary = {
        'model': 'sam_vit_b',
        'config': dict(pred_iou=args.pred_iou, stability=args.stability,
                       crop_n_layers=args.crop_n_layers,
                       crop_downscale=args.crop_downscale,
                       min_area_frac=args.min_area_frac,
                       strategy='union-bright + largest_cc'),
        'results': {},
    }

    # Load woundratio helper
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    for label, pairs in sets:
        agg_tp = agg_fp = agg_fn = 0
        per_image = []
        latencies = []
        # WRA-style accumulators: predicted wound fraction vs gt fraction
        wra_diffs = []
        for ip, gp in pairs:
            img, gt = load_pair(ip, gp)
            H, W = img.shape
            rgb = np.stack([img]*3, axis=-1)
            t0 = time.perf_counter()
            masks = amg.generate(rgb)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            wound, n_kept = union_bright_mask(masks, img, H, W, args.min_area_frac)
            tp, fp, fn = stats(wound, gt)
            i, d = iou_dice(tp, fp, fn)
            pred_frac = wound.sum() / float(H*W)
            gt_frac = gt.sum() / float(H*W)
            wra_diffs.append(abs(pred_frac - gt_frac))
            per_image.append({
                'image': ip.name, 'iou': i, 'dice': d,
                'n_amg_masks': len(masks), 'n_kept_in_union': n_kept,
                'pred_frac': pred_frac, 'gt_frac': gt_frac,
            })
            agg_tp += tp; agg_fp += fp; agg_fn += fn

        ds_iou, ds_dice = iou_dice(agg_tp, agg_fp, agg_fn)
        block = {
            'n': len(pairs),
            'mean_iou': float(np.mean([r['iou'] for r in per_image])),
            'mean_dice': float(np.mean([r['dice'] for r in per_image])),
            'dataset_iou': ds_iou,
            'dataset_dice': ds_dice,
            'wra_err_pct': float(np.mean(wra_diffs) * 100.0),
            'mean_latency_ms': float(np.mean(latencies)),
            'agg_tp': agg_tp, 'agg_fp': agg_fp, 'agg_fn': agg_fn,
            'per_image': per_image,
        }
        summary['results'][label] = block
        print(f'[{label}] n={block["n"]} mean_iou={block["mean_iou"]:.4f} '
              f'ds_iou={ds_iou:.4f} wra_err={block["wra_err_pct"]:.2f}% '
              f'lat={block["mean_latency_ms"]:.0f}ms', flush=True)

    # lowenstein_agg
    TP = FP = FN = 0
    all_pi = []
    all_wra = []
    for k in summary['results']:
        if not k.startswith('lowenstein/'):
            continue
        b = summary['results'][k]
        TP += b['agg_tp']; FP += b['agg_fp']; FN += b['agg_fn']
        all_pi += b['per_image']
        for r in b['per_image']:
            all_wra.append(abs(r['pred_frac'] - r['gt_frac']))
    if TP + FP + FN > 0:
        ds_iou, ds_dice = iou_dice(TP, FP, FN)
        summary['results']['lowenstein_agg'] = {
            'n': len(all_pi),
            'mean_iou': float(np.mean([r['iou'] for r in all_pi])) if all_pi else 0.0,
            'dataset_iou': ds_iou,
            'dataset_dice': ds_dice,
            'wra_err_pct': float(np.mean(all_wra) * 100.0) if all_wra else 0.0,
            'agg_tp': TP, 'agg_fp': FP, 'agg_fn': FN,
        }
        print(f'[lowenstein_agg] n={len(all_pi)} ds_iou={ds_iou:.4f} '
              f'mean_iou={summary["results"]["lowenstein_agg"]["mean_iou"]:.4f} '
              f'wra_err={summary["results"]["lowenstein_agg"]["wra_err_pct"]:.2f}%',
              flush=True)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[done] {args.out_json}', flush=True)


if __name__ == '__main__':
    main()
