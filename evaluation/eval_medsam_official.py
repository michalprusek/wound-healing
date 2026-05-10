"""MedSAM evaluation with OFFICIAL preprocessing per bowang-lab/MedSAM.

Differences from HF SamProcessor pipeline:
- Resize to 1024x1024 with skimage cubic + anti-aliasing (not SAM's longest-side+pad).
- Per-image min-max [0,1] normalization (NOT ImageNet mean/std).
- Bbox scaled into 1024x1024 space with scalar [W,H,W,H] factor.
- Logits resized back to native HxW with bilinear, threshold 0.5.

Strategies tested (full-image bbox dropped per audit):
- bbox_central_band: [0, 0.25*H, W, 0.75*H]
- bbox_brightspot: bbox of largest connected high-intensity region (Otsu top-percentile).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from skimage.transform import resize as sk_resize

INTERNAL_ROOT = Path('/data/AUTHOR/wounds/data/raw/dataset/test')
EXTERNAL_ROOT = Path('/data/AUTHOR/wounds/data/external_lowenstein/woundhealing_v3')
STAGES = ('stage_1', 'stage_2', 'stage_3')


def list_internal():
    return [(p, INTERNAL_ROOT / 'masks' / p.name)
            for p in sorted((INTERNAL_ROOT / 'images').iterdir())
            if (INTERNAL_ROOT / 'masks' / p.name).exists()]


def list_external(stage):
    out = []
    for p in sorted((EXTERNAL_ROOT / 'original' / stage).iterdir()):
        gt = EXTERNAL_ROOT / 'groundtruth' / stage / f'{p.stem}.png'
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
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return iou, dice


def get_brightspot_bbox(img):
    """Bbox around the largest connected high-intensity region (top-30%)."""
    thr = max(int(np.percentile(img, 70)), 100)
    fg = (img > thr).astype(np.uint8)
    fg = keep_largest(fg)
    if fg.sum() == 0:
        H, W = img.shape
        return [0, 0, W - 1, H - 1]
    ys, xs = np.where(fg > 0)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def get_central_band_bbox(H, W):
    """Central horizontal stripe where the scratch typically lives."""
    return [0, int(0.25 * H), W - 1, int(0.75 * H) - 1]


@torch.no_grad()
def medsam_official(model, dev, img_uint8, bbox_xyxy):
    """Official MedSAM forward pass per MedSAM_Inference.py."""
    H, W = img_uint8.shape
    img_3c = np.stack([img_uint8] * 3, axis=-1)

    # 1) skimage cubic + anti-aliasing resize -> 1024x1024
    img_1024 = sk_resize(img_3c, (1024, 1024), order=3,
                         preserve_range=True, anti_aliasing=True).astype(np.uint8)

    # 2) per-image min-max [0,1]
    img_1024 = (img_1024.astype(np.float32) - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), 1e-8, None)

    # 3) (1,3,1024,1024)
    x = torch.from_numpy(np.ascontiguousarray(img_1024)).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32)

    # 4) bbox scaled into 1024 space
    box_np = np.asarray(bbox_xyxy, dtype=np.float32)
    box_1024 = box_np / np.array([W, H, W, H], dtype=np.float32) * 1024.0
    box_t = torch.tensor(box_1024, device=dev).reshape(1, 1, 4)

    # 5) forward
    out = model(pixel_values=x, input_boxes=box_t, multimask_output=False)
    logits = out.pred_masks[:, 0]  # (1,1,256,256)
    prob = torch.sigmoid(logits)
    prob_full = F.interpolate(prob, size=(H, W), mode='bilinear', align_corners=False)
    return (prob_full[0, 0].cpu().numpy() > 0.5).astype(np.uint8)


STRATEGIES = ['bbox_central_band', 'bbox_brightspot']


def get_box(strat, img):
    H, W = img.shape
    if strat == 'bbox_central_band':
        return get_central_band_bbox(H, W)
    if strat == 'bbox_brightspot':
        return get_brightspot_bbox(img)
    raise ValueError(strat)


def wra_pair(pred, gt):
    """Wound-recovery accuracy: relative error in wound area fraction."""
    ap = float(pred.sum()) / pred.size
    ag = float(gt.sum()) / gt.size
    return abs(ap - ag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()

    from transformers import SamModel
    print('[load] wanglab/medsam-vit-base', flush=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SamModel.from_pretrained('wanglab/medsam-vit-base').to(dev).eval()

    sets = [('internal', list_internal())]
    for s in STAGES:
        sets.append((f'lowenstein/{s}', list_external(s)))

    summary = {
        'model': 'medsam-vit-base',
        'preprocessing': 'official_per_image_minmax_skimage1024',
        'strategies': STRATEGIES,
        'results': {},
    }

    for label, pairs in sets:
        summary['results'][label] = {}
        for strat in STRATEGIES:
            per_image = []
            agg_tp = agg_fp = agg_fn = 0
            wra_errs = []
            for ip, gp in pairs:
                img, gt = load_pair(ip, gp)
                box = get_box(strat, img)
                pred = medsam_official(model, dev, img, box)
                pred = keep_largest(pred)
                tp, fp, fn = stats(pred, gt)
                i, d = iou_dice(tp, fp, fn)
                w = wra_pair(pred, gt)
                per_image.append({'image': ip.name, 'iou': i, 'dice': d, 'wra_err': w})
                agg_tp += tp; agg_fp += fp; agg_fn += fn
                wra_errs.append(w)
            ds_iou, ds_dice = iou_dice(agg_tp, agg_fp, agg_fn)
            block = {
                'n': len(pairs),
                'mean_iou': float(np.mean([r['iou'] for r in per_image])) if per_image else 0.0,
                'dataset_iou': ds_iou,
                'dataset_dice': ds_dice,
                'mean_wra_err': float(np.mean(wra_errs)) if wra_errs else 0.0,
                'agg_tp': agg_tp, 'agg_fp': agg_fp, 'agg_fn': agg_fn,
                'per_image': per_image,
            }
            summary['results'][label][strat] = block
            print(f'[{label}/{strat}] n={block["n"]} mean_iou={block["mean_iou"]:.4f} '
                  f'ds_iou={block["dataset_iou"]:.4f} wra={block["mean_wra_err"]:.4f}',
                  flush=True)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[done] saved {args.out_json}', flush=True)


if __name__ == '__main__':
    main()
