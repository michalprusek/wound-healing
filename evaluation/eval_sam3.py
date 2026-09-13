"""SAM 3 (facebook/sam3) Promptable Concept Segmentation evaluator.

Strategies:
  cell_complement   : text='cell' (or 'cells'), union all instances, complement -> largest CC.
  cells_brightest   : text='cell', AND-postprocess: pick cell mask whose mean brightness is HIGHEST,
                      use complement of that single cell as the wound (consistency note: AMG-style).
  wound_direct      : text='wound' (or 'scratch'), pick brightest predicted instance (consistent with SAM/SAM2).
  wound_largest     : text='wound', pick largest predicted instance.

We sweep thresholds on the post-process step:
  default: threshold=0.5, mask_threshold=0.5
  relaxed: threshold=0.3, mask_threshold=0.0   (more permissive)

Usage:
  python eval_sam3.py --out-json results/sam3_relaxed.json --mode relaxed
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

# HF token discovery: HF_HOME may point elsewhere, but the user keeps the token at ~/.cache/...
_TOK = Path.home() / '.cache/huggingface/token'
if 'HF_TOKEN' not in os.environ and _TOK.exists():
    os.environ['HF_TOKEN'] = _TOK.read_text().strip()

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

INTERNAL_ROOT = Path('./data/data/raw/dataset/test')
EXTERNAL_ROOT = Path('./data/data/external_lowenstein/woundhealing_v3')
STAGES = ('stage_1', 'stage_2', 'stage_3')
HF_ID = 'facebook/sam3'

# --- IO helpers ---
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
    gt  = np.asarray(Image.open(g).convert('L'), dtype=np.uint8)
    rgb = np.stack([img]*3, axis=-1)
    return img, rgb, (gt > 127).astype(np.uint8)

def keep_largest(m):
    if m.sum() == 0: return m
    lab, _ = ndi.label(m, structure=np.ones((3,3), np.uint8))
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return (lab == sizes.argmax()).astype(np.uint8) if sizes.size else m

def stats(p, g):
    pb, gb = p.astype(bool), g.astype(bool)
    return int((pb & gb).sum()), int((pb & ~gb).sum()), int((~pb & gb).sum())

def iou_dice(tp, fp, fn):
    iou  = tp/(tp+fp+fn) if (tp+fp+fn) else 1.0
    dice = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 1.0
    return iou, dice

# --- SAM 3 helpers ---
def sam3_masks(model, processor, rgb_np, text, device, threshold, mask_threshold):
    """Run SAM3 with a text concept prompt. Returns list of bool HxW masks."""
    pil = Image.fromarray(rgb_np)
    inputs = processor(images=pil, text=text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = inputs.get('original_sizes')
    if hasattr(target_sizes, 'tolist'):
        target_sizes = target_sizes.tolist()
    res = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes,
    )[0]
    masks = res.get('masks', None)
    out = []
    if masks is not None and len(masks) > 0:
        for m in masks:
            arr = m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
            if arr.dtype != bool:
                arr = arr > 0.5
            out.append(arr.astype(bool))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--mode', choices=['default', 'relaxed'], default='relaxed')
    ap.add_argument('--cell-text', default='cell')
    ap.add_argument('--wound-text', default='wound')
    args = ap.parse_args()

    if args.mode == 'relaxed':
        threshold, mask_threshold = 0.3, 0.0
    else:
        threshold, mask_threshold = 0.5, 0.5

    from transformers import Sam3Processor, Sam3Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[load] {HF_ID} (mode={args.mode}, thr={threshold}, mask_thr={mask_threshold})', flush=True)
    model = Sam3Model.from_pretrained(HF_ID).to(device).eval()
    processor = Sam3Processor.from_pretrained(HF_ID)
    if device == 'cuda':
        try: model = model.half()
        except Exception: pass

    sets = [('internal', list_internal())]
    for s in STAGES:
        sets.append((f'lowenstein/{s}', list_external(s)))

    STRATS = ['cell_complement', 'cells_brightest']
    summary = {
        'model': 'facebook/sam3',
        'mode': args.mode,
        'post_process': {'threshold': threshold, 'mask_threshold': mask_threshold},
        'cell_text': args.cell_text,
        'wound_text': args.wound_text,
        'results': {},
    }

    for label, pairs in sets:
        agg = {s: {'tp': 0, 'fp': 0, 'fn': 0, 'per_image': []} for s in STRATS}
        for ip, gp in pairs:
            img, rgb, gt = load_pair(ip, gp)
            H, W = img.shape

            # --- text='cell' query ---
            try:
                cell_masks = sam3_masks(model, processor, rgb, args.cell_text,
                                        device, threshold, mask_threshold)
            except Exception as e:
                print(f'[err cell] {ip.name}: {e}', flush=True)
                cell_masks = []
            n_cells = len(cell_masks)

            fg = np.zeros(img.shape, dtype=bool)
            for s in cell_masks: fg |= s
            w_cc = keep_largest((~fg).astype(np.uint8))

            # cells_brightest: pick the cell instance with HIGHEST mean brightness (because in
            # bright-field the wound zone is bright; pick the brightest "cell" candidate as a
            # cell-region proxy; complement is wound). NOTE: this mirrors SAM/SAM2 'brightest'.
            if cell_masks:
                best, best_score = None, -1
                for s in cell_masks:
                    if s.sum() == 0: continue
                    score = float(img[s].mean())
                    if score > best_score:
                        best_score, best = score, s
                if best is not None:
                    w_cb = keep_largest((~best).astype(np.uint8))
                else:
                    w_cb = np.zeros_like(img, dtype=np.uint8)
            else:
                w_cb = np.zeros_like(img, dtype=np.uint8)

            # NOTE: 'wound'/'scratch' text queries return zero masks on bright-field wound
            # images (verified empirically); we skip them and only evaluate cell-based strategies.

            for sn, w in [('cell_complement', w_cc),
                          ('cells_brightest', w_cb)]:
                tp, fp, fn = stats(w, gt)
                i, d = iou_dice(tp, fp, fn)
                agg[sn]['per_image'].append({'image': ip.name, 'iou': i, 'dice': d, 'n_cells': n_cells})
                agg[sn]['tp'] += tp; agg[sn]['fp'] += fp; agg[sn]['fn'] += fn

        summary['results'][label] = {}
        for sn in STRATS:
            a = agg[sn]
            ds_iou, ds_dice = iou_dice(a['tp'], a['fp'], a['fn'])
            block = {
                'n': len(pairs),
                'mean_iou': float(np.mean([r['iou'] for r in a['per_image']])) if a['per_image'] else 0.0,
                'dataset_iou': ds_iou, 'dataset_dice': ds_dice,
                'agg_tp': a['tp'], 'agg_fp': a['fp'], 'agg_fn': a['fn'],
                'per_image': a['per_image'],
            }
            summary['results'][label][sn] = block
            print(f'[{label}/{sn}] mean={block["mean_iou"]:.4f} ds={block["dataset_iou"]:.4f}', flush=True)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[done] {args.out_json}', flush=True)

if __name__ == '__main__':
    main()
