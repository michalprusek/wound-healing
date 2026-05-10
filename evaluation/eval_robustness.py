"""Synthetic perturbation robustness on Löwenstein: 3 models × 7 perturbations.

Saves results/robustness.json + paper/img/robustness_curves.pdf.
"""
import sys, os, json, glob, shutil, tempfile
sys.path.insert(0, './data/scripts')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')


def main():
    from pathlib import Path
    import numpy as np
    import cv2
    import torch
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from train import Config
    from train_segformer_mit import MitSegmentor
    from eval_external import STAGES, list_pairs, load_pair, load_model, stats, iou_dice
    from failure_cases_top3 import predict_smp, predict_mit
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device('cuda')
    rng = np.random.default_rng(42)

    manet_ckpt = sorted(glob.glob('./data/runs/*arch_manet_s42*/best.ckpt'))[-1]
    mitb5_ckpt = './data/runs/20260508_203609_segformer_mitb5/best.ckpt'
    nnunet_dir = './data/nnunet_results/Dataset001_Wounds/nnUNetTrainer_250epochs__nnUNetPlans__2d'

    cfg = Config.from_yaml('./data/configs/arch_manet.yaml')
    print('[load] manet'); manet = load_model(cfg, manet_ckpt, device).eval()
    print('[load] mit_b5'); mit = MitSegmentor.load_from_checkpoint(mitb5_ckpt, map_location=device).to(device).eval()
    print('[load] nnUNet predictor')
    nn_predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=True, device=device, verbose=False,
        verbose_preprocessing=False, allow_tqdm=False)
    nn_predictor.initialize_from_trained_model_folder(
        nnunet_dir, use_folds=(0,), checkpoint_name='checkpoint_final.pth')

    # Perturbations
    def perturb(img_u8, name):
        if name == 'baseline':
            return img_u8.copy()
        kind, val = name.rsplit('_', 1)
        v = float(val)
        if kind == 'blur':
            k = int(2 * round(v) + 1)
            return cv2.GaussianBlur(img_u8, (k, k), v)
        if kind == 'bright':
            return np.clip(img_u8.astype(np.int16) + int(v), 0, 255).astype(np.uint8)
        if kind == 'noise':
            return np.clip(img_u8.astype(np.float32) + rng.normal(0, v, img_u8.shape), 0, 255).astype(np.uint8)
        raise ValueError(name)

    perturbations = ['baseline',
                     'blur_1', 'blur_2', 'blur_4',
                     'bright_-50', 'bright_50',
                     'noise_5', 'noise_10', 'noise_20']

    all_pairs = [(stage, ip, gp) for stage in STAGES for ip, gp in list_pairs(stage)]
    print(f'{len(all_pairs)} Löwenstein images')

    def run_nnunet_batch(images_dict):
        tmp = Path(tempfile.mkdtemp(prefix='nn_'))
        in_dir = tmp / 'in'; out_dir = tmp / 'out'
        in_dir.mkdir(); out_dir.mkdir()
        safe_stems = {}
        for stem, img in images_dict.items():
            safe = stem.replace(' ', '_').replace(',', '_')
            safe_stems[safe] = stem
            Image.fromarray(img, mode='L').save(in_dir / f'{safe}_0000.png')
        nn_predictor.predict_from_files(
            list_of_lists_or_source_folder=str(in_dir),
            output_folder_or_list_of_truncated_output_files=str(out_dir),
            save_probabilities=False, overwrite=True,
            num_processes_preprocessing=1, num_processes_segmentation_export=1)
        out = {}
        for safe, stem in safe_stems.items():
            p = out_dir / f'{safe}.png'
            if p.exists():
                arr = np.asarray(Image.open(p).convert('L'), dtype=np.uint8)
                out[stem] = (arr > 0).astype(np.uint8)
        shutil.rmtree(tmp, ignore_errors=True)
        return out

    results = {}
    for pname in perturbations:
        print(f'=== {pname}')
        per = {'manet': [], 'mit': [], 'nn': []}
        # Load + perturb all
        records = []
        for stage, ip, gp in all_pairs:
            img, gt = load_pair(ip, gp)
            img_p = perturb(img, pname)
            records.append((stage, ip.stem, img_p, gt, img))

        # SMP predictions
        for stage, stem, img_p, gt, _ in records:
            pm = predict_smp(manet, cfg, img_p, device); m = (pm > 0.5).astype(np.uint8)
            tp, fp, fn, _ = stats(m, gt); iou_m, _ = iou_dice(tp, fp, fn)
            pi = predict_mit(mit, img_p, device); mi = (pi > 0.5).astype(np.uint8)
            tp, fp, fn, _ = stats(mi, gt); iou_i, _ = iou_dice(tp, fp, fn)
            per['manet'].append(iou_m); per['mit'].append(iou_i)

        # nnUNet batch
        images_dict = {f'{s}_{stem}': img_p for s, stem, img_p, _, _ in records}
        nn_preds = run_nnunet_batch(images_dict)
        for s, stem, _, gt, _ in records:
            key = f'{s}_{stem}'
            pred = nn_preds.get(key)
            if pred is None or pred.shape != gt.shape:
                if pred is not None:
                    pi = Image.fromarray(pred * 255, mode='L').resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
                    pred = (np.asarray(pi) > 0).astype(np.uint8)
                else:
                    pred = np.zeros_like(gt, dtype=np.uint8)
            tp, fp, fn, _ = stats(pred, gt); iou_n, _ = iou_dice(tp, fp, fn)
            per['nn'].append(iou_n)

        means = {k: float(np.mean(v)) for k, v in per.items()}
        results[pname] = means
        print(f'  MAnet={means["manet"]:.3f}  MiT={means["mit"]:.3f}  nnU={means["nn"]:.3f}')

    out_json = './data/results/robustness.json'
    Path(out_json).parent.mkdir(exist_ok=True)
    json.dump(results, open(out_json, 'w'), indent=2)
    print(f'saved {out_json}')

    # Plot: 3 subplots blur / brightness / noise
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    base = results['baseline']
    plot_groups = [
        ('Gaussian blur ($\\sigma$)', ['baseline', 'blur_1', 'blur_2', 'blur_4'], [0, 1, 2, 4]),
        ('Brightness shift', ['bright_-50', 'baseline', 'bright_50'], [-50, 0, 50]),
        ('Gaussian noise ($\\sigma$)', ['baseline', 'noise_5', 'noise_10', 'noise_20'], [0, 5, 10, 20]),
    ]
    colors = {'manet': '#1f77b4', 'mit': '#d62728', 'nn': '#2ca02c'}
    labels = {'manet': 'MAnet', 'mit': 'MiT-B5', 'nn': 'nnU-Net'}
    for ax, (title, keys, xs) in zip(axes, plot_groups):
        for model in ['manet', 'mit', 'nn']:
            ys = [results[k][model] * 100 for k in keys]
            ax.plot(xs, ys, marker='o', color=colors[model], label=labels[model])
        ax.set_xlabel(title); ax.set_title(title); ax.grid(alpha=0.3)
    axes[0].set_ylabel('IoU [%]')
    axes[2].legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    out_pdf = '/tmp/robustness_curves.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_pdf.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f'saved {out_pdf}')


if __name__ == '__main__':
    main()
