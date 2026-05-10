# Wound Healing Segmentation — Anonymous Reproducibility Package

> Submitted to **IEEE ICIP 2026 COMIC Workshop** under double-blind review.
> All identifying information has been stripped.

This repository contains the source code and pointers to the model checkpoints + dataset used for our ICIP 2026 COMIC submission *"From Manual Annotation to Robust Inference in Wound Healing Microscopy."*

## Repository layout

```
anon_release/
├── training/             Training scripts + per-architecture configs (SMP + MiT-B5)
│   ├── train.py
│   ├── train_segformer_mit.py            (MiT-B5 trainer)
│   └── configs/                          (10 fair-recipe + 4 LOOCV configs)
├── classical/            Classical pre-annotation pipeline
│   ├── classical_pipeline.py             (Prewitt + adaptive threshold + Suzuki contours)
│   ├── eval_otsu.py                      (Otsu baseline)
│   └── eval_tscratch.py                  (TScratch curvelet baseline reimplementation)
├── evaluation/           Per-architecture and per-foundation-model eval
│   ├── eval_external.py                  (Löwenstein eval harness, shared utility)
│   ├── eval_cellpose.py                  (Cellpose-SAM, complement-of-cells)
│   ├── eval_sam_proper.py                (SAM ViT-B + AMG + union-bright)
│   ├── eval_sam2_proper.py               (SAM 2 Hiera-L + same strategy)
│   ├── eval_medsam_official.py           (MedSAM with official preprocessing + bbox prompts)
│   ├── eval_microsam_ais.py              (µSAM AIS workflow)
│   ├── eval_nnunet.py                    (nnU-Net inference wrapper)
│   ├── prepare_nnunet_dataset.py         (convert to Dataset001_Wounds format)
│   └── eval_robustness.py                (synthetic perturbation eval, Fig. S7)
├── statistics/           Aggregation + statistics
│   ├── aggregate_baselines.py            (baselines_table.json from per-image JSONs)
│   ├── eval_smp_to_json.py               (per-image inference dump for SMP runs)
│   ├── bootstrap_ci.py                   (1000-iter bootstrap CI for IoU)
│   └── bench.py                          (inference latency benchmark)
├── viz/                  Figure rendering
│   ├── failure_cases_top3.py             (Fig. 3 in main paper)
│   └── viz_foundation_qualitative.py     (foundation-model qualitative panel)
├── scripts/              Orchestrators
│   ├── loocv_run.sh                      (Tab. 2 reproduction)
│   ├── loocv_run_unet_fpn.sh             (per-architecture LOOCV variant)
│   └── prepare_loocv.sh                  (build per-cell-line dataset splits)
├── checksums.txt         SHA-256 for downloadable artifacts
├── requirements.txt      Python dependencies (core; foundation-model deps optional)
└── LICENSE               MIT
```

The annotation tool source code is not included; the full annotation platform will be released as a separate open-source repository upon paper acceptance (see paper Conclusion).

## Reviewer-accessible artifacts

| Artifact | Size | Location |
|---|---|---|
| Trained model checkpoints (Tab. 1 fair-recipe block: 10 SMP architectures + nnU-Net) | 2.4 GB | `<OSF view-only>/wounds-weights-fair-recipe.tar` |
| Dataset (446 frames + masks) | 366 MB | `<OSF view-only>/wounds-dataset.zip` |

LOOCV checkpoints (Tab. 2) are reproducible from `scripts/loocv_run.sh` in ~25 min on 2 NVIDIA L40S GPUs. The OSF view-only URL is provided in the paper Conclusion + the submission system's reproducibility field. Verify after download:
```bash
sha256sum -c checksums.txt
```

## Quickstart — reproducing Tab. 1

```bash
# 1. Clone (or download as ZIP from anonymous.4open.science)
git clone <this-repo>  &&  cd wound-healing-anon

# 2. Install core dependencies
pip install -r requirements.txt
# Foundation-model evals additionally need: cellpose>=4.1, segment-anything,
# transformers, micro-sam, nnunetv2 (uncomment in requirements.txt).

# 3. Fetch dataset + weights from anonymous host (OSF view-only URL in paper)
mkdir -p data checkpoints
# (download the 2 artifacts manually or via OSF API: dataset zip + fair-recipe tar)
unzip wounds-dataset.zip -d data/
tar -xf wounds-weights-fair-recipe.tar -C checkpoints/
sha256sum -c checksums.txt

# 4. Re-run individual rows of Tab. 1
python classical/eval_otsu.py     --eval both --out-json results/otsu.json
python classical/eval_tscratch.py --eval both --out-json results/tscratch.json
python classical/classical_pipeline.py --eval both --out-json results/classical.json
python evaluation/eval_cellpose.py    --model cpsam --flow_threshold 0.7 --eval both --out-json results/cellpose_sam.json
python evaluation/eval_sam_proper.py  --eval both --out-json results/sam_vitb.json
python evaluation/eval_sam2_proper.py --eval both --out-json results/sam2_hieral.json
python evaluation/eval_medsam_official.py --eval both --out-json results/medsam.json
python evaluation/eval_microsam_ais.py    --eval both --out-json results/microsam.json
python evaluation/eval_nnunet.py      --nnunet-dir checkpoints/nnunet_250ep --eval both --out-json results/nnunet.json
# Per-architecture supervised runs (SMP):
for cfg in training/configs/arch_unet.yaml training/configs/arch_manet.yaml training/configs/arch_fpn.yaml \
           training/configs/arch_deeplabv3plus.yaml training/configs/arch_linknet.yaml \
           training/configs/arch_pspnet.yaml training/configs/unetpp_v3_paperrecipe.yaml; do
    python statistics/eval_smp_to_json.py --config $cfg --ckpt checkpoints/$(basename $cfg .yaml).ckpt \
                                          --out-json results/$(basename $cfg .yaml).json
done
# MiT-B5 (different framework):
python statistics/eval_smp_to_json.py --segformer --config training/configs/arch_segformer_mitb5.yaml \
       --ckpt checkpoints/segformer_mitb5.ckpt --out-json results/segformer_mitb5.json

# 5. Aggregate into Tab. 1 + bootstrap CIs for Supp. S1
python statistics/aggregate_baselines.py --input-dir results --out-json results/baselines_table.json
python statistics/bootstrap_ci.py --input-dir results --n-iter 1000 --out-json results/bootstrap_ci.json
```

## Reproducing Tab. 2 (LOOCV)

```bash
# Build per-cell-line splits, then train 4 architectures × 4 holdouts (= 16 trainings)
bash scripts/prepare_loocv.sh
bash scripts/loocv_run.sh           # MAnet, U-Net, FPN, MiT-B5 across all 4 holdouts
# ~25 min on 2× L40S; runs train.py / train_segformer_mit.py per holdout.
```

## Reproducing Fig. 3 (failure cases) and Fig. S7 (robustness)

```bash
# Fig. 3 — worst Löwenstein cases for top-3 architectures
python viz/failure_cases_top3.py --manet-ckpt checkpoints/arch_manet_s42.ckpt \
    --mitb5-ckpt checkpoints/segformer_mitb5.ckpt \
    --nnunet-dir checkpoints/nnunet_250ep \
    --n 3 --out paper_figures/failure_cases_top3.pdf

# Fig. S7 — synthetic perturbation robustness curves
python evaluation/eval_robustness.py    # writes results/robustness.json + robustness_curves.pdf
```

## Inference latency (§4.2)

```bash
python statistics/bench.py --architectures unet+resnet50 unet+mitb5 manet+resnet50 \
    --image-size 256 --n-frames 20 --warmup 3 --gpu --out-json results/throughput.json
```

## License

- Code: MIT (see `LICENSE`)
- Dataset and weights: CC BY 4.0 (see `LICENSE-DATA` inside the dataset zip)

## Citation

(Anonymized for review.)

```bibtex
@inproceedings{anonymous2026wound,
  title={From Manual Annotation to Robust Inference in Wound Healing Microscopy},
  author={Anonymous},
  booktitle={IEEE ICIP 2026 COMIC Workshop},
  year={2026}
}
```
