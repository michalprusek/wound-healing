# Wound Healing Segmentation — Anonymous Reproducibility Package

> Submitted to **IEEE ICIP 2026 COMIC Workshop** under double-blind review.
> All identifying information has been stripped.

This repository contains the source code and a pointer to model checkpoints + dataset for our ICIP 2026 COMIC submission *"From Manual Annotation to Robust Inference in Wound Healing Microscopy."*

## What's in this repo

```
anon_release/
├── training/            Training scripts + per-architecture configs
├── classical/           Classical Prewitt + Suzuki contour pipeline + free-cell detection
├── evaluation/          Per-architecture and per-foundation-model evaluation scripts
├── annotation_tool/     React frontend + FastAPI backend (semi-automatic annotation platform)
├── viz/                 Figure rendering scripts (failure cases, qualitative panels)
├── requirements.txt     Python dependencies
└── LICENSE              MIT
```

## What's *not* in this repo (large artifacts)

| Artifact | Size | Location |
|---|---|---|
| Trained model checkpoints | ~3 GB | `[ANON_HOST]/wounds-weights.tar` (SHA-256: `[TBD]`) |
| Wound healing dataset (446 frames + masks) | ~350 MB | `[ANON_HOST]/wounds-dataset.zip` (SHA-256: `[TBD]`) |

Replace `[ANON_HOST]` with the URL provided in the paper Conclusion.

## Quickstart — reproducing Tab. 1

```bash
# 1. Clone (or anon.4open.science download)
git clone <this-repo-url>
cd wound-healing-anon

# 2. Install deps
pip install -r requirements.txt

# 3. Download dataset + weights from anonymous host
mkdir -p data checkpoints
curl -o wounds-dataset.zip "[ANON_HOST]/wounds-dataset.zip"
unzip wounds-dataset.zip -d data/
curl -o wounds-weights.tar "[ANON_HOST]/wounds-weights.tar"
tar -xf wounds-weights.tar -C checkpoints/

# 4. Verify checksums
sha256sum -c checksums.txt

# 5. Run evaluation suite for Tab. 1
python evaluation/eval_all.py \
    --data-root data/wounds_dataset_v1 \
    --checkpoints-root checkpoints \
    --output-json results/tab1_reproduction.json

# 6. Compare vs. paper Tab. 1
python evaluation/compare_to_paper.py results/tab1_reproduction.json
```

Expected output: per-method Dataset IoU and WRA Error matching paper Tab. 1 within ~0.5 pp (deterministic up to seed-42 trainings).

## Reproducing the LOOCV (Tab. 2)

```bash
# Each row of Tab. 2 = 4 trainings (one per held-out cell line)
# 4 architectures × 4 cell-lines = 16 trainings total (~25 min on 2× L40S)

bash scripts/run_loocv.sh --architectures unet manet fpn segformer_mitb5 \
    --cell-lines u2os miapaca mrc5 ufh001 \
    --data-root data/wounds_dataset_v1
```

## Reproducing foundation-model results

Foundation-model evaluation scripts handle their own model downloads (Cellpose, SAM, SAM 2, MedSAM, µSAM):

```bash
python evaluation/eval_cellpose.py    # Cellpose 4 (cpsam)
python evaluation/eval_sam_proper.py  # SAM ViT-B (relaxed AMG + union-bright)
python evaluation/eval_sam2_proper.py # SAM 2 Hiera-L
python evaluation/eval_medsam_official.py  # MedSAM with min-max normalization
python evaluation/eval_microsam_ais.py     # µSAM AIS workflow
```

## Annotation Tool

```bash
cd annotation_tool/
docker compose up
# Opens on http://localhost:8000 — drag-drop a frame to test
```

## License

Code: MIT (see LICENSE)
Dataset and weights: CC BY 4.0 (see LICENSE-DATA in the artifact bundle)

## Citation

(Anonymized for review)

```bibtex
@inproceedings{anonymous2026wound,
  title={From Manual Annotation to Robust Inference in Wound Healing Microscopy},
  author={Anonymous},
  booktitle={IEEE ICIP 2026 COMIC Workshop},
  year={2026}
}
```
