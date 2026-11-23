# Wound Healing Segmentation

Code, weights, and dataset for our cross-cell-line wound-healing segmentation paper (IEEE ICIP 2026 Workshop Learning Beyond Deep Learning (LBDL) II, *"Generalizing Wound Healing Segmentation: From Semi-Automatic Annotation to Robust Inference"*). The dataset and the trained weights are archived on Zenodo as **ScratchSeg**, the wound-healing assay module of [SpheroSeg](https://spherosegapp.utia.cas.cz/).

## Repository layout

```
wound-healing/
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
│   ├── eval_sam3.py                      (SAM 3 promptable concept segmentation)
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
│   ├── viz_foundation_qualitative.py     (foundation-model qualitative panel)
│   └── three_stages_panel.py             (per-stage qualitative panel, internal/Löwenstein)
├── scripts/              Orchestrators
│   ├── loocv_run.sh                      (Tab. 2 reproduction)
│   ├── loocv_run_unet_fpn.sh             (per-architecture LOOCV variant)
│   └── prepare_loocv.sh                  (build per-cell-line dataset splits)
├── checksums.txt         SHA-256 for downloadable artifacts
├── requirements.txt      Python dependencies (core; foundation-model deps optional)
└── LICENSE               MIT
```

The web annotation tool described in the paper is publicly available and deployed at <https://spherosegapp.utia.cas.cz/>.

## Downloadable artifacts

Archived on Zenodo with permanent DOIs:

- **Dataset:** ScratchSeg dataset, <https://doi.org/10.5281/zenodo.24416065> (CC BY 4.0)
- **Weights:** ScratchSeg trained weights, <https://doi.org/10.5281/zenodo.24416067> (licences below)

| Artifact | Record | Size | Purpose |
|---|---|---|---|
| `wounds-dataset.zip` | dataset | 349 MB | 445 frames + masks across 4 cell lines (U-2 OS 200, MiaPaca-2 147, UFH-001 52, MRC-5 46), with the paper's train/test split (401 / 44) |
| `wounds-weights-fair-recipe.tar` | weights | 2.2 GB | Tab. 1 fair-recipe: 10 SMP architectures (s42) + nnU-Net 250ep |
| `wounds-weights-loocv.tar` | weights | 4.3 GB | Tab. 2 LOOCV: 4 architectures × 4 cell-line holdouts (16 ckpts) |
| `wound_mitb5.ckpt` | weights | 324 MB | The model served by SpheroSeg (U-Net + MiT-B5); byte-identical to `fair_recipe/20260508_203609_segformer_mitb5.ckpt` in the fair-recipe tar |

`checksums.txt` lists the SHA-256 of all four files, the same values as `SHA256SUMS` in the two records. Verify after download:
```bash
sha256sum -c checksums.txt
```

The `README.md` inside `wounds-dataset.zip` predates acceptance and is kept byte-identical: it says 446 frames (MRC-5 47) and names the COMIC workshop. The archive holds 445 frames (MRC-5 46), and the paper appeared at LBDL II. The review-time OSF copy is no longer online; the Zenodo records replace it.

## Quickstart — reproducing Tab. 1

```bash
# 1. Clone (or download as ZIP)
git clone https://github.com/michalprusek/wound-healing  &&  cd wound-healing

# 2. Install core dependencies
pip install -r requirements.txt
# Foundation-model evals additionally need: cellpose>=4.1, segment-anything,
# transformers, micro-sam, nnunetv2 (uncomment in requirements.txt).

# 3. Fetch dataset + weights from Zenodo (records in §"Downloadable artifacts" above)
curl -L -o wounds-dataset.zip "https://zenodo.org/records/24416065/files/wounds-dataset.zip?download=1"
for f in wounds-weights-fair-recipe.tar wounds-weights-loocv.tar wound_mitb5.ckpt; do
    curl -L -o "$f" "https://zenodo.org/records/24416067/files/$f?download=1"
done
sha256sum -c checksums.txt
mkdir -p data checkpoints
unzip wounds-dataset.zip -d data/
tar -xf wounds-weights-fair-recipe.tar -C checkpoints/
tar -xf wounds-weights-loocv.tar -C checkpoints/

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

- Code: MIT (see `LICENSE`).
- Dataset: CC BY 4.0 (`LICENSE` inside `wounds-dataset.zip`).
- Weights: checkpoints with a MiT-B5 encoder (`wound_mitb5.ckpt`, the fair-recipe `segformer_mitb5` and the four `loocv_mitb5` files) were initialised from NVIDIA-derived ImageNet weights and are distributed under the NVIDIA Source Code License for SegFormer (non-commercial use only); all other checkpoints are under MIT. Full texts are in the `LICENSE` file of the weights record.

## Citation

```bibtex
@inproceedings{veskrna2026generalizing,
  title     = {Generalizing Wound Healing Segmentation: From Semi-Automatic Annotation to Robust Inference},
  author    = {Veškrna, Lukáš and Průšek, Michal and Novozámský, Adam and Šroubek, Filip and Svobodová Pavlíčková, Vladimíra and Rimpelová, Silvie},
  booktitle = {IEEE ICIP 2026 Workshop Learning Beyond Deep Learning (LBDL) II},
  address   = {Tampere, Finland},
  year      = {2026}
}

@dataset{veskrna2026scratchsegdata,
  title     = {ScratchSeg dataset: 445 bright-field wound-healing assay images of four cell lines with expert-corrected wound masks (wound-healing module of SpheroSeg)},
  author    = {Veškrna, Lukáš and Průšek, Michal and Novozámský, Adam and Šroubek, Filip and Svobodová Pavlíčková, Vladimíra and Rimpelová, Silvie},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.24416065}
}

@software{veskrna2026scratchsegweights,
  title     = {ScratchSeg: wound-healing assay module of SpheroSeg (SpheroSeg wound model, fair-recipe benchmark of 11 models and leave-one-cell-line-out models; trained weights)},
  author    = {Veškrna, Lukáš and Průšek, Michal and Novozámský, Adam and Šroubek, Filip and Svobodová Pavlíčková, Vladimíra and Rimpelová, Silvie},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.24416067}
}
```
