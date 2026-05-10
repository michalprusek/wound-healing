"""
Convert our wound-healing dataset (train/test/{images,masks}) to nnU-Net v2
raw dataset format.

nnUNet_raw layout:
  Dataset001_Wounds/
    dataset.json
    imagesTr/<case_id>_0000.png   (channel 0)
    labelsTr/<case_id>.png        (binary mask, 0/1)
    imagesTs/<case_id>_0000.png   (held-out test, optional)
    labelsTs/<case_id>.png        (optional but useful for eval)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path("./data/data/raw/dataset")
DST = Path("./data/nnunet_raw/Dataset001_Wounds")


def convert_split(split: str, dst_imgs: Path, dst_masks: Path):
    src_img_dir = SRC / split / "images"
    src_mask_dir = SRC / split / "masks"
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path in sorted(src_img_dir.iterdir()):
        stem = img_path.stem
        case_id = f"wound_{split}_{stem}".replace(" ", "_").replace(",", "_")
        img = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
        Image.fromarray(img, mode="L").save(dst_imgs / f"{case_id}_0000.png")
        msk_path = src_mask_dir / img_path.name
        if not msk_path.exists():
            print(f"WARN: missing mask {msk_path}")
            continue
        msk = np.asarray(Image.open(msk_path).convert("L"), dtype=np.uint8)
        msk_bin = (msk > 127).astype(np.uint8)
        Image.fromarray(msk_bin, mode="L").save(dst_masks / f"{case_id}.png")
        n += 1
    return n


def main():
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    n_train = convert_split("train", DST / "imagesTr", DST / "labelsTr")
    n_test = convert_split("test", DST / "imagesTs", DST / "labelsTs")

    info = {
        "channel_names": {"0": "grayscale_microscopy"},
        "labels": {"background": 0, "wound": 1},
        "numTraining": n_train,
        "numTest": n_test,
        "file_ending": ".png",
        "name": "Wounds",
        "description": "anonymized wound-healing scratch-assay binary segmentation",
    }
    with open(DST / "dataset.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[done] {DST}")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
