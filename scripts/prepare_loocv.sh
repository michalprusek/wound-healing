#!/usr/bin/env bash
# Create 4 leave-one-cell-line-out dataset directories using symlinks.
# Each directory has structure: train/{images,masks}/*.jpg + test/{images,masks}/*.jpg
#   - train: all images NOT from the held-out cell line
#   - test:  all images FROM the held-out cell line
set -uo pipefail

SRC=./data/data/raw/dataset
LOOCV_ROOT=./data/data/loocv

declare -A PREFIXES=(
    [u2os]="5min"
    [miapaca]="MiaPaca"
    [mrc5]="MRC-5"
    [ufh001]="UFH-001"
)

for cell in "${!PREFIXES[@]}"; do
    prefix="${PREFIXES[$cell]}"
    DST="$LOOCV_ROOT/holdout_${cell}"
    rm -rf "$DST"
    mkdir -p "$DST/train/images" "$DST/train/masks" "$DST/test/images" "$DST/test/masks"

    # train = all images NOT matching prefix
    for split_src in train test; do
        for img in "$SRC/$split_src/images"/*.jpg; do
            base=$(basename "$img")
            mask="$SRC/$split_src/masks/$base"
            if [[ "$base" == "$prefix"* ]]; then
                # held-out → goes to test
                ln -sf "$img" "$DST/test/images/$base"
                [ -f "$mask" ] && ln -sf "$mask" "$DST/test/masks/$base"
            else
                # train pool
                ln -sf "$img" "$DST/train/images/$base"
                [ -f "$mask" ] && ln -sf "$mask" "$DST/train/masks/$base"
            fi
        done
    done

    n_train_img=$(ls "$DST/train/images" | wc -l)
    n_train_msk=$(ls "$DST/train/masks" | wc -l)
    n_test_img=$(ls "$DST/test/images" | wc -l)
    n_test_msk=$(ls "$DST/test/masks" | wc -l)
    echo "  holdout=$cell: train=$n_train_img/$n_train_msk test=$n_test_img/$n_test_msk"
done

echo "[done] LOOCV directories ready under $LOOCV_ROOT/"
