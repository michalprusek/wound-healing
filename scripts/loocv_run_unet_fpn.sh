#!/usr/bin/env bash
# LOOCV expansion: U-Net + FPN (both ResNet-50) x 4 cell lines = 8 runs
# Run two trainings in parallel on GPU 0 (26 GB free, each needs ~8 GB).
# GPU 1 left alone (other user, only 10 GB free).
set -uo pipefail

ROOT="./data"
cd "$ROOT" && source .venv/bin/activate
export SSL_CERT_FILE="$(python -m certifi)"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

log() { echo "[loocv-uf] $(date +%H:%M:%S) $*"; }

launch() {
    local arch=$1; local cell=$2
    local name="loocv_arch_${arch}_holdout_${cell}"
    local TS="$(date +%Y%m%d_%H%M%S)_${name}"
    local RUN="$ROOT/runs/$TS"
    mkdir -p "$RUN"
    log "  start $arch/$cell -> $RUN"
    CUDA_VISIBLE_DEVICES=0 nohup python -u "$ROOT/scripts/train.py" \
        --config "$ROOT/configs/arch_${arch}.yaml" \
        --data-root "$ROOT/data/loocv/holdout_$cell" \
        --out-dir "$RUN" > "$RUN/train.log" 2>&1 &
    echo $!
}

run_pair() {
    local arch=$1; local cA=$2; local cB=$3
    log "=== pair $arch: $cA / $cB ==="
    pidA=$(launch $arch $cA)
    sleep 15
    pidB=$(launch $arch $cB)
    while kill -0 $pidA 2>/dev/null; do sleep 30; done
    log "  pidA done"
    while kill -0 $pidB 2>/dev/null; do sleep 30; done
    log "  pidB done"
}

for arch in unet fpn; do
    run_pair $arch u2os miapaca
    run_pair $arch mrc5 ufh001
done

log "all 8 LOOCV trainings finished"
