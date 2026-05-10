#!/usr/bin/env bash
# LOOCV for top-2 archs: arch_manet + segformer_mitb5 × 4 cell lines = 8 runs
# Parallel across 2 GPUs (one cell per GPU)
set -uo pipefail

ROOT="./data"
cd "$ROOT" && source .venv/bin/activate
export SSL_CERT_FILE="$(python -m certifi)"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

log() { echo "[loocv] $(date +%H:%M:%S) $*"; }

CELLS=(u2os miapaca mrc5 ufh001)

# Helper: launch one LOOCV training
launch_manet() {
    local cell=$1; local gpu=$2
    local TS="$(date +%Y%m%d_%H%M%S)_loocv_arch_manet_holdout_${cell}"
    local RUN="$ROOT/runs/$TS"
    mkdir -p "$RUN"
    CUDA_VISIBLE_DEVICES=$gpu nohup python -u "$ROOT/scripts/train.py" \
        --config "$ROOT/configs/arch_manet.yaml" \
        --data-root "$ROOT/data/loocv/holdout_$cell" \
        --out-dir "$RUN" > "$RUN/train.log" 2>&1 &
    echo $!
}

launch_mit_b5() {
    local cell=$1; local gpu=$2
    local TS="$(date +%Y%m%d_%H%M%S)_loocv_segformer_mitb5_holdout_${cell}"
    local RUN="$ROOT/runs/$TS"
    mkdir -p "$RUN"
    CUDA_VISIBLE_DEVICES=$gpu nohup python -u "$ROOT/scripts/train_segformer_mit.py" \
        --data-root "$ROOT/data/loocv/holdout_$cell" \
        --out-dir "$RUN" --arch unet --encoder mit_b5 \
        --batch-size 8 --accum 4 \
        > "$RUN/train.log" 2>&1 &
    echo $!
}

run_pair() {
    # $1 = launcher_fn name, $2 = cellA, $3 = cellB
    local fn=$1; local cA=$2; local cB=$3
    log "  $fn pair: $cA / $cB"
    pidA=$($fn $cA 0)
    sleep 5
    pidB=$($fn $cB 1)
    while kill -0 $pidA 2>/dev/null; do sleep 30; done
    while kill -0 $pidB 2>/dev/null; do sleep 30; done
    log "  pair done"
}

log "LOOCV: arch_manet × 4 cells"
run_pair launch_manet u2os miapaca
run_pair launch_manet mrc5 ufh001

log "LOOCV: segformer_mitb5 × 4 cells"
run_pair launch_mit_b5 u2os miapaca
run_pair launch_mit_b5 mrc5 ufh001

log "all 8 LOOCV trainings done"

# Run aggregator on whatever has been added
python "$ROOT/scripts/aggregate_baselines.py" --results-dir "$ROOT/results" 2>&1 | tail -10
log "DONE"
