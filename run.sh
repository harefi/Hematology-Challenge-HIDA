#!/usr/bin/env bash
# ------------------------------------------------------------------
# run.sh  –  Launch or resume baseline training in the background
# ------------------------------------------------------------------
set -Eeuo pipefail
cd "$(dirname "$0")"                      # always operate from repo root

# 1️⃣  Activate Conda
source "/<PATH to conda>/etc/profile.d/conda.sh"
conda activate hematol-22

echo "=============================================="
echo "  Baseline trainer launcher"
echo "=============================================="
echo "Choose run type:"
select MODE in "fresh" "resume"; do
    [[ -n "$MODE" ]] && break
done

case "$MODE" in
# ------------------------------------------------ fresh training run
fresh)
    ts=$(date +%Y%m%d_%H%M%S)
    out="results/baseline_${ts}"
    mkdir -p "$out"
    epochs=5                      # <<< adjust default here
    resume_arg=()                  # empty
    echo "→ New run will be stored in  $out"
    ;;
# ------------------------------------------------ resume existing run
resume)
    echo -n "Path to existing run folder (e.g. results/baseline_20250601_013731): "
    read -r out
    [[ -d "$out" ]] || { echo "ERROR: folder not found"; exit 1; }

    ckpt="$out/last_model.pth"
    [[ -f "$ckpt" ]] || { echo "ERROR: $ckpt not found"; exit 1; }

    echo -n "Target total epochs (must be > last epoch): "
    read -r epochs
    resume_arg=(--resume "$ckpt")
    echo "→ Resuming from  $ckpt"
    ;;
esac

# 2️⃣  Common training parameters
data_root="./Source"
batch_size=128
workers=32

# 3️⃣  Launch (unbuffered, detached)
log="$out/train_$(date +%H%M%S).log"
err="$out/train_$(date +%H%M%S).err"

nohup python -u scripts/train.py \
      --data_root "$data_root" \
      --output_dir "$out" \
      --epochs "$epochs" \
      --batch_size "$batch_size" \
      --num_workers "$workers" \
      "${resume_arg[@]}" \
      > "$log" 2> "$err" &

pid=$!
echo "=============================================="
echo "Training started in background (PID $pid)"
echo "Stdout → $log"
echo "Stderr → $err"
echo "Follow progress with:"
echo "  tail -f $log"
echo "=============================================="

