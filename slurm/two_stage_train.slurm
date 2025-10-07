#!/bin/bash
#SBATCH --job-name=two-stage-train-B-only
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail



# --------------------------- Paths (HPC) ---------------------------
PROJECT_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"


BB_DIR="${PROJECT_DIR}/bounding_box"
RUNS_DIR="${BB_DIR}/runs/detect"

# Clean split, augmented split, and prebuilt ROI split (from preprocess step)
DATA_ROOT="${BB_DIR}/data/yolo_split"       # clean (val/test)
AUG_ROOT="${BB_DIR}/data/yolo_split_aug"    # augmented (train)
ROI_OUT="${BB_DIR}/data/yolo_split_cupROI"  # prebuilt (used by Stage B)
# Prebuilt disc-only dataset (built by preprocess 'disc_only' stage)
DISC_ROOT="${BB_DIR}/data/yolo_split_disc_only"

# Optional output for two-stage inference
OUT_JSONL="${BB_DIR}/data/two_stage_boxes.jsonl"

# Experiment names (match script defaults)
STAGEA_NAME="stageA_disc_only"
STAGEB_NAME="stageB_cup_roi"


# Optional: avoid cache warnings on compute nodes
export XDG_CACHE_HOME="/scratch/st-ipor-1/cperez/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TMPDIR="${PROJECT_DIR}/tmp"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"

mkdir -p "${PROJECT_DIR}/logs" "${RUNS_DIR}" "${BB_DIR}/data"
cd "${PROJECT_DIR}"

# --------------------------- Env -----------------------------------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

echo "SLURM Job ID : ${SLURM_JOB_ID}"
echo "Node list    : ${SLURM_JOB_NODELIST}"
echo "Working dir  : $(pwd)"
echo "Runs dir     : ${RUNS_DIR}"

nvidia-smi -L || echo "nvidia-smi not available"

python - <<'PY'
import torch, ultralytics
print(f"[ENV] torch {torch.__version__} | cuda={torch.cuda.is_available()} | gpus={torch.cuda.device_count()}")
print(f"[ENV] ultralytics {ultralytics.__version__}")
PY

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"



# --------------------------- Helpers -------------------------------
die() { echo "[ERROR] $*" >&2; exit 1; }
check_file() { [[ -f "$1" ]] || die "$2 not found: $1"; }
check_dir()  { [[ -d "$1" ]] || die "$2 not found: $1"; }

# --------------------------- Stage A -------------------------------
run_stageA() {
  echo "[STAGE A] Train disc-only detector using PREBUILT disc-only dataset…"
  check_dir  "${DISC_ROOT}" "Prebuilt disc-only root"
  check_file "${DISC_ROOT}/od_only.yaml" "Disc-only YAML (od_only.yaml)"

  # Minimal CLI: just point --data_root at the disc-only root; no --aug_root needed.
  python "${BB_DIR}/train/train_stageA_disc_only.py" \
    --project_dir "${PROJECT_DIR}" \
    --data_root   "${DISC_ROOT}" \
    --model       "${BB_DIR}/weights/yolov8n.pt"

  local best="${RUNS_DIR}/${STAGEA_NAME}/weights/best.pt"
  check_file "${best}" "Stage A best checkpoint"
  echo "[STAGE A] Best checkpoint: ${best}"
}

# --------------------------- Stage B -------------------------------
run_stageB() {
  echo "[STAGE B] Train cup-only detector on prebuilt ROI dataset…"
  check_dir "${ROI_OUT}" "ROI dataset"

  # Explicitly pass project_dir so defaults resolve to BB_DIR/{data,runs,weights}
  # No other flags needed if you’re happy with defaults.
  python "${BB_DIR}/train/train_stageB_cup_roi.py" \
    --project_dir "${BB_DIR}"

  local best="${RUNS_DIR}/${STAGEB_NAME}/weights/best.pt"
  check_file "${best}" "Stage B best checkpoint"
  echo "[STAGE B] Best checkpoint: ${best}"
}

# --------------------------- Inference (optional) ------------------
run_infer_optional() {
  local test_images="${DATA_ROOT}/images/test"
  if [[ -d "${test_images}" ]]; then
    echo "[INFER] Two-stage inference on clean test images…"
    local bestA="${RUNS_DIR}/${STAGEA_NAME}/weights/best.pt"
    local bestB="${RUNS_DIR}/${STAGEB_NAME}/weights/best.pt"
    check_file "${bestA}" "Stage A best checkpoint"
    check_file "${bestB}" "Stage B best checkpoint"

    python "${BB_DIR}/train/infer_two_stage_to_boxes.py" \
      --stageA "${bestA}" \
      --stageB "${bestB}" \
      --images "${test_images}" \
      --out_jsonl "${OUT_JSONL}" \
      --od_pad_pct 0.08 \
      --confA 0.25 \
      --confB 0.10
    echo "[INFER] Wrote: ${OUT_JSONL}"
  else
    echo "[WARN] No clean test split at ${test_images}; skipping inference."
  fi
}

# --------------------------- Run -----------------------------------
# run_stageA
run_stageB
# run_infer_optional

echo "[DONE] Two-stage training completed at $(date)"