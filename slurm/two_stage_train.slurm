#!/bin/bash
#SBATCH --job-name=bb-two-stage
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --gpus=1
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca

set -euo pipefail

# --- Paths (adjust if your layout changes) ---
PROJECT_DIR="/scratch/st-ipor-1/cperez/medsam"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"
BB_DIR="${PROJECT_DIR}/bounding_box"
RUNS_DIR="${PROJECT_DIR}/runs/detect"
DATA_ROOT="${PROJECT_DIR}/papila_yolo"          # expects images/{train,val,test}, labels/{train,val,test}
ROI_OUT="${PROJECT_DIR}/papila_yolo_cupROI"     # will be created by ROI builder
OUT_JSONL="${BB_DIR}/two_stage_boxes.jsonl"

STAGEA_NAME="stageA_disc_only"
STAGEB_NAME="stageB_cup_roi"

mkdir -p "${PROJECT_DIR}/logs" "${RUNS_DIR}"

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Nodes       : $SLURM_JOB_NODELIST"
echo "Start       : $(date)"
echo "Working dir : ${SLURM_SUBMIT_DIR}"
echo "Project dir : ${PROJECT_DIR}"
echo "Runs dir    : ${RUNS_DIR}"
echo "Data root   : ${DATA_ROOT}"

echo "Visible GPUs (from nvidia-smi):"
nvidia-smi -L || echo "nvidia-smi not available"

# XLA flag (harmless if unused)
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

# --- Activate your per-project Conda env ---
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Helpful runtime info
python - <<'PY'
import torch, sys, ultralytics
print(f"[ENV] torch {torch.__version__} | cuda_available={torch.cuda.is_available()} | cuda_device_count={torch.cuda.device_count()}")
print(f"[ENV] ultralytics {ultralytics.__version__}")
PY

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

# --------------------------------------------------------------------
# Stage A — Train disc-only detector (filters labels to class=disc)
# --------------------------------------------------------------------
echo "[STAGE A] Training disc-only detector…"
python "${BB_DIR}/train_stageA_disc_only.py" \
  --data_root "${DATA_ROOT}" \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --project "${RUNS_DIR}" \
  --name "${STAGEA_NAME}"

STAGEA_BEST="${RUNS_DIR}/${STAGEA_NAME}/weights/best.pt"
if [[ ! -f "${STAGEA_BEST}" ]]; then
  echo "[ERROR] Stage A best checkpoint not found: ${STAGEA_BEST}" >&2
  exit 2
fi

# --------------------------------------------------------------------
# Build cup-ROI dataset (uses GT disc boxes)
# --------------------------------------------------------------------
echo "[DATA] Building cup ROI dataset at ${ROI_OUT}…"
python "${BB_DIR}/build_cup_roi_dataset.py" \
  --data_root "${DATA_ROOT}" \
  --out_root "${ROI_OUT}" \
  --pad_pct 0.10

# --------------------------------------------------------------------
# Stage B — Train cup-only detector on ROI crops
# --------------------------------------------------------------------
echo "[STAGE B] Training cup-only detector on ROIs…"
python "${BB_DIR}/train_stageB_cup_roi.py" \
  --data_root "${ROI_OUT}" \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --project "${RUNS_DIR}" \
  --name "${STAGEB_NAME}"

STAGEB_BEST="${RUNS_DIR}/${STAGEB_NAME}/weights/best.pt"
if [[ ! -f "${STAGEB_BEST}" ]]; then
  echo "[ERROR] Stage B best checkpoint not found: ${STAGEB_BEST}" >&2
  exit 3
fi

# --------------------------------------------------------------------
# Optional: two-stage inference on test split (if present)
# --------------------------------------------------------------------
TEST_IMAGES="${DATA_ROOT}/images/test"
if [[ -d "${TEST_IMAGES}" ]]; then
  echo "[INFER] Two-stage inference on test images…"
  python "${BB_DIR}/infer_two_stage_to_boxes.py" \
    --stageA "${STAGEA_BEST}" \
    --stageB "${STAGEB_BEST}" \
    --images "${TEST_IMAGES}" \
    --out_jsonl "${OUT_JSONL}" \
    --od_pad_pct 0.08 \
    --confA 0.25 \
    --confB 0.10
  echo "[INFER] Wrote: ${OUT_JSONL}"
else
  echo "[WARN] Test images not found at ${TEST_IMAGES}; skipping inference."
fi

echo "End: $(date)"
echo "[DONE] Two-stage pipeline completed."