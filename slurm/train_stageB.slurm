#!/bin/bash
#SBATCH --job-name=bb-stage_B
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --gpus=1
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca

set -euo pipefail

# --- Paths (adjust if your layout changes) ---
PROJECT_DIR="/arc/project/st-ipor-1/carlosp/fundus_data/Papila"
SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"


BB_DIR="${SCRATCH_DIR}/bounding_box"
RUNS_DIR="${BB_DIR}/runs/detect"
DATA_ROOT="${BB_DIR}/data/papila_yolo"          # expects images/{train,val,test}, labels/{train,val,test}
ROI_OUT="${BB_DIR}/data/papila_yolo_cupROI"     # will be created by ROI builder
OUT_JSONL="${BB_DIR}/data/two_stage_boxes.jsonl"

STAGEA_NAME="stageA_disc_only"
STAGEB_NAME="stageB_cup_roi"


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

# Optional: keep caches writable to avoid matplotlib/font warnings on compute nodes
export XDG_CACHE_HOME="/scratch/st-ipor-1/cperez/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"


# Helpful runtime info
python - <<'PY'
import torch, sys, ultralytics
print(f"[ENV] torch {torch.__version__} | cuda_available={torch.cuda.is_available()} | cuda_device_count={torch.cuda.device_count()}")
print(f"[ENV] ultralytics {ultralytics.__version__}")
PY

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

# --------------------------------------------------------------------
# Stage B — Train cup-only detector on ROI crops
# --------------------------------------------------------------------
echo "[STAGE B] Training cup-only detector on ROIs…"
python "${BB_DIR}/train/train_stageB_cup_roi.py" \
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