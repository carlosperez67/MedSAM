#!/bin/bash
#SBATCH --job-name=bb-stage_A
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

# ---------------- Paths ----------------
SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"

# Project layout (rooted inside bounding_box/)
BB_DIR="${SCRATCH_DIR}/bounding_box"
RUNS_DIR="${BB_DIR}/runs/detect"

# Unified YOLO split now lives here
DATA_ROOT="${BB_DIR}/data/yolo_split"

# ROI dataset will be created here
ROI_OUT="${BB_DIR}/data/yolo_split_cupROI"

# Local weights you uploaded
WEIGHTS="${BB_DIR}/weights/yolov8n.pt"

OUT_JSONL="${BB_DIR}/data/two_stage_boxes.jsonl"

STAGEA_NAME="stageA_disc_only"
STAGEB_NAME="stageB_cup_roi"

mkdir -p "${SCRATCH_DIR}/logs" "${RUNS_DIR}"

# ---------------- Env ----------------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

echo "SLURM Job ID : $SLURM_JOB_ID"
echo "Nodes        : $SLURM_JOB_NODELIST"
echo "Start        : $(date)"
echo "Project Dir  : $SCRATCH_DIR"
echo "Runs Dir     : $RUNS_DIR"
echo "Data Root    : $DATA_ROOT"
echo "Weights      : $WEIGHTS"

echo "Visible GPUs:"
nvidia-smi -L || echo "nvidia-smi not available"

python - <<'PY'
import torch, ultralytics
print(f"[ENV] torch {torch.__version__} | cuda_available={torch.cuda.is_available()} | cuda_device_count={torch.cuda.device_count()}")
print(f"[ENV] ultralytics {ultralytics.__version__}")
PY

# ---------------- Stage A ----------------
echo "[STAGE A] Training disc-only detector (val-only run with --train 0)…"
python "${BB_DIR}/train/train_stageA_disc_only.py" \
  --project_dir "${SCRATCH_DIR}" \
  --data_root   "${DATA_ROOT}" \
  --model       "${WEIGHTS}" \
  --epochs      100 \
  --imgsz       640 \
  --batch       32 \
  --project     "${RUNS_DIR}" \
  --name        "${STAGEA_NAME}" \
  --train       1

STAGEA_BEST="${RUNS_DIR}/${STAGEA_NAME}/weights/best.pt"
if [[ ! -f "${STAGEA_BEST}" ]]; then
  echo "[ERROR] Stage A best checkpoint not found: ${STAGEA_BEST}" >&2
  exit 2
fi

echo "Done  : $(date)"