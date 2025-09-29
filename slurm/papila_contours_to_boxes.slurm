#!/bin/bash
#SBATCH --job-name=papila_contours_to_boxes
#SBATCH --account=st-ipor-1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=0:30:00
# SBATCH --gpus=1
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca

set -euo pipefail

PROJECT_DIR="/arc/project/st-ipor-1/cperez/fundus_data/Papila"
SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"


BB_DIR="${SCRATCH_DIR}/bounding_box"
CONTOUR_DIR="${PROJECT_DIR}/Contours"
IMG_DIR="${PROJECT_DIR}/FundusImages"
OUT_DIR="${BB_DIR}/papila_boxes"

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
# Stage A — Train disc-only detector (filters labels to class=disc)
# --------------------------------------------------------------------
echo "[STAGE A] Training disc-only detector…"
python "${BB_DIR}/papila_contours_to_boxes.py" \
  --contour_dir "${CONTOUR_DIR}" \
  --image_dir ${IMG_DIR}
