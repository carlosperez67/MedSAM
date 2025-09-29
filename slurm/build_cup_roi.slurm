#!/bin/bash
#SBATCH --job-name=papila_build_cup_roi
#SBATCH --account=st-ipor-1
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

# --- Activate your per-project Conda env ---
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Optional: keep caches writable to avoid matplotlib/font warnings on compute nodes
export XDG_CACHE_HOME="/scratch/st-ipor-1/cperez/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"


export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1


PROJECT_DIR="/arc/project/st-ipor-1/carlosp/fundus_data/Papila"
SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"


BB_DIR="${SCRATCH_DIR}/bounding_box"
CONTOUR_DIR="${PROJECT_DIR}/Contours"
IMG_DIR="${PROJECT_DIR}/FundusImages"
OUT_DIR="${BB_DIR}/papila_boxes"


# --------------------------------------------------------------------
# Build cup-ROI dataset (uses GT disc boxes)
# --------------------------------------------------------------------
echo "[DATA] Building cup ROI dataset at ${BB_DIR}/papila_yolo_cupROI…"
 python "${BB_DIR}/build_cup_roi_dataset.py" \
  --data_root "${BB_DIR}/papila_yolo" \
  --out_root "${BB_DIR}/papila_yolo_cupROI"