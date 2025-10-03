#!/bin/bash
#SBATCH --job-name=build_cup_roi
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

# ---------------- PATHS ----------------
PROJECT_DIR="/arc/project/st-ipor-1/carlosp/fundus_data/SMDG-19"
SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"

BB_DIR="${SCRATCH_DIR}/bounding_box"

# ---------------- ENV SETUP ----------------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export XDG_CACHE_HOME="/scratch/st-ipor-1/cperez/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

# ---------------- JOB INFO ----------------
echo "SLURM Job ID : $SLURM_JOB_ID"
echo "Nodes        : $SLURM_JOB_NODELIST"
echo "Start        : $(date)"
echo "Project Dir  : $PROJECT_DIR"
echo "Scratch Dir  : $SCRATCH_DIR"

# ---------------- MAIN ----------------
echo "[DATA] Building cup ROI dataset..."
python "${BB_DIR}/build_cup_roi_dataset.py" \
  --project_dir "${SCRATCH_DIR}" \
  --data_root "${BB_DIR}/data/yolo_split" \
  --out_root "${BB_DIR}/data/yolo_split_cupROI" \
  --pad_pct 0.10 \
  --keep_negatives