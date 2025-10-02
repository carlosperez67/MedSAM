#!/bin/bash
#SBATCH --job-name=preprocess_smdg
#SBATCH --account=st-ipor-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca

set -euo pipefail

# --- Paths (HPC) ---
SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"
CONFIG="${SCRATCH_DIR}/bounding_box/preprocess/preprocess_pipeline_hpc.yaml"
PIPELINE="${SCRATCH_DIR}/bounding_box/preprocess/preprocess_pipeline.py"

mkdir -p "$(dirname "${CONFIG}")" "${SCRATCH_DIR}/bounding_box/data" "${SCRATCH_DIR}/MedSAM/logs"

# --- Conda env ---
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Optional: avoid cache warnings on compute nodes
export XDG_CACHE_HOME="/scratch/st-ipor-1/cperez/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

# --- Run the full preprocessing pipeline via YAML config ---
echo "[INFO] Using config: ${CONFIG}"
python "${PIPELINE}" --config "${CONFIG}"
# Or with srun:
# srun -u python "${PIPELINE}" --config "${CONFIG}"