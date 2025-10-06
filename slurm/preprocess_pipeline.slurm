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
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# --- Paths (HPC) ---
SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"
PIPELINE="${SCRATCH_DIR}/bounding_box/preprocess/preprocess_pipeline.py"
CONFIG="${SCRATCH_DIR}/bounding_box/preprocess/preprocess_pipeline_hpc.yaml"
AUG_YAML="${SCRATCH_DIR}/bounding_box/preprocess/augment_config_hpc.yaml"  # referenced by CONFIG

# Create dirs (fix logs path)
mkdir -p "$(dirname "${CONFIG}")" \
         "${SCRATCH_DIR}/bounding_box/data" \
         "${SCRATCH_DIR}/logs"

# --- Conda env ---
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Optional: avoid cache warnings on compute nodes
export XDG_CACHE_HOME="/scratch/st-ipor-1/cperez/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TMPDIR="${SCRATCH_DIR}/tmp"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"

# Threading (avoid oversubscription)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

# --- Sanity checks ---
if [[ ! -f "$PIPELINE" ]]; then
  echo "[ERR] Pipeline not found: $PIPELINE" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERR] Config not found: $CONFIG" >&2
  exit 1
fi

# If the config references an external augment YAML, make sure it exists
if [[ ! -f "$AUG_YAML" ]]; then
  echo "[WARN] Augment YAML not found at $AUG_YAML"
  echo "       If your CONFIG expects this file (aug_yaml: ...), create it first."
fi

# --- Run the full preprocessing pipeline via YAML config ---
echo "[INFO] Using config: ${CONFIG}"
srun -u python "${PIPELINE}" --config "${CONFIG}"