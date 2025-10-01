#!/bin/bash
#SBATCH --job-name=preprocess_smdg
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

PROJECT_DIR="/arc/project/st-ipor-1/carlosp/fundus_data/SMDG-19"
SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"


BB_DIR="${SCRATCH_DIR}/bounding_box"
FUNDUS_IMG_DIR="${PROJECT_DIR}/full-fundus/full-fundus"
OC_IMG_DIR="${PROJECT_DIR}/optic-disc/optic-disc"
OD_IMG_DIR="${PROJECT_DIR}/optic-cup/optic-cup"

OUT_DIR="${BB_DIR}/data"


# --- Activate your per-project Conda env ---
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Optional: keep caches writable to avoid matplotlib/font warnings on compute nodes
export XDG_CACHE_HOME="/scratch/st-ipor-1/cperez/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"


export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

# --------------------------------------------------------------------
# Stage A — Train disc-only detector (filters labels to class=disc)
# --------------------------------------------------------------------

python "${BB_DIR}/preprocess/mask_to_boxes.py" \
  --images "${FUNDUS_IMG_DIR}" \
  --disc_masks "${OD_IMG_DIR}" \
  --cup_masks "${OC_IMG_DIR}" \
  --cup_masks "${OC_IMG_DIR}" \
  --out_labels "${OUT_DIR}/labels" \
  --out_csv "${OUT_DIR}/labels_summary.csv"

python "${BB_DIR}/preprocess/visualize_bounding_boxes.py" \
  --labels_dir "${OUT_DIR}/labels" \
  --image_dir "${FUNDUS_IMG_DIR}" \
  --out_dir "${OUT_DIR}/viz_labels"

python "${BB_DIR}/dataset_splitter.py" \
  --img_dir ${IMG_DIR} \
  --labels_dir "${OUT_DIR}/labels" \
  --out_root  "${BB_DIR}/data/yolo_split"