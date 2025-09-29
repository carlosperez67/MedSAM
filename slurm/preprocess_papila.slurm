#!/bin/bash
#SBATCH --job-name=preprocess_papila
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

PROJECT_DIR="/arc/project/st-ipor-1/carlosp/fundus_data/Papila"
SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"


BB_DIR="${SCRATCH_DIR}/bounding_box"
CONTOUR_DIR="${PROJECT_DIR}/Contours"
IMG_DIR="${PROJECT_DIR}/FundusImages"
OUT_DIR="${BB_DIR}/papila_boxes"


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
echo ${CONTOUR_DIR}
python "${BB_DIR}/papila_contours_to_boxes.py" \
  --contour_dir "${CONTOUR_DIR}" \
  --image_dir ${IMG_DIR} \
  --out_dir "${OUT_DIR}"

python "${BB_DIR}/visualize_boxes.py" \
  --autoboxes "${OUT_DIR}/autoboxes_two.jsonl" \
  --out_dir "${OUT_DIR}/viz" \
  --image_dir ${IMG_DIR}

python "${BB_DIR}/dataset_splitter.py" \
  --img_dir ${IMG_DIR} \
  --labels_dir ${OUT_DIR}/yolo_labels \
  --out_root  "${BB_DIR}/papila_yolo"






