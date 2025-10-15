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
#SBATCH --mail-type=FAIL,END


set -euo pipefail

# ---------------- Paths ----------------
SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"          # project root on scratch
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam" # conda env

BB_DIR="${SCRATCH_DIR}/bounding_box"
RUNS_DIR="${BB_DIR}/runs/detect"

# Data locations (post-refactor)
DATA_ROOT="${BB_DIR}/data/yolo_split"            # full YOLO split (not used directly in Stage B)
ROI_OUT="${BB_DIR}/data/yolo_split_cupROI"       # ROI dataset produced by build_cup_roi_dataset.py

# Local weights (offline-safe)
WEIGHTS="${BB_DIR}/weights/yolov12x.pt"

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
echo "ROI Root     : $ROI_OUT"
echo "Weights      : $WEIGHTS"

echo "Visible GPUs:"
nvidia-smi -L || echo "nvidia-smi not available"

python - <<'PY'
import torch, ultralytics
print(f"[ENV] torch {torch.__version__} | cuda_available={torch.cuda.is_available()} | cuda_device_count={torch.cuda.device_count()}")
print(f"[ENV] ultralytics {ultralytics.__version__}")
PY

# ---------------- Sanity checks ----------------
if [[ ! -f "${WEIGHTS}" ]]; then
  echo "[ERROR] Weights not found at ${WEIGHTS}" >&2
  exit 2
fi

if [[ ! -f "${ROI_OUT}/cup_roi.yaml" ]]; then
  echo "[ERROR] ROI dataset missing cup_roi.yaml at ${ROI_OUT}. Build it first with build_cup_roi_dataset.py." >&2
  exit 2
fi

# ---------------- Stage B ----------------
echo "[STAGE B] Training cup-only detector on ROIs…"
python "${BB_DIR}/train/train_stageB_cup_roi.py" \
  --project_dir "${BB_DIR}" \
  --data_root   "${ROI_OUT}" \
  --model       "${WEIGHTS}" \
  --epochs      100 \
  --imgsz       640 \
  --batch       32 \
  --project     "${RUNS_DIR}" \
  --name        "${STAGEB_NAME}"

STAGEB_BEST="${RUNS_DIR}/${STAGEB_NAME}/weights/best.pt"
if [[ ! -f "${STAGEB_BEST}" ]]; then
  echo "[ERROR] Stage B best checkpoint not found: ${STAGEB_BEST}" >&2
  exit 3
fi

echo "Done  : $(date)"