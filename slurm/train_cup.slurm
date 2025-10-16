#!/bin/bash
#SBATCH --job-name=cup-roi-y12x
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --gpus=4
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail
set -x  # echo commands so you can see exactly what gets passed

SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"

BB_DIR="${SCRATCH_DIR}/bounding_box"
TRAIN_SCRIPT="${SCRATCH_DIR}/src/model/train_cup.py"
RUNS_DIR="${SCRATCH_DIR}/runs/detect"
ROI_ROOT="${BB_DIR}/data/yolo_split_cupROI"
WEIGHTS_LOCAL="${BB_DIR}/weights/yolo12x.pt"
EXP_NAME="stageB_cup_roi_y12x"

mkdir -p "${SCRATCH_DIR}/logs" "${RUNS_DIR}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Offline & quiet
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export ULTRALYTICS_HUB=False
export ULTRALYTICS_ANALYTICS=False

# Caches / temp
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export XDG_CONFIG_HOME="${SCRATCH_DIR}/.config"
export ULTRALYTICS_CONFIG_DIR="${XDG_CONFIG_HOME}/Ultralytics"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TMPDIR="${SCRATCH_DIR}/tmp"
mkdir -p "$XDG_CACHE_HOME" "$ULTRALYTICS_CONFIG_DIR/DDP" "$MPLCONFIGDIR" "$TMPDIR"

# Threading + allocator
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"

# NCCL (single node)
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=warn

# Choose devices by visibility rather than --device arg to avoid quoting issues
# This makes all GPUs visible; Ultralytics will DDP across them automatically.
CUDA_LIST=$(python - <<'PY'
import os
n = int(os.environ.get("SLURM_GPUS", "1"))
print(",".join(str(i) for i in range(n)))
PY
)
export CUDA_VISIBLE_DEVICES="${CUDA_LIST}"

nvidia-smi -L || true

# Sanity
[[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERR] Trainer not found: ${TRAIN_SCRIPT}" >&2; exit 2; }
[[ -d "${ROI_ROOT}"     ]] || { echo "[ERR] ROI root not found: ${ROI_ROOT}" >&2; exit 2; }
[[ -f "${WEIGHTS_LOCAL}" ]]|| { echo "[ERR] Local weights not found: ${WEIGHTS_LOCAL}" >&2; exit 2; }
if [[ -f "${ROI_ROOT}/cup_roi.yaml" && ! -f "${ROI_ROOT}/data.yaml" ]]; then
  ln -sf "${ROI_ROOT}/cup_roi.yaml" "${ROI_ROOT}/data.yaml"
fi
[[ -f "${ROI_ROOT}/data.yaml" ]] || { echo "[ERR] Missing data.yaml" >&2; exit 2; }

# SHOW the parser we expect (catches mismatched script versions)
python "${TRAIN_SCRIPT}" --help | sed -n '1,40p'

# Build args as an array (no line-continuation gotchas)
ARGS=(
  --data-root "${ROI_ROOT}"
  --runs-root "${RUNS_DIR}"
  --weights   "${WEIGHTS_LOCAL}"
  --family    auto
  --size      x
  --epochs    100
  --imgsz     512
  --batch     16
  --optimizer SGD
  --mosaic    0.0
  --freeze    5
  --amp       true
  --name      "${EXP_NAME}"
  --workers   "${SLURM_CPUS_PER_TASK}"
)

# Optional: ensure Unix line endings to avoid hidden \r
# dos2unix "${TRAIN_SCRIPT}" 2>/dev/null || true

# Train
cd /scratch/st-ipor-1/cperez/MedSAM
srun -u python -m src.model.train_cup "${ARGS[@]}"

BEST="${RUNS_DIR}/${EXP_NAME}/weights/best.pt"
[[ -f "${BEST}" ]] || { echo "[ERR] best.pt not found at ${BEST}" >&2; exit 3; }

echo "[OK] best.pt → ${BEST}"