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

SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"

BB_DIR="${SCRATCH_DIR}/bounding_box"
TRAIN_SCRIPT="${SCRATCH_DIR}/src/model/train_cup.py"     # your modern trainer
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
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TMPDIR="${SCRATCH_DIR}/tmp"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"

# Put Ultralytics config/DDP files on scratch (not $HOME)
export XDG_CONFIG_HOME="${SCRATCH_DIR}/.config"
export ULTRALYTICS_CONFIG_DIR="${XDG_CONFIG_HOME}/Ultralytics"   # extra safety, even if not required
mkdir -p "${ULTRALYTICS_CONFIG_DIR}/DDP"

# Threading
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# NCCL single-node sanity (often helps on clusters without IB)
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=warn

# allocator & offline
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export ULTRALYTICS_HUB=False
export ULTRALYTICS_ANALYTICS=False

echo "GPUs requested: ${SLURM_GPUS:-unknown}"
nvidia-smi -L || true

# Sanity
[[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERR] Trainer not found: ${TRAIN_SCRIPT}" >&2; exit 2; }
[[ -d "${ROI_ROOT}"     ]] || { echo "[ERR] ROI root not found: ${ROI_ROOT}" >&2; exit 2; }
[[ -f "${WEIGHTS_LOCAL}" ]]|| { echo "[ERR] Local weights not found: ${WEIGHTS_LOCAL}" >&2; exit 2; }
if [[ -f "${ROI_ROOT}/cup_roi.yaml" && ! -f "${ROI_ROOT}/data.yaml" ]]; then
  ln -sf "${ROI_ROOT}/cup_roi.yaml" "${ROI_ROOT}/data.yaml"
fi
[[ -f "${ROI_ROOT}/data.yaml" ]] || { echo "[ERR] Missing data.yaml" >&2; exit 2; }

# Build device list matching the number of visible GPUs: "0,1,2,3"
GPU_COUNT="${SLURM_GPUS:-1}"
if [[ "$GPU_COUNT" -lt 1 ]]; then GPU_COUNT=1; fi
DEVICES=$(python - <<PY
n = int("${GPU_COUNT}")
print(",".join(str(i) for i in range(n)))
PY
)

echo "[INFO] Using devices: $DEVICES"

srun -u python "${TRAIN_SCRIPT}" \
  --data-root "${ROI_ROOT}" \
  --runs-root "${RUNS_DIR}" \
  --weights   "${WEIGHTS_LOCAL}" \
  --family    auto \
  --size      x \
  --epochs    100 \
  --imgsz     512 \
  --batch     8 \                # per GPU (global=8*4=32)
  --accumulate 4 \               # effective global batch ≈ 128
  --optimizer SGD \
  --name      "${EXP_NAME}" \
  --workers   "${SLURM_CPUS_PER_TASK}" \
  --device    "${DEVICES}"

BEST="${RUNS_DIR}/${EXP_NAME}/weights/best.pt"
[[ -f "${BEST}" ]] || { echo "[ERR] best.pt not found at ${BEST}" >&2; exit 3; }

echo "[OK] best.pt → ${BEST}"