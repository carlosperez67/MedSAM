#!/bin/bash
#SBATCH --job-name=disc-y12x
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
set -x  # echo commands for debugging

# -------- Paths --------
SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"

BB_DIR="${SCRATCH_DIR}/bounding_box"
TRAIN_SCRIPT="${SCRATCH_DIR}/src/model/train_disc.py"
# 2-class YOLO split (images/{train,val,test}, labels/{...})
DATA_ROOT="${BB_DIR}/data/yolo_split"

# Local (offline) weights
WEIGHTS_LOCAL="${BB_DIR}/weights/yolo12x.pt"  # <- already copied to HPC

# Ultralytics runs live under project_dir/bounding_box/runs/detect by default
RUNS_DIR="${SCRATCH_DIR}/bounding_box/runs/detect"
EXP_NAME="stageA_disc_y12x"

mkdir -p "${SCRATCH_DIR}/logs" "${RUNS_DIR}"

# -------- Conda --------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# -------- Offline / Quiet --------
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export ULTRALYTICS_HUB=False
export ULTRALYTICS_ANALYTICS=False

# Put Ultralytics config/DDP files on scratch (not $HOME)
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export XDG_CONFIG_HOME="${SCRATCH_DIR}/.config"
export ULTRALYTICS_CONFIG_DIR="${XDG_CONFIG_HOME}/Ultralytics"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TMPDIR="${SCRATCH_DIR}/tmp"
mkdir -p "$XDG_CACHE_HOME" "$ULTRALYTICS_CONFIG_DIR/DDP" "$MPLCONFIGDIR" "$TMPDIR"

# -------- Threads & CUDA allocator --------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"

# -------- NCCL (single node) --------
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=warn

# Make all requested GPUs visible; Ultralytics will spawn DDP across them
CUDA_LIST=$(python - <<'PY'
import os
n = int(os.environ.get("SLURM_GPUS", "1"))
print(",".join(str(i) for i in range(n)))
PY
)
export CUDA_VISIBLE_DEVICES="${CUDA_LIST}"

echo "GPUs requested: ${SLURM_GPUS:-unknown}"
nvidia-smi -L || true

# -------- Sanity checks --------
[[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERR] Trainer not found: ${TRAIN_SCRIPT}" >&2; exit 2; }
[[ -d "${DATA_ROOT}/images" && -d "${DATA_ROOT}/labels" ]] || { echo "[ERR] Bad DATA_ROOT (needs images/ and labels/): ${DATA_ROOT}" >&2; exit 2; }
[[ -f "${WEIGHTS_LOCAL}" ]] || { echo "[ERR] Local weights not found: ${WEIGHTS_LOCAL}" >&2; exit 2; }

# -------- Show CLI (first lines) --------
python "${TRAIN_SCRIPT}" --help | sed -n '1,50p'

# -------- Build args safely --------
ARGS=(
  --project_dir "${SCRATCH_DIR}"      # makes runs at ${SCRATCH_DIR}/bounding_box/runs/detect
  --data_root   "${DATA_ROOT}"        # 2-class root; script will derive _disc_only/
  --weights     "${WEIGHTS_LOCAL}"    # local YOLOv12X file (offline)
  --family      auto
  --size        x
  --epochs      100
  --imgsz       640
  --batch       16                    # adjust if OOM; 12x is heavy
  --name        "${EXP_NAME}"
  --train       1
  --freeze      5                    # optional: freeze some early layers
  --amp         true                  # AMP on (good on V100)
  # Building toggles if you need them:
  # --copy_images                      # default is symlink
  # --drop_empty                       # drop empty labels after filtering
)

# -------- Train --------
cd /scratch/st-ipor-1/cperez/MedSAM
srun -u python -m src.model.train_disc "${ARGS[@]}"

BEST="${RUNS_DIR}/${EXP_NAME}/weights/best.pt"
[[ -f "${BEST}" ]] || { echo "[ERR] best.pt not found at ${BEST}" >&2; exit 3; }

echo "[OK] best.pt → ${BEST}"