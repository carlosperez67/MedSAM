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
set -x

# -------- Paths --------
SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"      # project root (contains src/)
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"

DATA_ROOT="${SCRATCH_DIR}/bounding_box/data/yolo_split"   # 2-class dataset root
WEIGHTS_LOCAL="${SCRATCH_DIR}/bounding_box/weights/yolo12x.pt"
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

# -------- Cache & config (on scratch, not $HOME) --------
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export XDG_CONFIG_HOME="${SCRATCH_DIR}/.config"
export ULTRALYTICS_CONFIG_DIR="${XDG_CONFIG_HOME}/Ultralytics"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TMPDIR="${SCRATCH_DIR}/tmp"
mkdir -p "$XDG_CACHE_HOME" "$ULTRALYTICS_CONFIG_DIR/DDP" "$MPLCONFIGDIR" "$TMPDIR"

# -------- Threads & allocator --------
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

# -------- Make project importable --------
export PYTHONPATH="${SCRATCH_DIR}:${PYTHONPATH:-}"

# -------- Choose GPUs by visibility; Ultralytics will DDP across them --------
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
[[ -d "${DATA_ROOT}/images" && -d "${DATA_ROOT}/labels" ]] || { echo "[ERR] Bad DATA_ROOT: ${DATA_ROOT}" >&2; exit 2; }
[[ -f "${WEIGHTS_LOCAL}" ]] || { echo "[ERR] Weights not found: ${WEIGHTS_LOCAL}" >&2; exit 2; }
[[ -d "${SCRATCH_DIR}/src" ]] || { echo "[ERR] No src/ under ${SCRATCH_DIR}" >&2; exit 2; }

# Optional: preflight import
python - <<'PY'
import sys, os
print("[PYTHONPATH]", os.environ.get("PYTHONPATH"))
import src, importlib
print("[CHECK] Imported 'src' OK from:", src.__file__)
importlib.import_module("src.model.train_disc")
print("[CHECK] Imported 'src.model.train_disc' OK")
PY

# -------- Train (run as module from project root) --------
cd "${SCRATCH_DIR}"

ARGS=(
  --project_dir "${SCRATCH_DIR}"
  --data_root   "${DATA_ROOT}"       # script will build <root>_disc_only/
  --weights     "${WEIGHTS_LOCAL}"   # local yolo12x.pt (offline)
  --family      auto
  --size        x
  --epochs      100
  --imgsz       640
  --batch       16                   # lower if OOM
  --name        "${EXP_NAME}"
  --train       1
  --freeze      5                    # optional
  --amp         true
  # --copy_images                       # optional build switch
  # --drop_empty                        # optional build switch
)

# Show CLI header
python -m src.model.train_disc --help | sed -n '1,60p'

# Launch
srun -u python -m src.model.train_disc "${ARGS[@]}"

BEST="${RUNS_DIR}/${EXP_NAME}/weights/best.pt"
[[ -f "${BEST}" ]] || { echo "[ERR] best.pt not found at ${BEST}" >&2; exit 3; }

echo "[OK] best.pt → ${BEST}"