#!/bin/bash
#SBATCH --job-name=cup-roi-y12x
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

# ---------------- Paths (adjust if your layout differs) ----------------
SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"   # your conda env

BB_DIR="${SCRATCH_DIR}/bounding_box"
TRAIN_SCRIPT="${SCRATCH_DIR}/src/model/train_cup"   # modern trainer
RUNS_DIR="${SCRATCH_DIR}/runs/detect"

# ROI dataset root (must contain images/, labels/, and data.yaml or cup_roi.yaml)
ROI_ROOT="${BB_DIR}/data/yolo_split_cupROI"

# Offline weights (already downloaded to HPC)
WEIGHTS_LOCAL="${BB_DIR}/weights/yolo12x.pt"   # <- your local YOLO12X weight file

# Ultralytics run name
EXP_NAME="stageB_cup_roi_y12x"

# ---------------- Create dirs ----------------
mkdir -p "${SCRATCH_DIR}/logs" "${RUNS_DIR}"

# ---------------- Conda env ----------------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# ---------------- Keep everything offline / quiet ----------------
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export ULTRALYTICS_HUB=False
export ULTRALYTICS_ANALYTICS=False

# Caches & temp dirs (local to scratch)
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TMPDIR="${SCRATCH_DIR}/tmp"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"

# Threading
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
# Reduce CUDA alloc fragmentation on some drivers
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "SLURM Job ID : $SLURM_JOB_ID"
echo "Nodes        : $SLURM_JOB_NODELIST"
echo "Start        : $(date)"
echo "Project Dir  : $SCRATCH_DIR"
echo "Runs Dir     : $RUNS_DIR"
echo "ROI Root     : $ROI_ROOT"
echo "Weights      : $WEIGHTS_LOCAL"

echo "Visible GPUs:"
nvidia-smi -L || echo "nvidia-smi not available"

python - <<'PY'
import torch, ultralytics, os
print(f"[ENV] torch {torch.__version__} | cuda={torch.cuda.is_available()} | gpus={torch.cuda.device_count()}")
print(f"[ENV] ultralytics {ultralytics.__version__}")
PY

# ---------------- Sanity checks ----------------
[[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERR] Trainer not found: ${TRAIN_SCRIPT}" >&2; exit 2; }
[[ -d "${ROI_ROOT}" ]]     || { echo "[ERR] ROI root not found: ${ROI_ROOT}" >&2; exit 2; }
[[ -f "${WEIGHTS_LOCAL}" ]]|| { echo "[ERR] Local weights not found: ${WEIGHTS_LOCAL}" >&2; exit 2; }

# Ensure dataset YAML is named data.yaml (modern trainer expects 'data.yaml')
if [[ -f "${ROI_ROOT}/cup_roi.yaml" && ! -f "${ROI_ROOT}/data.yaml" ]]; then
  echo "[INFO] Linking ${ROI_ROOT}/cup_roi.yaml -> ${ROI_ROOT}/data.yaml"
  ln -sf "${ROI_ROOT}/cup_roi.yaml" "${ROI_ROOT}/data.yaml"
fi
[[ -f "${ROI_ROOT}/data.yaml" ]] || { echo "[ERR] Missing ${ROI_ROOT}/data.yaml (or cup_roi.yaml)" >&2; exit 2; }

# Quick offline load test (fail early if yolo12x.pt is incompatible with ultralytics in this env)
python - <<PY
from ultralytics import YOLO
w = r"${WEIGHTS_LOCAL}"
try:
    YOLO(w)
    print("[CHECK] Successfully loaded weights:", w)
except Exception as e:
    raise SystemExit(f"[ERR] Failed to load {w}: {e}")
PY

# ---------------- Train (Stage B) ----------------
echo "[STAGE B] Training cup-only detector on ROIs with YOLO12X (offline)…"

# Notes:
# - --weights: absolute path to local yolo12x.pt (offline)
# - --family auto and --size x are harmless here since --weights is explicit
# - Tune --batch / --imgsz if you see OOM on your GPU
srun -u python "${TRAIN_SCRIPT}" \
  --data-root "${ROI_ROOT}" \
  --runs-root "${RUNS_DIR}" \
  --weights   "${WEIGHTS_LOCAL}" \
  --family    auto \
  --size      x \
  --epochs    100 \
  --imgsz     640 \
  --batch     32 \
  --name      "${EXP_NAME}" \
  --workers   "${SLURM_CPUS_PER_TASK}"

BEST="${RUNS_DIR}/${EXP_NAME}/weights/best.pt"
if [[ ! -f "${BEST}" ]]; then
  echo "[ERROR] Training finished but best.pt not found at ${BEST}" >&2
  exit 3
fi

echo "[OK] best.pt → ${BEST}"
echo "Done: $(date)"