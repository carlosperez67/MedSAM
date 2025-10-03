#!/bin/bash
#SBATCH --job-name=bb-stage_A-val
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --gpus=1
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca

set -euo pipefail

# --- Paths (adjust if your layout changes) ---
SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"

BB_DIR="${SCRATCH_DIR}/bounding_box"
RUNS_DIR="${BB_DIR}/runs/detect"
DATA_ROOT="${BB_DIR}/papila_yolo"                 # original split dataset
DISC_ONLY_ROOT="${DATA_ROOT}_disc_only"           # created by Stage-A training script
YAML_PATH="${DISC_ONLY_ROOT}/od_only.yaml"
STAGEA_NAME="stageA_disc_only"
MODEL_PATH="${RUNS_DIR}/${STAGEA_NAME}/weights/best.pt"

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node(s)     : $SLURM_JOB_NODELIST"
echo "Start       : $(date)"
echo "SCRATCH_DIR : ${SCRATCH_DIR}"
echo "BB_DIR      : ${BB_DIR}"

echo "Visible GPUs (from nvidia-smi):"
nvidia-smi -L || echo "nvidia-smi not available"

# --- Activate your conda env (offline safe) ---
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Optional: writable caches on compute node
export XDG_CACHE_HOME="/scratch/st-ipor-1/cperez/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

# Sanity checks
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[ERROR] Missing model weights at ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${YAML_PATH}" ]]; then
  echo "[ERROR] Missing dataset yaml at ${YAML_PATH} (did you run Stage-A to create *_disc_only?)" >&2
  exit 3
fi

# --- Validation only (no training) ---
python - <<'PY'
import os, yaml
from ultralytics import YOLO
import torch

yaml_path = os.environ["YAML_PATH"]
model_path = os.environ["MODEL_PATH"]

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device={device}")
print(f"[INFO] Model: {model_path}")
print(f"[INFO] Data : {yaml_path}")

with open(yaml_path, "r") as f:
    y = yaml.safe_load(f)

model = YOLO(model_path)

if "test" in y:
    print("[VAL] Running on TEST split…")
    results = model.val(data=yaml_path, split="test", imgsz=640, device=device)
else:
    print("[VAL] TEST split not found; running on VAL split…")
    results = model.val(data=yaml_path, imgsz=640, device=device)

print("[VAL] Done. Metrics summary:")
try:
    # Ultralytics returns a Results object; printing will show a brief summary.
    print(results)
except Exception:
    pass
PY