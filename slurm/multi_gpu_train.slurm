#!/bin/bash
#SBATCH --job-name=bb-train-flex
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
# Default layout below works for BOTH modes:
# - Parallel A&B: 2 tasks, 1 GPU per task (two trainings at once)
# - DDP: we will still request 2 GPUs and use both for one stage (set DDP_STAGE and NUM_DDP_GPUS=2)
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail

# ========================= Mode switches =========================
# Choose ONE of the following:
# 1) Run Stage A and Stage B in PARALLEL (1 GPU each). (Default)
PARALLEL_AB=${PARALLEL_AB:-1}

# 2) Run ONE stage with DDP (multi-GPU) using NUM_DDP_GPUS GPUs.
#    Set DDP_STAGE to "A" or "B". When using DDP, PARALLEL_AB must be 0.
DDP_STAGE=${DDP_STAGE:-""}         # "", "A", or "B"
NUM_DDP_GPUS=${NUM_DDP_GPUS:-2}    # how many GPUs to use for the chosen stage

# (Tip) If you want more speed for DDP, submit with more GPUs:
#   sbatch --export=ALL,PARALLEL_AB=0,DDP_STAGE=B,NUM_DDP_GPUS=2  this_script.sh
# For parallel mode:
#   sbatch --export=ALL,PARALLEL_AB=1 this_script.sh

# ========================= Paths (HPC) =========================
PROJECT_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"

BB_DIR="${PROJECT_DIR}/bounding_box"
RUNS_DIR="${BB_DIR}/runs/detect"

# Datasets
DATA_ROOT="${BB_DIR}/data/yolo_split"          # clean split (val/test)
AUG_ROOT="${BB_DIR}/data/yolo_split_aug"       # augmented split (train for A when building)
ROI_OUT="${BB_DIR}/data/yolo_split_cupROI"     # prebuilt ROI for Stage B
DISC_ROOT="${BB_DIR}/data/yolo_split_disc_only" # prebuilt disc-only dataset (preferred for Stage A)

# Optional two-stage inference output
OUT_JSONL="${BB_DIR}/data/two_stage_boxes.jsonl"

# Experiment names (match your trainers’ defaults unless you override --name)
STAGEA_NAME="stageA_disc_only"
STAGEB_NAME="stageB_cup_roi"

# ========================= Caches / temp =========================
export XDG_CACHE_HOME="/scratch/st-ipor-1/cperez/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TMPDIR="${PROJECT_DIR}/tmp"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"
mkdir -p "${PROJECT_DIR}/logs" "${RUNS_DIR}" "${BB_DIR}/data"

cd "${PROJECT_DIR}"

# ========================= Env =========================
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

echo "SLURM Job ID : ${SLURM_JOB_ID}"
echo "Node list    : ${SLURM_JOB_NODELIST}"
echo "Working dir  : $(pwd)"
echo "Runs dir     : ${RUNS_DIR}"
nvidia-smi -L || echo "nvidia-smi not available"

python - <<'PY'
import torch, ultralytics
print(f"[ENV] torch {torch.__version__} | cuda={torch.cuda.is_available()} | gpus={torch.cuda.device_count()}")
print(f"[ENV] ultralytics {ultralytics.__version__}")
PY

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

# ========================= Helpers =========================
die() { echo "[ERROR] $*" >&2; exit 1; }
check_file() { [[ -f "$1" ]] || die "$2 not found: $1"; }
check_dir()  { [[ -d "$1" ]] || die "$2 not found: $1"; }

make_devices_csv() {
  # Build "0,1,...,N-1" for given N (default: NUM_DDP_GPUS)
  local n="${1:-$NUM_DDP_GPUS}"
  if [[ "$n" -le 1 ]]; then echo "0"; return; fi
  local out="0"
  for ((i=1;i<n;i++)); do out+=",${i}"; done
  echo "$out"
}

# ========================= Stage A =========================
run_stageA() {
  echo "[STAGE A] Train disc-only detector…"

  if [[ -d "${DISC_ROOT}" && -f "${DISC_ROOT}/od_only.yaml" ]]; then
    echo "[STAGE A] Using PREBUILT disc-only dataset at ${DISC_ROOT}"
    python "${BB_DIR}/train/train_stageA_disc_only.py" \
      --project_dir "${PROJECT_DIR}" \
      --data_root   "${DISC_ROOT}" \
      --model       "${BB_DIR}/weights/yolov8n.pt"
  else
    echo "[STAGE A] Prebuilt disc-only dataset NOT found, falling back to clean+aug roots."
    check_dir "${DATA_ROOT}" "Clean YOLO split"
    check_dir "${AUG_ROOT}"  "Augmented YOLO split"
    python "${BB_DIR}/train/train_stageA_disc_only.py" \
      --project_dir "${PROJECT_DIR}" \
      --data_root   "${DATA_ROOT}" \
      --aug_root    "${AUG_ROOT}" \
      --train_splits train \
      --model       "${BB_DIR}/weights/yolov8n.pt"
  fi

  local best="${RUNS_DIR}/${STAGEA_NAME}/weights/best.pt"
  check_file "${best}" "Stage A best checkpoint"
  echo "[STAGE A] Best checkpoint: ${best}"
}

# ========================= Stage B =========================
run_stageB() {
  echo "[STAGE B] Train cup-only detector on prebuilt ROI dataset…"
  check_dir "${ROI_OUT}" "ROI dataset"

  python "${BB_DIR}/train/train_stageB_cup_roi.py" \
    --project_dir "${PROJECT_DIR}"

  local best="${RUNS_DIR}/${STAGEB_NAME}/weights/best.pt"
  check_file "${best}" "Stage B best checkpoint"
  echo "[STAGE B] Best checkpoint: ${best}"
}

export -f run_stageA run_stageB die check_file check_dir make_devices_csv

# ========================= Dispatcher =========================
if [[ "${PARALLEL_AB}" == "1" && -z "${DDP_STAGE}" ]]; then
  echo "[MODE] PARALLEL: launching Stage A and Stage B concurrently (1 GPU each)"
  # two tasks, each gets its own GPU via SLURM binding
  srun --ntasks=1 --gpus-per-task=1 bash -lc 'run_stageA' &
  srun --ntasks=1 --gpus-per-task=1 bash -lc 'run_stageB' &
  wait
elif [[ -n "${DDP_STAGE}" && "${PARALLEL_AB}" == "0" ]]; then
  # Multi-GPU mode for ONE stage (NUM_DDP_GPUS GPUs)
  DEVICES="$(make_devices_csv "${NUM_DDP_GPUS}")"
  export YOLO_DEVICES="${DEVICES}"
  echo "[MODE] DDP: training Stage ${DDP_STAGE} on devices: ${YOLO_DEVICES}"

  if [[ "${DDP_STAGE}" == "A" ]]; then
    run_stageA
  elif [[ "${DDP_STAGE}" == "B" ]]; then
    run_stageB
  else
    die "DDP_STAGE must be 'A' or 'B' (got '${DDP_STAGE}')"
  fi
else
  echo "[MODE] SEQUENTIAL: running Stage B only (default path here)"
  # change below if you prefer Stage A only when no mode is set
  run_stageB
fi

# ========================= (Optional) Inference =========================
run_infer_optional() {
  local test_images="${DATA_ROOT}/images/test"
  if [[ -d "${test_images}" ]]; then
    echo "[INFER] Two-stage inference on clean test images…"
    local bestA="${RUNS_DIR}/${STAGEA_NAME}/weights/best.pt"
    local bestB="${RUNS_DIR}/${STAGEB_NAME}/weights/best.pt"
    check_file "${bestA}" "Stage A best checkpoint"
    check_file "${bestB}" "Stage B best checkpoint"

    python "${BB_DIR}/train/infer_two_stage_to_boxes.py" \
      --stageA "${bestA}" \
      --stageB "${bestB}" \
      --images "${test_images}" \
      --out_jsonl "${OUT_JSONL}" \
      --od_pad_pct 0.08 \
      --confA 0.25 \
      --confB 0.10
    echo "[INFER] Wrote: ${OUT_JSONL}"
  else
    echo "[WARN] No clean test split at ${test_images}; skipping inference."
  fi
}

# Uncomment if you want an automatic inference after training
# run_infer_optional

echo "[DONE] Training finished at $(date)"