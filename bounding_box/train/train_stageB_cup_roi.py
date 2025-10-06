#!/usr/bin/env python3
# train_stageB_cup_roi.py
"""
Train a cup-only detector on ROI crops (built by build_cup_roi_dataset.py).

Key features
------------
- Modular, typed config via dataclass
- Clear path resolution + checks
- Train-time augmentations (Ultralytics built-ins)
  * train-only; val/test are never augmented
  * all knobs exposed via CLI (with safe defaults)
- Validates on test if present, else val

Expected layout (defaults; override via CLI):
PROJECT_DIR/
  bounding_box/data/yolo_split_cupROI/   # images/{train,val,test}, labels/{...}, cup_roi.yaml
  weights/yolov8n.pt
  bounding_box/runs/detect/
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from ultralytics import YOLO

# Local helper: returns "0" if CUDA, else "cpu"
from device_utils import ultralytics_device_arg


# ------------------------- Utilities -------------------------

def _expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _require_file(p: Path, desc: str) -> None:
    if not p.exists():
        raise SystemExit(f"[ERR] {desc} not found: {p}")

def _require_dir(p: Path, desc: str) -> None:
    if not p.exists():
        raise SystemExit(f"[ERR] {desc} not found: {p}")


# ------------------------- Config -------------------------

@dataclass
class CupROITrainConfig:
    # Paths
    project_dir: Path
    data_root: Path
    weights: Path
    runs_root: Path
    # Training
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    exp_name: str = "stageB_cup_roi"
    workers: int = 8
    seed: int = 1337
    # Augmentations (Ultralytics built-ins; train-only)
    hsv_h: float = 0.015   # hue gain
    hsv_s: float = 0.7     # saturation gain
    hsv_v: float = 0.4     # value gain
    degrees: float = 10.0  # rotation
    translate: float = 0.10
    scale: float = 0.50    # scale gain range ± (e.g., 0.5 ⇒ 50%)
    shear: float = 2.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 0.50
    mixup: float = 0.10
    copy_paste: float = 0.0
    erasing: float = 0.4   # random erasing prob
    # Optimization
    optimizer: str = "AdamW"
    cos_lr: bool = True
    patience: int = 50
    pretrained: bool = True
    # Misc
    device: Optional[str] = None  # set at runtime


# ------------------------- Core helpers -------------------------

def build_config_from_cli() -> CupROITrainConfig:
    ap = argparse.ArgumentParser(description="Train cup-only YOLO on ROI crops with train-time augmentations.")

    # Paths
    ap.add_argument("--project_dir", default=".", help="Project root.")
    ap.add_argument("--data_root", default=None,
                    help="Cup-ROI YOLO root (defaults to PROJECT_DIR/bounding_box/data/yolo_split_cupROI)")
    ap.add_argument("--model", default=None,
                    help="Weights path (defaults to PROJECT_DIR/weights/yolov8n.pt)")
    ap.add_argument("--project", default=None,
                    help="Ultralytics runs root (defaults to PROJECT_DIR/bounding_box/runs/detect)")

    # Training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="stageB_cup_roi")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)

    # Augmentations (train-only)
    ap.add_argument("--hsv_h", type=float, default=0.015)
    ap.add_argument("--hsv_s", type=float, default=0.70)
    ap.add_argument("--hsv_v", type=float, default=0.40)
    ap.add_argument("--degrees", type=float, default=10.0)
    ap.add_argument("--translate", type=float, default=0.10)
    ap.add_argument("--scale", type=float, default=0.50)
    ap.add_argument("--shear", type=float, default=2.0)
    ap.add_argument("--perspective", type=float, default=0.0)
    ap.add_argument("--flipud", type=float, default=0.0)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--mosaic", type=float, default=0.50)
    ap.add_argument("--mixup", type=float, default=0.10)
    ap.add_argument("--copy_paste", type=float, default=0.0)
    ap.add_argument("--erasing", type=float, default=0.40)

    # Optimization knobs
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--cos_lr", action="store_true", default=True)
    ap.add_argument("--no-cos_lr", dest="cos_lr", action="store_false")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--pretrained", type=int, default=1)

    args = ap.parse_args()

    project_dir = _expand(args.project_dir)
    # Default locations relative to project_dir
    default_data = project_dir / "bounding_box" / "data" / "yolo_split_cupROI"
    default_runs = project_dir / "bounding_box" / "runs" / "detect"
    default_wts  = project_dir / "weights" / "yolov8n.pt"

    data_root = _expand(args.data_root) if args.data_root else default_data
    runs_root = _expand(args.project) if args.project else default_runs
    weights   = _expand(args.model) if args.model else default_wts

    # Basic checks
    _require_dir(data_root, "ROI dataset root")
    _require_file(data_root / "cup_roi.yaml", "cup_roi.yaml")
    _require_file(weights, "model weights")
    _ensure_dir(runs_root)

    return CupROITrainConfig(
        project_dir=project_dir,
        data_root=data_root,
        weights=weights,
        runs_root=runs_root,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        exp_name=args.name,
        workers=args.workers,
        seed=args.seed,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
        optimizer=args.optimizer,
        cos_lr=bool(args.cos_lr),
        patience=args.patience,
        pretrained=bool(args.pretrained),
        device=None,  # set below
    )


def build_model(cfg: CupROITrainConfig) -> YOLO:
    model = YOLO(str(cfg.weights))
    return model


def train(model: YOLO, cfg: CupROITrainConfig, yaml_path: Path) -> None:
    """
    Train with Ultralytics augmentations enabled for the training dataloader only.
    Validation/test remain unaugmented automatically.
    """
    overrides = dict(
        data=str(yaml_path),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        project=str(cfg.runs_root),
        name=cfg.exp_name,
        workers=cfg.workers,
        seed=cfg.seed,
        optimizer=cfg.optimizer,
        cos_lr=cfg.cos_lr,
        patience=cfg.patience,
        pretrained=cfg.pretrained,
        single_cls=True,     # cup-only
        # --- Train-time augs (train loader only) ---
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        shear=cfg.shear,
        perspective=cfg.perspective,
        flipud=cfg.flipud,
        fliplr=cfg.fliplr,
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        copy_paste=cfg.copy_paste,
        erasing=cfg.erasing,
    )
    model.train(**overrides)


def validate(model: YOLO, yaml_path: Path, imgsz: int, device: str) -> None:
    y = yaml.safe_load(yaml_path.read_text())
    if "test" in y:
        model.val(data=str(yaml_path), split="test", imgsz=imgsz, device=device)
    else:
        model.val(data=str(yaml_path), imgsz=imgsz, device=device)


# ------------------------- Main -------------------------

def main() -> None:
    cfg = build_config_from_cli()
    cfg.device = ultralytics_device_arg()  # "0" or "cpu"

    yaml_path = cfg.data_root / "cup_roi.yaml"
    print(f"[INFO] Using dataset YAML: {yaml_path}")
    print(f"[INFO] Runs dir: {cfg.runs_root}")
    print(f"[INFO] Device: {cfg.device}")

    model = build_model(cfg)
    train(model, cfg, yaml_path)
    print("[INFO] Validating…")
    validate(model, yaml_path, cfg.imgsz, cfg.device)
    print("[OK] Done.")


if __name__ == "__main__":
    main()