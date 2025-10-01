#!/usr/bin/env python3
# train_stageB_cup_roi.py
import argparse, os
from pathlib import Path
from ultralytics import YOLO
from device_utils import ultralytics_device_arg

"""
Train a cup-only detector on ROI crops produced by build_cup_roi_dataset.py.

Project layout (defaults; can be overridden):
  /Users/carlosperez/PycharmProjects/MedSAM/bounding_box/
    data/yolo_split_cupROI/     # input dataset (images/{train,val,test}, labels/{...}, cup_roi.yaml)
    weights/yolov8n.pt          # local weights (must exist, no internet download)
    runs/detect/                # Ultralytics output root
    train/train_stageB_cup_roi.py
"""

def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project_dir",
        default="/Users/carlosperez/PycharmProjects/MedSAM/bounding_box",
        help="Project root directory"
    )
    ap.add_argument(
        "--data_root",
        default=None,
        help="Cup-ROI YOLO dataset root (defaults to PROJECT_DIR/data/yolo_split_cupROI)"
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Path to weights (defaults to PROJECT_DIR/weights/yolov8n.pt)"
    )
    ap.add_argument(
        "--project",
        default=None,
        help="Ultralytics runs root (defaults to PROJECT_DIR/runs/detect)"
    )

    # training args
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="stageB_cup_roi")
    args = ap.parse_args()

    PROJECT_DIR = _expand(args.project_dir)
    data_root = _expand(args.data_root) if args.data_root else (PROJECT_DIR / "data" / "yolo_split_cupROI")
    model_path = _expand(args.model) if args.model else (PROJECT_DIR / "weights" / "yolov8n.pt")
    runs_root = _expand(args.project) if args.project else (PROJECT_DIR / "runs" / "detect")

    yaml_path = data_root / "cup_roi.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"[ERR] Missing {yaml_path}. Run build_cup_roi_dataset.py first.")

    if not model_path.exists():
        raise SystemExit(f"[ERR] Model weights not found: {model_path}. Put yolov8n.pt there or pass --model /abs/path.pt")

    runs_root.mkdir(parents=True, exist_ok=True)

    device = ultralytics_device_arg()
    model = YOLO(str(model_path))

    # training
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(runs_root),
        name=args.name,
        cos_lr=True,
        optimizer="AdamW",
        pretrained=True,
        patience=50
    )

    # validation (prefer test if exists)
    import yaml
    y = yaml.safe_load(yaml_path.read_text())
    if "test" in y:
        model.val(data=str(yaml_path), split="test", imgsz=args.imgsz, device=device)
    else:
        model.val(data=str(yaml_path), imgsz=args.imgsz, device=device)

if __name__ == "__main__":
    main()