#!/usr/bin/env python3
# train_cup_roi.py
import argparse
from pathlib import Path
from ultralytics import YOLO
from bounding_box.train.device_utils import ultralytics_device_arg

"""
Train a cup-only detector on disc-ROI crops produced by build_cup_roi_dataset.py.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./papila_yolo_cupROI")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="stageB_cup_roi")
    args = ap.parse_args()

    yaml_path = Path(args.data_root)/"cup_roi.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"Missing {yaml_path}. Run build_cup_roi_dataset.py first.")

    device = ultralytics_device_arg()
    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        cos_lr=True,
        optimizer="adamw",
        pretrained=True,
        patience=50
    )
    # Optional test split
    try:
        model.val(data=str(yaml_path), split="test", imgsz=args.imgsz, device=device)
    except Exception:
        pass

if __name__ == "__main__":
    main()