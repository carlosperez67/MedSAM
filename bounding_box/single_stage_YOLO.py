#!/usr/bin/env python3
# train_yolo_odoc.py
"""
Train a single-stage object detector for OD/OC (YOLO family, 2 classes).

Prereqs
-------
Run the splitter first so you have:
DATASET_ROOT/
  images/{train,val,test}
  labels/{train,val,test}

Defaults
--------
- DATASET_ROOT = ./papila_yolo
- classes      = ["disc", "cup"]
- model        = yolov8n.pt (changeable via CLI)

Usage
-----
python train_yolo_odoc.py \
  --data_root ./papila_yolo \
  --model yolov8n.pt \
  --epochs 100 --imgsz 640 \
  --batch 16 \
  --device auto

For inference:
  yolo detect predict model=./runs/detect/train/weights/best.pt source=./papila_yolo/images/test conf=0.1 iou=0.5
"""

import argparse
import os
from pathlib import Path

import torch
import yaml


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./papila_yolo", help="Root with images/ and labels/ splits")
    ap.add_argument("--names", nargs="+", default=["disc","cup"], help="Class names order (0,1)")
    ap.add_argument("--model", default="yolov8n.pt", help="Ultralytics model to start from")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.01, help="Initial LR (optional)")
    ap.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    ap.add_argument("--project", default="runs/detect", help="Ultralytics project folder")
    ap.add_argument("--name", default="train", help="Run name")
    return ap.parse_args()

def main():
    args = parse_args()
    from ultralytics import YOLO

    data_root = Path(args.data_root)
    for sub in ["images/train","images/val","labels/train","labels/val"]:
        if not (data_root / sub).exists():
            raise SystemExit(f"Missing dataset subdir: {data_root/sub}")

    # Prepare data YAML
    data_yaml = {
        "path": str(data_root.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "names": args.names
    }
    yaml_path = data_root / "odoc.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    print(f"[OK] Wrote data yaml: {yaml_path}")

    # Load model
    model = YOLO(args.model)   # e.g., 'yolov8n.pt'

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # Train
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        device=device,
        project=args.project,
        name=args.name,
        # reasonable defaults for medical small objects:
        cos_lr=True,
        optimizer="AdamW",
        pretrained=True
    )

    # Validate on test split if present
    test_images = data_root / "images/test"
    test_labels = data_root / "labels/test"
    if test_images.exists() and test_labels.exists():
        results = model.val(data=str(yaml_path), split="test", imgsz=args.imgsz, device=args.device)
        print("[OK] Test metrics:", results)

if __name__ == "__main__":
    main()