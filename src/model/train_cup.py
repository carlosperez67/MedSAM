#!/usr/bin/env python3
# train_stageB_cup_roi_modern.py
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from ultralytics import YOLO

# --- tiny utils ---
def _expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _need(p: Path, what: str) -> None:
    if not p.exists():
        raise SystemExit(f"[ERR] {what} not found: {p}")

def _auto_device() -> str:
    # returns "0" if CUDA visible else "cpu" (simple heuristic)
    import torch
    if torch.cuda.is_available(): return "0"
    try:
        if torch.backends.mps.is_available(): return "mps"  # mac
    except Exception:
        pass
    return "cpu"

# --- config ---
@dataclass
class CupROITrainCfg:
    # required paths
    data_root: Path                  # should contain cup_roi.yaml + images/labels
    runs_root: Path
    # model selection
    amp: bool = True
    accumulate: int = 1  # gradient accumulation steps
    freeze: int = 0  # freeze first N layers (0 = none)
    recompute: bool = False  # if your Ultralytics build supports gradient checkpointing
    weights: Optional[str] = None    # e.g., /path/best.pt or yolo12x.pt
    family: str = "auto"             # auto|yolo12|yolo11|yolov8
    size: str = "x"                  # n|s|m|l|x

    # training knobs
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    name: str = "stageB_cup_roi_modern"
    workers: int = 8
    seed: int = 1337
    optimizer: str = "AdamW"
    cos_lr: bool = True
    patience: int = 50
    pretrained: bool = True

    # augs (train loader only; Ultralytics built-ins)
    hsv_h: float = 0.015
    hsv_s: float = 0.70
    hsv_v: float = 0.40
    degrees: float = 10.0
    translate: float = 0.10
    scale: float = 0.50
    shear: float = 2.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 0.50
    mixup: float = 0.10
    copy_paste: float = 0.0
    erasing: float = 0.40

    device: Optional[str] = None     # "0"|"cpu"|"mps" etc.

# --- trainer ---
class CupROITrainer:
    def __init__(self, cfg: CupROITrainCfg):
        self.cfg = cfg
        self.yaml_path = cfg.data_root / "data.yaml"
        _need(cfg.data_root, "ROI dataset root")
        _need(self.yaml_path, "data.yaml")
        _ensure_dir(cfg.runs_root)
        self.device = cfg.device or _auto_device()
        self.model = YOLO(self._resolve_weights())

    # prefer newest, biggest model by name
    def _resolve_weights(self) -> str:
        if self.cfg.weights:
            return str(_expand(self.cfg.weights))
        fam = (self.cfg.family or "auto").lower()
        size = (self.cfg.size or "x").lower()
        # attempt order by recency -> stability
        if fam in ("auto", "yolo12"):
            return f"yolo12{size}.pt"   # will download if supported
        if fam in ("yolo11",):
            return f"yolo11{size}.pt"
        # conservative fallback
        return f"yolov8{size}.pt"

    def train(self) -> None:
        c = self.cfg
        overrides = dict(
            data=str(self.yaml_path),
            epochs=c.epochs,
            imgsz=c.imgsz,
            batch=c.batch,
            device=self.device,
            project=str(c.runs_root),
            name=c.name,
            workers=c.workers,
            seed=c.seed,
            optimizer=c.optimizer,
            cos_lr=c.cos_lr,
            patience=c.patience,
            pretrained=c.pretrained,   # use pretrained backbone/heads, fine-tune on single class
            single_cls=True,           # cup-only
            amp = c.amp,
            accumulate = c.accumulate,
            freeze = c.freeze,
            recompute = c.recompute,
            # train-time augs
            hsv_h=c.hsv_h, hsv_s=c.hsv_s, hsv_v=c.hsv_v,
            degrees=c.degrees, translate=c.translate, scale=c.scale,
            shear=c.shear, perspective=c.perspective,
            flipud=c.flipud, fliplr=c.fliplr,
            mosaic=c.mosaic, mixup=c.mixup, copy_paste=c.copy_paste, erasing=c.erasing,
        )
        print(f"[INFO] Using weights: {self.model.ckpt_path or 'hub-pretrained'}")
        print(f"[INFO] Device: {self.device}")
        self.model.train(**overrides)

    def validate(self) -> None:
        y = yaml.safe_load(self.yaml_path.read_text())
        split = "test" if "test" in y else "val"
        self.model.val(data=str(self.yaml_path), split=split, imgsz=self.cfg.imgsz, device=self.device)

# --- CLI ---
def parse_args() -> CupROITrainCfg:
    ap = argparse.ArgumentParser(description="Train a cup-only detector on ROI crops with the newest YOLO family.")
    ap.add_argument("--data-root", default=None,
                    help="Root containing cup_roi.yaml and images/labels. Default: ./bounding_box/data/yolo_split_cupROI")
    ap.add_argument("--runs-root", default=None,
                    help="Ultralytics runs root. Default: ./bounding_box/runs/detect")
    ap.add_argument("--weights", default=None,
                    help="Explicit weights path or hub tag (e.g., yolo12x.pt). If omitted, picks largest available.")
    ap.add_argument("--family", default="auto", choices=["auto","yolo12","yolo11","yolov8"])
    ap.add_argument("--size", default="x", choices=["n","s","m","l","x"])

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accumulate", type=int, default=1)
    ap.add_argument("--freeze", type=int, default=0)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--amp", type=lambda v: str(v).lower() not in {"0", "false", "no"}, default=True)
    ap.add_argument("--name", default="stageB_cup_roi_modern")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)

    # augs
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

    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--cos-lr", action="store_true", default=True)
    ap.add_argument("--no-cos-lr", dest="cos_lr", action="store_false")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--device", default=None)

    args = ap.parse_args()

    project = Path(".").resolve()
    default_data = project / "bounding_box" / "data" / "yolo_split_cupROI"
    default_runs = project / "bounding_box" / "runs" / "detect"

    data_root = _expand(args.data_root) if args.data_root else default_data
    runs_root = _expand(args.runs_root) if args.runs_root else default_runs

    _need(data_root, "ROI dataset root")
    _need(data_root / "data.yaml", "data.yaml")
    _ensure_dir(runs_root)

    return CupROITrainCfg(
        data_root=data_root,
        runs_root=runs_root,
        weights=args.weights,
        family=args.family,
        size=args.size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        accumulate=args.accumulate, freeze = args.freeze,
        recompute = bool(args.recompute), amp = bool(args.amp),
        mosaic = args.mosaic,
        name=args.name,
        workers=args.workers,
        seed=args.seed,
        hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
        degrees=args.degrees, translate=args.translate, scale=args.scale,
        shear=args.shear, perspective=args.perspective,
        flipud=args.flipud, fliplr=args.fliplr,
        mixup=args.mixup, copy_paste=args.copy_paste, erasing=args.erasing,
        optimizer=args.optimizer, cos_lr=bool(args.cos_lr),
        patience=args.patience, pretrained=bool(args.pretrained),
        device=args.device,
    )

def main():
    cfg = parse_args()
    trainer = CupROITrainer(cfg)
    trainer.train()
    print("[INFO] Validating…")
    trainer.validate()
    print("[OK] Done.")

if __name__ == "__main__":
    main()