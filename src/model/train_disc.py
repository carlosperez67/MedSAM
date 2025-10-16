#!/usr/bin/env python3
# train_disc_only.py
"""
OOP refactor: Train a disc-only detector from a 2-class YOLO dataset (0=disc, 1=cup).

Additions
---------
- Modern model selection (YOLOv12/YOLOv11/YOLOv8):
  * --weights /path/to/weights.pt  OR a hub tag like yolo12x.pt
  * --family {auto,yolo12,yolo11,yolov8} with --size {n,s,m,l,x}
- AMP and layer freezing:
  * --amp true/false, --freeze N
- Backward compatible with --model (mapped to --weights).

Behavior
--------
- If <data_root>/data.yaml exists, use it as a precomputed disc-only dataset.
- Otherwise derive <data_root>_disc_only/ by filtering labels to class 0 (disc).
- Optional resume:
  * --resume auto         -> runs/<name>/weights/last.pt
  * --resume /abs/path/last.pt
- Validate on 'test' if present, else 'val'.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import yaml
from ultralytics import YOLO

from src.utils import ultralytics_device_arg, place, split_has_data, ensure_dir

# ------------------------------
# tiny local utils
# ------------------------------
def _expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()

def _looks_like_tag(s: str) -> bool:
    # crude check for things like "yolo12x.pt", "yolo11l.pt", "yolov8s.pt"
    s = s.lower()
    return s.endswith(".pt") and ("yolo" in s)

# ============================================================
# Config & simple utilities
# ============================================================

@dataclass
class DiscOnlyConfig:
    # core paths
    project_dir: Path
    data_root: Path                   # base YOLO root OR a precomputed disc-only root
    runs_root: Path

    # model selection
    weights: Optional[str]            # explicit path or hub tag (yolo12x.pt)
    family: str                       # auto|yolo12|yolo11|yolov8
    size: str                         # n|s|m|l|x
    amp: bool                         # AMP on/off
    freeze: int                       # freeze first N layers (0 = none)

    # optional augmented source for train split(s)
    aug_root: Optional[Path]
    train_splits: List[str]

    # building options
    copy_images: bool
    drop_empty: bool

    # training
    epochs: int
    imgsz: int
    batch: int
    exp_name: str
    do_train: bool

    # resume
    resume: Optional[str]             # "", "auto", or path to last.pt


class DiscOnlyDatasetBuilder:
    """Builds (or reuses) a disc-only YOLO dataset and returns (root, data_yaml_path)."""

    DISC_YAML_NAME = "data.yaml"  # expected filename inside disc-only root

    @staticmethod
    def disc_yaml_in(root: Path) -> Path:
        return root / DiscOnlyDatasetBuilder.DISC_YAML_NAME

    @staticmethod
    def has_precomputed(root: Path) -> bool:
        return DiscOnlyDatasetBuilder.disc_yaml_in(root).exists()

    @staticmethod
    def _filter_label_lines_to_disc(lines: Iterable[str]) -> List[str]:
        keep: List[str] = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            try:
                cls = int(float(parts[0]))
            except Exception:
                continue
            if cls == 0:
                keep.append(ln)
        return keep

    @classmethod
    def _filter_labels_dir_to_disc(
        cls, src_lbl_dir: Path, dst_lbl_dir: Path, drop_empty: bool
    ) -> int:
        ensure_dir(dst_lbl_dir)
        written = 0
        for lbl in sorted(src_lbl_dir.glob("*.txt")):
            lines_in = [ln for ln in lbl.read_text().splitlines()]
            kept = cls._filter_label_lines_to_disc(lines_in)
            if not kept and drop_empty:
                continue
            (dst_lbl_dir / lbl.name).write_text(
                "\n".join(kept) + ("\n" if kept else "")
            )
            written += 1
        return written

    @classmethod
    def build(
        cls,
        base_root: Path,
        copy_images: bool,
        drop_empty: bool,
        aug_root: Optional[Path] = None,
        train_splits: List[str] = ("train",),
    ) -> Tuple[Path, Path]:
        """
        Creates a disc-only dataset next to base_root:
          dst_root = base_root.parent / f"{base_root.name}_disc_only"
        For splits in train_splits (when aug_root provided), source from aug_root; otherwise use base_root.
        Returns (dst_root, data_yaml_path)
        """
        dst_root = base_root.parent / f"{base_root.name}_disc_only"
        ensure_dir(dst_root)

        for split in ("train", "val", "test"):
            src_root = aug_root if (aug_root is not None and split in set(train_splits)) else base_root
            src_img_dir = src_root / "images" / split
            src_lbl_dir = src_root / "labels" / split
            if not src_img_dir.exists() or not src_lbl_dir.exists():
                continue

            # images
            for img in sorted(src_img_dir.glob("*")):
                place(img, dst_root / "images" / split / img.name, copy_images)

            # labels -> only disc
            cls._filter_labels_dir_to_disc(src_lbl_dir, dst_root / "labels" / split, drop_empty)

        # dataset YAML
        data_yaml = {
            "path": str(dst_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": ["disc"],
        }
        if split_has_data(dst_root, "test"):
            data_yaml["test"] = "images/test"

        yaml_path = cls.disc_yaml_in(dst_root)
        yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False))
        return dst_root, yaml_path


# ============================================================
# Training / Validation encapsulation
# ============================================================

class YoloTrainer:
    """Handles model construction, resume logic, training, and validation."""

    def __init__(self, cfg: DiscOnlyConfig) -> None:
        self.cfg = cfg
        self.device = ultralytics_device_arg()  # "0" or "cpu" (DDP via env if multi-GPU)
        self.model: YOLO | None = None
        self.resuming: bool = False

    def _find_last_ckpt(self) -> Optional[Path]:
        p = self.cfg.runs_root / self.cfg.exp_name / "weights" / "last.pt"
        return p if p.exists() else None

    def _resolve_weights_tag(self) -> str:
        """Pick hub tag based on family/size (largest by default)."""
        fam = (self.cfg.family or "auto").lower()
        size = (self.cfg.size or "x").lower()
        if fam in ("auto", "yolo12"):
            return f"yolo12{size}.pt"
        if fam == "yolo11":
            return f"yolo11{size}.pt"
        return f"yolov8{size}.pt"

    def _build_from_weights(self, weights_spec: str) -> YOLO:
        """Accepts either a local path or a hub tag."""
        p = Path(weights_spec)
        if p.exists():
            return YOLO(str(p))
        # If it's not a local file, treat as tag (requires internet unless you’ve pre-downloaded to CWD)
        try:
            return YOLO(weights_spec)
        except Exception as e:
            raise SystemExit(
                f"[ERR] Could not load weights '{weights_spec}'. "
                f"If you are offline, pass a LOCAL .pt path. Underlying error: {e}"
            )

    def prepare_model(self) -> None:
        """Apply resume policy or start from provided/selected weights."""
        last_ckpt: Optional[Path] = None
        if self.cfg.resume:
            if str(self.cfg.resume).lower() == "auto":
                last_ckpt = self._find_last_ckpt()
            else:
                last_ckpt = _expand(self.cfg.resume)

        if last_ckpt and last_ckpt.exists():
            print(f"[INFO] Resuming from: {last_ckpt}")
            self.model = YOLO(str(last_ckpt))
            self.resuming = True
        else:
            if self.cfg.resume:
                print(f"[WARN] --resume requested but checkpoint not found. "
                      f"Starting fresh. (Searched: {last_ckpt})")
            weights_spec = (
                self.cfg.weights if self.cfg.weights
                else self._resolve_weights_tag()
            )
            print(f"[INFO] Starting from weights: {weights_spec}")
            self.model = self._build_from_weights(weights_spec)
            self.resuming = False

    def train(self, data_yaml: Path) -> None:
        if not self.cfg.do_train:
            return
        assert self.model is not None
        print(f"[INFO] Training… (epochs={self.cfg.epochs}, imgsz={self.cfg.imgsz}, "
              f"batch={self.cfg.batch}, resume={self.resuming}, amp={self.cfg.amp}, freeze={self.cfg.freeze})")
        self.model.train(
            data=str(data_yaml),
            epochs=self.cfg.epochs,
            imgsz=self.cfg.imgsz,
            batch=self.cfg.batch,
            device=self.device,
            project=str(self.cfg.runs_root),
            name=self.cfg.exp_name,
            cos_lr=True,
            optimizer="AdamW",
            pretrained=not self.resuming,   # avoid re-init when resuming
            patience=50,
            single_cls=True,                # disc-only
            resume=self.resuming,
            amp=self.cfg.amp,
            freeze=self.cfg.freeze,
        )

    def validate(self, data_yaml: Path) -> None:
        assert self.model is not None
        print("[INFO] Validating…")
        y = yaml.safe_load(data_yaml.read_text())
        if "test" in y:
            self.model.val(data=str(data_yaml), split="test", imgsz=self.cfg.imgsz, device=self.device)
        else:
            self.model.val(data=str(data_yaml), imgsz=self.cfg.imgsz, device=self.device)
        print("[OK] Done.")


# ============================================================
# Orchestrator
# ============================================================

class DiscOnlyPipeline:
    """High-level pipeline: dataset resolve/build → train → validate."""

    def __init__(self, cfg: DiscOnlyConfig) -> None:
        self.cfg = cfg
        self.data_yaml: Optional[Path] = None

    def _assert_yolo_root(self, root: Path, name: str) -> None:
        if not (root / "images").exists() or not (root / "labels").exists():
            raise SystemExit(f"[ERR] Not a YOLO dataset root ({name}): {root} (missing images/ or labels/)")

    def resolve_dataset(self) -> Path:
        """
        Returns the disc-only dataset root and sets self.data_yaml.
        If cfg.data_root already has data.yaml, use it; else build derived dataset.
        """
        # Basic sanity on provided base data_root
        self._assert_yolo_root(self.cfg.data_root, "data_root")

        # Optional sanity on aug_root
        if self.cfg.aug_root is not None:
            self._assert_yolo_root(self.cfg.aug_root, "aug_root")

        # Use precomputed if present
        if DiscOnlyDatasetBuilder.has_precomputed(self.cfg.data_root):
            od_yaml = DiscOnlyDatasetBuilder.disc_yaml_in(self.cfg.data_root)
            self.data_yaml = od_yaml
            print(f"[INFO] Using precomputed disc-only dataset: {self.cfg.data_root}")
            print(f"[INFO] Dataset YAML: {od_yaml}")
            return self.cfg.data_root

        # Otherwise build derived dataset
        print(f"[INFO] Building disc-only dataset from: {self.cfg.data_root}")
        out_root, od_yaml = DiscOnlyDatasetBuilder.build(
            base_root=self.cfg.data_root,
            copy_images=self.cfg.copy_images,
            drop_empty=self.cfg.drop_empty,
            aug_root=self.cfg.aug_root,
            train_splits=list(self.cfg.train_splits),
        )
        self.data_yaml = od_yaml
        print(f"[OK] Disc-only dataset: {out_root}")
        print(f"[OK] Dataset YAML:      {od_yaml}")
        return out_root

    def run(self) -> None:
        dataset_root = self.resolve_dataset()
        assert self.data_yaml is not None and dataset_root.exists()

        trainer = YoloTrainer(self.cfg)
        trainer.prepare_model()
        trainer.train(self.data_yaml)
        trainer.validate(self.data_yaml)


# ============================================================
# CLI
# ============================================================

def _parse_train_splits(val: str | List[str]) -> List[str]:
    if isinstance(val, list):
        return [str(s).strip() for s in val if str(s).strip()]
    return [s.strip() for s in str(val).split(",") if s.strip()]

def build_config_from_cli() -> DiscOnlyConfig:
    ap = argparse.ArgumentParser(
        description="Train a disc-only detector from a 2-class YOLO dataset (0=disc,1=cup)."
    )

    ap.add_argument("--project_dir", default=".", help="Project root.")
    ap.add_argument("--data_root", default=None,
                    help="YOLO dataset root. If it already contains data.yaml (disc-only), "
                         "it is used directly; otherwise a derived '<data_root>_disc_only' is built.")
    ap.add_argument("--aug_root", default=None,
                    help="Augmented YOLO dataset root (used ONLY for the splits listed in --train_splits).")
    ap.add_argument("--train_splits", default="train",
                    help="Comma list or YAML-style list of splits to pull from aug_root (default: 'train').")

    # Model selection (NEW)
    ap.add_argument("--weights", default=None,
                    help="Explicit weights path or hub tag (e.g., yolo12x.pt). If omitted, selects by --family/--size.")
    ap.add_argument("--family", default="auto", choices=["auto","yolo12","yolo11","yolov8"],
                    help="Model family to auto-select from when --weights is not given.")
    ap.add_argument("--size", default="x", choices=["n","s","m","l","x"],
                    help="Model size to auto-select (default: x, largest).")

    # Back-compat alias
    ap.add_argument("--model", default=None,
                    help="[Deprecated] Same as --weights. Kept for backward compatibility.")

    # Training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="stageA_disc_only", help="Ultralytics experiment name.")
    ap.add_argument("--train", type=int, default=1, help="1=train+val, 0=val-only.")
    ap.add_argument("--freeze", type=int, default=0, help="Freeze first N layers (0 = none).")
    ap.add_argument("--amp", type=lambda v: str(v).lower() not in {"0","false","no"}, default=True,
                    help="Enable/disable AMP (default: True).")

    # Resume
    ap.add_argument("--resume", default="", help="Path to last.pt OR 'auto' to use runs/<name>/weights/last.pt")

    # Building options
    ap.add_argument("--copy_images", action="store_true", help="Copy images instead of symlinking (default: symlink).")
    ap.add_argument("--drop_empty", action="store_true", help="Drop label files that become empty after filtering.")

    args = ap.parse_args()

    # Resolve roots
    project_dir = _expand(args.project_dir)
    data_root = _expand(args.data_root) if args.data_root else (project_dir / "data" / "yolo_split")
    aug_root = _expand(args.aug_root) if args.aug_root else None
    train_splits = _parse_train_splits(args.train_splits)

    # Runs root
    runs_root = project_dir / "bounding_box" / "runs" / "detect"
    ensure_dir(runs_root)

    # Choose weights: prefer --weights; else --model (deprecated); else family/size auto-select
    weights = args.weights or args.model

    return DiscOnlyConfig(
        project_dir=project_dir,
        data_root=data_root,
        runs_root=runs_root,
        weights=(str(weights) if weights else None),
        family=str(args.family),
        size=str(args.size),
        amp=bool(args.amp),
        freeze=int(args.freeze),
        aug_root=aug_root,
        train_splits=train_splits,
        copy_images=bool(args.copy_images),
        drop_empty=bool(args.drop_empty),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        exp_name=str(args.name),
        do_train=bool(args.train),
        resume=(str(args.resume) if args.resume else ""),
    )


def main() -> None:
    cfg = build_config_from_cli()
    pipeline = DiscOnlyPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()