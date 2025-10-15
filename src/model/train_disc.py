#!/usr/bin/env python3
# train_disc_only.py
"""
OOP refactor: Train a disc-only detector from a 2-class YOLO dataset (0=disc, 1=cup).

Features
--------
- Precomputed disc-only support: if <data_root>/data.yaml exists, use it.
- Otherwise, derive a disc-only dataset next to <data_root>: <data_root>_disc_only/
  * Images symlinked (default) or copied
  * Labels filtered to keep only class 0 (disc)
  * Optionally drop images whose filtered label becomes empty
  * Writes data.yaml with names: ["disc"]
- Optional training-time resume:
  * --resume auto         -> runs/<name>/weights/last.pt if present
  * --resume /path/last.pt
- Validates on 'test' if present, else 'val'.

CLI examples
------------
# Basic
python train_stageA_disc_only.py --project_dir /path/to/MedSAM

# From existing disc-only root (already contains data.yaml)
python train_stageA_disc_only.py --data_root /path/to/yolo_split_disc_only

# With augmented train split for building
python train_stageA_disc_only.py \
  --project_dir /path/to/MedSAM \
  --aug_root   /path/to/yolo_split_aug \
  --train_splits train

# Resume training from last.pt (auto)
python train_stageA_disc_only.py --resume auto
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import yaml
from ultralytics import YOLO

from src.model.device_utils import ultralytics_device_arg


# ============================================================
# Config & simple utilities
# ============================================================

@dataclass
class DiscOnlyConfig:
    # core paths
    project_dir: Path
    data_root: Path                   # base YOLO root OR a precomputed disc-only root
    model_path: Path
    runs_root: Path

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


def _expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _place(src: Path, dst: Path, copy_files: bool) -> None:
    """Copy or symlink src → dst."""
    _ensure_dir(dst.parent)
    if copy_files:
        shutil.copy2(src, dst)
    else:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())


def _split_has_data(root: Path, split: str) -> bool:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    if not (img_dir.exists() and lbl_dir.exists()):
        return False
    try:
        return any(img_dir.iterdir()) and any(lbl_dir.iterdir())
    except Exception:
        return False


# ============================================================
# Dataset builder (disc-only)
# ============================================================

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
        _ensure_dir(dst_lbl_dir)
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
        _ensure_dir(dst_root)

        for split in ("train", "val", "test"):
            src_root = aug_root if (aug_root is not None and split in set(train_splits)) else base_root
            src_img_dir = src_root / "images" / split
            src_lbl_dir = src_root / "labels" / split
            if not src_img_dir.exists() or not src_lbl_dir.exists():
                continue

            # images
            for img in sorted(src_img_dir.glob("*")):
                _place(img, dst_root / "images" / split / img.name, copy_images)

            # labels -> only disc
            cls._filter_labels_dir_to_disc(src_lbl_dir, dst_root / "labels" / split, drop_empty)

        # dataset YAML
        data_yaml = {
            "path": str(dst_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": ["disc"],
        }
        if _split_has_data(dst_root, "test"):
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
        self.device = ultralytics_device_arg()
        self.model: YOLO | None = None
        self.resuming: bool = False

    def _find_last_ckpt(self) -> Optional[Path]:
        p = self.cfg.runs_root / self.cfg.exp_name / "weights" / "last.pt"
        return p if p.exists() else None

    def _build_from_weights(self, weights: Path) -> YOLO:
        if not weights.exists():
            raise SystemExit(f"[ERR] Model weights not found at: {weights}")
        return YOLO(str(weights))

    def prepare_model(self) -> None:
        """Apply resume policy or start from provided weights."""
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
            self.model = self._build_from_weights(self.cfg.model_path)
            self.resuming = False

    def train(self, data_yaml: Path) -> None:
        if not self.cfg.do_train:
            return
        assert self.model is not None
        print(f"[INFO] Training… (epochs={self.cfg.epochs}, imgsz={self.cfg.imgsz}, "
              f"batch={self.cfg.batch}, resume={self.resuming})")
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
            single_cls=True,
            resume=self.resuming,
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
    ap.add_argument("--model", default=None,
                    help="Weights path (defaults to PROJECT_DIR/weights/yolov8n.pt).")
    ap.add_argument("--project", default=None,
                    help="Ultralytics runs root (defaults to PROJECT_DIR/bounding_box/runs/detect).")

    # Training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="stageA_disc_only", help="Ultralytics experiment name.")
    ap.add_argument("--train", type=int, default=1, help="1=train+val, 0=val-only.")

    # Resume
    ap.add_argument("--resume", default="", help="Path to last.pt OR 'auto' to use runs/<name>/weights/last.pt")

    # Building options
    ap.add_argument("--copy_images", action="store_true", help="Copy images instead of symlinking (default: symlink).")
    ap.add_argument("--drop_empty", action="store_true", help="Drop label files that become empty after filtering.")

    args = ap.parse_args()

    project_dir = _expand(args.project_dir)
    data_root = _expand(args.data_root) if args.data_root else (project_dir / "data" / "yolo_split")
    aug_root = _expand(args.aug_root) if args.aug_root else None
    train_splits = _parse_train_splits(args.train_splits)

    # defaults
    model_path = _expand(args.model) if args.model else (project_dir / "weights" / "yolov8n.pt")
    runs_root = _expand(args.project) if args.project else (project_dir / "bounding_box" / "runs" / "detect")
    _ensure_dir(runs_root)

    return DiscOnlyConfig(
        project_dir=project_dir,
        data_root=data_root,
        model_path=model_path,
        runs_root=runs_root,
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