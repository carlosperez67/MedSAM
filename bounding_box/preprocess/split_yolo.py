#!/usr/bin/env python3
# split_yolo.py
"""
Patient-wise YOLO dataset splitter (modular, wrapper-friendly).

What it does
------------
- Takes a *flat* folder of YOLO label files (one .txt per image stem) and a
  root folder of fundus images (recursively searched).
- Ensures all images from the same patient go to the same split (train/val/test).
- Creates a YOLO-ready folder tree with images/ and labels/ subfolders.
- Writes a CSV manifest of every (image, label, split).
- (Optional) Writes a data.yaml for Ultralytics.

Assumptions
-----------
- Each label file is named <stem>.txt using the same <stem> as the image file.
- A "patient id" can be derived from the stem using either:
  * a user-supplied regex with ONE capturing group (--patient_regex), OR
  * the fallback "last-dash" rule: keep everything before the last '-' in the stem.

Project-rooted defaults (overridable)
-------------------------------------
PROJECT_DIR/
  bounding_box/data/labels/          # input labels_dir (default)
  # images_root must be provided unless PROJECT_DIR/bounding_box/data/images exists
  bounding_box/data/yolo_split/      # out_root (default)

Examples
--------
# Minimal (explicit image root)
python split_yolo.py \
  --project_dir /path/to/MedSAM \
  --images_root /datasets/SMDG-19/full-fundus \
  --labels_dir  /path/to/MedSAM/bounding_box/data/labels \
  --out_root    /path/to/MedSAM/bounding_box/data/yolo_split \
  --write_yaml

# With regex (capture patient-id from stem, e.g., 'SMDG-0001-OS' -> 'SMDG-0001')
python split_yolo.py ... --patient_regex "^(.*)-[A-Z]{2}$"

# Copy instead of symlink (e.g., if training on a system that disallows symlinks)
python split_yolo.py ... --copy
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ------------------------- Constants -------------------------

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# ------------------------- Small utils -----------------------

def _expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def log(msg: str) -> None:
    print(msg, flush=True)

# ------------------------- Dataclass -------------------------

@dataclass
class SplitConfig:
    project_dir: Path
    images_root: Path
    labels_dir: Path
    out_root: Path
    val_frac: float
    test_frac: float
    seed: int
    copy_files: bool
    patient_regex: Optional[str]
    write_yaml: bool

# ------------------------- Core helpers ----------------------

def enumerate_images_recursive(images_root: Path) -> Dict[str, Path]:
    """
    Recursively find images under images_root and return stem->image_path.
    If multiple files share the same stem, the first encountered is kept.
    """
    stem_to_img: Dict[str, Path] = {}
    for ext in IMG_EXTS:
        for p in images_root.rglob(f"*{ext}"):
            stem_to_img.setdefault(p.stem, p)
    return stem_to_img

def derive_patient_id(stem: str, regex: Optional[re.Pattern]) -> str:
    """
    Patient ID derivation:
      - If regex is provided, use its FIRST capturing group.
      - Else, use the "last-dash" rule: keep everything before the last '-'.
      - Fallback to using the entire stem.
    """
    if regex:
        m = regex.search(stem)
        if m:
            return m.group(1)
    if "-" in stem:
        return stem[: stem.rfind("-")]
    return stem

def place(src: Path, dst: Path, copy_files: bool) -> None:
    """Copy or symlink from src -> dst. Overwrites existing dst (file) if present."""
    _ensure_dir(dst.parent)
    if copy_files:
        shutil.copy2(src, dst)
    else:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

def partition_patient_ids(pids: List[str], val_frac: float, test_frac: float, seed: int) -> Tuple[set, set, set]:
    """Shuffle patients and split into train/val/test by fractions."""
    rng = random.Random(seed)
    rng.shuffle(pids)
    n = len(pids)
    n_val = int(val_frac * n)
    n_test = int(test_frac * n)
    val_ids = set(pids[:n_val])
    test_ids = set(pids[n_val : n_val + n_test])
    train_ids = set(pids[n_val + n_test :])
    return train_ids, val_ids, test_ids

def write_manifest(manifest_path: Path, rows: List[dict]) -> None:
    cols = ["patient_id", "stem", "image_path", "label_path", "split"]
    _ensure_dir(manifest_path.parent)
    with manifest_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

def maybe_write_yaml(out_root: Path, write_yaml: bool, names: List[str] = None) -> None:
    if not write_yaml:
        return
    names = names or ["disc", "cup"]
    # Add test only if present and non-empty
    test_imgs = out_root / "images" / "test"
    test_lbls = out_root / "labels" / "test"
    yaml_lines = [
        f"path: {out_root.resolve()}",
        "train: images/train",
        "val: images/val",
    ]
    if test_imgs.exists() and any(test_imgs.iterdir()) and test_lbls.exists() and any(test_lbls.iterdir()):
        yaml_lines.append("test: images/test")
    yaml_lines.append(f"names: {names}")
    (out_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

# ------------------------- Runner ----------------------------

def run_split(cfg: SplitConfig) -> None:
    # Basic checks
    if not cfg.labels_dir.exists():
        raise SystemExit(f"[ERR] labels_dir not found: {cfg.labels_dir}")
    if not cfg.images_root.exists():
        raise SystemExit(f"[ERR] images_root not found: {cfg.images_root}")

    # Build stem -> image map
    stem_to_img = enumerate_images_recursive(cfg.images_root)
    if not stem_to_img:
        raise SystemExit(f"[ERR] No images found under {cfg.images_root}")

    # Collect labels and pair with images
    label_files = sorted(cfg.labels_dir.glob("*.txt"))
    if not label_files:
        raise SystemExit(f"[ERR] No labels found under {cfg.labels_dir}")

    regex = re.compile(cfg.patient_regex, re.I) if cfg.patient_regex else None

    # Group (image,label) pairs by patient id
    patients: Dict[str, List[Tuple[Path, Path]]] = {}
    missing_images = 0
    for lp in label_files:
        st = lp.stem
        ip = stem_to_img.get(st)
        if not ip:
            missing_images += 1
            continue
        pid = derive_patient_id(st, regex)
        patients.setdefault(pid, []).append((ip, lp))

    if not patients:
        raise SystemExit("[ERR] After matching labels to images, nothing remained. Check stems/paths.")

    if missing_images:
        log(f"[WARN] {missing_images} label file(s) had no matching image by stem and were skipped.")

    # Split patient IDs
    pids = list(patients.keys())
    train_ids, val_ids, test_ids = partition_patient_ids(pids, cfg.val_frac, cfg.test_frac, cfg.seed)

    # Materialize split
    rows: List[dict] = []
    for pid, pairs in patients.items():
        split = "train" if pid in train_ids else "val" if pid in val_ids else "test"
        for ip, lp in pairs:
            # manifest row
            rows.append(
                {
                    "patient_id": pid,
                    "stem": ip.stem,
                    "image_path": str(ip.resolve()),
                    "label_path": str(lp.resolve()),
                    "split": split,
                }
            )
            # place files
            place(ip, cfg.out_root / "images" / split / ip.name, cfg.copy_files)
            place(lp, cfg.out_root / "labels" / split / f"{ip.stem}.txt", cfg.copy_files)

    # Write outputs
    write_manifest(cfg.out_root / "split_manifest.csv", rows)
    maybe_write_yaml(cfg.out_root, cfg.write_yaml, names=["disc", "cup"])

    # Report
    log(f"[OK] Patients total: {len(pids)}  |  train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    log(f"[OK] Manifest: {cfg.out_root / 'split_manifest.csv'}")
    if cfg.write_yaml:
        log(f"[OK] data.yaml: {cfg.out_root / 'data.yaml'}")

# ------------------------- CLI --------------------------------

def build_cfg_from_cli() -> SplitConfig:
    ap = argparse.ArgumentParser(description="Split a YOLO dataset by patient id (patient-wise separation).")

    # Project-rooted defaults
    ap.add_argument("--project_dir", default=".", help="Project root. Defaults derive from here.")

    # I/O
    ap.add_argument("--images_root", default=None,
                    help="Fundus image root (recursively searched). "
                         "If omitted, defaults to PROJECT_DIR/bounding_box/data/images (must exist).")
    ap.add_argument("--labels_dir", default=None,
                    help="Folder with YOLO .txt labels. Default: PROJECT_DIR/bounding_box/data/labels")
    ap.add_argument("--out_root",   default=None,
                    help="Output dataset root. Default: PROJECT_DIR/bounding_box/data/yolo_split")

    # Split ratios
    ap.add_argument("--val_frac", type=float, default=0.15, help="Validation fraction (by patient).")
    ap.add_argument("--test_frac", type=float, default=0.15, help="Test fraction (by patient).")
    ap.add_argument("--seed", type=int, default=1337,         help="Random seed for shuffling patients.")

    # Behavior
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlinking.")
    ap.add_argument("--patient_regex", default="",
                    help="Regex with ONE capturing group for patient_id (optional).")
    ap.add_argument("--write_yaml", action="store_true", help="Also write Ultralytics data.yaml.")

    args = ap.parse_args()

    project_dir = _expand(args.project_dir)

    # Derive defaults from project_dir
    default_labels = project_dir / "bounding_box" / "data" / "labels"
    default_images = project_dir / "bounding_box" / "data" / "images"
    default_out    = project_dir / "bounding_box" / "data" / "yolo_split"

    images_root = _expand(args.images_root) if args.images_root else default_images
    labels_dir  = _expand(args.labels_dir)  if args.labels_dir  else default_labels
    out_root    = _expand(args.out_root)    if args.out_root    else default_out

    # If images_root default doesn't exist, force user to provide it
    if not images_root.exists():
        raise SystemExit(
            f"[ERR] images_root not found: {images_root}\n"
            f"      Provide --images_root, or create {default_images}."
        )

    _ensure_dir(out_root)

    return SplitConfig(
        project_dir=project_dir,
        images_root=images_root,
        labels_dir=labels_dir,
        out_root=out_root,
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
        copy_files=bool(args.copy),
        patient_regex=args.patient_regex or None,
        write_yaml=bool(args.write_yaml),
    )

def main() -> None:
    cfg = build_cfg_from_cli()
    run_split(cfg)

if __name__ == "__main__":
    main()