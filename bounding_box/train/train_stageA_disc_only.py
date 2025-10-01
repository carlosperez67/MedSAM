#!/usr/bin/env python3
# train_stageA_disc_only.py
import argparse, os, yaml, shutil
from pathlib import Path
from ultralytics import YOLO
from device_utils import ultralytics_device_arg

"""
Train a disc-only detector from an existing 2-class YOLO dataset.
Assumes original labels use 0=disc, 1=cup.

Project layout (defaults; can be overridden):
  PROJECT_DIR/
    data/yolo_split/            # input YOLO dataset root (images/{train,val,test}, labels/{...})
    weights/yolov8n.pt          # local weights
    runs/detect/                 # Ultralytics output root

Outputs:
  <data_root>_disc_only/
    images/{...}   (symlinks or copies)
    labels/{...}   (filtered to disc only)
    od_only.yaml
"""

def _split_has_data(root: Path, split: str) -> bool:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    if not (img_dir.exists() and lbl_dir.exists()):
        return False
    try:
        has_img = any(img_dir.iterdir())
        has_lbl = any(lbl_dir.iterdir())
    except Exception:
        return False
    return has_img and has_lbl

def _place(src: Path, dst: Path, copy_files: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(src, dst)
    else:
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())

def filter_labels_to_disc(src_lbl_dir: Path, dst_lbl_dir: Path, drop_empty: bool) -> int:
    """
    Copy labels keeping only class 0 (disc). If drop_empty, skip writing files with no disc lines.
    Returns number of labels written.
    """
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for txt in sorted(src_lbl_dir.glob("*.txt")):
        lines_in = [ln.strip() for ln in txt.read_text().splitlines() if ln.strip()]
        keep = []
        for line in lines_in:
            parts = line.split()
            try:
                cls = int(float(parts[0]))
            except Exception:
                continue
            if cls == 0:
                keep.append(line)
        if not keep and drop_empty:
            continue
        (dst_lbl_dir / txt.name).write_text("\n".join(keep) + ("\n" if keep else ""))
        written += 1
    return written

def make_disc_only_dataset(root: Path, copy_images: bool, drop_empty: bool) -> Path:
    out = root.parent / f"{root.name}_disc_only"
    # ensure root exists (prevents FileNotFound on YAML write)
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        src_img_dir = root / "images" / split
        src_lbl_dir = root / "labels" / split
        if not src_img_dir.exists() or not src_lbl_dir.exists():
            continue

        # images
        for img in sorted(src_img_dir.glob("*")):
            dst = out / "images" / split / img.name
            _place(img, dst, copy_images)

        # labels (filtered)
        filter_labels_to_disc(src_lbl_dir, out / "labels" / split, drop_empty=drop_empty)

    # write YAML; include 'test' only if it truly has data
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "names": ["disc"],
    }
    if _split_has_data(out, "test"):
        data_yaml["test"] = "images/test"

    (out / "od_only.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return out

def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()

def main():
    ap = argparse.ArgumentParser()
    # Single source of truth for paths
    ap.add_argument("--project_dir", default=".", help="Root working directory for project")
    # Allow overrides; if omitted, these are derived from project_dir
    ap.add_argument("--data_root", default=None, help="YOLO dataset root (defaults to PROJECT_DIR/data/yolo_split)")
    ap.add_argument("--model", default=None, help="Weights path (defaults to PROJECT_DIR/weights/yolov8n.pt)")
    ap.add_argument("--project", default=None, help="Ultralytics runs root (defaults to PROJECT_DIR/runs/detect)")

    # Training args
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="stageA_disc_only")
    ap.add_argument("--train", type=int, default=1, help="1=train+val, 0=val-only")

    # Data curation
    ap.add_argument("--copy_images", action="store_true", help="Copy images instead of symlinking")
    ap.add_argument("--drop_empty", action="store_true", help="Drop images whose disc-only label becomes empty")
    args = ap.parse_args()

    # Resolve base
    PROJECT_DIR = _expand(args.project_dir)

    # Derive paths from project_dir if not provided
    data_root = _expand(args.data_root) if args.data_root else (PROJECT_DIR / "data" / "yolo_split")
    model_path = _expand(args.model) if args.model else (PROJECT_DIR / "weights" / "yolov8n.pt")
    runs_root = _expand(args.project) if args.project else (PROJECT_DIR / "runs" / "detect")

    # Basic checks / creates
    if not (data_root / "images").exists() or not (data_root / "labels").exists():
        raise SystemExit(f"[ERR] Not a YOLO dataset root: {data_root} (missing images/ or labels/)")
    runs_root.mkdir(parents=True, exist_ok=True)

    # Build disc-only derivative
    out_root = make_disc_only_dataset(data_root, copy_images=args.copy_images, drop_empty=args.drop_empty)
    yaml_path = out_root / "od_only.yaml"

    # Device selection (CUDA→'0', else 'cpu') compatible with Ultralytics
    device = ultralytics_device_arg()

    # Weights must be local (HPC offline-safe)
    if not model_path.exists():
        raise SystemExit(f"[ERR] Model weights not found at {model_path}. Put yolov8n.pt there or pass --model /abs/path.pt")

    model = YOLO(str(model_path))

    if args.train:
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
            patience=50,
        )

    # Validate: prefer test if present, else val
    y = yaml.safe_load(yaml_path.read_text())
    if "test" in y:
        model.val(data=str(yaml_path), split="test", imgsz=args.imgsz, device=device)
    else:
        model.val(data=str(yaml_path), imgsz=args.imgsz, device=device)

if __name__ == "__main__":
    main()