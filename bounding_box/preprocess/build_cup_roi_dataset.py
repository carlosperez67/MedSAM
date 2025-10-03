#!/usr/bin/env python3
# build_cup_roi_dataset.py
"""
Create a 'cup-in-ROI' YOLO dataset derived from a full 2-class YOLO dataset.
- Input labels: 0=disc, 1=cup
- For each image, make a SQUARE crop centered on the disc box with padding
- Re-map any cup boxes that intersect the ROI into the crop coordinate frame
- Output is a SINGLE-CLASS dataset (0=cup) with its own data.yaml

Modular & wrapper-friendly:
- All logic split into helpers
- `run_build_cup_roi(cfg)` can be called directly from an umbrella pipeline

Project-rooted defaults (overridable):
  PROJECT_DIR/
    bounding_box/data/yolo_split/        # input YOLO (images/{train,val,test}, labels/{...})
    bounding_box/data/yolo_split_cupROI/ # output dataset (this script writes here)

Usage (defaults)
----------------
python build_cup_roi_dataset.py --project_dir /path/to/project
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Iterable, Dict

import cv2
from PIL import Image
import yaml

# --------------------- Constants ---------------------

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# --------------------- Config ------------------------

@dataclass
class CupROIConfig:
    project_dir: Path
    data_root: Path             # input YOLO2-class root
    out_root: Path              # output single-class (cup) root
    pad_pct: float              # padding (fraction of max(W,H)) around disc ROI
    keep_negatives: bool        # keep crop even if no cup lands inside ROI
    splits: List[str]           # which splits to process (must exist)

# --------------------- IO helpers --------------------

def _expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _iter_images(folder: Path, exts: Iterable[str] = IMG_EXTS) -> List[Path]:
    if not folder.exists():
        return []
    out: List[Path] = []
    for ext in exts:
        out.extend(folder.glob(f"*{ext}"))
    return sorted(out)

def _has_nonempty_split(root: Path, split: str) -> bool:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    return img_dir.exists() and lbl_dir.exists() and any(img_dir.iterdir()) and any(lbl_dir.iterdir())

# --------------------- Geometry ----------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def read_yolo_boxes(txt_path: Path) -> List[List[float]]:
    """Return rows of [cls, cx, cy, w, h] (floats). Empty if file missing or unparsable."""
    if not txt_path.exists():
        return []
    rows: List[List[float]] = []
    for line in txt_path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        try:
            rows.append([float(x) for x in parts[:5]])
        except Exception:
            continue
    return rows

def yolo_to_xyxy_int(row: List[float], W: int, H: int) -> Tuple[int, int, int, int, int]:
    """Convert [cls,cx,cy,w,h] (normalized) → (cls,x1,y1,x2,y2) in integer pixels."""
    cls, cx, cy, bw, bh = row[:5]
    x1 = (cx - bw / 2.0) * W
    y1 = (cy - bh / 2.0) * H
    x2 = (cx + bw / 2.0) * W
    y2 = (cy + bh / 2.0) * H
    return int(cls), int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def ensure_square_expand(x1: int, y1: int, x2: int, y2: int, pad_frac: float, W: int, H: int):
    """Square ROI around (x1,y1,x2,y2) with extra padding as fraction of max(W,H)."""
    w = x2 - x1
    h = y2 - y1
    side = max(w, h)
    extra = int(round(pad_frac * max(W, H)))
    side = side + 2 * extra
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    nx1 = clamp(cx - side // 2, 0, W - 1)
    ny1 = clamp(cy - side // 2, 0, H - 1)
    nx2 = clamp(nx1 + side, 1, W)
    ny2 = clamp(ny1 + side, 1, H)
    # If we clipped on far side, shift back to keep square size if possible
    if nx2 - nx1 != side:
        nx1 = clamp(W - side, 0, W - 1); nx2 = W
    if ny2 - ny1 != side:
        ny1 = clamp(H - side, 0, H - 1); ny2 = H
    return nx1, ny1, nx2, ny2

def box_intersection(a, b) -> Optional[Tuple[int,int,int,int]]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1); y1 = max(ay1, by1)
    x2 = min(ax2, bx2); y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def xyxy_to_yolo(box, W, H) -> Tuple[float,float,float,float]:
    x1, y1, x2, y2 = box
    bw = (x2 - x1); bh = (y2 - y1)
    cx = x1 + bw / 2.0; cy = y1 + bh / 2.0
    return cx / W, cy / H, bw / W, bh / H

# --------------------- Core logic ---------------------

def parse_disc_and_cups(rows: List[List[float]], W: int, H: int) -> Tuple[Optional[Tuple[int,int,int,int]], List[Tuple[int,int,int,int]]]:
    """
    From YOLO rows, return disc box (class 0) and list of cup boxes (class 1),
    both in absolute xyxy integer pixels.
    """
    disc_box: Optional[Tuple[int,int,int,int]] = None
    cup_boxes: List[Tuple[int,int,int,int]] = []
    for r in rows:
        cls, x1, y1, x2, y2 = yolo_to_xyxy_int(r, W, H)
        if cls == 0:
            disc_box = (x1, y1, x2, y2)
        elif cls == 1:
            cup_boxes.append((x1, y1, x2, y2))
    return disc_box, cup_boxes

def transform_cups_into_roi(cup_boxes: List[Tuple[int,int,int,int]],
                            roi: Tuple[int,int,int,int],
                            roi_W: int, roi_H: int) -> List[str]:
    """
    Intersect each cup with ROI, translate → ROI-local, convert to YOLO.
    Returns list of label lines for single-class cup dataset: "0 cx cy w h".
    """
    rx1, ry1, rx2, ry2 = roi
    out_lines: List[str] = []
    for cup in cup_boxes:
        inter = box_intersection(cup, roi)
        if inter is None:
            continue
        ix1, iy1, ix2, iy2 = inter
        # ROI-local coords
        lx1 = ix1 - rx1; ly1 = iy1 - ry1
        lx2 = ix2 - rx1; ly2 = iy2 - ry1
        cx, cy, bw, bh = xyxy_to_yolo((lx1, ly1, lx2, ly2), roi_W, roi_H)
        out_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return out_lines

def process_one_image(ip: Path, lbl_dir: Path, out_img_dir: Path, out_lbl_dir: Path, cfg: CupROIConfig) -> bool:
    """
    Process one full image:
      - read labels
      - compute disc-centered square ROI (with padding)
      - crop image to ROI
      - remap cup boxes into ROI
      - write crop + label (always if has cups; only if keep_negatives when no cups)
    Returns True if a crop was written.
    """
    # read dims
    try:
        with Image.open(ip) as im:
            W, H = im.size
    except Exception:
        return False

    # labels
    rows = read_yolo_boxes(lbl_dir / f"{ip.stem}.txt")
    if not rows:
        return False

    disc_box, cup_boxes = parse_disc_and_cups(rows, W, H)
    if disc_box is None:
        return False

    # ROI
    rx1, ry1, rx2, ry2 = ensure_square_expand(*disc_box, pad_frac=cfg.pad_pct, W=W, H=H)
    roi = (rx1, ry1, rx2, ry2)

    # crop
    im_bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
    if im_bgr is None:
        return False
    crop = im_bgr[ry1:ry2, rx1:rx2].copy()
    cH, cW = crop.shape[:2]

    # cups → ROI YOLO
    out_lines = transform_cups_into_roi(cup_boxes, roi, cW, cH)

    if not out_lines and not cfg.keep_negatives:
        return False

    # write outputs
    _ensure_dir(out_img_dir)
    _ensure_dir(out_lbl_dir)
    cv2.imwrite(str(out_img_dir / f"{ip.stem}_roi.png"), crop)
    (out_lbl_dir / f"{ip.stem}_roi.txt").write_text("\n".join(out_lines) + ("\n" if out_lines else ""))

    return True

def write_cup_yaml(out_root: Path) -> Path:
    """Write a minimal Ultralytics data.yaml for the cup-only dataset."""
    yaml_path = out_root / "cup_roi.yaml"
    ds = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "names": ["cup"],
    }
    if _has_nonempty_split(out_root, "test"):
        ds["test"] = "images/test"
    yaml_path.write_text(yaml.safe_dump(ds, sort_keys=False))
    return yaml_path

# --------------------- Runner ------------------------

def run_build_cup_roi(cfg: CupROIConfig) -> None:
    # Prepare output dirs
    for split in cfg.splits:
        _ensure_dir(cfg.out_root / "images" / split)
        _ensure_dir(cfg.out_root / "labels" / split)

    total, kept = 0, 0

    for split in cfg.splits:
        img_dir = cfg.data_root / "images" / split
        lbl_dir = cfg.data_root / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            # Skip non-existent split silently
            continue

        out_img_dir = cfg.out_root / "images" / split
        out_lbl_dir = cfg.out_root / "labels" / split

        for ip in _iter_images(img_dir):
            total += 1
            if process_one_image(ip, lbl_dir, out_img_dir, out_lbl_dir, cfg):
                kept += 1

    yaml_path = write_cup_yaml(cfg.out_root)

    print(f"[OK] Built ROI dataset at: {cfg.out_root}")
    print(f"  images processed: {total} | crops kept: {kept}")
    print(f"[OK] Wrote data yaml: {yaml_path}")

# --------------------- CLI ---------------------------

def build_cfg_from_cli() -> CupROIConfig:
    ap = argparse.ArgumentParser(description="Build cup-in-ROI (single-class) dataset from 2-class YOLO.")
    ap.add_argument("--project_dir", default=".", help="Project working directory (root of your repo).")

    # Allow overrides; if omitted, derive from project_dir
    ap.add_argument("--data_root", default=None,
                    help="Input YOLO root. Default: {PROJECT_DIR}/bounding_box/data/yolo_split")
    ap.add_argument("--out_root",  default=None,
                    help="Output ROI dataset root. Default: {PROJECT_DIR}/bounding_box/data/yolo_split_cupROI")

    ap.add_argument("--pad_pct", type=float, default=0.10, help="Padding (fraction of max image dim) around disc ROI")
    ap.add_argument("--keep_negatives", action="store_true", help="Keep ROI crops with no cup visible")
    ap.add_argument("--splits", nargs="+", default=["train","val","test"], help="Splits to process")

    args = ap.parse_args()

    project_dir = _expand(args.project_dir)
    data_root = _expand(args.data_root) if args.data_root else (project_dir / "bounding_box" / "data" / "yolo_split")
    out_root  = _expand(args.out_root)  if args.out_root  else (project_dir / "bounding_box" / "data" / "yolo_split_cupROI")

    return CupROIConfig(
        project_dir = project_dir,
        data_root   = data_root,
        out_root    = out_root,
        pad_pct     = float(args.pad_pct),
        keep_negatives = bool(args.keep_negatives),
        splits      = list(args.splits),
    )

def main():
    cfg = build_cfg_from_cli()
    run_build_cup_roi(cfg)

if __name__ == "__main__":
    main()