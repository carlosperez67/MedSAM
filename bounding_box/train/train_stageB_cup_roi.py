#!/usr/bin/env python3
# build_cup_roi_dataset.py
"""
Create a 'cup-in-ROI' YOLO dataset derived from a full 2-class YOLO dataset.
- Input labels: 0=disc, 1=cup
- For each image, make a SQUARE crop centered on the disc box with padding
- Re-map any cup boxes that intersect the ROI into the crop coordinate frame
- Output is a SINGLE-CLASS dataset (0=cup) with its own data.yaml

Project-rooted paths
--------------------
You can pass --project_dir once; defaults are derived from it:
  PROJECT_DIR/
    data/yolo_split/            # input YOLO (images/{train,val,test}, labels/{...})
    data/yolo_split_cupROI/     # output dataset (this script writes here)

Usage (defaults)
----------------
python build_cup_roi_dataset.py --project_dir /path/to/project
"""

import argparse, os, math, shutil
from pathlib import Path
from typing import Tuple, List

import cv2
from PIL import Image
import yaml

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# --------------------- IO helpers ---------------------

def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()

def _iter_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    out = []
    for ext in IMG_EXTS:
        out.extend(folder.glob(f"*{ext}"))
    return sorted(out)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# --------------------- Geometry -----------------------

def clamp(v, lo, hi): return max(lo, min(hi, v))

def read_yolo_boxes(txt_path: Path) -> List[List[float]]:
    if not txt_path.exists():
        return []
    rows = []
    for line in txt_path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        try:
            rows.append([float(x) for x in parts[:5]])  # cls cx cy bw bh
        except Exception:
            continue
    return rows

def yolo_to_xyxy(row, w, h) -> Tuple[int, int, int, int, int]:
    cls, cx, cy, bw, bh = row[:5]
    x1 = (cx - bw / 2.0) * w
    y1 = (cy - bh / 2.0) * h
    x2 = (cx + bw / 2.0) * w
    y2 = (cy + bh / 2.0) * h
    return int(cls), int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def ensure_square_expand(x1, y1, x2, y2, pad_frac, W, H):
    """Square ROI around (x1,y1,x2,y2) with extra padding as frac of max(W,H)."""
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

def box_intersection(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1); y1 = max(ay1, by1)
    x2 = min(ax2, bx2); y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def xyxy_to_yolo(box, W, H):
    x1, y1, x2, y2 = box
    bw = (x2 - x1); bh = (y2 - y1)
    cx = x1 + bw / 2.0; cy = y1 + bh / 2.0
    return cx / W, cy / H, bw / W, bh / H

# --------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    # Single source of truth for paths
    ap.add_argument("--project_dir", default=".",
                    help="Project working directory. Defaults are derived from here.")
    # Allow overrides; if omitted, they come from project_dir
    ap.add_argument("--data_root", default=None,
                    help="Input YOLO root (defaults to PROJECT_DIR/data/yolo_split)")
    ap.add_argument("--out_root",  default=None,
                    help="Output ROI dataset root (defaults to PROJECT_DIR/data/yolo_split_cupROI)")

    ap.add_argument("--pad_pct", type=float, default=0.10,
                    help="Padding (fraction of max image dim) around disc ROI")
    ap.add_argument("--keep_negatives", action="store_true",
                    help="Keep ROI crops with no cup visible")
    ap.add_argument("--splits", nargs="+", default=["train","val","test"],
                    help="Splits to process (existing ones will be used)")
    args = ap.parse_args()

    PROJECT_DIR = _expand(args.project_dir)
    data_root = _expand(args.data_root) if args.data_root else (PROJECT_DIR / "data" / "yolo_split")
    out_root  = _expand(args.out_root)  if args.out_root  else (PROJECT_DIR / "data" / "yolo_split_cupROI")

    # Prepare output dirs
    for split in args.splits:
        _ensure_dir(out_root / "images" / split)
        _ensure_dir(out_root / "labels" / split)

    total, kept = 0, 0

    for split in args.splits:
        img_dir = data_root / "images" / split
        lbl_dir = data_root / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            # Skip non-existent split silently
            continue

        for img_path in _iter_images(img_dir):
            total += 1

            with Image.open(img_path) as im:
                W, H = im.size

            rows = read_yolo_boxes(lbl_dir / f"{img_path.stem}.txt")
            if not rows:
                continue

            disc_box = None
            cup_boxes = []
            for r in rows:
                cls, x1, y1, x2, y2 = yolo_to_xyxy(r, W, H)
                if cls == 0:
                    disc_box = (x1, y1, x2, y2)
                elif cls == 1:
                    cup_boxes.append((x1, y1, x2, y2))

            if disc_box is None:
                continue

            # Build square ROI around disc
            rx1, ry1, rx2, ry2 = ensure_square_expand(*disc_box, pad_frac=args.pad_pct, W=W, H=H)
            roi = (rx1, ry1, rx2, ry2)

            # Crop image to ROI
            im_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if im_bgr is None:
                continue
            crop = im_bgr[ry1:ry2, rx1:rx2].copy()
            cH, cW = crop.shape[:2]

            # Transform cup boxes into ROI-local coords
            out_lines = []
            for cup in cup_boxes:
                inter = box_intersection(cup, roi)
                if inter is None:
                    continue
                ix1, iy1, ix2, iy2 = inter
                lx1 = ix1 - rx1; ly1 = iy1 - ry1
                lx2 = ix2 - rx1; ly2 = iy2 - ry1
                cx, cy, bw, bh = xyxy_to_yolo((lx1, ly1, lx2, ly2), cW, cH)
                # cup-only dataset: class 0
                out_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not out_lines and not args.keep_negatives:
                continue

            # Write crop + label
            cv_out = out_root / "images" / split / f"{img_path.stem}_roi.png"
            lb_out = out_root / "labels" / split / f"{img_path.stem}_roi.txt"
            cv2.imwrite(str(cv_out), crop)
            lb_out.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))

            kept += 1

    # Write YAML for cup-only dataset
    yaml_path = out_root / "cup_roi.yaml"
    ds = {"path": str(out_root.resolve()),
          "train": "images/train",
          "val": "images/val",
          "names": ["cup"]}
    # Add test only if it exists and is non-empty
    test_img_dir = out_root / "images" / "test"
    test_lbl_dir = out_root / "labels" / "test"
    if test_img_dir.exists() and any(test_img_dir.iterdir()) and test_lbl_dir.exists() and any(test_lbl_dir.iterdir()):
        ds["test"] = "images/test"

    yaml_path.write_text(yaml.safe_dump(ds, sort_keys=False))

    print(f"[OK] Built ROI dataset at: {out_root}")
    print(f"  images processed: {total} | crops kept: {kept}")
    print(f"[OK] Wrote data yaml: {yaml_path}")

if __name__ == "__main__":
    main()