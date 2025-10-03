#!/usr/bin/env python3
# viz_yolo_two_boxes.py
"""
Visualize OD/OC (disc/cup) YOLO boxes on fundus images.

What it does
------------
- Loads YOLO labels (0=disc, 1=cup) with normalized cx,cy,w,h
- Resolves the source image by stem
- Draws translucent boxes (OD=blue-ish, OC=green)
- (Optional) Saves per-class crops and side-by-side montage

Modular & wrapper-friendly
--------------------------
- All logic is split into small helper functions
- A single `run_viz(cfg)` entry point can be called by an external orchestrator

Project-rooted defaults (overridable)
-------------------------------------
PROJECT_DIR/
  bounding_box/data/yolo_split/
    images/{train,val,test}/*.png|jpg|...
    labels/{train,val,test}/*.txt
  bounding_box/data/viz_two_boxes/{train,val,test}/   # this script writes here by default

Examples
--------
# Use project-rooted defaults on the "val" split, visualize 40 random images
python viz_yolo_two_boxes.py --project_dir /path/to/MedSAM --split val --sample 40

# Override explicit folders
python viz_yolo_two_boxes.py \
  --images_dir /data/fundus/images/val \
  --labels_dir /data/fundus/labels/val \
  --out_dir    /tmp/viz_two_boxes/val \
  --sample 12 --save_crops --make_montage --verbose
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
from PIL import Image

# ------------------------ Constants ------------------------

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# Colors (BGR)
COLOR_DISC = (255, 96, 32)   # blue-ish/orange mix for OD
COLOR_CUP  = (64, 192, 64)   # green for OC

# ------------------------ Config ---------------------------

@dataclass
class VizConfig:
    # Path strategy
    project_dir: Path
    split: str
    images_dir: Path
    labels_dir: Path
    out_dir: Path

    # Viz options
    sample: int
    seed: int
    thickness: int
    alpha: float
    save_crops: bool
    crop_pad: int
    make_montage: bool
    verbose: bool

    # Resolution helpers (optional override)
    img_exts: Tuple[str, ...] = IMG_EXTS

# ------------------------ Small utils ----------------------

def _expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def auto_thickness(h: int, w: int) -> int:
    return max(2, int(0.002 * max(h, w)))

# ------------------------ I/O helpers ----------------------

def try_resolve_image(stem: str, image_dir: Path, exts: Iterable[str]) -> Optional[Path]:
    """Try stem + each extension; fallback to any stem.*."""
    for ext in exts:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    cands = sorted(image_dir.glob(f"{stem}.*"))
    return cands[0] if cands else None

def read_image_bgr(path: Path) -> Optional[cv2.Mat]:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return im

def read_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (W, H)

def list_label_stems(labels_dir: Path) -> List[str]:
    files = sorted(labels_dir.glob("*.txt"))
    return [p.stem for p in files]

# ------------------------ Label parsing --------------------

def yolo_line_to_xyxy(line: str, W: int, H: int) -> Optional[Tuple[int, int, int, int, int]]:
    """'cls cx cy w h' → (cls, x1,y1,x2,y2) int pixels (clamped) or None if malformed."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx  = float(parts[1]) * W
        cy  = float(parts[2]) * H
        bw  = float(parts[3]) * W
        bh  = float(parts[4]) * H
        x1 = int(round(cx - bw / 2.0)); y1 = int(round(cy - bh / 2.0))
        x2 = int(round(cx + bw / 2.0)); y2 = int(round(cy + bh / 2.0))
        x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2)); y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return (cls, x1, y1, x2, y2)
    except Exception:
        return None

def load_yolo_boxes_for_stem(labels_dir: Path, stem: str, W: int, H: int) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Returns box dict with optional keys 'disc' and 'cup'.
    - If multiple of a class exist, keeps the FIRST occurrence.
    """
    out: Dict[str, Tuple[int, int, int, int]] = {}
    lp = labels_dir / f"{stem}.txt"
    if not lp.exists():
        return out
    for line in lp.read_text().splitlines():
        if not line.strip():
            continue
        parsed = yolo_line_to_xyxy(line, W, H)
        if not parsed:
            continue
        cls, x1, y1, x2, y2 = parsed
        if cls == 0 and "disc" not in out:
            out["disc"] = (x1, y1, x2, y2)
        elif cls == 1 and "cup" not in out:
            out["cup"] = (x1, y1, x2, y2)
    return out

# ------------------------ Drawing -------------------------

def draw_box(img: cv2.Mat,
             box: Optional[Tuple[int, int, int, int]],
             color: Tuple[int, int, int],
             label: str,
             thickness: int,
             alpha: float = 0.25) -> cv2.Mat:
    """Draw translucent filled rectangle + border + label."""
    if not box:
        return img
    x1, y1, x2, y2 = map(int, box)
    overlay = img.copy()
    # fill
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    # border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)
    # caption
    font_scale = 0.45
    font_thick = 1
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
    bx2 = min(img.shape[1] - 1, x1 + tw + 8)
    by2 = min(img.shape[0] - 1, y1 + th + 8)
    cv2.rectangle(img, (x1, y1), (bx2, by2), color, thickness=-1)
    cv2.putText(img, label, (x1 + 4, y1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
    return img

def crop_region(img: cv2.Mat, box: Optional[Tuple[int, int, int, int]], pad: int = 0) -> Optional[cv2.Mat]:
    if not box:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()

def side_by_side(left_bgr: cv2.Mat, right_bgr: cv2.Mat) -> cv2.Mat:
    lh, lw = left_bgr.shape[:2]
    rh, rw = right_bgr.shape[:2]
    if lh != rh:
        right_bgr = cv2.resize(right_bgr, (int(rw * (lh / rh)), lh), interpolation=cv2.INTER_AREA)
    return cv2.hconcat([left_bgr, right_bgr])

# ------------------------ Core per-item --------------------

def visualize_one(stem: str, cfg: VizConfig) -> bool:
    """
    Visualize one image by stem.
    Returns True if something was written to disk.
    """
    img_path = try_resolve_image(stem, cfg.images_dir, cfg.img_exts)
    if not img_path:
        if cfg.verbose:
            print(f"[WARN] No image found for stem '{stem}'")
        return False

    img = read_image_bgr(img_path)
    if img is None:
        if cfg.verbose:
            print(f"[WARN] Failed to read image: {img_path}")
        return False

    W, H = img.shape[1], img.shape[0]
    boxes = load_yolo_boxes_for_stem(cfg.labels_dir, stem, W, H)
    if not boxes:
        if cfg.verbose:
            print(f"[INFO] No boxes for stem '{stem}' (empty label?)")
        return False

    th = cfg.thickness if cfg.thickness > 0 else auto_thickness(H, W)
    ann = img.copy()
    ann = draw_box(ann, boxes.get("disc"), COLOR_DISC, "OD (disc)", th, alpha=cfg.alpha)
    ann = draw_box(ann, boxes.get("cup"),  COLOR_CUP,  "OC (cup)",  th, alpha=cfg.alpha)

    # save annotated
    _ensure_dir(cfg.out_dir)
    out_png = cfg.out_dir / f"{stem}_two_boxes.png"
    cv2.imwrite(str(out_png), ann)
    if cfg.verbose:
        print(f"[OK] {out_png}")

    # optional crops
    if cfg.save_crops:
        od_crop_dir = cfg.out_dir / "crops" / "od"
        oc_crop_dir = cfg.out_dir / "crops" / "oc"
        _ensure_dir(od_crop_dir)
        _ensure_dir(oc_crop_dir)

        od_crop = crop_region(img, boxes.get("disc"), pad=cfg.crop_pad)
        oc_crop = crop_region(img, boxes.get("cup"),  pad=cfg.crop_pad)
        if od_crop is not None:
            cv2.imwrite(str(od_crop_dir / f"{stem}_od.png"), od_crop)
        if oc_crop is not None:
            cv2.imwrite(str(oc_crop_dir / f"{stem}_oc.png"), oc_crop)

    # optional montage
    if cfg.make_montage:
        montage = side_by_side(img, ann)
        cv2.imwrite(str(cfg.out_dir / f"{stem}_montage.png"), montage)

    return True

# ------------------------ Runner --------------------------

def run_viz(cfg: VizConfig) -> None:
    # Derive list of stems
    stems = list_label_stems(cfg.labels_dir)
    if not stems:
        print(f"[ERROR] No label files found under {cfg.labels_dir}")
        return

    random.seed(cfg.seed)
    if cfg.sample > 0 and cfg.sample < len(stems):
        stems = random.sample(stems, cfg.sample)

    # Ensure output folder
    _ensure_dir(cfg.out_dir)

    wrote = 0
    for stem in stems:
        if visualize_one(stem, cfg):
            wrote += 1

    print(f"[DONE] Visualized {wrote}/{len(stems)} file(s) → {cfg.out_dir}")

# ------------------------ CLI -----------------------------

def build_cfg_from_cli() -> VizConfig:
    ap = argparse.ArgumentParser(description="Visualize OD/OC YOLO boxes on fundus images.")
    # Project-rooted pathing
    ap.add_argument("--project_dir", default=".", help="Project root (defaults derive from here).")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split to visualize.")

    # Optional overrides (use these OR the project-rooted defaults)
    ap.add_argument("--images_dir", default=None, help="Explicit images dir (overrides project-rooted default).")
    ap.add_argument("--labels_dir", default=None, help="Explicit labels dir (overrides project-rooted default).")
    ap.add_argument("--out_dir",    default=None, help="Explicit output dir (overrides project-rooted default).")

    # Viz behavior
    ap.add_argument("--sample", type=int, default=12, help="Random sample size (>=count to process all).")
    ap.add_argument("--seed",   type=int, default=1337, help="Random seed for sampling.")
    ap.add_argument("--thickness", type=int, default=0, help="Line thickness (0=auto).")
    ap.add_argument("--alpha",     type=float, default=0.25, help="Fill transparency 0..1.")
    ap.add_argument("--save_crops", action="store_true", help="Also save OD/OC crops.")
    ap.add_argument("--crop_pad",   type=int, default=0, help="Extra pixels to pad around crops.")
    ap.add_argument("--make_montage", action="store_true", help="Save side-by-side montage.")
    ap.add_argument("--verbose", action="store_true", help="Print per-file diagnostics.")
    args = ap.parse_args()

    project_dir = _expand(args.project_dir)
    # Project-rooted defaults
    default_data_root = project_dir / "bounding_box" / "data" / "yolo_split"
    default_images = default_data_root / "images" / args.split
    default_labels = default_data_root / "labels" / args.split
    default_out    = project_dir / "bounding_box" / "data" / "viz_two_boxes" / args.split

    images_dir = _expand(args.images_dir) if args.images_dir else default_images
    labels_dir = _expand(args.labels_dir) if args.labels_dir else default_labels
    out_dir    = _expand(args.out_dir)    if args.out_dir    else default_out

    return VizConfig(
        project_dir=project_dir,
        split=args.split,
        images_dir=images_dir,
        labels_dir=labels_dir,
        out_dir=out_dir,
        sample=args.sample,
        seed=args.seed,
        thickness=args.thickness,
        alpha=args.alpha,
        save_crops=args.save_crops,
        crop_pad=args.crop_pad,
        make_montage=args.make_montage,
        verbose=args.verbose,
    )

def main():
    cfg = build_cfg_from_cli()
    run_viz(cfg)

if __name__ == "__main__":
    main()