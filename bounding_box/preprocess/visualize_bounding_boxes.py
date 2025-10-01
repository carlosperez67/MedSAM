#!/usr/bin/env python3
"""
viz_yolo_two_boxes.py — Visualize OD/OC boxes produced by masks_to_yolo_boxes_smdg.py

Reads:  YOLO label TXT files (0=disc, 1=cup) with normalized cx,cy,bw,bh
Finds:  Source fundus image by stem
Draws:  OD (disc, blue) and OC (cup, green)
Saves:  annotated PNGs; optional per-class crops; optional side-by-side montage.

Defaults:
  - labels_dir : ./labels
  - images_dir : /Users/carlosperez/OneDrive_UBC/Ipek_Carlos/GlaucomaDatasets/SMDG-19/full-fundus/full-fundus
  - out_dir    : ./viz_labels

Usage:
  python viz_yolo_two_boxes.py
  python viz_yolo_two_boxes.py --sample 12 --save_crops --make_montage
"""

import argparse, os, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from PIL import Image

DATA_ROOT = "/Users/carlosperez/OneDrive_UBC/Ipek_Carlos/GlaucomaDatasets"
DEF_IMAGES = f"{DATA_ROOT}/SMDG-19/full-fundus/full-fundus"
DEF_LABELS = "./../data/labels"
DEF_OUTDIR = "./../data/viz_labels"

IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

# ----------------- helpers -----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Visualize OD/OC YOLO boxes from masks_to_yolo_boxes_smdg.py")
    ap.add_argument("--labels_dir", default=DEF_LABELS, help="Directory containing YOLO .txt labels")
    ap.add_argument("--image_dir", default=DEF_IMAGES, help="Directory containing fundus images")
    ap.add_argument("--out_dir",    default=DEF_OUTDIR, help="Output directory for visualizations")
    ap.add_argument("--img_exts",   nargs="+", default=IMG_EXTS, help="Image extensions to try when resolving stems")
    ap.add_argument("--sample", type=int, default=12, help="Random sample size (>=count to process all)")
    ap.add_argument("--seed",   type=int, default=1337, help="Random seed for sampling")
    ap.add_argument("--thickness", type=int, default=0, help="Line thickness (0=auto)")
    ap.add_argument("--alpha",     type=float, default=0.25, help="Fill transparency 0..1")
    ap.add_argument("--save_crops", action="store_true", help="Also save OD/OC crops to subfolders")
    ap.add_argument("--crop_pad",   type=int, default=0, help="Extra pixels to pad around the crop")
    ap.add_argument("--make_montage", action="store_true",
                    help="Save side-by-side montage: [original | annotated]")
    ap.add_argument("--verbose", action="store_true", help="Print per-file diagnostics")
    return ap.parse_args()

def auto_thickness(h: int, w: int) -> int:
    return max(2, int(0.002 * max(h, w)))

def try_resolve_image(stem: str, image_dir: Path, img_exts: List[str]) -> Optional[Path]:
    for ext in img_exts:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # fallback: any stem.*
    cands = sorted(image_dir.glob(f"{stem}.*"))
    return cands[0] if cands else None

def read_image_size(p: Path) -> Tuple[int, int]:
    with Image.open(p) as im:
        return im.size  # (W,H)

def yolo_to_xyxy(line: str, W: int, H: int) -> Optional[Tuple[int,int,int,int,int]]:
    """
    Parse a YOLO label line: 'cls cx cy bw bh' (normalized) → (cls, x1,y1,x2,y2) in pixels.
    Returns None if the line is malformed.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx  = float(parts[1]) * W
        cy  = float(parts[2]) * H
        bw  = float(parts[3]) * W
        bh  = float(parts[4]) * H
        x1 = int(round(cx - bw/2.0)); y1 = int(round(cy - bh/2.0))
        x2 = int(round(cx + bw/2.0)); y2 = int(round(cy + bh/2.0))
        # clamp
        x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
        x2 = max(0, min(W-1, x2)); y2 = max(0, min(H-1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return cls, x1, y1, x2, y2
    except Exception:
        return None

def load_yolo_boxes_for_stem(labels_dir: Path, stem: str, W: int, H: int) -> Dict[str, Tuple[int,int,int,int]]:
    """
    Load a label file <stem>.txt and return dict with optional keys 'disc' and 'cup'.
    class 0 → 'disc', class 1 → 'cup'
    """
    out: Dict[str, Tuple[int,int,int,int]] = {}
    lp = labels_dir / f"{stem}.txt"
    if not lp.exists():
        return out
    lines = [ln for ln in lp.read_text().strip().splitlines() if ln.strip()]
    for ln in lines:
        parsed = yolo_to_xyxy(ln, W, H)
        if not parsed:
            continue
        cls, x1, y1, x2, y2 = parsed
        if cls == 0:
            out["disc"] = (x1,y1,x2,y2)
        elif cls == 1:
            out["cup"]  = (x1,y1,x2,y2)
    return out

def draw_box(img, box, color, label, thickness, alpha=0.25):
    if not box:
        return img
    x1, y1, x2, y2 = map(int, box)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)
    font_scale = 0.4
    font_thick = 1

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
    bx2 = min(img.shape[1] - 1, x1 + tw + 6)
    by2 = min(img.shape[0] - 1, y1 + th + 6)
    cv2.rectangle(img, (x1, y1), (bx2, by2), color, thickness=-1)
    cv2.putText(img, label, (x1 + 3, y1 + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
    return img

def crop_region(img, box, pad=0):
    if not box:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w-1, x2 + pad); y2 = min(h-1, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()

def make_side_by_side(left_bgr, right_bgr):
    lh, lw = left_bgr.shape[:2]
    rh, rw = right_bgr.shape[:2]
    if lh != rh:
        scale = lh / rh
        right_bgr = cv2.resize(right_bgr, (int(rw*scale), lh), interpolation=cv2.INTER_AREA)
    return cv2.hconcat([left_bgr, right_bgr])

# ----------------- main -----------------

def main():
    args = parse_args()
    labels_dir = Path(args.labels_dir)
    images_dir = Path(args.images_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # optional crop dirs
    crops_dir = out_dir / "crops"
    if args.save_crops:
        (crops_dir / "od").mkdir(parents=True, exist_ok=True)
        (crops_dir / "oc").mkdir(parents=True, exist_ok=True)

    # collect stems from labels_dir
    label_files = sorted([p for p in labels_dir.glob("*.txt")])
    if not label_files:
        print(f"[ERROR] No label files found under {labels_dir}")
        return

    stems = [p.stem for p in label_files]
    random.seed(args.seed)
    if args.sample < len(stems):
        stems = random.sample(stems, args.sample)

    for stem in stems:
        img_path = try_resolve_image(stem, images_dir, args.img_exts)
        if not img_path:
            if args.verbose:
                print(f"[WARN] No image found for stem '{stem}'")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            if args.verbose:
                print(f"[WARN] Failed to read image: {img_path}")
            continue

        W, H = img.shape[1], img.shape[0]
        boxes = load_yolo_boxes_for_stem(labels_dir, stem, W, H)
        if not boxes:
            if args.verbose:
                print(f"[INFO] No boxes for stem '{stem}' (empty label?)")
            continue

        thick = args.thickness if args.thickness > 0 else auto_thickness(H, W)

        # BGR colors
        blue  = (255, 96,  32)  # OD (disc)
        green = (64,  192, 64)  # OC (cup)

        ann = img.copy()
        ann = draw_box(ann, boxes.get("disc"), blue,  "OD (disc)", thick, alpha=args.alpha)
        ann = draw_box(ann, boxes.get("cup"),  green, "OC (cup)",  thick, alpha=args.alpha)

        out_png = out_dir / f"{stem}_two_boxes.png"
        cv2.imwrite(str(out_png), ann)
        if args.verbose:
            print(f"[OK] {out_png}")

        if args.save_crops:
            od_crop = crop_region(img, boxes.get("disc"), pad=args.crop_pad)
            oc_crop = crop_region(img, boxes.get("cup"),  pad=args.crop_pad)
            if od_crop is not None:
                cv2.imwrite(str(crops_dir/"od"/f"{stem}_od.png"), od_crop)
            if oc_crop is not None:
                cv2.imwrite(str(crops_dir/"oc"/f"{stem}_oc.png"), oc_crop)

        if args.make_montage:
            montage = make_side_by_side(img, ann)
            cv2.imwrite(str(out_dir / f"{stem}_montage.png"), montage)

if __name__ == "__main__":
    main()