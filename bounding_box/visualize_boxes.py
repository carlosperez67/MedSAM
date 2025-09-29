#!/usr/bin/env python3
"""
viz_two_boxes.py — Visualize PapilaDB OD/OC bounding boxes (two-box version)

Reads:  autoboxes_two.jsonl  (from papila_contours_to_two_boxes.py)
Draws : OD (disc, blue) and OC (cup, green)
Saves : annotated PNGs; optional per-class crops; optional side-by-side montage.

Defaults:
  - autoboxes_two.jsonl: ./papila_boxes_two/autoboxes_two.jsonl
  - fallback image_dir : /Users/carlosperez/OneDrive_UBC/Ipek_Carlos/GlaucomaDatasets/Papila_db/FundusImages
  - outputs            : ./papila_boxes_two/viz/

Usage:
  python viz_two_boxes.py
  python viz_two_boxes.py --sample 12 --save_crops --make_montage
"""

import argparse, os, json, random
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
from PIL import Image

DEF_IMAGE_DIR = "/Users/carlosperez/OneDrive_UBC/Ipek_Carlos/GlaucomaDatasets/Papila_db/FundusImages"
DEF_AUTOBOXES = "./papila_boxes_two/autoboxes_two.jsonl"
DEF_OUT_DIR   = "./papila_boxes_two/viz"

def parse_args():
    ap = argparse.ArgumentParser(description="Visualize OD/OC boxes and optionally save crops/montage.")
    ap.add_argument("--autoboxes", default=DEF_AUTOBOXES, help="Path to autoboxes_two.jsonl")
    ap.add_argument("--out_dir",   default=DEF_OUT_DIR,   help="Output directory for visualizations")
    ap.add_argument("--image_dir", default=DEF_IMAGE_DIR, help="Fallback image dir if JSONL lacks absolute paths")
    ap.add_argument("--img_exts",  nargs="+", default=[".png",".jpg",".jpeg",".tif",".tiff"],
                    help="Candidate image extensions for fallback resolution")
    ap.add_argument("--sample", type=int, default=8, help="Random sample size (use >= total to process all)")
    ap.add_argument("--seed",   type=int, default=1337, help="Random seed for sampling")
    ap.add_argument("--thickness", type=int, default=0, help="Line thickness (0 = auto)")
    ap.add_argument("--alpha",     type=float, default=0.25, help="Fill transparency 0..1")
    ap.add_argument("--save_crops", action="store_true", help="Also save OD/OC crops to subfolders")
    ap.add_argument("--crop_pad",   type=int, default=0, help="Extra pixels to pad around the crop")
    ap.add_argument("--make_montage", action="store_true",
                    help="Save side-by-side montage: [original | annotated]")
    return ap.parse_args()

def auto_thickness(h: int, w: int) -> int:
    return max(2, int(0.002 * max(h, w)))

def try_resolve_image(stem: str, image_dir: str, img_exts: List[str]) -> Optional[str]:
    for ext in img_exts:
        p = Path(image_dir) / f"{stem}{ext}"
        if p.exists():
            return str(p)
    return None

def draw_box(img, box, color, label, thickness, alpha=0.25):
    if box is None:
        return img
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1]-1, x2), min(img.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return img

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    bx2 = min(img.shape[1]-1, x1 + tw + 8)
    by2 = min(img.shape[0]-1, y1 + th + 8)
    cv2.rectangle(img, (x1, y1), (bx2, by2), color, thickness=-1)
    cv2.putText(img, label, (x1 + 4, y1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return img

def crop_region(img, box, pad=0):
    if box is None:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w-1, x2 + pad); y2 = min(h-1, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()

def make_side_by_side(left_bgr, right_bgr):
    """Return a simple side-by-side (same height) montage."""
    lh, lw = left_bgr.shape[:2]
    rh, rw = right_bgr.shape[:2]
    if lh != rh:
        # scale right to left's height
        scale = lh / rh
        right_bgr = cv2.resize(right_bgr, (int(rw*scale), lh), interpolation=cv2.INTER_AREA)
    return cv2.hconcat([left_bgr, right_bgr])

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir).mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    (crops_dir / "od").mkdir(parents=True, exist_ok=True) if args.save_crops else None
    (crops_dir / "oc").mkdir(parents=True, exist_ok=True) if args.save_crops else None

    # Load JSONL
    records = []
    with open(args.autoboxes, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception:
                continue
    if not records:
        print(f"[ERROR] No records found in {args.autoboxes}")
        return

    random.seed(args.seed)
    sample = records if args.sample >= len(records) else random.sample(records, args.sample)

    for rec in sample:
        img_path = rec.get("image") or ""
        stem = rec.get("stem") or Path(img_path).stem

        if not img_path or not Path(img_path).exists():
            img_path = try_resolve_image(stem, args.image_dir, args.img_exts)
            if not img_path:
                print(f"[WARN] Could not resolve image for stem '{stem}'. Skipping.")
                continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        h, w = img.shape[:2]
        thick = args.thickness if args.thickness > 0 else auto_thickness(h, w)

        # Colors (BGR)
        blue  = (255, 96,  32)  # OD (disc)
        green = (64,  192, 64)  # OC (cup)

        ann = img.copy()
        ann = draw_box(ann, rec.get("box_od"), blue,  "OD (disc)", thick, alpha=args.alpha)
        ann = draw_box(ann, rec.get("box_oc"), green, "OC (cup)",  thick, alpha=args.alpha)

        # Save annotated overlay
        out_png = out_dir / f"{stem}_two_boxes.png"
        cv2.imwrite(str(out_png), ann)
        print(f"[OK] {out_png}")

        # Optional crops
        if args.save_crops:
            od_crop = crop_region(img, rec.get("box_od"), pad=args.crop_pad)
            oc_crop = crop_region(img, rec.get("box_oc"), pad=args.crop_pad)
            if od_crop is not None:
                cv2.imwrite(str(crops_dir/"od"/f"{stem}_od.png"), od_crop)
            if oc_crop is not None:
                cv2.imwrite(str(crops_dir/"oc"/f"{stem}_oc.png"), oc_crop)

        # Optional original|annotated montage
        if args.make_montage:
            montage = make_side_by_side(img, ann)
            cv2.imwrite(str(out_dir / f"{stem}_montage.png"), montage)

if __name__ == "__main__":
    main()