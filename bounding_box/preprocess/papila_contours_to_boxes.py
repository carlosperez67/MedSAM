#!/usr/bin/env python3
"""
Papila DB: Convert contour .txt files to TWO bounding boxes:
- box_od (disc)
- box_oc (cup)

Key differences from prior script
---------------------------------
- No union box is produced.
- Optional strictness: --require_both (skip records lacking either class)
- Optional fail-fast: --fail_on_missing (error if any image lacks OD or OC)

Assumptions (unchanged)
-----------------------
- Contour files follow:
    <stem>_cup_exp1.txt
    <stem>_cup_e_exp2.txt
    <stem>_disc_exp1.txt
- Each file has lines "<x> <y>" (floats), one point per line.
- Images are optional; if present they enable %-based padding and clamping.

Default Paths (PapilaDB on your OneDrive)
-----------------------------------------
- Contours: /Users/carlosperez/OneDrive_UBC/Ipek_Carlos/GlaucomaDatasets/Papila_db/ExpertsSegmentations/Contours
- Images  : /Users/carlosperez/OneDrive_UBC/Ipek_Carlos/GlaucomaDatasets/Papila_db/FundusImages
- Outputs : ./papila_boxes_two (relative to CWD)

Outputs
-------
1) autoboxes_two.jsonl : {"image","stem","box_od","box_oc"}
2) boxes_two.csv       : image,stem,box_od_x1,box_od_y1,box_od_x2,box_od_y2,box_oc_x1,...
3) (optional) YOLO labels under out_dir/yolo_labels (0=disc, 1=cup)

Usage
-----
python papila_contours_to_two_boxes.py \
  --strategy union \
  --pad_px 8 \
  --pad_pct 0.05 \
  --make_yolo \
  --require_both
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

FNAME_RE = re.compile(
    r"""^(?P<stem>.+?)_            # image stem
        (?P<class>cup|disc)        # class token
        (?:_e)?_                   # optional '_e'
        exp\d+                     # 'exp1', 'exp2', ...
        \.txt$""",
    re.VERBOSE | re.IGNORECASE
)

CLASS_TO_ID = {"disc": 0, "cup": 1}

def parse_args():
    ap = argparse.ArgumentParser(description="Papila contours -> two class boxes (OD & OC)")
    ap.add_argument("--contour_dir", default="/Users/carlosperez/OneDrive_UBC/Ipek_Carlos/GlaucomaDatasets/Papila_db/ExpertsSegmentations/Contours",
                    help="Directory with contour .txt files")
    ap.add_argument("--image_dir", default="/Users/carlosperez/OneDrive_UBC/Ipek_Carlos/GlaucomaDatasets/Papila_db/FundusImages",
                    help="Directory with images (optional; enables % padding and clamping)")
    ap.add_argument("--out_dir", default="./../data/papila_boxes", help="Output directory")

    ap.add_argument("--strategy", choices=["union", "largest"], default="union",
                    help="Combine multiple contours per class: union of boxes or largest-area polygon")
    ap.add_argument("--pad_px", type=int, default=8, help="Absolute padding (pixels) per side")
    ap.add_argument("--pad_pct", type=float, default=0.01,
                    help="Relative padding per side as a fraction of max(image_w, image_h)")
    ap.add_argument("--make_yolo", default="true", action="store_true", help="Also write YOLO label files")
    ap.add_argument("--require_both", action="store_true",
                    help="Only write rows where BOTH OD and OC boxes exist after padding")
    ap.add_argument("--fail_on_missing", action="store_true",
                    help="Exit with error if any stem lacks OD or OC")
    ap.add_argument("--img_exts", nargs="+", default=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                    help="Candidate image extensions under --image_dir")
    return ap.parse_args()

def read_contour_txt(path: str) -> Optional[np.ndarray]:
    pts = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0]); y = float(parts[1])
                pts.append((x, y))
            except ValueError:
                continue
    if not pts:
        return None
    return np.array(pts, dtype=np.float64)

def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]; y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def bbox_from_points(points: np.ndarray) -> Tuple[int, int, int, int]:
    x1 = int(np.floor(points[:, 0].min()))
    y1 = int(np.floor(points[:, 1].min()))
    x2 = int(np.ceil(points[:, 0].max()))
    y2 = int(np.ceil(points[:, 1].max()))
    return x1, y1, x2, y2

def union_boxes(boxes: List[Tuple[int,int,int,int]]) -> Optional[Tuple[int,int,int,int]]:
    if not boxes:
        return None
    xs1, ys1, xs2, ys2 = zip(*boxes)
    return min(xs1), min(ys1), max(xs2), max(ys2)

def clamp_box(box, w: int, h: int) -> Tuple[int,int,int,int]:
    x1,y1,x2,y2 = box
    return max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)

def pad_box(box, pad_px: int, img_wh: Optional[Tuple[int,int]], pad_pct: float) -> Tuple[int,int,int,int]:
    x1,y1,x2,y2 = box
    pad = pad_px
    if img_wh and pad_pct > 0.0:
        w,h = img_wh
        p = int(round(pad_pct * max(w,h)))
        pad = max(pad, p)
    return x1 - pad, y1 - pad, x2 + pad, y2 + pad

def find_image_path(stem: str, image_dir: Optional[str], img_exts: List[str]) -> Optional[str]:
    if not image_dir:
        return None
    for ext in img_exts:
        p = os.path.join(image_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

def boxes_to_yolo_lines(boxes_by_class: Dict[str, Tuple[int,int,int,int]], w: int, h: int) -> List[str]:
    lines = []
    for cls_name, box in boxes_by_class.items():
        if box is None:
            continue
        cls_id = CLASS_TO_ID.get(cls_name.lower())
        if cls_id is None:
            continue
        x1,y1,x2,y2 = box
        bw = max(0, x2-x1); bh = max(0, y2-y1)
        if bw <= 0 or bh <= 0:
            continue
        cx = x1 + bw/2.0; cy = y1 + bh/2.0
        lines.append(f"{cls_id} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}")
    return lines

def main():
    args = parse_args()
    print(args)
    os.makedirs(args.out_dir, exist_ok=True)

    # Collect contours: {stem: {"disc": [pts...], "cup": [pts...]}}
    contours: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    for root, _, files in os.walk(args.contour_dir):
        for fn in files:
            if not fn.lower().endswith(".txt"):
                continue
            m = FNAME_RE.match(fn)
            if not m:
                continue
            stem = m.group("stem")
            cls  = m.group("class").lower()  # 'disc' or 'cup'
            path = os.path.join(root, fn)
            pts = read_contour_txt(path)
            if pts is not None and pts.shape[0] >= 3:
                contours[stem][cls].append(pts)

    stems = sorted(contours.keys())
    if not stems:
        print("No matching contour files found. Check naming and --contour_dir.")
        return

    jsonl_path = os.path.join(args.out_dir, "autoboxes_two.jsonl")
    csv_path   = os.path.join(args.out_dir, "boxes_two.csv")
    yolo_dir   = os.path.join(args.out_dir, "yolo_labels") if args.make_yolo else None
    if yolo_dir:
        os.makedirs(yolo_dir, exist_ok=True)

    csv_cols = [
        "image","stem",
        "box_od_x1","box_od_y1","box_od_x2","box_od_y2",
        "box_oc_x1","box_oc_y1","box_oc_x2","box_oc_y2",
    ]

    missing_any = False

    with open(jsonl_path, "w") as jf, open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_cols); writer.writeheader()

        for stem in stems:
            cls_dict = contours[stem]

            def choose_box(list_of_pts: List[np.ndarray]) -> Optional[Tuple[int,int,int,int]]:
                if not list_of_pts:
                    return None
                if args.strategy == "largest":
                    areas = [polygon_area(p) for p in list_of_pts]
                    pts = list_of_pts[int(np.argmax(areas))]
                    return bbox_from_points(pts)
                else:
                    boxes = [bbox_from_points(p) for p in list_of_pts]
                    return union_boxes(boxes)

            box_od = choose_box(cls_dict.get("disc", []))
            box_oc = choose_box(cls_dict.get("cup", []))

            img_path = find_image_path(stem, args.image_dir, args.img_exts)
            img_w = img_h = None
            if img_path and PIL_AVAILABLE:
                try:
                    with Image.open(img_path) as im:
                        img_w, img_h = im.size
                except Exception:
                    img_w = img_h = None

            def pad_and_clamp(box):
                if box is None:
                    return None
                pb = pad_box(box, args.pad_px, (img_w, img_h) if (img_w and img_h) else None, args.pad_pct)
                if img_w and img_h:
                    return clamp_box(pb, img_w, img_h)
                return pb

            box_od = pad_and_clamp(box_od)
            box_oc = pad_and_clamp(box_oc)

            if args.require_both and (box_od is None or box_oc is None):
                missing_any = True
                continue

            jf.write(json.dumps({
                "image": img_path if img_path else stem,
                "stem": stem,
                "box_od": box_od,
                "box_oc": box_oc
            }) + "\n")

            row = {
                "image": img_path if img_path else "",
                "stem": stem,
                "box_od_x1": "", "box_od_y1": "", "box_od_x2": "", "box_od_y2": "",
                "box_oc_x1": "", "box_oc_y1": "", "box_oc_x2": "", "box_oc_y2": "",
            }
            if box_od: row.update({"box_od_x1": box_od[0], "box_od_y1": box_od[1], "box_od_x2": box_od[2], "box_od_y2": box_od[3]})
            if box_oc: row.update({"box_oc_x1": box_oc[0], "box_oc_y1": box_oc[1], "box_oc_x2": box_oc[2], "box_oc_y2": box_oc[3]})
            writer.writerow(row)

            if yolo_dir and img_w and img_h:
                yolo_lines = {}
                if box_od: yolo_lines["disc"] = box_od
                if box_oc: yolo_lines["cup"]  = box_oc
                lines = boxes_to_yolo_lines(yolo_lines, img_w, img_h)
                with open(os.path.join(yolo_dir, stem + ".txt"), "w") as lf:
                    lf.write("\n".join(lines))

    print(f"[OK] Wrote JSONL: {jsonl_path}")
    print(f"[OK] Wrote CSV  : {csv_path}")
    if yolo_dir:
        print(f"[OK] Wrote YOLO labels to: {yolo_dir}")
    if args.require_both and missing_any:
        print("[INFO] Some stems skipped because one of OD/OC was missing (per --require_both).")
    if args.fail_on_missing and missing_any:
        raise SystemExit("[FAIL] Missing OD or OC for at least one stem.")

if __name__ == "__main__":
    main()