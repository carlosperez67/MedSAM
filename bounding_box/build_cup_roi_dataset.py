#!/usr/bin/env python3
# build_cup_roi_dataset.py
import argparse, os, math, shutil
from pathlib import Path
from typing import Tuple, List

import cv2
from PIL import Image

"""
Create a 'cup-in-ROI' YOLO dataset:
- Reads full images + labels (0=disc, 1=cup)
- Makes square crops around disc boxes with padding
- Rewrites cup boxes into crop coordinates (single class: 0=cup)
- Skips samples with no cup visible inside ROI (optional)

Output structure:
DATA_ROOT_cupROI/
  images/{train,val,test}/*.png
  labels/{train,val,test}/*.txt
  cup_roi.yaml   # names: ["cup"]
"""

IMG_EXTS = (".png",".jpg",".jpeg",".tif",".tiff")

def read_yolo_boxes(txt_path: Path) -> List[List[float]]:
    if not txt_path.exists(): return []
    rows = []
    for line in txt_path.read_text().strip().splitlines():
        if not line.strip(): continue
        parts = line.strip().split()
        rows.append([float(x) for x in parts])
    return rows

def yolo_to_xyxy(row, w, h) -> Tuple[int,int,int,int,int]:
    cls, cx, cy, bw, bh = row[:5]
    x1 = (cx - bw/2.0)*w; y1 = (cy - bh/2.0)*h
    x2 = (cx + bw/2.0)*w; y2 = (cy + bh/2.0)*h
    return int(cls), int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def clamp(v, lo, hi): return max(lo, min(hi, v))

def ensure_square_expand(x1,y1,x2,y2, pad, W,H):
    # get square side as max(width,height) then add pad fraction of max(W,H)
    w = x2 - x1; h = y2 - y1
    side = max(w, h)
    extra = int(round(pad * max(W, H)))
    side = side + 2*extra
    # center
    cx = (x1 + x2)//2; cy = (y1 + y2)//2
    nx1 = clamp(cx - side//2, 0, W-1)
    ny1 = clamp(cy - side//2, 0, H-1)
    nx2 = clamp(nx1 + side, 1, W)
    ny2 = clamp(ny1 + side, 1, H)
    # re-clamp if overflow on far side
    if nx2 - nx1 != side:
        nx1 = clamp(W - side, 0, W-1); nx2 = W
    if ny2 - ny1 != side:
        ny1 = clamp(H - side, 0, H-1); ny2 = H
    return nx1, ny1, nx2, ny2

def box_intersection(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1); y1 = max(ay1, by1)
    x2 = min(ax2, bx2); y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1: return None
    return (x1,y1,x2,y2)

def xyxy_to_yolo(box, W, H):
    x1,y1,x2,y2 = box
    bw = (x2 - x1); bh = (y2 - y1)
    cx = x1 + bw/2.0; cy = y1 + bh/2.0
    return cx/W, cy/H, bw/W, bh/H

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./papila_yolo", help="Full-image YOLO dataset root")
    ap.add_argument("--out_root",  default="./papila_yolo_cupROI", help="Output cup-ROI dataset root")
    ap.add_argument("--pad_pct",   type=float, default=0.10, help="Padding (fraction of max image dim) around disc")
    ap.add_argument("--keep_negatives", action="store_true", help="Keep ROI crops with no cup visible")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)
    for split in ("train","val","test"):
        (out_root/"images"/split).mkdir(parents=True, exist_ok=True)
        (out_root/"labels"/split).mkdir(parents=True, exist_ok=True)

    total, kept = 0, 0
    for split in ("train","val","test"):
        img_dir = data_root/"images"/split
        lbl_dir = data_root/"labels"/split
        for img_path in sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]):
            total += 1
            W,H = Image.open(img_path).size
            rows = read_yolo_boxes(lbl_dir/f"{img_path.stem}.txt")
            # find disc box (class 0); pick top-most row (no scores in GT)
            disc = None; cups=[]
            for r in rows:
                cls,x1,y1,x2,y2 = yolo_to_xyxy(r,W,H)
                if cls == 0: disc = (x1,y1,x2,y2)
                elif cls == 1: cups.append((x1,y1,x2,y2))
            if disc is None:
                continue
            # build square ROI around disc with padding
            rx1,ry1,rx2,ry2 = ensure_square_expand(*disc, pad=args.pad_pct, W=W, H=H)
            roi = (rx1,ry1,rx2,ry2)

            # crop image
            im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            crop = im[ry1:ry2, rx1:rx2].copy()
            cH, cW = crop.shape[:2]

            # transform cups into ROI coords
            out_lines=[]
            for cup in cups:
                inter = box_intersection(cup, roi)
                if inter is None:
                    continue
                ix1,iy1,ix2,iy2 = inter
                # translate to ROI local
                lx1 = ix1 - rx1; ly1 = iy1 - ry1
                lx2 = ix2 - rx1; ly2 = iy2 - ry1
                cx,cy,bw,bh = xyxy_to_yolo((lx1,ly1,lx2,ly2), cW, cH)
                # single class dataset: 0=cup
                out_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not out_lines and not args.keep_negatives:
                continue  # skip crops with no cup inside ROI

            # write crop + label
            cv_out = out_root/"images"/split/f"{img_path.stem}_roi.png"
            lb_out = out_root/"labels"/split/f"{img_path.stem}_roi.txt"
            cv2.imwrite(str(cv_out), crop)
            with open(lb_out, "w") as f:
                f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

            kept += 1

    # YAML for cup-only dataset
    names = ["cup"]
    yaml_path = out_root/"cup_roi.yaml"
    import yaml
    with open(yaml_path,"w") as f:
        yaml.safe_dump({"path": str(out_root.resolve()), "train":"images/train", "val":"images/val", "names": names}, f, sort_keys=False)

    print(f"[OK] Built ROI dataset at: {out_root}")
    print(f"  images processed: {total} | crops kept: {kept}")
    print(f"[OK] Wrote data yaml: {yaml_path}")

if __name__ == "__main__":
    main()