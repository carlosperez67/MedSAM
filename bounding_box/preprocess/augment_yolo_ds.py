#!/usr/bin/env python3
# augment_yolo_ds.py
import argparse, os
from pathlib import Path
import albumentations as A
import cv2
import shutil
import yaml
import math
import random

# ----------------- Utils -----------------
def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(folder: Path, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff")):
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in exts])

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ----------------- Augmentations -----------------
def build_transform(oh, ow):
    # Albumentations v2-friendly API; returns bboxes and class_labels
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                shear=(-5, 5),
                fit_output=False,
                p=0.7,
            ),
            A.RandomResizedCrop(size=(oh, ow), scale=(0.9, 1.0), ratio=(0.9, 1.1), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2,
        ),
    )

def _sanitize_yolo_boxes(boxes):
    """
    Clamp YOLO boxes to [0,1] by converting to xyxy, clamping, then back to YOLO.
    boxes: list of [cx, cy, w, h]
    """
    clean = []
    for cx, cy, w, h in boxes:
        cx = float(cx); cy = float(cy); w = float(w); h = float(h)
        # xyxy normalized
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        # clamp
        x1 = clamp(x1, 0.0, 1.0)
        y1 = clamp(y1, 0.0, 1.0)
        x2 = clamp(x2, 0.0, 1.0)
        y2 = clamp(y2, 0.0, 1.0)
        w2 = max(0.0, x2 - x1)
        h2 = max(0.0, y2 - y1)
        if w2 <= 1e-6 or h2 <= 1e-6:
            continue
        cx2 = (x1 + x2) / 2.0
        cy2 = (y1 + y2) / 2.0
        clean.append([cx2, cy2, w2, h2])
    return clean

# ----------------- Geometry / conversions -----------------
def yolo_to_xyxy_abs(box_yolo, W, H):
    cx, cy, w, h = box_yolo
    x1 = (cx - w/2.0) * W
    y1 = (cy - h/2.0) * H
    x2 = (cx + w/2.0) * W
    y2 = (cy + h/2.0) * H
    return [x1, y1, x2, y2]

def xyxy_abs_to_yolo(box_xyxy, W, H):
    x1, y1, x2, y2 = box_xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 0 or bh <= 0:
        return None
    cx = x1 + bw/2.0
    cy = y1 + bh/2.0
    return [cx / W, cy / H, bw / W, bh / H]

def box_i_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1); y1 = max(ay1, by1)
    x2 = min(ax2, bx2); y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, None
    return (x2 - x1) * (y2 - y1), [x1, y1, x2, y2]

# ----------------- Tiling -----------------
def tile_image_and_boxes(im, boxes_yolo, labels, tile_size=512, overlap=0.2, min_tile_vis=0.2, keep_empty=False):
    """
    Split the image into overlapping square tiles and project boxes into each tile.
    - boxes_yolo: list of [cx, cy, w, h] normalized to full image
    - labels    : list of class ids (same length)
    Returns list of tuples: (tile_img, tile_boxes_yolo, tile_labels, suffix)
    """
    H, W = im.shape[:2]
    ts = int(tile_size)
    ts = max(16, ts)
    stride = max(1, int(round(ts * (1.0 - overlap))))

    abs_boxes = [yolo_to_xyxy_abs(b, W, H) for b in boxes_yolo]
    tiles = []
    for y0 in range(0, max(1, H - ts + 1), stride):
        for x0 in range(0, max(1, W - ts + 1), stride):
            x1 = x0 + ts
            y1 = y0 + ts
            if x1 > W:
                x0 = max(0, W - ts); x1 = W
            if y1 > H:
                y0 = max(0, H - ts); y1 = H
            tile_rect = [x0, y0, x1, y1]
            tile_w = x1 - x0; tile_h = y1 - y0
            if tile_w <= 1 or tile_h <= 1:
                continue

            t_boxes_yolo, t_labels = [], []
            for b_xyxy, cls in zip(abs_boxes, labels):
                box_area = max(0.0, (b_xyxy[2] - b_xyxy[0])) * max(0.0, (b_xyxy[3] - b_xyxy[1]))
                if box_area <= 0:
                    continue
                inter_area, inter = box_i_area(b_xyxy, tile_rect)
                if inter_area <= 0:
                    continue
                vis = inter_area / (box_area + 1e-12)
                if vis < min_tile_vis:
                    continue
                ix1, iy1, ix2, iy2 = inter
                lx1 = ix1 - x0; ly1 = iy1 - y0
                lx2 = ix2 - x0; ly2 = iy2 - y0
                yolo_local = xyxy_abs_to_yolo([lx1, ly1, lx2, ly2], tile_w, tile_h)
                if yolo_local is None:
                    continue
                t_boxes_yolo.append(yolo_local)
                t_labels.append(cls)

            if t_boxes_yolo or keep_empty:
                tile_img = im[y0:y1, x0:x1].copy()
                suffix = f"t{y0}_{x0}"
                tiles.append((tile_img, t_boxes_yolo, t_labels, suffix))
    return tiles

# ----------------- Zoom crops (object-centric) -----------------
def square_crop_bounds(cx, cy, side, W, H):
    """Return [x0,y0,x1,y1] for a square of given side centered near (cx,cy) clamped to image."""
    half = side / 2.0
    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    x1 = x0 + int(round(side)); y1 = y0 + int(round(side))
    # clamp to borders (shift window if needed)
    if x0 < 0:
        x1 -= x0; x0 = 0
    if y0 < 0:
        y1 -= y0; y0 = 0
    if x1 > W:
        shift = x1 - W
        x0 = max(0, x0 - shift); x1 = W
    if y1 > H:
        shift = y1 - H
        y0 = max(0, y0 - shift); y1 = H
    # final sanity
    x0 = clamp(x0, 0, max(0, W-1)); y0 = clamp(y0, 0, max(0, H-1))
    x1 = clamp(x1, 1, W); y1 = clamp(y1, 1, H)
    return [x0, y0, x1, y1]

def make_zoom_crops(
    im, boxes_yolo, labels,
    zoom_scales, zoom_per_obj=1, zoom_on="both",
    zoom_out_size=640, zoom_jitter=0.05, min_vis=0.2, keep_empty=False
):
    """
    For each chosen object, make square crops at multiple scales centered (with jitter) on the object.
    - zoom_scales: list of fractions of the *shorter* side (e.g., [0.5, 0.65])
    - zoom_per_obj: how many random jitters per object per scale
    - zoom_on: 'disc' (0), 'cup' (1), 'both', or 'any'
    - zoom_out_size: if >0, resize crop to (S,S). If 0, keep native crop size.
    - Returns: list of (crop_img, crop_boxes_yolo, crop_labels, suffix)
    """
    H, W = im.shape[:2]
    short_side = min(W, H)
    # Convert full boxes to absolute xyxy once
    abs_boxes = [yolo_to_xyxy_abs(b, W, H) for b in boxes_yolo]

    def select_indices():
        if zoom_on == "disc":
            return [i for i,l in enumerate(labels) if l == 0]
        elif zoom_on == "cup":
            return [i for i,l in enumerate(labels) if l == 1]
        elif zoom_on == "both":
            return list(range(len(labels)))
        elif zoom_on == "any":
            return list(range(len(labels))) if labels else []
        else:
            return list(range(len(labels)))

    idxs = select_indices()
    results = []

    for i in idxs:
        x1, y1, x2, y2 = abs_boxes[i]
        # object center
        ocx = (x1 + x2) / 2.0
        ocy = (y1 + y2) / 2.0

        for s in zoom_scales:
            side = max(16, int(round(s * short_side)))  # square crop side
            for k in range(zoom_per_obj):
                jx = (random.uniform(-zoom_jitter, zoom_jitter)) * side
                jy = (random.uniform(-zoom_jitter, zoom_jitter)) * side
                cx = ocx + jx
                cy = ocy + jy
                rx0, ry0, rx1, ry1 = square_crop_bounds(cx, cy, side, W, H)
                roi_w = rx1 - rx0; roi_h = ry1 - ry0
                if roi_w <= 1 or roi_h <= 1:
                    continue

                # compute boxes inside crop
                crop_boxes_yolo, crop_labels = [], []
                for b_xyxy, cls in zip(abs_boxes, labels):
                    box_area = max(0.0, (b_xyxy[2] - b_xyxy[0])) * max(0.0, (b_xyxy[3] - b_xyxy[1]))
                    if box_area <= 0:
                        continue
                    inter_area, inter = box_i_area(b_xyxy, [rx0, ry0, rx1, ry1])
                    if inter_area <= 0:
                        continue
                    vis = inter_area / (box_area + 1e-12)
                    if vis < min_vis:
                        continue
                    ix1, iy1, ix2, iy2 = inter
                    lx1 = ix1 - rx0; ly1 = iy1 - ry0
                    lx2 = ix2 - rx0; ly2 = iy2 - ry0
                    crop_yolo = xyxy_abs_to_yolo([lx1, ly1, lx2, ly2], roi_w, roi_h)
                    if crop_yolo is None:
                        continue
                    crop_boxes_yolo.append(crop_yolo)
                    crop_labels.append(cls)

                if not crop_boxes_yolo and not keep_empty:
                    continue

                crop_img = im[ry0:ry1, rx0:rx1].copy()
                out_suffix = f"z{i}_s{int(round(s*100))}_j{k}"

                # optional resize to square
                if zoom_out_size and zoom_out_size > 0:
                    S = int(zoom_out_size)
                    crop_img = cv2.resize(crop_img, (S, S), interpolation=cv2.INTER_AREA)
                    # YOLO boxes are already normalized to crop; resizing doesn't change normalized values

                results.append((crop_img, crop_boxes_yolo, crop_labels, out_suffix))

    return results

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="Augment a YOLO dataset (disc/cup) with flips/affine/crops + optional tiling + object-centric zoom crops.")
    ap.add_argument("--project_dir", default="", help="Base project directory (root of MedSAM).")
    ap.add_argument("--data_root", default=None,
        help="YOLO dataset root. Default: {PROJECT_DIR}/bounding_box/data/yolo_split")
    ap.add_argument("--out_root", default=None,
        help="Output augmented dataset root. Default: {PROJECT_DIR}/bounding_box/data/yolo_split_aug")
    ap.add_argument("--multiplier", type=int, default=2,
        help="How many augmented copies to create per image.")
    ap.add_argument("--out_ext", default=".jpg",
        help="Output image extension for augmented/tiles/zooms (e.g., .jpg, .png).")
    ap.add_argument("--write_yaml", action="store_true",
        help="Write a data.yaml pointing to the augmented dataset.")

    # Tiling controls
    ap.add_argument("--enable_tiling", action="store_true",
        help="Enable overlapping tiling to create additional crops per image.")
    ap.add_argument("--tile_size", type=int, default=512, help="Square tile size in pixels.")
    ap.add_argument("--tile_overlap", type=float, default=0.2,
        help="Overlap fraction between tiles (0.0–0.95). stride = size*(1-overlap)")
    ap.add_argument("--min_tile_vis", type=float, default=0.2,
        help="Minimum fraction of an object's area that must fall inside a tile to keep it.")
    ap.add_argument("--keep_empty_tiles", action="store_true",
        help="Keep tiles that have no boxes.")
    ap.add_argument("--tile_from_aug", action="store_true",
        help="If set, also tile each augmented image (not just the original).")

    # Zoom crops
    ap.add_argument("--enable_zoom_crops", action="store_true",
        help="Enable object-centric zoom crops (square crops at smaller scales).")
    ap.add_argument("--zoom_scales", default="0.5,0.7,0.85",
        help="Comma-separated scales (fractions of shorter side) for square crops, e.g. '0.5,0.7'. Smaller = more zoom.")
    ap.add_argument("--zoom_per_obj", type=int, default=1,
        help="How many jittered crops per object per scale.")
    ap.add_argument("--zoom_on", default="both", choices=["disc","cup","both","any"],
        help="Which objects to center zooms on.")
    ap.add_argument("--zoom_out_size", type=int, default=640,
        help="Resize each zoom crop to this square size (pixels). 0 = keep native.")
    ap.add_argument("--zoom_jitter", type=float, default=0.05,
        help="Random center jitter as a fraction of crop side.")
    ap.add_argument("--zoom_min_vis", type=float, default=0.2,
        help="Minimum fraction of an object's area that must be inside the zoom crop.")
    ap.add_argument("--zoom_from_aug", action="store_true",
        help="If set, also make zoom crops from the augmented images/labels.")
    # argparse additions
    ap.add_argument("--zoom_keep_empty", action="store_true",
                    help="Keep zoom crops even if they contain no boxes (negatives).")

    args = ap.parse_args()

    # Parse scales
    zoom_scales = [float(s) for s in args.zoom_scales.split(",") if s.strip()]
    zoom_scales = [s for s in zoom_scales if s > 0 and s <= 1.0]

    PROJECT_DIR = _expand(args.project_dir)
    data_root = _expand(args.data_root) if args.data_root else (PROJECT_DIR / "bounding_box" / "data" / "yolo_split")
    out_root  = _expand(args.out_root)  if args.out_root  else (PROJECT_DIR / "bounding_box" / "data" / "yolo_split_aug")

    print(f"[INFO] Input : {data_root}")
    print(f"[INFO] Output: {out_root}")

    for split in ("train", "val", "test"):
        in_img_dir  = data_root / "images" / split
        in_lbl_dir  = data_root / "labels" / split
        out_img_dir = out_root  / "images" / split
        out_lbl_dir = out_root  / "labels" / split

        if not in_img_dir.exists() or not in_lbl_dir.exists():
            print(f"[WARN] Missing split '{split}' in input dataset; skipping.")
            continue

        _ensure_dir(out_img_dir)
        _ensure_dir(out_lbl_dir)

        for img_path in list_images(in_img_dir):
            lbl_path = in_lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            # read YOLO boxes (cls cx cy w h)
            boxes, labels = [], []
            lines = [ln.strip() for ln in lbl_path.read_text().splitlines() if ln.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                boxes.append([x, y, w, h])
                labels.append(cls)
            boxes = _sanitize_yolo_boxes(boxes)

            im = cv2.imread(str(img_path))
            if im is None:
                print(f"[WARN] Failed to read image: {img_path}")
                continue
            h, w = im.shape[:2]
            transform = build_transform(h, w)

            # --- (A) write original (respect --out_ext) ---
            orig_img_out = out_img_dir / f"{img_path.stem}{args.out_ext}"
            if args.out_ext.lower() == img_path.suffix.lower():
                shutil.copy2(img_path, orig_img_out)
            else:
                cv2.imwrite(str(orig_img_out), im)
            (out_lbl_dir / f"{img_path.stem}.txt").write_text(
                "\n".join(f"{l} {' '.join(f'{x:.6f}' for x in b)}" for b, l in zip(boxes, labels)) + ("\n" if boxes else "")
            )

            # --- (B) optional tiling of the original ---
            if args.enable_tiling:
                tiles = tile_image_and_boxes(
                    im, boxes, labels,
                    tile_size=args.tile_size,
                    overlap=args.tile_overlap,
                    min_tile_vis=args.min_tile_vis,
                    keep_empty=args.keep_empty_tiles,
                )
                for (timg, tboxes, tlabs, suffix) in tiles:
                    t_img_name = f"{img_path.stem}_{suffix}{args.out_ext}"
                    t_lbl_name = f"{img_path.stem}_{suffix}.txt"
                    cv2.imwrite(str(out_img_dir / t_img_name), timg)
                    (out_lbl_dir / t_lbl_name).write_text(
                        "\n".join(f"{l} {' '.join(f'{x:.6f}' for x in b)}" for b, l in zip(tboxes, tlabs)) + ("\n" if tboxes else "")
                    )

            # --- (C) optional zoom crops of the original ---
            if args.enable_zoom_crops and boxes:
                zooms = make_zoom_crops(
                    im, boxes, labels,
                    zoom_scales=zoom_scales,
                    zoom_per_obj=args.zoom_per_obj,
                    zoom_on=args.zoom_on,
                    zoom_out_size=args.zoom_out_size,
                    zoom_jitter=args.zoom_jitter,
                    min_vis=args.zoom_min_vis,
                    keep_empty=args.zoom_keep_empty,  # <-- was False
                )
                for (zimg, zboxes, zlabs, suffix) in zooms:
                    z_img_name = f"{img_path.stem}_{suffix}{args.out_ext}"
                    z_lbl_name = f"{img_path.stem}_{suffix}.txt"
                    cv2.imwrite(str(out_img_dir / z_img_name), zimg)
                    (out_lbl_dir / z_lbl_name).write_text(
                        "\n".join(f"{l} {' '.join(f'{x:.6f}' for x in b)}" for b, l in zip(zboxes, zlabs)) + ("\n" if zboxes else "")
                    )

            # --- (D) augmented variants (+ optional tiling/zooms) ---
            for k in range(args.multiplier):
                if boxes:
                    aug = transform(image=im, bboxes=boxes, class_labels=labels)
                    aug_img, aug_boxes, aug_labels = aug["image"], aug["bboxes"], aug["class_labels"]
                else:
                    aug = transform(image=im, bboxes=[], class_labels=[])
                    aug_img, aug_boxes, aug_labels = aug["image"], [], []

                aug_img_name = f"{img_path.stem}_aug{k}{args.out_ext}"
                aug_lbl_name = f"{img_path.stem}_aug{k}.txt"
                cv2.imwrite(str(out_img_dir / aug_img_name), aug_img)
                (out_lbl_dir / aug_lbl_name).write_text(
                    "\n".join(f"{l} {' '.join(f'{x:.6f}' for x in b)}" for b, l in zip(aug_boxes, aug_labels)) + ("\n" if aug_boxes else "")
                )

                if args.enable_tiling and args.tile_from_aug:
                    tiles = tile_image_and_boxes(
                        aug_img, aug_boxes, aug_labels,
                        tile_size=args.tile_size,
                        overlap=args.tile_overlap,
                        min_tile_vis=args.min_tile_vis,
                        keep_empty=args.keep_empty_tiles,
                    )
                    for (timg, tboxes, tlabs, suffix) in tiles:
                        t_img_name = f"{img_path.stem}_aug{k}_{suffix}{args.out_ext}"
                        t_lbl_name = f"{img_path.stem}_aug{k}_{suffix}.txt"
                        cv2.imwrite(str(out_img_dir / t_img_name), timg)
                        (out_lbl_dir / t_lbl_name).write_text(
                            "\n".join(f"{l} {' '.join(f'{x:.6f}' for x in b)}" for b, l in zip(tboxes, tlabs)) + ("\n" if tboxes else "")
                        )

                if args.enable_zoom_crops and args.zoom_from_aug and aug_boxes:
                    zooms = make_zoom_crops(
                        aug_img, aug_boxes, aug_labels,
                        zoom_scales=zoom_scales,
                        zoom_per_obj=args.zoom_per_obj,
                        zoom_on=args.zoom_on,
                        zoom_out_size=args.zoom_out_size,
                        zoom_jitter=args.zoom_jitter,
                        min_vis=args.zoom_min_vis,
                        keep_empty=False,
                    )
                    for (zimg, zboxes, zlabs, suffix) in zooms:
                        z_img_name = f"{img_path.stem}_aug{k}_{suffix}{args.out_ext}"
                        z_lbl_name = f"{img_path.stem}_aug{k}_{suffix}.txt"
                        cv2.imwrite(str(out_img_dir / z_img_name), zimg)
                        (out_lbl_dir / z_lbl_name).write_text(
                            "\n".join(f"{l} {' '.join(f'{x:.6f}' for x in b)}" for b, l in zip(zboxes, zlabs)) + ("\n" if zboxes else "")
                        )

    # Optional: write a data.yaml for the augmented dataset
    if args.write_yaml:
        _ensure_dir(out_root)
        yaml_path = out_root / "data.yaml"
        names = ["disc", "cup"]
        ds = {
            "path": str(out_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": names
        }
        test_img_dir = out_root / "images" / "test"
        test_lbl_dir = out_root / "labels" / "test"
        if test_img_dir.exists() and any(test_img_dir.iterdir()) and test_lbl_dir.exists() and any(test_lbl_dir.iterdir()):
            ds["test"] = "images/test"
        yaml_path.write_text(yaml.safe_dump(ds, sort_keys=False))
        print(f"[OK] Wrote data.yaml -> {yaml_path}")

    print(f"[OK] Augmented dataset written to: {out_root}")

if __name__ == "__main__":
    main()