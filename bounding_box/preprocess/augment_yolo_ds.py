#!/usr/bin/env python3
# augment_yolo_ds.py
"""
Augment a YOLO (disc=0, cup=1) dataset with:
  1) Geometric & photometric transforms (Albumentations)
  2) Fixed-size overlapping tiling (sliding window)
  3) Object-centric zoom crops (centered near objects)
  4) Multi-scale sliding zoom sweeps (cover the whole image at several scales)

New: --enable_zoom_sweep creates zoomed-in crops over EVERY part of the image
(including sides and corners) by sliding a square window at multiple scales.

Default dataset locations (relative to --project_dir):
  {PROJECT_DIR}/bounding_box/data/yolo_split        (input)
  {PROJECT_DIR}/bounding_box/data/yolo_split_aug    (output)

Examples
--------
# Full coverage, keep negatives:
python augment_yolo_ds.py \
  --project_dir /path/to/MedSAM \
  --splits train \
  --enable_zoom_sweep --zoom_sweep_scales 0.35,0.5,0.7 \
  --zoom_sweep_overlap 0.25 --zoom_sweep_keep_empty \
  --include_images_without_labels

# All augmentations:
python augment_yolo_ds.py \
  --project_dir /path/to/MedSAM \
  --splits train \
  --enable_tiling --tile_size 512 --tile_overlap 0.2 --keep_empty_tiles \
  --enable_zoom_crops --zoom_scales 0.5,0.7,0.85 --zoom_per_obj 2 --zoom_keep_empty \
  --enable_zoom_sweep --zoom_sweep_scales 0.35,0.5 --zoom_sweep_overlap 0.3 --zoom_sweep_keep_empty
"""

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import albumentations as A
import cv2
import yaml

# ---------------------------- I/O utils ----------------------------

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in IMG_EXTS])

def read_image(path: Path) -> Optional["cv2.Mat"]:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)

def write_image(path: Path, image: "cv2.Mat") -> None:
    _ensure_dir(path.parent)
    cv2.imwrite(str(path), image)

# ---------------------------- Label utils --------------------------

def read_yolo_labels(lbl_path: Path) -> Tuple[List[List[float]], List[int]]:
    """Return (boxes, labels) where boxes are [cx,cy,w,h] normalized to [0,1]."""
    boxes, labels = [], []
    if not lbl_path.exists():
        return boxes, labels
    lines = [ln.strip() for ln in lbl_path.read_text().splitlines() if ln.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        boxes.append([x, y, w, h])
        labels.append(cls)
    return boxes, labels

def write_yolo_labels(lbl_path: Path, boxes: List[List[float]], labels: List[int]) -> None:
    _ensure_dir(lbl_path.parent)
    if not boxes:
        # Write empty file to explicitly mark a negative sample
        lbl_path.write_text("")
        return
    lbl_path.write_text(
        "\n".join(f"{l} " + " ".join(f"{x:.6f}" for x in b) for b, l in zip(boxes, labels)) + "\n"
    )

# ---------------------------- Geometry helpers ---------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

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

def square_crop_bounds(cx, cy, side, W, H):
    half = side / 2.0
    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    x1 = x0 + int(round(side)); y1 = y0 + int(round(side))
    if x0 < 0:  x1 -= x0; x0 = 0
    if y0 < 0:  y1 -= y0; y0 = 0
    if x1 > W:
        shift = x1 - W; x0 = max(0, x0 - shift); x1 = W
    if y1 > H:
        shift = y1 - H; y0 = max(0, y0 - shift); y1 = H
    x0 = clamp(x0, 0, max(0, W-1)); y0 = clamp(y0, 0, max(0, H-1))
    x1 = clamp(x1, 1, W);           y1 = clamp(y1, 1, H)
    return [x0, y0, x1, y1]

def sanitize_yolo_boxes(boxes: List[List[float]]) -> List[List[float]]:
    """Clamp to [0,1] by xyxy -> clamp -> yolo, dropping degenerate boxes."""
    clean = []
    for cx, cy, w, h in boxes:
        cx = float(cx); cy = float(cy); w = float(w); h = float(h)
        x1 = cx - w/2.0; y1 = cy - h/2.0
        x2 = cx + w/2.0; y2 = cy + h/2.0
        x1 = clamp(x1, 0.0, 1.0); y1 = clamp(y1, 0.0, 1.0)
        x2 = clamp(x2, 0.0, 1.0); y2 = clamp(y2, 0.0, 1.0)
        w2 = max(0.0, x2 - x1); h2 = max(0.0, y2 - y1)
        if w2 <= 1e-6 or h2 <= 1e-6:
            continue
        cx2 = (x1 + x2) / 2.0; cy2 = (y1 + y2) / 2.0
        clean.append([cx2, cy2, w2, h2])
    return clean

# ---------------------------- Albumentations -----------------------

def build_transform(out_h: int, out_w: int) -> A.Compose:
    """Aug pipeline that returns 'image', 'bboxes', and 'class_labels' (YOLO format)."""
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
            A.RandomResizedCrop(size=(out_h, out_w), scale=(0.9, 1.0), ratio=(0.9, 1.1), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.2),
    )

# ---------------------------- Tiling (fixed size) ------------------

def tile_image_and_boxes(
    im, boxes_yolo, labels,
    tile_size=512, overlap=0.2, min_tile_vis=0.2, keep_empty=False
):
    """
    Split the image into overlapping square tiles of fixed pixel size.
    Returns list of (tile_img, tile_boxes_yolo, tile_labels, suffix)
    """
    H, W = im.shape[:2]
    ts = max(16, int(tile_size))
    stride = max(1, int(round(ts * (1.0 - overlap))))

    abs_boxes = [yolo_to_xyxy_abs(b, W, H) for b in boxes_yolo]
    tiles = []
    for y0 in range(0, max(1, H - ts + 1), stride):
        for x0 in range(0, max(1, W - ts + 1), stride):
            x1 = x0 + ts; y1 = y0 + ts
            if x1 > W: x0 = max(0, W - ts); x1 = W
            if y1 > H: y0 = max(0, H - ts); y1 = H
            tile_rect = [x0, y0, x1, y1]
            tile_w = x1 - x0; tile_h = y1 - y0
            if tile_w <= 1 or tile_h <= 1:
                continue

            t_boxes_yolo, t_labels = [], []
            for b_xyxy, cls in zip(abs_boxes, labels):
                box_area = max(0.0, (b_xyxy[2] - b_xyxy[0])) * max(0.0, (b_xyxy[3] - b_xyxy[1]))
                if box_area <= 0: continue
                inter_area, inter = box_i_area(b_xyxy, tile_rect)
                if inter_area <= 0: continue
                vis = inter_area / (box_area + 1e-12)
                if vis < min_tile_vis: continue
                ix1, iy1, ix2, iy2 = inter
                lx1 = ix1 - x0; ly1 = iy1 - y0
                lx2 = ix2 - x0; ly2 = iy2 - y0
                yolo_local = xyxy_abs_to_yolo([lx1, ly1, lx2, ly2], tile_w, tile_h)
                if yolo_local is None: continue
                t_boxes_yolo.append(yolo_local)
                t_labels.append(cls)

            if t_boxes_yolo or keep_empty:
                tile_img = im[y0:y1, x0:x1].copy()
                suffix = f"t{y0}_{x0}"
                tiles.append((tile_img, t_boxes_yolo, t_labels, suffix))
    return tiles

# -------- Multi-scale sliding "zoom sweep" (covers the whole image) --------

def multiscale_zoom_sweep(
    im, boxes_yolo, labels,
    scales: List[float], overlap: float = 0.25, min_vis: float = 0.2,
    keep_empty: bool = False, out_size: int = 640
):
    """
    Slide a square window at multiple scales (fractions of the shorter side)
    over the entire image (including edges). For each crop, project boxes.
    Returns list of (crop_img, crop_boxes_yolo, crop_labels, suffix)
    """
    H, W = im.shape[:2]
    short_side = min(W, H)
    abs_boxes = [yolo_to_xyxy_abs(b, W, H) for b in boxes_yolo]
    results = []

    for s in scales:
        ts = max(16, int(round(s * short_side)))
        stride = max(1, int(round(ts * (1.0 - overlap))))

        # Sweep just like tiling, but size depends on scale
        for y0 in range(0, max(1, H - ts + 1), stride):
            for x0 in range(0, max(1, W - ts + 1), stride):
                x1 = x0 + ts; y1 = y0 + ts
                if x1 > W: x0 = max(0, W - ts); x1 = W
                if y1 > H: y0 = max(0, H - ts); y1 = H
                tile_rect = [x0, y0, x1, y1]
                tile_w = x1 - x0; tile_h = y1 - y0
                if tile_w <= 1 or tile_h <= 1:
                    continue

                c_boxes, c_labels = [], []
                for b_xyxy, cls in zip(abs_boxes, labels):
                    box_area = max(0.0, (b_xyxy[2] - b_xyxy[0])) * max(0.0, (b_xyxy[3] - b_xyxy[1]))
                    if box_area <= 0: continue
                    inter_area, inter = box_i_area(b_xyxy, tile_rect)
                    if inter_area <= 0:
                        continue
                    vis = inter_area / (box_area + 1e-12)
                    if vis < min_vis:
                        continue
                    ix1, iy1, ix2, iy2 = inter
                    lx1 = ix1 - x0; ly1 = iy1 - y0
                    lx2 = ix2 - x0; ly2 = iy2 - y0
                    yolo_local = xyxy_abs_to_yolo([lx1, ly1, lx2, ly2], tile_w, tile_h)
                    if yolo_local is None: continue
                    c_boxes.append(yolo_local)
                    c_labels.append(cls)

                if c_boxes or keep_empty:
                    crop = im[y0:y1, x0:x1].copy()
                    if out_size and out_size > 0:
                        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
                        # boxes are normalized to crop; resizing doesn't change normalized values
                    suffix = f"ms{int(round(s*100))}_{y0}_{x0}"
                    results.append((crop, c_boxes, c_labels, suffix))
    return results

# ---------------------------- Object-centric zoom crops ------------

def make_zoom_crops(
    im, boxes_yolo, labels,
    zoom_scales: List[float], zoom_per_obj=1, zoom_on="both",
    zoom_out_size=640, zoom_jitter=0.05, min_vis=0.2, keep_empty=False
):
    """
    For each selected object, make square crops at given scales, with jitter.
    Returns list of (crop_img, crop_boxes_yolo, crop_labels, suffix)
    """
    H, W = im.shape[:2]
    short_side = min(W, H)
    abs_boxes = [yolo_to_xyxy_abs(b, W, H) for b in boxes_yolo]

    def select_indices():
        if   zoom_on == "disc": return [i for i,l in enumerate(labels) if l == 0]
        elif zoom_on == "cup":  return [i for i,l in enumerate(labels) if l == 1]
        elif zoom_on == "both": return list(range(len(labels)))
        elif zoom_on == "any":  return list(range(len(labels))) if labels else []
        else:                   return list(range(len(labels)))

    idxs = select_indices()
    results = []

    for i in idxs:
        x1, y1, x2, y2 = abs_boxes[i]
        ocx = (x1 + x2) / 2.0
        ocy = (y1 + y2) / 2.0

        for s in zoom_scales:
            side = max(16, int(round(s * short_side)))
            for k in range(zoom_per_obj):
                jx = (random.uniform(-zoom_jitter, zoom_jitter)) * side
                jy = (random.uniform(-zoom_jitter, zoom_jitter)) * side
                cx = ocx + jx; cy = ocy + jy
                rx0, ry0, rx1, ry1 = square_crop_bounds(cx, cy, side, W, H)
                roi_w = rx1 - rx0; roi_h = ry1 - ry0
                if roi_w <= 1 or roi_h <= 1:
                    continue

                crop_boxes_yolo, crop_labels = [], []
                for b_xyxy, cls in zip(abs_boxes, labels):
                    box_area = max(0.0, (b_xyxy[2] - b_xyxy[0])) * max(0.0, (b_xyxy[3] - b_xyxy[1]))
                    if box_area <= 0: continue
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
                    if crop_yolo is None: continue
                    crop_boxes_yolo.append(crop_yolo)
                    crop_labels.append(cls)

                if not crop_boxes_yolo and not keep_empty:
                    continue

                crop_img = im[ry0:ry1, rx0:rx1].copy()
                if zoom_out_size and zoom_out_size > 0:
                    crop_img = cv2.resize(crop_img, (int(zoom_out_size), int(zoom_out_size)), interpolation=cv2.INTER_AREA)
                suffix = f"z{i}_s{int(round(s*100))}_{k}"
                results.append((crop_img, crop_boxes_yolo, crop_labels, suffix))
    return results

# ---------------------------- Config dataclass ---------------------

@dataclass
class AugConfig:
    project_dir: Path
    data_root: Path
    out_root: Path
    out_ext: str
    splits: List[str]
    multiplier: int
    include_images_without_labels: bool

    # tiling (fixed size)
    enable_tiling: bool
    tile_size: int
    tile_overlap: float
    min_tile_vis: float
    keep_empty_tiles: bool
    tile_from_aug: bool

    # object-centric zoom crops
    enable_zoom_crops: bool
    zoom_scales: List[float]
    zoom_per_obj: int
    zoom_on: str
    zoom_out_size: int
    zoom_jitter: float
    zoom_min_vis: float
    zoom_from_aug: bool
    zoom_keep_empty: bool

    # multi-scale zoom sweep (whole-image)
    enable_zoom_sweep: bool
    zoom_sweep_scales: List[float]
    zoom_sweep_overlap: float
    zoom_sweep_min_vis: float
    zoom_sweep_keep_empty: bool
    zoom_sweep_out_size: int
    zoom_sweep_from_aug: bool

    # misc
    write_yaml: bool
    seed: int

# ---------------------------- Processing ---------------------------

def dump_variant(
    out_img_dir: Path, out_lbl_dir: Path, stem: str, suffix: str, ext: str,
    img, boxes, labels
):
    img_p = out_img_dir / f"{stem}_{suffix}{ext}"
    lbl_p = out_lbl_dir / f"{stem}_{suffix}.txt"
    write_image(img_p, img)
    write_yolo_labels(lbl_p, boxes, labels)

def process_one_image(
    img_path: Path, lbl_path: Path, cfg: AugConfig,
    out_img_dir: Path, out_lbl_dir: Path, transform: A.Compose
):
    """Process a single (image,label) pair: write original, tiles, zooms, zoom sweeps, and augmented variants."""
    im = read_image(img_path)
    if im is None:
        print(f"[WARN] Failed to read image: {img_path}")
        return

    stem = img_path.stem
    boxes, labels = read_yolo_labels(lbl_path)
    boxes = sanitize_yolo_boxes(boxes)

    # (A) original
    orig_img_out = out_img_dir / f"{stem}{cfg.out_ext}"
    if cfg.out_ext.lower() == img_path.suffix.lower():
        _ensure_dir(orig_img_out.parent)
        shutil.copy2(img_path, orig_img_out)
    else:
        write_image(orig_img_out, im)
    write_yolo_labels(out_lbl_dir / f"{stem}.txt", boxes, labels)

    # (B) tiling (fixed size)
    if cfg.enable_tiling:
        tiles = tile_image_and_boxes(
            im, boxes, labels,
            tile_size=cfg.tile_size,
            overlap=cfg.tile_overlap,
            min_tile_vis=cfg.min_tile_vis,
            keep_empty=cfg.keep_empty_tiles,
        )
        for (timg, tboxes, tlabs, suffix) in tiles:
            dump_variant(out_img_dir, out_lbl_dir, stem, suffix, cfg.out_ext, timg, tboxes, tlabs)

    # (C) object-centric zoom crops
    if cfg.enable_zoom_crops:
        zooms = make_zoom_crops(
            im, boxes, labels,
            zoom_scales=cfg.zoom_scales,
            zoom_per_obj=cfg.zoom_per_obj,
            zoom_on=cfg.zoom_on,
            zoom_out_size=cfg.zoom_out_size,
            zoom_jitter=cfg.zoom_jitter,
            min_vis=cfg.zoom_min_vis,
            keep_empty=cfg.zoom_keep_empty,
        )
        for (zimg, zboxes, zlabs, suffix) in zooms:
            dump_variant(out_img_dir, out_lbl_dir, stem, suffix, cfg.out_ext, zimg, zboxes, zlabs)

    # (D) multi-scale sliding zoom sweep (covers whole image, multiple scales)
    if cfg.enable_zoom_sweep:
        sweeps = multiscale_zoom_sweep(
            im, boxes, labels,
            scales=cfg.zoom_sweep_scales,
            overlap=cfg.zoom_sweep_overlap,
            min_vis=cfg.zoom_sweep_min_vis,
            keep_empty=cfg.zoom_sweep_keep_empty,
            out_size=cfg.zoom_sweep_out_size,
        )
        for (cimg, cboxes, clabs, suffix) in sweeps:
            dump_variant(out_img_dir, out_lbl_dir, stem, suffix, cfg.out_ext, cimg, cboxes, clabs)

    # (E) augmented variants (+ optional tiling/zooms/zoom-sweeps from aug)
    for k in range(cfg.multiplier):
        if boxes:
            aug = transform(image=im, bboxes=boxes, class_labels=labels)
            aug_img, aug_boxes, aug_labels = aug["image"], aug["bboxes"], aug["class_labels"]
        else:
            aug = transform(image=im, bboxes=[], class_labels=[])
            aug_img, aug_boxes, aug_labels = aug["image"], [], []

        a_suffix = f"aug{k}"
        dump_variant(out_img_dir, out_lbl_dir, stem, a_suffix, cfg.out_ext, aug_img, aug_boxes, aug_labels)

        # tiling from augmented
        if cfg.enable_tiling and cfg.tile_from_aug:
            tiles = tile_image_and_boxes(
                aug_img, aug_boxes, aug_labels,
                tile_size=cfg.tile_size,
                overlap=cfg.tile_overlap,
                min_tile_vis=cfg.min_tile_vis,
                keep_empty=cfg.keep_empty_tiles,
            )
            for (timg, tboxes, tlabs, suffix) in tiles:
                dump_variant(out_img_dir, out_lbl_dir, stem, f"aug{k}_{suffix}", cfg.out_ext, timg, tboxes, tlabs)

        # object-centric zooms from augmented
        if cfg.enable_zoom_crops and cfg.zoom_from_aug:
            zooms = make_zoom_crops(
                aug_img, aug_boxes, aug_labels,
                zoom_scales=cfg.zoom_scales,
                zoom_per_obj=cfg.zoom_per_obj,
                zoom_on=cfg.zoom_on,
                zoom_out_size=cfg.zoom_out_size,
                zoom_jitter=cfg.zoom_jitter,
                min_vis=cfg.zoom_min_vis,
                keep_empty=cfg.zoom_keep_empty,
            )
            for (zimg, zboxes, zlabs, suffix) in zooms:
                dump_variant(out_img_dir, out_lbl_dir, stem, f"aug{k}_{suffix}", cfg.out_ext, zimg, zboxes, zlabs)

        # multi-scale zoom sweep from augmented
        if cfg.enable_zoom_sweep and cfg.zoom_sweep_from_aug:
            sweeps = multiscale_zoom_sweep(
                aug_img, aug_boxes, aug_labels,
                scales=cfg.zoom_sweep_scales,
                overlap=cfg.zoom_sweep_overlap,
                min_vis=cfg.zoom_sweep_min_vis,
                keep_empty=cfg.zoom_sweep_keep_empty,
                out_size=cfg.zoom_sweep_out_size,
            )
            for (cimg, cboxes, clabs, suffix) in sweeps:
                dump_variant(out_img_dir, out_lbl_dir, stem, f"aug{k}_{suffix}", cfg.out_ext, cimg, cboxes, clabs)

def process_split(split: str, cfg: AugConfig):
    in_img_dir  = cfg.data_root / "images" / split
    in_lbl_dir  = cfg.data_root / "labels" / split
    out_img_dir = cfg.out_root  / "images" / split
    out_lbl_dir = cfg.out_root  / "labels" / split

    if not in_img_dir.exists():
        print(f"[WARN] Missing images split '{split}' in input dataset; skipping.")
        return
    if not in_lbl_dir.exists():
        print(f"[WARN] Missing labels split '{split}' in input dataset; skipping.")
        return

    _ensure_dir(out_img_dir); _ensure_dir(out_lbl_dir)
    count = 0
    for img_path in list_images(in_img_dir):
        lbl_path = in_lbl_dir / f"{img_path.stem}.txt"

        # Include images without labels? (negatives)
        if not lbl_path.exists() and not cfg.include_images_without_labels:
            continue

        # Build transform sized to this image
        im = read_image(img_path)
        if im is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue
        H, W = im.shape[:2]
        transform = build_transform(H, W)

        # If no labels exist but we want to include negatives:
        if not lbl_path.exists():
            # Create an empty temp label file path for the function
            (out_lbl_dir / f"{img_path.stem}.txt").write_text("")
            # Still run all augmentations that allow empty crops
            process_one_image(img_path, lbl_path, cfg, out_img_dir, out_lbl_dir, transform)
        else:
            process_one_image(img_path, lbl_path, cfg, out_img_dir, out_lbl_dir, transform)
        count += 1

    print(f"[OK] Split '{split}': processed {count} images → {out_img_dir.parent}")

# ---------------------------- CLI / Main ---------------------------

def parse_args() -> AugConfig:
    ap = argparse.ArgumentParser(description="YOLO dataset augmenter with tiles, object zooms, and multi-scale zoom sweeps.")
    ap.add_argument("--project_dir", default=".", help="Base project directory (root of MedSAM).")
    ap.add_argument("--data_root", default=None,
        help="YOLO dataset root. Default: {PROJECT_DIR}/bounding_box/data/yolo_split")
    ap.add_argument("--out_root", default=None,
        help="Output augmented dataset root. Default: {PROJECT_DIR}/bounding_box/data/yolo_split_aug")
    ap.add_argument("--splits", nargs="+", default=["train","val","test"],
        help="Which splits to process (default: all)")

    ap.add_argument("--multiplier", type=int, default=2,
        help="How many augmented copies to create per image.")
    ap.add_argument("--out_ext", default=".jpg",
        help="Output image extension for augmented/tiles/zooms (e.g., .jpg, .png).")
    ap.add_argument("--include_images_without_labels", action="store_true",
        help="If set, include images even when no label file exists (creates negatives).")

    # Tiling (fixed size)
    ap.add_argument("--enable_tiling", action="store_true", help="Enable overlapping tiling (fixed pixel size).")
    ap.add_argument("--tile_size", type=int, default=512, help="Square tile size (pixels).")
    ap.add_argument("--tile_overlap", type=float, default=0.2, help="Overlap fraction (0–0.95).")
    ap.add_argument("--min_tile_vis", type=float, default=0.2, help="Min frac of obj area required to keep in tile.")
    ap.add_argument("--keep_empty_tiles", action="store_true", help="Keep tiles that have no boxes.")
    ap.add_argument("--tile_from_aug", action="store_true", help="Also tile each augmented image.")

    # Object-centric zoom crops
    ap.add_argument("--enable_zoom_crops", action="store_true", help="Enable object-centric zoom crops.")
    ap.add_argument("--zoom_scales", default="0.5,0.7,0.85",
        help="Comma-separated scales (fractions of shorter side), e.g. '0.5,0.7'.")
    ap.add_argument("--zoom_per_obj", type=int, default=1, help="How many jittered crops per object per scale.")
    ap.add_argument("--zoom_on", default="both", choices=["disc","cup","both","any"], help="Which classes to center crops on.")
    ap.add_argument("--zoom_out_size", type=int, default=640, help="Resize zoom crop to this SxS size (0=keep native).")
    ap.add_argument("--zoom_jitter", type=float, default=0.05, help="Center jitter as fraction of crop side.")
    ap.add_argument("--zoom_min_vis", type=float, default=0.2, help="Min frac of obj area required to keep in crop.")
    ap.add_argument("--zoom_from_aug", action="store_true", help="Also create zoom crops from augmented images.")
    ap.add_argument("--zoom_keep_empty", action="store_true", help="Keep zoom crops even if they contain no boxes.")

    # Multi-scale sliding zoom sweep (whole-image coverage)
    ap.add_argument("--enable_zoom_sweep", action="store_true", help="Enable multi-scale sliding zoom sweeps over entire image.")
    ap.add_argument("--zoom_sweep_scales", default="0.35,0.5,0.7",
        help="Comma-separated scales (fractions of shorter side) for sweep windows.")
    ap.add_argument("--zoom_sweep_overlap", type=float, default=0.25,
        help="Overlap fraction for the sweep windows (0–0.95).")
    ap.add_argument("--zoom_sweep_min_vis", type=float, default=0.2,
        help="Min frac of obj area required to keep in a sweep crop.")
    ap.add_argument("--zoom_sweep_keep_empty", action="store_true",
        help="Keep sweep crops even if they contain no boxes.")
    ap.add_argument("--zoom_sweep_out_size", type=int, default=640,
        help="Resize sweep crops to this SxS size (0=keep native).")
    ap.add_argument("--zoom_sweep_from_aug", action="store_true",
        help="Also create zoom sweep crops from augmented images.")

    ap.add_argument("--write_yaml", action="store_true", help="Write a data.yaml for the augmented dataset.")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed.")
    args = ap.parse_args()

    random.seed(args.seed)

    PROJECT_DIR = _expand(args.project_dir)
    data_root = _expand(args.data_root) if args.data_root else (PROJECT_DIR / "bounding_box" / "data" / "yolo_split")
    out_root  = _expand(args.out_root)  if args.out_root  else (PROJECT_DIR / "bounding_box" / "data" / "yolo_split_aug")

    zoom_scales = [float(s) for s in args.zoom_scales.split(",") if s.strip()]
    zoom_scales = [s for s in zoom_scales if 0 < s <= 1.0]

    sweep_scales = [float(s) for s in args.zoom_sweep_scales.split(",") if s.strip()]
    sweep_scales = [s for s in sweep_scales if 0 < s <= 1.0]

    return AugConfig(
        project_dir=PROJECT_DIR,
        data_root=data_root,
        out_root=out_root,
        out_ext=args.out_ext,
        splits=args.splits,
        multiplier=args.multiplier,
        include_images_without_labels=args.include_images_without_labels,

        enable_tiling=args.enable_tiling,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        min_tile_vis=args.min_tile_vis,
        keep_empty_tiles=args.keep_empty_tiles,
        tile_from_aug=args.tile_from_aug,

        enable_zoom_crops=args.enable_zoom_crops,
        zoom_scales=zoom_scales,
        zoom_per_obj=args.zoom_per_obj,
        zoom_on=args.zoom_on,
        zoom_out_size=args.zoom_out_size,
        zoom_jitter=args.zoom_jitter,
        zoom_min_vis=args.zoom_min_vis,
        zoom_from_aug=args.zoom_from_aug,
        zoom_keep_empty=args.zoom_keep_empty,

        enable_zoom_sweep=args.enable_zoom_sweep,
        zoom_sweep_scales=sweep_scales,
        zoom_sweep_overlap=args.zoom_sweep_overlap,
        zoom_sweep_min_vis=args.zoom_sweep_min_vis,
        zoom_sweep_keep_empty=args.zoom_sweep_keep_empty,
        zoom_sweep_out_size=args.zoom_sweep_out_size,
        zoom_sweep_from_aug=args.zoom_sweep_from_aug,

        write_yaml=args.write_yaml,
        seed=args.seed,
    )

def maybe_write_yaml(cfg: AugConfig):
    if not cfg.write_yaml:
        return
    _ensure_dir(cfg.out_root)
    yaml_path = cfg.out_root / "data.yaml"
    ds = {
        "path": str(cfg.out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": ["disc", "cup"],
    }
    test_img_dir = cfg.out_root / "images" / "test"
    test_lbl_dir = cfg.out_root / "labels" / "test"
    if test_img_dir.exists() and any(test_img_dir.iterdir()) and test_lbl_dir.exists() and any(test_lbl_dir.iterdir()):
        ds["test"] = "images/test"
    yaml_path.write_text(yaml.safe_dump(ds, sort_keys=False))
    print(f"[OK] Wrote data.yaml → {yaml_path}")

def main():
    cfg = parse_args()
    print(f"[INFO] Input : {cfg.data_root}")
    print(f"[INFO] Output: {cfg.out_root}")

    for split in cfg.splits:
        process_split(split, cfg)

    maybe_write_yaml(cfg)
    print(f"[OK] Augmented dataset written to: {cfg.out_root}")

if __name__ == "__main__":
    main()