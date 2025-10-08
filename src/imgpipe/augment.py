# src/imgpipe/augment.py
from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import albumentations as A
import cv2
import yaml

from .utils import ensure_dir


# ----------------------------- I/O helpers -----------------------------

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _expand(p: Path | str | None) -> Optional[Path]:
    if p in (None, "", False):
        return None
    return Path(os.path.expanduser(str(p))).resolve()


def _list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in _IMG_EXTS])


def _read_image(path: Path) -> Optional["cv2.Mat"]:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _write_image(path: Path, image: "cv2.Mat") -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image)


# ----------------------------- YOLO label helpers ----------------------

def _read_yolo_labels(lbl_path: Path) -> Tuple[List[List[float]], List[int]]:
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


def _write_yolo_labels(lbl_path: Path, boxes: List[List[float]], labels: List[int]) -> None:
    ensure_dir(lbl_path.parent)
    if not boxes:
        # Explicit negatives are allowed
        lbl_path.write_text("")
        return
    lbl_path.write_text(
        "\n".join(f"{l} " + " ".join(f"{x:.6f}" for x in b) for b, l in zip(boxes, labels)) + "\n"
    )


# ----------------------------- Geometry helpers -----------------------

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _yolo_to_xyxy_abs(box_yolo, W, H):
    cx, cy, w, h = box_yolo
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    return [x1, y1, x2, y2]


def _xyxy_abs_to_yolo(box_xyxy, W, H):
    x1, y1, x2, y2 = box_xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 0 or bh <= 0:
        return None
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return [cx / W, cy / H, bw / W, bh / H]


def _box_i_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, None
    return (x2 - x1) * (y2 - y1), [x1, y1, x2, y2]


def _square_crop_bounds(cx, cy, side, W, H):
    half = side / 2.0
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = x0 + int(round(side))
    y1 = y0 + int(round(side))
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > W:
        shift = x1 - W
        x0 = max(0, x0 - shift)
        x1 = W
    if y1 > H:
        shift = y1 - H
        y0 = max(0, y0 - shift)
        y1 = H
    x0 = _clamp(x0, 0, max(0, W - 1))
    y0 = _clamp(y0, 0, max(0, H - 1))
    x1 = _clamp(x1, 1, W)
    y1 = _clamp(y1, 1, H)
    return [x0, y0, x1, y1]


def _sanitize_yolo_boxes(boxes: List[List[float]]) -> List[List[float]]:
    """Clamp to [0,1] by xyxy -> clamp -> yolo, dropping degenerate boxes."""
    clean = []
    for cx, cy, w, h in boxes:
        cx = float(cx)
        cy = float(cy)
        w = float(w)
        h = float(h)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        x1 = _clamp(x1, 0.0, 1.0)
        y1 = _clamp(y1, 0.0, 1.0)
        x2 = _clamp(x2, 0.0, 1.0)
        y2 = _clamp(y2, 0.0, 1.0)
        w2 = max(0.0, x2 - x1)
        h2 = max(0.0, y2 - y1)
        if w2 <= 1e-6 or h2 <= 1e-6:
            continue
        cx2 = (x1 + x2) / 2.0
        cy2 = (y1 + y2) / 2.0
        clean.append([cx2, cy2, w2, h2])
    return clean


# ----------------------------- Config dataclass ----------------------

@dataclass
class AugmentSettings:
    out_ext: str = ".jpg"
    multiplier: int = 2
    include_images_without_labels: bool = False
    seed: int = 1337
    write_yaml: bool = True

    transform: Dict[str, Any] = None  # Albumentations dict
    tiling: Dict[str, Any] = None     # fixed-size tiling
    zoom_crops: Dict[str, Any] = None # object-centric zooms
    zoom_sweep: Dict[str, Any] = None # multi-scale sweep

    @classmethod
    def from_yaml(cls, path: Path) -> "AugmentSettings":
        if not path.exists():
            raise SystemExit(f"[ERR] YAML config not found: {path}")
        with open(path, "r") as f:
            y = yaml.safe_load(f) or {}
        if not isinstance(y, dict):
            raise SystemExit(f"[ERR] YAML at {path} must be a mapping (key: value).")

        return cls(
            out_ext=str(y.get("out_ext", ".jpg")),
            multiplier=int(y.get("multiplier", 2)),
            include_images_without_labels=bool(y.get("include_images_without_labels", False)),
            seed=int(y.get("seed", 1337)),
            write_yaml=bool(y.get("write_yaml", True)),
            transform=dict(y.get("transform", {}) or {}),
            tiling=dict(y.get("tiling", {}) or {}),
            zoom_crops=dict(y.get("zoom_crops", {}) or {}),
            zoom_sweep=dict(y.get("zoom_sweep", {}) or {}),
        )


# ----------------------------- Augmentor class ----------------------

class Augmentor:
    """
    Full-feature YOLO augmenter (OOP) with feature parity to legacy augment_yolo_ds.py:
      1) Albumentations geometric & photometric transforms
      2) Fixed-size overlapping tiling (sliding window)
      3) Object-centric zoom crops (centered near objects)
      4) Multi-scale sliding zoom sweeps

    All hyperparameters live in a YAML file (e.g., cfg.raw['aug_yaml']).
    """

    def __init__(self, aug_yaml: Path):
        self.settings = AugmentSettings.from_yaml(_expand(aug_yaml) or aug_yaml)
        random.seed(self.settings.seed)

    # -------- Albumentations builder --------
    def _build_transform(self, H: int, W: int) -> A.Compose:
        tcfg = self.settings.transform or {}

        hflip_p = float(tcfg.get("hflip_p", 0.5))
        vflip_p = float(tcfg.get("vflip_p", 0.0))

        aff = tcfg.get("affine", {}) or {}
        aff_p = float(aff.get("p", 0.7))
        scale = tuple(aff.get("scale", [0.9, 1.1]))
        trans = tuple(aff.get("translate_percent", [-0.05, 0.05]))
        rotate = tuple(aff.get("rotate", [-15, 15]))
        shear = tuple(aff.get("shear", [-5, 5]))

        rrc = tcfg.get("random_resized_crop", {}) or {}
        rrc_p = float(rrc.get("p", 0.5))
        rrc_scl = tuple(rrc.get("scale", [0.9, 1.0]))
        rrc_rat = tuple(rrc.get("ratio", [0.9, 1.1]))

        cj = tcfg.get("color_jitter", {}) or {}
        cj_p = float(cj.get("p", 0.5))
        cj_b = float(cj.get("brightness", 0.2))
        cj_c = float(cj.get("contrast", 0.2))
        cj_s = float(cj.get("saturation", 0.2))
        cj_h = float(cj.get("hue", 0.1))

        return A.Compose(
            [
                A.HorizontalFlip(p=hflip_p),
                A.VerticalFlip(p=vflip_p),
                A.Affine(
                    scale=scale,
                    translate_percent=trans,
                    rotate=rotate,
                    shear=shear,
                    fit_output=False,
                    p=aff_p,
                ),
                A.RandomResizedCrop(size=(H, W), scale=rrc_scl, ratio=rrc_rat, p=rrc_p),
                A.ColorJitter(
                    brightness=cj_b,
                    contrast=cj_c,
                    saturation=cj_s,
                    hue=cj_h,
                    p=cj_p,
                ),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.2,
            ),
        )

    # -------- Tiling (fixed size) --------
    def _tile_image_and_boxes(
        self,
        im,
        boxes_yolo,
        labels,
        *,
        tile_size=512,
        overlap=0.2,
        min_tile_vis=0.2,
        keep_empty=False,
    ):
        H, W = im.shape[:2]
        ts = max(16, int(tile_size))
        stride = max(1, int(round(ts * (1.0 - overlap))))

        abs_boxes = [_yolo_to_xyxy_abs(b, W, H) for b in boxes_yolo]
        tiles = []
        for y0 in range(0, max(1, H - ts + 1), stride):
            for x0 in range(0, max(1, W - ts + 1), stride):
                x1 = x0 + ts
                y1 = y0 + ts
                if x1 > W:
                    x0 = max(0, W - ts)
                    x1 = W
                if y1 > H:
                    y0 = max(0, H - ts)
                    y1 = H
                tile_rect = [x0, y0, x1, y1]
                tile_w = x1 - x0
                tile_h = y1 - y0
                if tile_w <= 1 or tile_h <= 1:
                    continue

                t_boxes_yolo, t_labels = [], []
                for b_xyxy, cls in zip(abs_boxes, labels):
                    box_area = max(0.0, (b_xyxy[2] - b_xyxy[0])) * max(0.0, (b_xyxy[3] - b_xyxy[1]))
                    if box_area <= 0:
                        continue
                    inter_area, inter = _box_i_area(b_xyxy, tile_rect)
                    if inter_area <= 0:
                        continue
                    vis = inter_area / (box_area + 1e-12)
                    if vis < min_tile_vis:
                        continue
                    ix1, iy1, ix2, iy2 = inter
                    lx1 = ix1 - x0
                    ly1 = iy1 - y0
                    lx2 = ix2 - x0
                    ly2 = iy2 - y0
                    yolo_local = _xyxy_abs_to_yolo([lx1, ly1, lx2, ly2], tile_w, tile_h)
                    if yolo_local is None:
                        continue
                    t_boxes_yolo.append(yolo_local)
                    t_labels.append(cls)

                if t_boxes_yolo or keep_empty:
                    tile_img = im[y0:y1, x0:x1].copy()
                    suffix = f"t{y0}_{x0}"
                    tiles.append((tile_img, t_boxes_yolo, t_labels, suffix))
        return tiles

    # -------- Multi-scale sliding zoom sweep --------
    def _multiscale_zoom_sweep(
        self,
        im,
        boxes_yolo,
        labels,
        *,
        scales: List[float],
        overlap: float = 0.25,
        min_vis: float = 0.2,
        keep_empty: bool = False,
        out_size: int = 640,
    ):
        H, W = im.shape[:2]
        short_side = min(W, H)
        abs_boxes = [_yolo_to_xyxy_abs(b, W, H) for b in boxes_yolo]
        results = []

        for s in scales:
            ts = max(16, int(round(s * short_side)))
            stride = max(1, int(round(ts * (1.0 - overlap))))

            for y0 in range(0, max(1, H - ts + 1), stride):
                for x0 in range(0, max(1, W - ts + 1), stride):
                    x1 = x0 + ts
                    y1 = y0 + ts
                    if x1 > W:
                        x0 = max(0, W - ts)
                        x1 = W
                    if y1 > H:
                        y0 = max(0, H - ts)
                        y1 = H
                    tile_rect = [x0, y0, x1, y1]
                    tile_w = x1 - x0
                    tile_h = y1 - y0
                    if tile_w <= 1 or tile_h <= 1:
                        continue

                    c_boxes, c_labels = [], []
                    for b_xyxy, cls in zip(abs_boxes, labels):
                        box_area = max(0.0, (b_xyxy[2] - b_xyxy[0])) * max(0.0, (b_xyxy[3] - b_xyxy[1]))
                        if box_area <= 0:
                            continue
                        inter_area, inter = _box_i_area(b_xyxy, tile_rect)
                        if inter_area <= 0:
                            continue
                        vis = inter_area / (box_area + 1e-12)
                        if vis < min_vis:
                            continue
                        ix1, iy1, ix2, iy2 = inter
                        lx1 = ix1 - x0
                        ly1 = iy1 - y0
                        lx2 = ix2 - x0
                        ly2 = iy2 - y0
                        yolo_local = _xyxy_abs_to_yolo([lx1, ly1, lx2, ly2], tile_w, tile_h)
                        if yolo_local is None:
                            continue
                        c_boxes.append(yolo_local)
                        c_labels.append(cls)

                    if c_boxes or keep_empty:
                        crop = im[y0:y1, x0:x1].copy()
                        if out_size and out_size > 0:
                            crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
                        suffix = f"ms{int(round(s*100))}_{y0}_{x0}"
                        results.append((crop, c_boxes, c_labels, suffix))
        return results

    # -------- Object-centric zoom crops --------
    def _zoom_crops(
        self,
        im,
        boxes_yolo,
        labels,
        *,
        zoom_scales: List[float],
        zoom_per_obj=1,
        zoom_on="both",
        zoom_out_size=640,
        zoom_jitter=0.05,
        min_vis=0.2,
        keep_empty=False,
    ):
        H, W = im.shape[:2]
        short_side = min(W, H)
        abs_boxes = [_yolo_to_xyxy_abs(b, W, H) for b in boxes_yolo]

        def _select_indices():
            if zoom_on == "disc":
                return [i for i, l in enumerate(labels) if l == 0]
            if zoom_on == "cup":
                return [i for i, l in enumerate(labels) if l == 1]
            if zoom_on == "both":
                return list(range(len(labels)))
            if zoom_on == "any":
                return list(range(len(labels))) if labels else []
            return list(range(len(labels)))

        idxs = _select_indices()
        out = []

        for i in idxs:
            x1, y1, x2, y2 = abs_boxes[i]
            ocx = (x1 + x2) / 2.0
            ocy = (y1 + y2) / 2.0

            for s in zoom_scales:
                side = max(16, int(round(s * short_side)))
                for k in range(zoom_per_obj):
                    jx = (random.uniform(-zoom_jitter, zoom_jitter)) * side
                    jy = (random.uniform(-zoom_jitter, zoom_jitter)) * side
                    cx = ocx + jx
                    cy = ocy + jy
                    rx0, ry0, rx1, ry1 = _square_crop_bounds(cx, cy, side, W, H)
                    roi_w = rx1 - rx0
                    roi_h = ry1 - ry0
                    if roi_w <= 1 or roi_h <= 1:
                        continue

                    crop_boxes_yolo, crop_labels = [], []
                    for b_xyxy, cls in zip(abs_boxes, labels):
                        box_area = max(0.0, (b_xyxy[2] - b_xyxy[0])) * max(0.0, (b_xyxy[3] - b_xyxy[1]))
                        if box_area <= 0:
                            continue
                        inter_area, inter = _box_i_area(b_xyxy, [rx0, ry0, rx1, ry1])
                        if inter_area <= 0:
                            continue
                        vis = inter_area / (box_area + 1e-12)
                        if vis < min_vis:
                            continue
                        ix1, iy1, ix2, iy2 = inter
                        lx1 = ix1 - rx0
                        ly1 = iy1 - ry0
                        lx2 = ix2 - rx0
                        ly2 = iy2 - ry0
                        crop_yolo = _xyxy_abs_to_yolo([lx1, ly1, lx2, ly2], roi_w, roi_h)
                        if crop_yolo is None:
                            continue
                        crop_boxes_yolo.append(crop_yolo)
                        crop_labels.append(cls)

                    if not crop_boxes_yolo and not keep_empty:
                        continue

                    crop_img = im[ry0:ry1, rx0:rx1].copy()
                    if zoom_out_size and zoom_out_size > 0:
                        crop_img = cv2.resize(
                            crop_img, (int(zoom_out_size), int(zoom_out_size)), interpolation=cv2.INTER_AREA
                        )
                    suffix = f"z{i}_s{int(round(s*100))}_{k}"
                    out.append((crop_img, crop_boxes_yolo, crop_labels, suffix))
        return out

    # -------- Internals for writing variants --------
    def _dump_variant(
        self,
        out_img_dir: Path,
        out_lbl_dir: Path,
        stem: str,
        suffix: str,
        ext: str,
        img,
        boxes,
        labels,
    ):
        img_p = out_img_dir / f"{stem}_{suffix}{ext}"
        lbl_p = out_lbl_dir / f"{stem}_{suffix}.txt"
        _write_image(img_p, img)
        _write_yolo_labels(lbl_p, boxes, labels)

    # -------- One (image,label) pair end-to-end --------
    def _process_one(
        self,
        img_path: Path,
        lbl_path: Path,
        out_img_dir: Path,
        out_lbl_dir: Path,
        transform: A.Compose,
        *,
        prefer_copy: bool,
    ):
        s = self.settings

        im = _read_image(img_path)
        if im is None:
            print(f"[WARN] Failed to read image: {img_path}")
            return

        stem = img_path.stem
        boxes, labels = _read_yolo_labels(lbl_path)
        boxes = _sanitize_yolo_boxes(boxes)

        # (A) original
        orig_img_out = out_img_dir / f"{stem}{s.out_ext}"
        if s.out_ext.lower() == img_path.suffix.lower() and prefer_copy:
            ensure_dir(orig_img_out.parent)
            shutil.copy2(img_path, orig_img_out)
        else:
            _write_image(orig_img_out, im)
        _write_yolo_labels(out_lbl_dir / f"{stem}.txt", boxes, labels)

        # (B) tiling
        if bool(s.tiling.get("enable", False)):
            tiles = self._tile_image_and_boxes(
                im,
                boxes,
                labels,
                tile_size=int(s.tiling.get("tile_size", 512)),
                overlap=float(s.tiling.get("tile_overlap", 0.2)),
                min_tile_vis=float(s.tiling.get("min_tile_vis", 0.2)),
                keep_empty=bool(s.tiling.get("keep_empty_tiles", False)),
            )
            for (timg, tboxes, tlabs, suffix) in tiles:
                self._dump_variant(out_img_dir, out_lbl_dir, stem, suffix, s.out_ext, timg, tboxes, tlabs)

        # (C) object-centric zoom crops
        if bool(s.zoom_crops.get("enable", False)):
            zooms = self._zoom_crops(
                im,
                boxes,
                labels,
                zoom_scales=[float(x) for x in s.zoom_crops.get("scales", [0.5, 0.7])],
                zoom_per_obj=int(s.zoom_crops.get("per_obj", 1)),
                zoom_on=str(s.zoom_crops.get("on", "both")),
                zoom_out_size=int(s.zoom_crops.get("out_size", 640)),
                zoom_jitter=float(s.zoom_crops.get("jitter", 0.05)),
                min_vis=float(s.zoom_crops.get("min_vis", 0.2)),
                keep_empty=bool(s.zoom_crops.get("keep_empty", False)),
            )
            for (zimg, zboxes, zlabs, suffix) in zooms:
                self._dump_variant(out_img_dir, out_lbl_dir, stem, suffix, s.out_ext, zimg, zboxes, zlabs)

        # (D) multi-scale sweep
        if bool(s.zoom_sweep.get("enable", False)):
            sweeps = self._multiscale_zoom_sweep(
                im,
                boxes,
                labels,
                scales=[float(x) for x in s.zoom_sweep.get("scales", [0.35, 0.5, 0.7])],
                overlap=float(s.zoom_sweep.get("overlap", 0.25)),
                min_vis=float(s.zoom_sweep.get("min_vis", 0.2)),
                keep_empty=bool(s.zoom_sweep.get("keep_empty", False)),
                out_size=int(s.zoom_sweep.get("out_size", 640)),
            )
            for (cimg, cboxes, clabs, suffix) in sweeps:
                self._dump_variant(out_img_dir, out_lbl_dir, stem, suffix, s.out_ext, cimg, cboxes, clabs)

        # (E) augmented variants (+ optional tiling/zooms/sweeps from aug)
        H, W = im.shape[:2]
        aug_transform = self._build_transform(H, W)

        for k in range(int(s.multiplier)):
            if boxes:
                aug = aug_transform(image=im, bboxes=boxes, class_labels=labels)
                aug_img, aug_boxes, aug_labels = aug["image"], aug["bboxes"], aug["class_labels"]
            else:
                aug = aug_transform(image=im, bboxes=[], class_labels=[])
                aug_img, aug_boxes, aug_labels = aug["image"], [], []

            a_suffix = f"aug{k}"
            self._dump_variant(out_img_dir, out_lbl_dir, stem, a_suffix, s.out_ext, aug_img, aug_boxes, aug_labels)

            # tiling from augmented
            if bool(s.tiling.get("enable", False)) and bool(s.tiling.get("from_aug", False)):
                tiles = self._tile_image_and_boxes(
                    aug_img,
                    aug_boxes,
                    aug_labels,
                    tile_size=int(s.tiling.get("tile_size", 512)),
                    overlap=float(s.tiling.get("tile_overlap", 0.2)),
                    min_tile_vis=float(s.tiling.get("min_tile_vis", 0.2)),
                    keep_empty=bool(s.tiling.get("keep_empty_tiles", False)),
                )
                for (timg, tboxes, tlabs, suffix) in tiles:
                    self._dump_variant(
                        out_img_dir, out_lbl_dir, stem, f"aug{k}_{suffix}", s.out_ext, timg, tboxes, tlabs
                    )

            # zoom crops from augmented
            if bool(s.zoom_crops.get("enable", False)) and bool(s.zoom_crops.get("from_aug", False)):
                zooms = self._zoom_crops(
                    aug_img,
                    aug_boxes,
                    aug_labels,
                    zoom_scales=[float(x) for x in s.zoom_crops.get("scales", [0.5, 0.7])],
                    zoom_per_obj=int(s.zoom_crops.get("per_obj", 1)),
                    zoom_on=str(s.zoom_crops.get("on", "both")),
                    zoom_out_size=int(s.zoom_crops.get("out_size", 640)),
                    zoom_jitter=float(s.zoom_crops.get("jitter", 0.05)),
                    min_vis=float(s.zoom_crops.get("min_vis", 0.2)),
                    keep_empty=bool(s.zoom_crops.get("keep_empty", False)),
                )
                for (zimg, zboxes, zlabs, suffix) in zooms:
                    self._dump_variant(
                        out_img_dir, out_lbl_dir, stem, f"aug{k}_{suffix}", s.out_ext, zimg, zboxes, zlabs
                    )

            # multi-scale sweep from augmented
            if bool(s.zoom_sweep.get("enable", False)) and bool(s.zoom_sweep.get("from_aug", False)):
                sweeps = self._multiscale_zoom_sweep(
                    aug_img,
                    aug_boxes,
                    aug_labels,
                    scales=[float(x) for x in s.zoom_sweep.get("scales", [0.35, 0.5, 0.7])],
                    overlap=float(s.zoom_sweep.get("overlap", 0.25)),
                    min_vis=float(s.zoom_sweep.get("min_vis", 0.2)),
                    keep_empty=bool(s.zoom_sweep.get("keep_empty", False)),
                    out_size=int(s.zoom_sweep.get("out_size", 640)),
                )
                for (cimg, cboxes, clabs, suffix) in sweeps:
                    self._dump_variant(
                        out_img_dir, out_lbl_dir, stem, f"aug{k}_{suffix}", s.out_ext, cimg, cboxes, clabs
                    )

    # -------- Split processor --------
    def _process_split(self, in_root: Path, out_root: Path, split: str, *, prefer_copy: bool) -> None:
        in_img_dir = in_root / "images" / split
        in_lbl_dir = in_root / "labels" / split
        out_img_dir = out_root / "images" / split
        out_lbl_dir = out_root / "labels" / split

        if not in_img_dir.exists():
            print(f"[WARN] Missing images split '{split}'; skipping.")
            return
        if not in_lbl_dir.exists():
            print(f"[WARN] Missing labels split '{split}'; skipping.")
            return

        ensure_dir(out_img_dir)
        ensure_dir(out_lbl_dir)

        count = 0
        for img_path in _list_images(in_img_dir):
            lbl_path = in_lbl_dir / f"{img_path.stem}.txt"

            if not lbl_path.exists() and not self.settings.include_images_without_labels:
                continue

            # If negatives are included, ensure an empty label file exists so downstream code behaves consistently
            if not lbl_path.exists() and self.settings.include_images_without_labels:
                ensure_dir(lbl_path.parent)
                lbl_path.write_text("")

            # build once per image (use image size)
            im = _read_image(img_path)
            if im is None:
                print(f"[WARN] Failed to read image: {img_path}")
                continue
            H, W = im.shape[:2]
            transform = self._build_transform(H, W)

            self._process_one(
                img_path=img_path,
                lbl_path=lbl_path,
                out_img_dir=out_img_dir,
                out_lbl_dir=out_lbl_dir,
                transform=transform,
                prefer_copy=prefer_copy,
            )
            count += 1

        print(f"[OK] Split '{split}': processed {count} images → {out_img_dir.parent}")

    # -------- Public API for pipeline --------
    def run_from_yolo_root(
        self,
        *,
        data_root: Path,
        out_root: Path,
        splits: Iterable[str],
        prefer_copy: bool = False,
    ) -> None:
        print(f"[AUG] Input : {data_root}")
        print(f"[AUG] Output: {out_root}")
        for s in ("images", "labels"):
            ensure_dir(out_root / s / "train")  # prime dirs

        for split in splits:
            self._process_split(data_root, out_root, split, prefer_copy=prefer_copy)

        self._maybe_write_data_yaml(out_root)
        print(f"[OK] Augmented dataset written to: {out_root}")

    # -------- Write data.yaml --------
    def _maybe_write_data_yaml(self, out_root: Path) -> None:
        if not self.settings.write_yaml:
            return
        ensure_dir(out_root)
        yaml_path = out_root / "data.yaml"
        ds = {
            "path": str(out_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": ["disc", "cup"],
        }
        test_img_dir = out_root / "images" / "test"
        test_lbl_dir = out_root / "labels" / "test"
        if test_img_dir.exists() and any(test_img_dir.iterdir()) and test_lbl_dir.exists() and any(test_lbl_dir.iterdir()):
            ds["test"] = "images/test"
        yaml_path.write_text(yaml.safe_dump(ds, sort_keys=False))
        print(f"[OK] Wrote data.yaml → {yaml_path}")