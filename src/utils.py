# src.device_utils.py

# device_utils.py
from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io as skio, transform as sktf
from torch.accelerator import is_available, device_count

def need(p: Path, what: str) -> None:
    if not p.exists():
        raise SystemExit(f"[ERR] {what} not found: {p}")

# src/device_utils.py

import os
import torch

def ultralytics_device_arg() -> str:
    """
    Decide the `device` argument for Ultralytics:
      - If YOLO_DEVICES is set (e.g., "0,1,2,3"), return it (DDP).
      - Else if CUDA_VISIBLE_DEVICES is set, use all *visible* GPUs ("0,1,...").
      - Else if CUDA is available, use "0".
      - Else if Apple MPS is available, use "mps".
      - Else "cpu".
    """
    # 1) Respect explicit multi-GPU setting
    env = os.getenv("YOLO_DEVICES")
    if env:
        return env

    # 2) Use all visible GPUs if CUDA_VISIBLE_DEVICES is set
    cvd = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd not in ("", "-1"):
        # Map to local ordinals 0..N-1
        n = len([x for x in cvd.split(",") if x.strip() != ""])
        return ",".join(str(i) for i in range(n)) if n > 1 else "0"

    # 3) Native CUDA check
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "0"

    # 4) Apple GPU
    try:
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except Exception:
        pass

    # 5) CPU fallback
    return "cpu"

# ======================================================================
# Small helpers (device, I/O, geometry, metrics)
# ======================================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def expand(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()

def place(src: Path, dst: Path, copy_files: bool) -> None:
    """Copy or symlink src → dst."""
    ensure_dir(dst.parent)
    if copy_files:
        shutil.copy2(src, dst)
    else:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

def load_image_bgr(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return im

def save_mask_png(path: Path, mask_bool: np.ndarray) -> None:
    ensure_dir(path.parent)
    skio.imsave(str(path), (mask_bool.astype(np.uint8) * 255), check_contrast=False)

def save_viz(path: Path, viz_bgr: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), viz_bgr)

def mask_vertical_height(mask: np.ndarray) -> int:
    ys = np.where(mask > 0)[0]
    if ys.size == 0:
        return 0
    return int(ys.max() - ys.min() + 1)

def cdr_from_masks(disc_mask: Optional[np.ndarray], cup_mask: Optional[np.ndarray]) -> Optional[float]:
    if disc_mask is None or cup_mask is None:
        return None
    dh = mask_vertical_height(disc_mask)
    if dh <= 0:
        return None
    ch = mask_vertical_height(cup_mask)
    return float(ch) / float(dh)

def corners_inside(mask: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box_xyxy
    H, W = mask.shape[:2]
    x1 = np.clip(x1, 0, W - 1); x2 = np.clip(x2, 0, W - 1)
    y1 = np.clip(y1, 0, H - 1); y2 = np.clip(y2, 0, H - 1)
    pts = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    for (x, y) in pts:
        if mask[int(y), int(x)] == 0:
            return False
    return True

def tight_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    return (x1, y1, x2 + 1, y2 + 1)  # half-open

def shrink_box_to_fit_mask(mask: np.ndarray,
                           base_box: Tuple[int, int, int, int],
                           step_frac: float = 0.02,
                           max_iter: int = 200,
                           min_side_px: int = 8) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = map(float, base_box)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    H, W = mask.shape[:2]
    for _ in range(max_iter):
        bx1, by1, bx2, by2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        bx1 = max(0, min(bx1, W - 1)); bx2 = max(1, min(bx2, W))
        by1 = max(0, min(by1, H - 1)); by2 = max(1, min(by2, H))
        if (bx2 - bx1) < min_side_px or (by2 - by1) < min_side_px:
            return None
        if corners_inside(mask, (bx1, by1, bx2, by2)):
            return (bx1, by1, bx2, by2)
        w = (x2 - x1) * (1.0 - step_frac)
        h = (y2 - y1) * (1.0 - step_frac)
        x1 = cx - w / 2.0; x2 = cx + w / 2.0
        y1 = cy - h / 2.0; y2 = cy + h / 2.0
    return None

def overlay_masks_and_boxes(
    img_bgr: np.ndarray,
    disc_mask: Optional[np.ndarray],
    cup_mask: Optional[np.ndarray],
    disc_box: Optional[Tuple[int,int,int,int]],
    cup_box: Optional[Tuple[int,int,int,int]],
    cdr_text: Optional[str] = None,
    alpha: float = 0.4
) -> np.ndarray:
    out = img_bgr.copy()
    if disc_mask is not None:
        overlay = out.copy()
        overlay[disc_mask > 0] = (0, 255, 255)  # yellow
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    if cup_mask is not None:
        overlay = out.copy()
        overlay[cup_mask > 0] = (255, 0, 255)  # magenta
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    if disc_box is not None:
        x1, y1, x2, y2 = disc_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
    if cup_box is not None:
        x1, y1, x2, y2 = cup_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 200), 2)
    if cdr_text:
        cv2.putText(out, cdr_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(out, cdr_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def make_side_by_side(
    img_bgr: np.ndarray,
    pred_disc: Optional[np.ndarray], pred_cup: Optional[np.ndarray],
    gt_disc: Optional[np.ndarray],   gt_cup: Optional[np.ndarray],
    pred_text: str, gt_text: str
) -> np.ndarray:
    left  = overlay_masks_and_boxes(img_bgr, pred_disc, pred_cup, None, None, cdr_text=pred_text)
    right = overlay_masks_and_boxes(img_bgr, gt_disc, gt_cup, None, None, cdr_text=gt_text)
    return np.hstack([left, right])

def dice(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    if pred is None or gt is None:
        return None
    predb = (pred > 0).astype(np.uint8)
    gtb = (gt > 0).astype(np.uint8)
    inter = (predb & gtb).sum()
    s = predb.sum() + gtb.sum()
    if s == 0:
        return None
    return 2.0 * inter / float(s)

def split_has_data(root: Path, split: str) -> bool:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    if not (img_dir.exists() and lbl_dir.exists()):
        return False
    try:
        return any(img_dir.iterdir()) and any(lbl_dir.iterdir())
    except Exception:
        return False

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

def ensure_parent(p: Path) -> None:
    ensure_dir(p.parent)


def safe_link_or_copy(src: Path, dst: Path, prefer_copy: bool = False) -> None:
    """Symlink if allowed; otherwise copy. Overwrites existing."""
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if prefer_copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(src, dst)


def list_files_with_ext(root: Path, exts: Iterable[str] = IMG_EXTS, recursive: bool = True) -> List[Path]:
    exts = tuple(e.lower() for e in exts)
    if recursive:
        return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    return sorted([p for p in root.iterdir() if p.suffix.lower() in exts])


def stem_map_by_first_match(root: Path, exts: Iterable[str] = IMG_EXTS) -> Dict[str, Path]:
    """Map file stem → first matching path under root (recursive)."""
    out: Dict[str, Path] = {}
    for p in list_files_with_ext(root, exts, recursive=True):
        out.setdefault(p.stem, p)
    return out

def gen_uid(n: int = 12) -> str:
    return uuid.uuid4().hex[:n]


def read_image_size(img_path: Path) -> Tuple[int, int]:
    """Return (width, height) without fully decoding if possible."""
    try:
        from PIL import Image  # type: ignore
        with Image.open(img_path) as im:
            return im.size  # (W, H)
    except Exception:
        pass
    try:
        import cv2  # type: ignore
        im = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise ValueError("cv2.imread returned None")
        h, w = im.shape[:2]
        return w, h
    except Exception:
        pass
    try:
        import imageio.v3 as iio  # type: ignore
        im = iio.imread(str(img_path))
        h, w = im.shape[:2]
        return w, h
    except Exception as e:
        raise RuntimeError(f"Unable to read image size for {img_path}: {e}") from e

def ensure_bool_mask(arr: np.ndarray) -> np.ndarray:
    return arr.astype(bool, copy=False)

def xyxy_to_xc_yc_wh(x1: float, y1: float, x2: float, y2: float):
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0
    return xc, yc, w, h

def xc_yc_wh_to_xyxy(xc: float, yc: float, w: float, h: float):
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return x1, y1, x2, y2
