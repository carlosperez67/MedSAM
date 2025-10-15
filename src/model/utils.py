# device_utils.py

# device_utils.py
from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io as skio, transform as sktf


def ultralytics_device_arg() -> str:
    """
    Returns a device argument for Ultralytics .train():
      - If YOLO_DEVICES is set (e.g. "0,1" or "0,1,2,3"), return it (enables DDP).
      - Else if CUDA is visible, return "0".
      - Else return "cpu".
    """
    env = os.getenv("YOLO_DEVICES")
    if env:
        return env  # Ultralytics treats "0,1" as multi-GPU (DDP)
    # simple fallback
    if is_available() and device_count() > 0:
        return "0"
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