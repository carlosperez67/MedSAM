from __future__ import annotations

import uuid
from typing import Tuple

import numpy as np

import os
import shutil
from pathlib import Path
from typing import Iterable, Dict, List


IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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

