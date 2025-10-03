#!/usr/bin/env python3
# masks_to_boxes.py — convert segmentation masks → YOLO boxes (disc=0, cup=1)
"""
Refactored, modular script to generate YOLO labels from disc/cup binary masks.

Key behavior:
- Processes ONLY images that have at least one segmentation (disc or cup).
- If --require_both is set, only keeps images with BOTH valid disc & cup boxes.
- Can filter by dataset name parsed from filename stem ("<dataset>-<patient>").
- NEW: Name-based include/exclude filters for images and masks.
- Writes one YOLO .txt per image; writes a CSV summary with coordinates & flags.

Designed for composition:
- All core steps are split into small helpers.
- A single `run_masks_to_boxes(cfg)` entrypoint can be invoked from a wrapper.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Iterable, Dict

import numpy as np
import cv2
from PIL import Image

# ----------------------------- Defaults -----------------------------

IMG_EXTS_DEFAULT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
DATA_ROOT = "/Users/carlosperez/OneDrive_UBC/Ipek_Carlos/GlaucomaDatasets"

# ----------------------------- Config -------------------------------

@dataclass
class M2BConfig:
    images: Path
    disc_masks: Path
    cup_masks: Path
    out_labels: Path
    out_csv: Path

    img_exts: Tuple[str, ...]
    pad_pct: float
    min_area_px: int
    largest_only: bool
    require_both: bool

    exclude_datasets: List[str]
    include_datasets: List[str]

    # Name-based filters (case-insensitive substrings)
    exclude_name_contains: List[str]
    include_name_contains: List[str]

    recursive: bool
    verbose: bool

# ------------------------- Small utilities --------------------------

def name_ok(filename: str, ex_subs: List[str], in_subs: List[str]) -> bool:
    """Return True if filename passes include/exclude substring filters."""
    nm = filename.lower()
    if ex_subs and any(s in nm for s in ex_subs):
        return False
    if in_subs and not any(s in nm for s in in_subs):
        return False
    return True

def expand(p: str | Path) -> Path:
    """Expand ~ and resolve to absolute Path."""
    return Path(p).expanduser().resolve()

def read_image_size(p: Path) -> Tuple[int, int]:
    """Return (W, H) without loading full pixels into RAM."""
    with Image.open(p) as im:
        return im.size

def list_images(images_dir: Path, exts: Iterable[str], recursive: bool) -> List[Path]:
    """Enumerate images under a directory with optional recursion."""
    exts = tuple(e.lower() for e in exts)
    if recursive:
        paths: List[Path] = []
        for e in exts:
            paths.extend(images_dir.rglob(f"*{e}"))
        return sorted(paths)
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])

def stem_to_dataset_patient(stem: str) -> Tuple[str, str]:
    """
    Parse "<dataset>-<patient>" → ("dataset", "patient").
    If no dash is present, returns ("unknown", stem).
    """
    if "-" not in stem:
        return "unknown", stem
    i = stem.rfind("-")
    return stem[:i], stem[i + 1 :]

def first_existing_with_stem(folder: Path, stem: str) -> Optional[Path]:
    """Find first file in folder whose basename matches 'stem' with common image extensions."""
    for ext in [".png",".PNG",".jpg",".JPG",".jpeg",".JPEG",".tif",".tiff",".TIF",".TIFF"]:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    cands = sorted(folder.glob(f"{stem}.*"))
    return cands[0] if cands else None

# --------------------------- Mask handling --------------------------

def load_binary_mask(p: Optional[Path]) -> Optional[np.ndarray]:
    """
    Robust mask loader:
    - Accepts 1ch, 3ch, or 4ch (alpha) images
    - Foreground = any non-zero
    Returns uint8 binary mask or None.
    """
    if not p or not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None

    if m.ndim == 2:
        return (m > 0).astype(np.uint8)

    if m.ndim == 3:
        if m.shape[2] == 3:
            return (np.any(m > 0, axis=2)).astype(np.uint8)
        if m.shape[2] == 4:
            rgb = np.any(m[:, :, :3] > 0, axis=2)
            alpha = m[:, :, 3] > 0
            return (rgb | alpha).astype(np.uint8)

    mm = np.squeeze(m)
    return (mm > 0).astype(np.uint8)

def largest_component_mask(bin_mask: Optional[np.ndarray], min_area_px: int, largest_only: bool) -> Optional[np.ndarray]:
    """
    Keep only components ≥ min_area_px. If largest_only, keep only the largest valid component.
    Returns a binary mask or None (if nothing valid).
    """
    if bin_mask is None:
        return None
    if bin_mask.sum() == 0:
        return None

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return None  # only background

    fg_idx = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area_px]
    if not fg_idx:
        return None
    if largest_only and len(fg_idx) > 1:
        i_max = max(fg_idx, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        fg_idx = [i_max]
    return np.isin(labels, fg_idx).astype(np.uint8)

# ------------------------ Boxes & formatting ------------------------

def bbox_from_mask(mask: Optional[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) inclusive; None if invalid/empty."""
    if mask is None or mask.sum() == 0:
        return None
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def pad_clamp_xyxy(box: Optional[Tuple[int,int,int,int]], pad_pct: float, W: int, H: int) -> Optional[Tuple[int,int,int,int]]:
    """Pad a box by pad_pct*max(W,H), clamp to [0..W-1/H-1]; return None if box is None or degenerate."""
    if box is None:
        return None
    x1, y1, x2, y2 = box
    if pad_pct > 0:
        p = int(round(pad_pct * max(W, H)))
        x1 -= p; y1 -= p; x2 += p; y2 += p
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W - 1, x2); y2 = min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def yolo_line_from_xyxy(box: Tuple[int,int,int,int], W: int, H: int, cls_id: int) -> str:
    """Format a single YOLO line 'cls cx cy w h' with normalized floats."""
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return ""
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f"{cls_id} {cx / W:.6f} {cy / H:.6f} {bw / W:.6f} {bh / H:.6f}"

def write_label(out_dir: Path, stem: str, lines: List[str]) -> None:
    """Write YOLO .txt label file for stem."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.txt").write_text("\n".join([ln for ln in lines if ln]) + ("\n" if lines else ""))

# ------------------------- Single-image step ------------------------

def process_one_image(
    img_path: Path,
    disc_dir: Path,
    cup_dir: Path,
    cfg: M2BConfig
) -> Optional[Dict]:
    """
    Process a single image:
    - Read masks (if present)
    - Post-process masks
    - Create YOLO lines & a CSV row dict
    Returns a row dict (for CSV) or None if filtered out.
    """
    stem = img_path.stem
    dataset, patient = stem_to_dataset_patient(stem)

    # include/exclude dataset filtering
    if cfg.include_datasets and (dataset not in cfg.include_datasets):
        return None
    if dataset in cfg.exclude_datasets:
        return None

    # Find mask paths
    disc_mp = first_existing_with_stem(disc_dir, stem)
    cup_mp  = first_existing_with_stem(cup_dir,  stem)

    # Apply name filters to mask filenames themselves
    if disc_mp and not name_ok(disc_mp.name, cfg.exclude_name_contains, cfg.include_name_contains):
        disc_mp = None
    if cup_mp and not name_ok(cup_mp.name, cfg.exclude_name_contains, cfg.include_name_contains):
        cup_mp = None

    # Skip if neither mask exists (after name filters)
    if disc_mp is None and cup_mp is None:
        return {"_skip_reason": "no_mask"}

    # If require_both, both must be present
    if cfg.require_both and (disc_mp is None or cup_mp is None):
        return {"_skip_reason": "require_both_missing_one"}

    # Image size
    try:
        W, H = read_image_size(img_path)
    except Exception:
        return {"_skip_reason": "unreadable_image"}

    # Load masks → QC → largest components
    disc_mask = load_binary_mask(disc_mp) if disc_mp else None
    cup_mask  = load_binary_mask(cup_mp)  if cup_mp  else None

    dmask = largest_component_mask(disc_mask, cfg.min_area_px, cfg.largest_only) if disc_mask is not None else None
    cmask = largest_component_mask(cup_mask,  cfg.min_area_px, cfg.largest_only) if cup_mask  is not None else None

    # Boxes with padding/clamp
    dbox = pad_clamp_xyxy(bbox_from_mask(dmask), cfg.pad_pct, W, H) if dmask is not None else None
    cbox = pad_clamp_xyxy(bbox_from_mask(cmask), cfg.pad_pct, W, H) if cmask is not None else None

    # require_both implies both valid boxes required
    if cfg.require_both and (dbox is None or cbox is None):
        return {"_skip_reason": "require_both_no_valid_box"}

    # If not require_both, accept if at least one valid box exists
    if dbox is None and cbox is None:
        return {"_skip_reason": "no_valid_box"}

    # Compose YOLO lines
    lines: List[str] = []
    if dbox is not None:
        lines.append(yolo_line_from_xyxy(dbox, W, H, cls_id=0))  # disc
    if cbox is not None:
        lines.append(yolo_line_from_xyxy(cbox, W, H, cls_id=1))  # cup

    # CSV row
    row = {
        "stem": stem, "image_path": str(img_path), "W": W, "H": H,
        "dataset": dataset, "patient": patient,
        "disc_x1": "", "disc_y1": "", "disc_x2": "", "disc_y2": "",
        "cup_x1": "",  "cup_y1":  "", "cup_x2":  "", "cup_y2":  "",
        "disc_mask_found": int(disc_mp is not None), "cup_mask_found": int(cup_mp is not None),
        "disc_kept": int(dbox is not None), "cup_kept": int(cbox is not None),
        "_yolo_lines": lines,  # attached for caller
    }
    if dbox is not None:
        row.update({"disc_x1": dbox[0], "disc_y1": dbox[1], "disc_x2": dbox[2], "disc_y2": dbox[3]})
    if cbox is not None:
        row.update({"cup_x1": cbox[0], "cup_y1": cbox[1], "cup_x2": cbox[2], "cup_y2": cbox[3]})
    return row

# ------------------------- Orchestration layer ----------------------

def run_masks_to_boxes(cfg: M2BConfig) -> None:
    """Main pipeline: scan images, process each, write labels + CSV summary."""
    if cfg.verbose:
        print(f"[CFG] images={cfg.images}")
        print(f"[CFG] disc_masks={cfg.disc_masks}")
        print(f"[CFG] cup_masks={cfg.cup_masks}")
        print(f"[CHK] exist? images={cfg.images.exists()} disc={cfg.disc_masks.exists()} cup={cfg.cup_masks.exists()}")

    img_paths = list_images(cfg.images, cfg.img_exts, cfg.recursive)

    # Apply filename-based filtering to images
    if cfg.exclude_name_contains or cfg.include_name_contains:
        img_paths = [
            p for p in img_paths
            if name_ok(p.name, cfg.exclude_name_contains, cfg.include_name_contains)
        ]

    cols = [
        "stem","image_path","W","H","dataset","patient",
        "disc_x1","disc_y1","disc_x2","disc_y2",
        "cup_x1","cup_y1","cup_x2","cup_y2",
        "disc_mask_found","cup_mask_found","disc_kept","cup_kept"
    ]
    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    skipped_no_mask = 0
    skipped_require_both = 0
    skipped_no_valid_box = 0

    with open(cfg.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for ip in img_paths:
            total += 1
            row = process_one_image(ip, cfg.disc_masks, cfg.cup_masks, cfg)

            # row is None → filtered out by include/exclude (dataset filters)
            if row is None:
                continue

            # handle skip reasons
            if "_skip_reason" in row:
                reason = row["_skip_reason"]
                if reason == "no_mask":
                    skipped_no_mask += 1
                elif reason in ("require_both_missing_one", "require_both_no_valid_box"):
                    skipped_require_both += 1
                elif reason in ("no_valid_box", "unreadable_image"):
                    skipped_no_valid_box += 1
                continue

            # write label + CSV
            lines = row.pop("_yolo_lines", [])
            write_label(cfg.out_labels, row["stem"], lines)
            writer.writerow(row)
            kept += 1

    print(f"[OK] Images scanned           : {total}")
    print(f"[OK] Labels written (kept)    : {kept}")
    print(f"[SKIP] No mask present        : {skipped_no_mask}")
    if cfg.require_both:
        print(f"[SKIP] Missing one (require_both): {skipped_require_both}")
    print(f"[SKIP] No valid box after QC  : {skipped_no_valid_box}")
    print(f"[OK] Labels directory         : {cfg.out_labels}")
    print(f"[OK] Summary CSV              : {cfg.out_csv}")

# ------------------------------ CLI --------------------------------

def parse_args() -> M2BConfig:
    ap = argparse.ArgumentParser(
        description="SMDG masks → YOLO boxes (disc=0, cup=1); process only images with segmentations."
    )
    # Paths
    ap.add_argument("--images",     default=f"{DATA_ROOT}/SMDG-19/full-fundus/full-fundus", help="Fundus image directory")
    ap.add_argument("--disc_masks", default=f"{DATA_ROOT}/SMDG-19/optic-disc/optic-disc",  help="Optic disc mask directory")
    ap.add_argument("--cup_masks",  default=f"{DATA_ROOT}/SMDG-19/optic-cup/optic-cup",    help="Optic cup mask directory")
    ap.add_argument("--out_labels", default="./../data/labels",             help="Output folder for YOLO .txt labels")
    ap.add_argument("--out_csv",    default="./../data/labels_summary.csv", help="CSV summary path")

    # Processing policy
    ap.add_argument("--img_exts",    nargs="+", default=list(IMG_EXTS_DEFAULT), help="Accepted image extensions")
    ap.add_argument("--pad_pct",     type=float, default=0.0,  help="Padding as fraction of max(W,H)")
    ap.add_argument("--min_area_px", type=int,   default=25,    help="Remove mask blobs smaller than this (pixels)")
    ap.add_argument("--largest_only", action="store_true", help="If multiple components remain, keep only the largest")
    ap.add_argument("--require_both", action="store_true",
                    help="If set, only keep images where BOTH disc and cup masks exist (and yield valid boxes).")

    # Dataset filtering
    ap.add_argument("--exclude_datasets", default="", help="Comma-separated dataset names to exclude")
    ap.add_argument("--include_datasets", default="", help="Comma-separated dataset names to include exclusively (whitelist)")

    # Name-based filtering (images & masks)
    ap.add_argument("--exclude_name_contains", default="",
                    help="Comma-separated substrings (case-insensitive). Skip any image/mask whose filename contains any of these.")
    ap.add_argument("--include_name_contains", default="",
                    help="Comma-separated substrings (case-insensitive). If set, keep only files whose filename contains any of these.")

    # Misc
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders under --images")
    ap.add_argument("--verbose",   action="store_true", help="Print selected paths and early counts")

    args = ap.parse_args()

    return M2BConfig(
        images       = expand(args.images),
        disc_masks   = expand(args.disc_masks),
        cup_masks    = expand(args.cup_masks),
        out_labels   = expand(args.out_labels),
        out_csv      = expand(args.out_csv),

        img_exts     = tuple(args.img_exts),
        pad_pct      = float(args.pad_pct),
        min_area_px  = int(args.min_area_px),
        largest_only = bool(args.largest_only),
        require_both = bool(args.require_both),

        exclude_datasets=[s.strip() for s in args.exclude_datasets.split(",") if s.strip()],
        include_datasets=[s.strip() for s in args.include_datasets.split(",") if s.strip()],

        exclude_name_contains=[s.strip().lower() for s in args.exclude_name_contains.split(",") if s.strip()],
        include_name_contains=[s.strip().lower() for s in args.include_name_contains.split(",") if s.strip()],

        recursive    = bool(args.recursive),
        verbose      = bool(args.verbose),
    )

# ----------------------------- Entrypoint ---------------------------

def main():
    cfg = parse_args()
    run_masks_to_boxes(cfg)

if __name__ == "__main__":
    main()