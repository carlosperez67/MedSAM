#!/usr/bin/env python3
# masks_to_boxes.py — process ONLY images that have a segmentation
import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import cv2
from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

DATA_ROOT = "/Users/carlosperez/OneDrive_UBC/Ipek_Carlos/GlaucomaDatasets"

def parse_args():
    ap = argparse.ArgumentParser(description="SMDG: binary masks → YOLO boxes (disc=0, cup=1); process only images with segmentations.")
    # Paths
    ap.add_argument("--images",     default=f"{DATA_ROOT}/SMDG-19/full-fundus/full-fundus", help="Fundus image directory")
    ap.add_argument("--disc_masks", default=f"{DATA_ROOT}/SMDG-19/optic-disc/optic-disc",  help="Optic disc mask directory")
    ap.add_argument("--cup_masks",  default=f"{DATA_ROOT}/SMDG-19/optic-cup/optic-cup",    help="Optic cup mask directory")
    ap.add_argument("--out_labels", default="./../data/labels",            help="Output folder for YOLO .txt labels")
    ap.add_argument("--out_csv",    default="./../data/labels_summary.csv", help="CSV summary path")

    # Processing policy
    ap.add_argument("--img_exts",    nargs="+", default=list(IMG_EXTS), help="Accepted image extensions")
    ap.add_argument("--pad_pct",     type=float, default=0.0, help="Padding as fraction of max(W,H)")
    ap.add_argument("--min_area_px", type=int, default=25,   help="Remove mask blobs smaller than this (pixels)")
    ap.add_argument("--largest_only", action="store_true",   help="If multiple components remain, keep only the largest")
    ap.add_argument("--require_both", action="store_true",
                    help="If set, only keep images where BOTH disc and cup masks exist (and yield valid boxes).")

    # Dataset filtering
    ap.add_argument("--exclude_datasets", default="", help="Comma-separated dataset names to exclude")
    ap.add_argument("--include_datasets", default="", help="Comma-separated dataset names to include exclusively (whitelist)")

    # Misc
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders under --images")
    ap.add_argument("--verbose",   action="store_true", help="Print selected paths and early counts")
    return ap.parse_args()

# ---------- helpers ----------

def read_image_size(p: Path) -> Tuple[int, int]:
    with Image.open(p) as im:
        return im.size  # (W,H)

def load_binary_mask(p: Optional[Path]) -> Optional[np.ndarray]:
    """Robust mask loader: foreground if any channel or alpha is non-zero."""
    if not p or not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 2:
        return (m > 0).astype(np.uint8)
    if m.ndim == 3 and m.shape[2] == 3:
        return (np.any(m > 0, axis=2)).astype(np.uint8)
    if m.ndim == 3 and m.shape[2] == 4:
        rgb = np.any(m[:, :, :3] > 0, axis=2)
        alpha = m[:, :, 3] > 0
        return (rgb | alpha).astype(np.uint8)
    mm = np.squeeze(m)
    return (mm > 0).astype(np.uint8)

def largest_component_mask(bin_mask: np.ndarray, min_area_px: int, largest_only: bool) -> Optional[np.ndarray]:
    if bin_mask is None or bin_mask.sum() == 0:
        return None
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return None
    fg = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area_px]
    if not fg:
        return None
    if largest_only and len(fg) > 1:
        i_max = max(fg, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        fg = [i_max]
    return np.isin(labels, fg).astype(np.uint8)

def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if mask is None or mask.sum() == 0:
        return None
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def pad_clamp_xyxy(box, pad_pct: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    if pad_pct > 0:
        p = int(round(pad_pct * max(W, H)))
        x1 -= p; y1 -= p; x2 += p; y2 += p
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W - 1, x2); y2 = min(H - 1, y2)
    return x1, y1, x2, y2

def yolo_line_from_xyxy(box, W: int, H: int, cls_id: int) -> str:
    x1, y1, x2, y2 = box
    bw = x2 - x1; bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return ""
    cx = x1 + bw / 2.0; cy = y1 + bh / 2.0
    return f"{cls_id} {cx / W:.6f} {cy / H:.6f} {bw / W:.6f} {bh / H:.6f}"

def write_label(out_dir: Path, stem: str, lines: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

def stem_to_dataset_patient(stem: str) -> Tuple[str, str]:
    if "-" not in stem:
        return "unknown", stem
    i = stem.rfind("-")
    return stem[:i], stem[i+1:]

def first_existing_with_stem(folder: Path, stem: str) -> Optional[Path]:
    for ext in [".png",".PNG",".jpg",".JPG",".jpeg",".JPEG",".tif",".tiff",".TIF",".TIFF"]:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    cands = sorted(folder.glob(f"{stem}.*"))
    return cands[0] if cands else None

def enumerate_images(images_dir: Path, exts: Tuple[str, ...], recursive: bool) -> List[Path]:
    if recursive:
        paths = []
        for e in exts:
            paths.extend(images_dir.glob(f"**/*{e}"))
        return sorted(paths)
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])

# ---------- main ----------

def main():
    args = parse_args()

    images_dir = Path(args.images)
    disc_dir   = Path(args.disc_masks)
    cup_dir    = Path(args.cup_masks)
    out_labels = Path(args.out_labels)
    out_csv    = Path(args.out_csv)

    if args.verbose:
        print(f"[CFG] images={images_dir}  disc={disc_dir}  cup={cup_dir}")
        print(f"[CHK] exist? images={images_dir.exists()} disc={disc_dir.exists()} cup={cup_dir.exists()}")

    exclude = {s.strip() for s in args.exclude_datasets.split(",") if s.strip()}
    include = {s.strip() for s in args.include_datasets.split(",") if s.strip()}

    img_paths = enumerate_images(images_dir, tuple(x.lower() for x in args.img_exts), args.recursive)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["stem","image_path","W","H","dataset","patient",
            "disc_x1","disc_y1","disc_x2","disc_y2",
            "cup_x1","cup_y1","cup_x2","cup_y2",
            "disc_mask_found","cup_mask_found","disc_kept","cup_kept"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        total = 0
        kept = 0
        skipped_no_mask = 0
        skipped_require_both = 0
        skipped_no_valid_box = 0

        for ip in img_paths:
            total += 1
            stem = ip.stem
            dataset, patient = stem_to_dataset_patient(stem)

            if include and dataset not in include:
                continue
            if dataset in exclude:
                continue

            # find corresponding masks
            disc_mp = first_existing_with_stem(disc_dir, stem)
            cup_mp  = first_existing_with_stem(cup_dir,  stem)

            # Skip if neither mask exists (your new requirement)
            if disc_mp is None and cup_mp is None:
                skipped_no_mask += 1
                continue

            # If require both, skip when either is missing
            if args.require_both and (disc_mp is None or cup_mp is None):
                skipped_require_both += 1
                continue

            # image size
            try:
                W, H = read_image_size(ip)
            except Exception:
                # treat unreadable as no valid box
                skipped_no_valid_box += 1
                continue

            # load masks
            disc_mask = load_binary_mask(disc_mp) if disc_mp else None
            cup_mask  = load_binary_mask(cup_mp)  if cup_mp  else None

            # post-process & boxes
            dmask = largest_component_mask(disc_mask, args.min_area_px, args.largest_only) if disc_mask is not None else None
            cmask = largest_component_mask(cup_mask,  args.min_area_px, args.largest_only) if cup_mask  is not None else None

            dbox = pad_clamp_xyxy(bbox_from_mask(dmask), args.pad_pct, W, H) if dmask is not None and dmask.any() else None
            cbox = pad_clamp_xyxy(bbox_from_mask(cmask), args.pad_pct, W, H) if cmask is not None and cmask.any() else None

            # If require_both, both boxes must be valid
            if args.require_both and (dbox is None or cbox is None):
                skipped_no_valid_box += 1
                continue

            # If not require_both, accept if at least one valid box exists
            if (dbox is None) and (cbox is None):
                skipped_no_valid_box += 1
                continue

            # Write label ONLY when we have a segmentation-derived box
            lines = []
            if dbox is not None:
                lines.append(yolo_line_from_xyxy(dbox, W, H, cls_id=0))
            if cbox is not None:
                lines.append(yolo_line_from_xyxy(cbox, W, H, cls_id=1))
            write_label(out_labels, stem, lines)
            kept += 1

            # CSV row
            row = {"stem": stem, "image_path": str(ip), "W": W, "H": H,
                   "dataset": dataset, "patient": patient,
                   "disc_x1": "", "disc_y1": "", "disc_x2": "", "disc_y2": "",
                   "cup_x1": "", "cup_y1": "", "cup_x2": "", "cup_y2": "",
                   "disc_mask_found": int(disc_mp is not None), "cup_mask_found": int(cup_mp is not None),
                   "disc_kept": int(dbox is not None), "cup_kept": int(cbox is not None)}
            if dbox is not None:
                row.update({"disc_x1": dbox[0], "disc_y1": dbox[1], "disc_x2": dbox[2], "disc_y2": dbox[3]})
            if cbox is not None:
                row.update({"cup_x1": cbox[0], "cup_y1": cbox[1], "cup_x2": cbox[2], "cup_y2": cbox[3]})
            writer.writerow(row)

    print(f"[OK] Images scanned           : {total}")
    print(f"[OK] Labels written (kept)    : {kept}")
    print(f"[SKIP] No mask present        : {skipped_no_mask}")
    if args.require_both:
        print(f"[SKIP] Missing one (require_both): {skipped_require_both}")
    print(f"[SKIP] No valid box after QC  : {skipped_no_valid_box}")
    print(f"[OK] Labels directory         : {out_labels}")
    print(f"[OK] Summary CSV              : {out_csv}")

if __name__ == "__main__":
    main()