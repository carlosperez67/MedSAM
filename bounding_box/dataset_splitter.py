#!/usr/bin/env python3
# split_papila_yolo.py
"""
Split Papila fundus images + YOLO labels into train/val/test (patient-wise).

Assumptions
-----------
- Images live under IMG_DIR (default: Papila FundusImages).
- YOLO label TXT files (0=disc, 1=cup) live under LABELS_DIR; one .txt per image stem.
- Stems look like 'RET002OD' or similar; patient id is parsed as the leading token like 'RET002'.
  (Regex is configurable; default handles 'RET<digits>(OD|OS)'.)

Outputs
-------
DATASET_ROOT/
  images/{train,val,test}/*.{png|jpg|...}
  labels/{train,val,test}/*.txt

Usage
-----
python split_papila_yolo.py \
  --img_dir   /Users/.../Papila_db/FundusImages \
  --labels_dir ./papila_boxes_two/yolo_labels \
  --out_root  ./papila_yolo \
  --val_frac  0.15 --test_frac 0.15 \
  --seed 1337 --copy
"""

import argparse
import os
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir",
        default="/Users/carlosperez/OneDrive_UBC/Ipek_Carlos/GlaucomaDatasets/Papila_db/FundusImages")
    ap.add_argument("--labels_dir",
        default="./papila_boxes_two/yolo_labels")
    ap.add_argument("--out_root", default="./papila_yolo")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--copy", action="store_true",
        help="Copy files instead of symlinking (default is symlink).")
    ap.add_argument("--patient_regex", default=r"^(RET\d+)",
        help="Regex to extract patient id from stem (default matches 'RET002' in 'RET002OD').")
    return ap.parse_args()

def stem_of(p: Path) -> str:
    return p.stem  # filename without extension

def find_images(img_dir: Path) -> List[Path]:
    out = []
    for ext in IMG_EXTS:
        out.extend(img_dir.glob(f"*{ext}"))
    return sorted(out)

def main():
    args = parse_args()
    random.seed(args.seed)

    img_dir = Path(args.img_dir)
    lbl_dir = Path(args.labels_dir)
    out_root = Path(args.out_root)

    if not img_dir.exists():
        raise SystemExit(f"Image dir not found: {img_dir}")
    if not lbl_dir.exists():
        raise SystemExit(f"Labels dir not found: {lbl_dir}")

    images = find_images(img_dir)
    if not images:
        raise SystemExit("No images found.")

    # Map: patient_id -> list of stems (image files that have a label)
    pat_re = re.compile(args.patient_regex, re.IGNORECASE)
    patients: Dict[str, List[Tuple[Path, Path]]] = {}
    kept = 0

    for ip in images:
        st = stem_of(ip)
        # label txt expected to be named <stem>.txt
        lp = lbl_dir / f"{st}.txt"
        if not lp.exists():
            # silently skip images without labels
            continue
        m = pat_re.match(st)
        if not m:
            # if no patient id parsed, treat whole stem as patient id (fallback)
            pid = st
        else:
            pid = m.group(1)
        patients.setdefault(pid, []).append((ip, lp))
        kept += 1

    if kept == 0:
        raise SystemExit("No (image,label) pairs found. Check labels_dir and file stems.")

    pids = list(patients.keys())
    random.shuffle(pids)

    n = len(pids)
    n_test = int(round(args.test_frac * n))
    n_val  = int(round(args.val_frac  * n))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise SystemExit("Invalid split sizes; adjust val_frac/test_frac.")

    pid_train = set(pids[:n_train])
    pid_val   = set(pids[n_train:n_train+n_val])
    pid_test  = set(pids[n_train+n_val:])

    # Prepare folders
    for split in ("train","val","test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    import shutil

    def place(src: Path, dst: Path):
        if args.copy:
            shutil.copy2(src, dst)
        else:
            # create symlink relative to destination
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())

    # Move/symlink files by split
    for pid, pairs in patients.items():
        split = "train" if pid in pid_train else "val" if pid in pid_val else "test"
        for ip, lp in pairs:
            dst_img = out_root / "images" / split / ip.name
            dst_lbl = out_root / "labels" / split / f"{ip.stem}.txt"
            place(ip, dst_img)
            place(lp, dst_lbl)

    # Summary
    def count_items(split):
        return len(list((out_root/"images"/split).glob("*")))
    print("Patients:", len(pids), "| train/val/test:", len(pid_train), len(pid_val), len(pid_test))
    print("Images  :", {s: count_items(s) for s in ("train","val","test")})
    print(f"[OK] Wrote split under: {out_root}")

if __name__ == "__main__":
    main()