#!/usr/bin/env python3
# split_yolo_by_patient.py
"""
Generic YOLO dataset splitter with patient-wise separation.

Defaults assume you’ve already generated YOLO labels with your earlier
Papila / SMDG scripts into a single `labels` directory, and images are
stored under a unified `images` folder.

This ensures the same patient (OD + OS, disc + cup) stays entirely in
train, val, or test.

Outputs
-------
OUT_ROOT/
  images/{train,val,test}
  labels/{train,val,test}
  split_manifest.csv
  (optional) data.yaml for Ultralytics
"""

import argparse, csv, random, re, shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

IMG_EXTS = (".png",".jpg",".jpeg",".tif",".tiff")
DATA_ROOT = "/Users/carlosperez/OneDrive_UBC/Ipek_Carlos/GlaucomaDatasets"


# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Split YOLO dataset by patient id.")
    ap.add_argument("--images_root",     default=f"{DATA_ROOT}/SMDG-19/full-fundus/full-fundus", help="Fundus image directory")
    ap.add_argument("--labels_dir",
        default="./../data/labels", help="Folder with YOLO label .txt files.")
    ap.add_argument("--out_root",
        default="./../data/yolo_split", help="Output dataset root.")
    ap.add_argument("--val_frac", type=float, default=0.15,
        help="Fraction of patients to assign to validation.")
    ap.add_argument("--test_frac", type=float, default=0.15,
        help="Fraction of patients to assign to test.")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed.")
    ap.add_argument("--copy", action="store_true",
        help="Copy instead of symlink.")
    ap.add_argument("--patient_regex",
        default="", help="Regex with one capturing group for patient_id.")
    ap.add_argument("--write_yaml", action="store_true",
        help="Also write Ultralytics data.yaml.")
    return ap.parse_args()

# ---------------- Helpers ----------------
def enumerate_images_recursive(images_root: Path) -> Dict[str, Path]:
    stem_to_img = {}
    for ext in IMG_EXTS:
        for p in images_root.rglob(f"*{ext}"):
            stem_to_img.setdefault(p.stem, p)
    return stem_to_img

def derive_patient_id(stem: str, regex: Optional[re.Pattern]) -> str:
    if regex:
        m = regex.search(stem)
        if m: return m.group(1)
    if "-" in stem:  # last-dash rule
        return stem[:stem.rfind("-")]
    return stem

def place(src: Path, dst: Path, copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        if dst.exists(): dst.unlink()
        dst.symlink_to(src.resolve())

# ---------------- Main ----------------
def main():
    args = parse_args()
    random.seed(args.seed)
    img_root = Path(args.images_root)
    lbl_root = Path(args.labels_dir)
    out_root = Path(args.out_root)

    if not img_root.exists() or not lbl_root.exists():
        raise SystemExit("Missing images_root or labels_dir.")

    stem_to_img = enumerate_images_recursive(img_root)
    label_files = sorted(lbl_root.glob("*.txt"))
    if not label_files:
        raise SystemExit("No labels found.")

    regex = re.compile(args.patient_regex, re.I) if args.patient_regex else None

    patients: Dict[str, List[Tuple[Path, Path]]] = {}
    for lp in label_files:
        st = lp.stem
        ip = stem_to_img.get(st)
        if not ip: continue
        pid = derive_patient_id(st, regex)
        patients.setdefault(pid, []).append((ip, lp))

    pids = list(patients.keys())
    random.shuffle(pids)
    n_total = len(pids)
    n_val, n_test = int(args.val_frac*n_total), int(args.test_frac*n_total)
    val_ids = set(pids[:n_val])
    test_ids = set(pids[n_val:n_val+n_test])
    train_ids = set(pids[n_val+n_test:])

    manifest = out_root/"split_manifest.csv"
    cols = ["patient_id","stem","image_path","label_path","split"]
    out_root.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader()
        for pid, pairs in patients.items():
            split = "train" if pid in train_ids else "val" if pid in val_ids else "test"
            for ip, lp in pairs:
                wr.writerow({"patient_id": pid,"stem": ip.stem,
                             "image_path": str(ip.resolve()),"label_path": str(lp.resolve()),
                             "split": split})
                place(ip, out_root/"images"/split/ip.name, args.copy)
                place(lp, out_root/"labels"/split/f"{ip.stem}.txt", args.copy)

    if args.write_yaml:
        (out_root/"data.yaml").write_text(
            f"path: {out_root.resolve()}\n"
            "train: images/train\nval: images/val\ntest: images/test\n"
            "names: ['disc','cup']\n"
        )

    print(f"[OK] Patients: {len(pids)} | train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"[OK] Manifest: {manifest}")
    if args.write_yaml: print(f"[OK] data.yaml: {out_root/'data.yaml'}")

if __name__ == "__main__":
    main()