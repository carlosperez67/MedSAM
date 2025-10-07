#!/usr/bin/env python3
# build_disc_only_dataset.py
from __future__ import annotations
import argparse, os, shutil, yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

def _xp(p:str|Path)->Path: return Path(os.path.expanduser(str(p))).resolve()
def _ens(d:Path)->None: d.mkdir(parents=True, exist_ok=True)

def _place(src:Path, dst:Path, copy_files:bool)->None:
    _ens(dst.parent)
    if copy_files: shutil.copy2(src, dst)
    else:
        if dst.exists() or dst.is_symlink(): dst.unlink()
        dst.symlink_to(src.resolve())

def _split_has_data(root: Path, split: str) -> bool:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    return img_dir.exists() and lbl_dir.exists() and any(img_dir.iterdir()) and any(lbl_dir.iterdir())

def _filter_label_lines_to_disc(lines: Iterable[str]) -> List[str]:
    keep = []
    for ln in lines:
        ln = ln.strip()
        if not ln: continue
        parts = ln.split()
        try: cls = int(float(parts[0]))
        except Exception: continue
        if cls == 0: keep.append(ln)
    return keep

def _filter_labels_dir_to_disc(src_lbl_dir: Path, dst_lbl_dir: Path, drop_empty: bool) -> int:
    _ens(dst_lbl_dir)
    written = 0
    for lbl in sorted(src_lbl_dir.glob("*.txt")):
        lines_in = [ln for ln in lbl.read_text().splitlines()]
        kept = _filter_label_lines_to_disc(lines_in)
        if not kept and drop_empty: continue
        (dst_lbl_dir / lbl.name).write_text("\n".join(kept) + ("\n" if kept else ""))
        written += 1
    return written

@dataclass
class DiscOnlyArgs:
    base_root: Path                   # clean YOLO split
    out_root: Path                    # target: <...>/yolo_split_disc_only
    aug_root: Optional[Path]          # optional: use for selected train splits
    train_splits: tuple[str,...]      # e.g., ("train",)
    copy_images: bool
    drop_empty: bool

def build_disc_only_dataset(a: DiscOnlyArgs) -> Path:
    _ens(a.out_root)
    for split in ("train", "val", "test"):
        src_root = a.aug_root if (a.aug_root and split in set(a.train_splits)) else a.base_root
        src_img = src_root / "images" / split
        src_lbl = src_root / "labels" / split
        if not src_img.exists() or not src_lbl.exists(): continue
        for img in sorted(src_img.glob("*")):
            _place(img, a.out_root / "images" / split / img.name, a.copy_images)
        _filter_labels_dir_to_disc(src_lbl, a.out_root / "labels" / split, a.drop_empty)

    data_yaml = {
        "path": str(a.out_root.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "names": ["disc"],
    }
    if _split_has_data(a.out_root, "test"): data_yaml["test"] = "images/test"
    (a.out_root / "od_only.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return a.out_root

def main():
    ap = argparse.ArgumentParser(description="Build disc-only YOLO dataset from 2-class labels (0=disc,1=cup).")
    ap.add_argument("--base_root", required=True, help="Clean YOLO split")
    ap.add_argument("--out_root",  required=True, help="Output disc-only root")
    ap.add_argument("--aug_root",  default=None,  help="Augmented YOLO split (used for --train_splits)")
    ap.add_argument("--train_splits", default="train", help="Comma list; which splits source from aug_root")
    ap.add_argument("--copy_images", action="store_true")
    ap.add_argument("--drop_empty", action="store_true")
    args = ap.parse_args()

    train_splits = tuple(s.strip() for s in str(args.train_splits).split(",") if s.strip())
    build_disc_only_dataset(DiscOnlyArgs(
        base_root=_xp(args.base_root),
        out_root=_xp(args.out_root),
        aug_root=_xp(args.aug_root) if args.aug_root else None,
        train_splits=train_splits,
        copy_images=bool(args.copy_images),
        drop_empty=bool(args.drop_empty),
    ))
    print("[OK] Disc-only dataset built.")

if __name__ == "__main__":
    main()