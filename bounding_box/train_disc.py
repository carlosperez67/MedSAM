#!/usr/bin/env python3
# train_stageA_disc_only.py
import argparse, os, yaml
from pathlib import Path
from ultralytics import YOLO
from device_utils import ultralytics_device_arg

"""
Train a disc-only detector from an existing 2-class YOLO dataset.
It auto-generates a temporary 'disc-only' dataset YAML that points to the
same image splits but uses filtered label copies containing ONLY class 0 (disc).

DATA_ROOT/
  images/{train,val,test}
  labels/{train,val,test}    # original 2-class labels (0=disc,1=cup)

Outputs:
  DATA_ROOT_disc_only/
    images/{...}  (symlinks)
    labels/{...}  (filtered to disc only)
"""

def filter_labels_to_disc(src_lbl_dir: Path, dst_lbl_dir: Path):
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    for txt in sorted(src_lbl_dir.glob("*.txt")):
        keep = []
        for line in txt.read_text().strip().splitlines():
            if not line.strip():
                continue
            parts = line.strip().split()
            cls = int(float(parts[0]))
            if cls == 0:  # disc only
                keep.append(line)
        (dst_lbl_dir / txt.name).write_text("\n".join(keep) + ("\n" if keep else ""))

def make_disc_only_dataset(root: Path) -> Path:
    out = root.parent / f"{root.name}_disc_only"
    # symlink images, filter labels
    for split in ("train","val","test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        # images: symlink
        for img in (root/"images"/split).glob("*"):
            dst = out/"images"/split/img.name
            if dst.exists(): dst.unlink()
            dst.symlink_to(img.resolve())
        # labels: copy filtered
        src_lbl = root/"labels"/split
        if src_lbl.exists():
            filter_labels_to_disc(src_lbl, out/"labels"/split)
    # write YAML
    names = ["disc"]
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "names": names
    }
    with open(out/"od_only.yaml","w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./papila_yolo")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="stageA_disc_only")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root  = make_disc_only_dataset(data_root)
    yaml_path = out_root/"od_only.yaml"

    device = ultralytics_device_arg()
    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        cos_lr=True,
        optimizer="AdamW",
        pretrained=True,
        patience=50
    )
    # Optional test
    test_dir = out_root/"images"/"test"
    if test_dir.exists():
        model.val(data=str(yaml_path), split="test", imgsz=args.imgsz, device=device)

if __name__ == "__main__":
    main()