#!/usr/bin/env python3
# infer_two_stage_to_boxes.py
import argparse, json
from pathlib import Path
import cv2
from PIL import Image
from ultralytics import YOLO
from device_utils import ultralytics_device_arg

def clamp(v, lo, hi): return max(lo, min(hi, v))
def pad_box(x1,y1,x2,y2, pad, W,H):
    px = int(round(pad*max(W,H)))
    return clamp(x1-px,0,W-1), clamp(y1-px,0,H-1), clamp(x2+px,1,W), clamp(y2+px,1,H)

def best_box(yolo_results, cls: int, W: int, H: int):
    """Return highest-score box for class 'cls' in xyxy ints, or None."""
    if not yolo_results or not yolo_results[0].boxes: return None
    boxes = yolo_results[0].boxes
    if boxes.cls is None: return None
    # filter by class
    import torch
    cls_tensor = boxes.cls.int()
    conf = boxes.conf if boxes.conf is not None else torch.ones_like(cls_tensor, dtype=torch.float)
    xyxy = boxes.xyxy
    keep = (cls_tensor == cls)
    if keep.sum() == 0: return None
    idx = conf[keep].argmax()
    # map idx back to original index
    indices = torch.nonzero(keep, as_tuple=False).squeeze(1)
    j = indices[idx]
    x1,y1,x2,y2 = xyxy[j].tolist()
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stageA", required=True, help="Path to disc model (e.g., runs/detect/stageA_disc_only/weights/best.pt)")
    ap.add_argument("--stageB", required=True, help="Path to cup model  (e.g., runs/detect/stageB_cup_roi/weights/best.pt)")
    ap.add_argument("--images", required=True, help="Folder of full images to run on")
    ap.add_argument("--out_jsonl", default="./two_stage_boxes.jsonl")
    ap.add_argument("--od_pad_pct", type=float, default=0.08, help="Pad OD box before making ROI")
    ap.add_argument("--confA", type=float, default=0.25, help="Stage A disc conf")
    ap.add_argument("--confB", type=float, default=0.10, help="Stage B cup conf (ROI)")
    ap.add_argument("--iouA",  type=float, default=0.50)
    ap.add_argument("--iouB",  type=float, default=0.50)
    args = ap.parse_args()

    device = ultralytics_device_arg()
    A = YOLO(args.stageA)
    B = YOLO(args.stageB)

    img_dir = Path(args.images)
    out = []
    for ip in sorted(img_dir.iterdir()):
        if ip.suffix.lower() not in (".png",".jpg",".jpeg",".tif",".tiff"): continue
        W,H = Image.open(ip).size

        # Stage A: disc
        ra = A.predict(source=str(ip), conf=args.confA, iou=args.iouA, imgsz=640, device=device, verbose=False)
        od = best_box(ra, cls=0, W=W, H=H)  # class 0=disc
        if od is None:
            out.append({"image": str(ip), "box_od": None, "box_oc": None})
            continue

        # Pad and crop ROI
        x1,y1,x2,y2 = od
        x1,y1,x2,y2 = pad_box(x1,y1,x2,y2, args.od_pad_pct, W,H)
        full = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        crop = full[y1:y2, x1:x2].copy()
        cH,cW = crop.shape[:2]

        # Stage B: cup on ROI crop
        tmp = str(ip.with_suffix("")) + "_TMP_ROI.png"
        cv2.imwrite(tmp, crop)
        rb = B.predict(source=tmp, conf=args.confB, iou=args.iouB, imgsz=640, device=device, verbose=False)
        Path(tmp).unlink(missing_ok=True)

        oc_local = best_box(rb, cls=0, W=cW, H=cH)  # stage B has 0=cup
        if oc_local is None:
            out.append({"image": str(ip), "box_od": [x1,y1,x2,y2], "box_oc": None})
            continue

        lx1,ly1,lx2,ly2 = oc_local
        oc = [x1+lx1, y1+ly1, x2-(cW-lx2), y2-(cH-ly2)]
        out.append({"image": str(ip), "box_od": [x1,y1,x2,y2], "box_oc": oc})

    with open(args.out_jsonl,"w") as f:
        for r in out:
            f.write(json.dumps(r)+"\n")
    print(f"[OK] Wrote: {args.out_jsonl}")

if __name__ == "__main__":
    main()