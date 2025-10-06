#!/usr/bin/env python3
# infer_two_stage_to_boxes.py
import argparse, json, os
from pathlib import Path
import cv2
from PIL import Image
from ultralytics import YOLO
from bounding_box.train.device_utils import ultralytics_device_arg

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def pad_box(x1, y1, x2, y2, pad, W, H):
    px = int(round(pad * max(W, H)))
    return (
        clamp(x1 - px, 0, W - 1),
        clamp(y1 - px, 0, H - 1),
        clamp(x2 + px, 1, W),
        clamp(y2 + px, 1, H),
    )

def best_box(yolo_results, cls: int):
    """Return highest-score box for class 'cls' in xyxy ints, or None."""
    if not yolo_results or not yolo_results[0].boxes:
        return None
    boxes = yolo_results[0].boxes
    if boxes.cls is None:
        return None
    import torch
    keep = (boxes.cls.int() == cls)
    if keep.sum() == 0:
        return None
    conf = boxes.conf if boxes.conf is not None else torch.ones_like(boxes.cls, dtype=torch.float)
    idx_rel = conf[keep].argmax()
    idx_abs = torch.nonzero(keep, as_tuple=False).squeeze(1)[idx_rel]
    x1, y1, x2, y2 = boxes.xyxy[idx_abs].tolist()
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def auto_thickness(h: int, w: int) -> int:
    return max(2, int(0.002 * max(h, w)))

def draw_box(img, box, color, label=None, thickness=2, alpha=0.20):
    if box is None:
        return img
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return img
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.5
        tw, th = cv2.getTextSize(label, font, fs, 1)[0]
        bx2 = min(img.shape[1] - 1, x1 + tw + 8)
        by2 = min(img.shape[0] - 1, y1 + th + 8)
        cv2.rectangle(img, (x1, y1), (bx2, by2), color, thickness=-1)
        cv2.putText(img, label, (x1 + 4, y1 + th + 2), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return img

def main():
    ap = argparse.ArgumentParser()
    # --- defaults updated ---
    ap.add_argument("--stageA", default="/Users/carlosperez/PycharmProjects/MedSAM/bounding_box/runs/detect/stageA_disc_only/weights/best.pt",
        help="Path to disc model checkpoint")
    ap.add_argument("--stageB", default="/Users/carlosperez/PycharmProjects/MedSAM/bounding_box/runs/detect/stageB_cup_roi/weights/best.pt",
        help="Path to cup model checkpoint")
    ap.add_argument("--images", default="/Users/carlosperez/PycharmProjects/MedSAM/bounding_box/data/test/og",
        help="Folder of full images to run on (default: /Users/carlosperez/PycharmProjects/MedSAM/bounding_box/data/test)")
    ap.add_argument("--out_jsonl", default="/Users/carlosperez/PycharmProjects/MedSAM/bounding_box/data/test/two_stage_boxes.jsonl")
    ap.add_argument("--od_pad_pct", type=float, default=0.08)
    ap.add_argument("--confA", type=float, default=0.25)
    ap.add_argument("--confB", type=float, default=0.10)
    ap.add_argument("--iouA",  type=float, default=0.50)
    ap.add_argument("--iouB",  type=float, default=0.50)
    ap.add_argument("--viz_dir", default="/Users/carlosperez/PycharmProjects/MedSAM/bounding_box/data/test/results",
        help="If set, save annotated predictions to this folder")
    ap.add_argument("--save_roi", action="store_true",
        help="Also save the ROI crop with cup box overlaid")
    args = ap.parse_args()

    device = ultralytics_device_arg()
    A = YOLO(args.stageA)
    B = YOLO(args.stageB)

    img_dir = Path(args.images)
    if not img_dir.exists():
        raise SystemExit(f"[ERR] Image folder not found: {img_dir}")

    img_dir = Path(args.images)
    out = []
    if args.viz_dir:
        Path(args.viz_dir).mkdir(parents=True, exist_ok=True)
        if args.save_roi:
            (Path(args.viz_dir) / "roi").mkdir(parents=True, exist_ok=True)

    for ip in sorted(img_dir.iterdir()):
        if ip.suffix.lower() not in IMG_EXTS:
            continue

        # read image + dims
        full_bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if full_bgr is None:
            continue
        H, W = full_bgr.shape[:2]

        # Stage A: disc on full image
        ra = A.predict(source=str(ip), conf=args.confA, iou=args.iouA, imgsz=640, device=device, verbose=False)
        od = best_box(ra, cls=0)  # 0=disc
        if od is None:
            out.append({"image": str(ip), "box_od": None, "box_oc": None})
            # optional: still dump a red "no detection" overlay
            if args.viz_dir:
                vis = full_bgr.copy()
                cv2.putText(vis, "OD not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imwrite(str(Path(args.viz_dir) / f"{ip.stem}_pred.png"), vis)
            continue

        x1, y1, x2, y2 = od
        # pad ROI around disc
        x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, args.od_pad_pct, W, H)
        roi = (x1, y1, x2, y2)
        crop = full_bgr[y1:y2, x1:x2].copy()
        cH, cW = crop.shape[:2]

        # Stage B: cup in ROI crop
        tmp = str(ip.with_suffix("")) + "_TMP_ROI.png"
        cv2.imwrite(tmp, crop)
        rb = B.predict(source=tmp, conf=args.confB, iou=args.iouB, imgsz=640, device=device, verbose=False)
        Path(tmp).unlink(missing_ok=True)

        oc_local = best_box(rb, cls=0)  # 0=cup (ROI coordinates)
        if oc_local is None:
            out.append({"image": str(ip), "box_od": [x1, y1, x2, y2], "box_oc": None})

            if args.viz_dir:
                vis = full_bgr.copy()
                th = auto_thickness(H, W)
                vis = draw_box(vis, roi, (255, 128, 0), "OD-ROI", thickness=th)  # orange-ish for ROI
                vis = draw_box(vis, od, (255, 96, 32), "OD", thickness=th)  # blue/orange palette
                cv2.imwrite(str(Path(args.viz_dir) / f"{ip.stem}_pred.png"), vis)

                if args.save_roi:
                    roi_vis = draw_box(crop.copy(), None, (0, 0, 0))  # no-op to copy interface
                    cv2.putText(roi_vis, "Cup not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    cv2.imwrite(str(Path(args.viz_dir) / "roi" / f"{ip.stem}_roi.png"), roi_vis)
            continue

        # translate cup box back to full-image coords
        lx1, ly1, lx2, ly2 = oc_local
        oc = [x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2]  # translate ROI → full
        out.append({"image": str(ip), "box_od": [x1, y1, x2, y2], "box_oc": oc})

        # Visualization
        if args.viz_dir:
            th = auto_thickness(H, W)
            vis = full_bgr.copy()
            vis = draw_box(vis, roi, (255, 128, 0), "OD-ROI", thickness=th)
            vis = draw_box(vis, od, (255, 96, 32), "OD", thickness=th)
            vis = draw_box(vis, oc, (64, 192, 64), "OC", thickness=th)
            cv2.imwrite(str(Path(args.viz_dir) / f"{ip.stem}_pred.png"), vis)

            if args.save_roi:
                roi_vis = crop.copy()
                # draw OC in ROI-local coords
                roi_vis = draw_box(roi_vis, (lx1, ly1, lx2, ly2), (64, 192, 64), "OC", thickness=auto_thickness(cH, cW))
                cv2.imwrite(str(Path(args.viz_dir) / "roi" / f"{ip.stem}_roi.png"), roi_vis)

    # Write JSONL
    with open(args.out_jsonl, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    print(f"[OK] Wrote: {args.out_jsonl}")
    if args.viz_dir:
        print(f"[OK] Visualizations saved under: {args.viz_dir}{' (+ /roi)' if args.save_roi else ''}")

if __name__ == "__main__":
    main()