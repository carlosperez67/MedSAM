#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io as skio, transform as sktf

# ---- MedSAM ----
from segment_anything import sam_model_registry
from src.imgpipe.binary_mask_ref import BinaryMaskRef

# ---- Your OOP pieces ----
from src.imgpipe.enums import LabelType, Structure
from src.imgpipe.image import Image
from src.imgpipe.bounding_box import BoundingBox
from src.imgpipe.collector import DatasetCollector, group_by_subject
from src.imgpipe.utils import ensure_dir

# NOTE: adjust this import path if your repo structure differs
from src.model.predict_bounding_box import BoundingBoxPredictor, LabelWriter


# ======================================================================
# Small helpers (device, I/O, geometry)
# ======================================================================




# ======================================================================
# MedSAM core
# ======================================================================

@dataclass
class MedSAMModel:
    model: torch.nn.Module
    device: torch.device

def _pick_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        if device_arg.lower() == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if device_arg != "cpu" and torch.cuda.is_available():
            return torch.device(device_arg if device_arg != "cuda" else "cuda:0")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_medsam(checkpoint: Path, device: torch.device, variant: str = "vit_b") -> MedSAMModel:
    model = sam_model_registry[variant](checkpoint=str(checkpoint))
    model = model.to(device)
    model.eval()
    return MedSAMModel(model=model, device=device)

@torch.no_grad()
def _embed_image_1024(msam: MedSAMModel, img_3c: np.ndarray) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
    H, W, _ = img_3c.shape
    img_1024 = sktf.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)
    img_1024_t = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(msam.device)
    emb = msam.model.image_encoder(img_1024_t)  # (1,256,64,64)
    return emb, H, W, img_1024_t

@torch.no_grad()
def medsam_infer(msam: MedSAMModel,
                 img_embed: torch.Tensor,
                 box_xyxy: Tuple[int, int, int, int],
                 H: int, W: int) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    box_np = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    box_1024 = box_np / np.array([W, H, W, H], dtype=np.float32) * 1024.0

    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B,1,4)

    sparse_embeddings, dense_embeddings = msam.model.prompt_encoder(
        points=None, boxes=box_torch, masks=None
    )
    low_res_logits, _ = msam.model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=msam.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)  # (1,1,256,256)
    pred = F.interpolate(low_res_pred, size=(H, W), mode="bilinear", align_corners=False)
    mask = (pred.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8)
    return mask


# ======================================================================
# High-level pipeline
# ======================================================================

@dataclass
class Args:
    images_root: Path
    weights: Path            # YOLO weights
    medsam_ckpt: Path        # MedSAM checkpoint
    out_dir: Path            # SINGLE output root

    # derived subpaths (filled at runtime)
    out_labels: Path
    out_disc: Path
    out_cup: Path
    out_viz: Path
    csv_path: Path

    include: Tuple[str, ...]
    exclude: Tuple[str, ...]
    recursive: bool
    subset_n: int
    subset_seed: int

    conf: float
    iou: float
    device: Optional[str]
    overwrite: bool
    save_viz: bool


# --- collection (DatasetCollector needs a config-like object) ----

@dataclass
class _CollectCfg:
    project_dir: Path
    images_root: Path
    disc_masks: Optional[Path] = None
    cup_masks: Optional[Path] = None
    include_name_contains: Optional[List[str]] = None
    exclude_name_contains: Optional[List[str]] = None
    recursive: bool = True

def collect_images(args: Args) -> List[Image]:
    cfg = _CollectCfg(
        project_dir=args.images_root.parent if args.images_root.parent else Path("."),
        images_root=args.images_root,
        disc_masks=None,
        cup_masks=None,
        include_name_contains=list(args.include) if args.include else None,
        exclude_name_contains=list(args.exclude) if args.exclude else None,
        recursive=args.recursive,
    )
    coll = DatasetCollector(cfg)  # type: ignore[arg-type]
    ds = coll.collect()

    # Optional patient-wise subset (deterministic)
    if args.subset_n and args.subset_n > 0:
        by = group_by_subject(ds.images)
        keys = list(by.keys())
        rng = np.random.RandomState(args.subset_seed)
        rng.shuffle(keys)
        picked: List[Image] = []
        for k in keys:
            if len(picked) >= args.subset_n:
                break
            picked.extend(by[k])
        ds.images = picked[:args.subset_n]

    return ds.images


# --- cup box from disc mask ---

def build_cup_box_from_disc_mask(disc_mask: np.ndarray,
                                 fallback_box: Tuple[int,int,int,int]) -> Optional[Tuple[int,int,int,int]]:
    tight = _tight_bbox_from_mask(disc_mask)
    if tight is None:
        return None
    box = _shrink_box_to_fit_mask(disc_mask, tight, step_frac=0.02, max_iter=300)
    if box is None:
        box = _shrink_box_to_fit_mask(disc_mask, fallback_box, step_frac=0.02, max_iter=300)
    return box


# --- per-image workflow ---

def _process_one_image(
    img: Image,
    bb_pred: BoundingBoxPredictor,
    msam: MedSAMModel,
    lbl_writer: LabelWriter,
    out_disc_dir: Path,
    out_cup_dir: Path,
    out_viz_dir: Path,
    save_viz: bool = True,
) -> Tuple[Optional[float], Optional[Path], Optional[Path], Optional[Path]]:
    """
    Returns (cd_ratio, disc_mask_path, cup_mask_path, viz_path).
    """

    # A) Predict disc bounding box with YOLO (single-stage)
    disc_box = bb_pred.predict_one_image_to_box(img)  # -> BoundingBox | None
    if disc_box is None:
        return None, None, None, None

    img.set_box(Structure.DISC, LabelType.PRED, disc_box)
    lbl_writer.write(img, require_both=False)

    # Load full image RGB for MedSAM embedding + viz
    img_bgr = _load_image_bgr(img.image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    emb, H, W, _ = _embed_image_1024(msam, img_rgb)

    # B) MedSAM with disc bounding box -> pred_disc_mask
    dxyxy = (int(round(disc_box.x1)), int(round(disc_box.y1)),
             int(round(disc_box.x2)), int(round(disc_box.y2)))
    pred_disc_mask = medsam_infer(msam, emb, dxyxy, H, W)
    img.set_mask(Structure.DISC, LabelType.PRED, BinaryMaskRef(array=pred_disc_mask))

    # Save disc mask
    seg_disc_path = out_disc_dir / f"{img.image_path.stem}.png"
    _save_mask_png(seg_disc_path, pred_disc_mask)

    # C) Build cup box fully inside disc mask
    fallback_box = dxyxy
    cup_xyxy = build_cup_box_from_disc_mask(pred_disc_mask, fallback_box)
    if cup_xyxy is None:
        cdr_text = "CDR: N/A"
        viz_path = (out_viz_dir / f"{img.image_path.stem}_viz.jpg") if save_viz else None
        if viz_path:
            viz = _overlay_masks_and_boxes(img_bgr, pred_disc_mask, None, dxyxy, None, cdr_text=cdr_text)
            _save_viz(viz_path, viz)
        return None, seg_disc_path, None, viz_path

    cup_box = BoundingBox(*map(float, cup_xyxy))
    img.set_box(Structure.CUP, LabelType.PRED, cup_box)

    # D) MedSAM with cup bounding box -> pred_cup_mask
    pred_cup_mask = medsam_infer(msam, emb, cup_xyxy, H, W)
    img.set_mask(Structure.CUP, LabelType.PRED, BinaryMaskRef(array=pred_cup_mask))

    seg_cup_path = out_cup_dir / f"{img.image_path.stem}.png"
    _save_mask_png(seg_cup_path, pred_cup_mask)

    # E) Cup-to-Disc vertical ratio
    disc_h = _mask_vertical_height(pred_disc_mask)
    cup_h  = _mask_vertical_height(pred_cup_mask)
    cd_ratio: Optional[float] = None
    if disc_h > 0:
        cd_ratio = float(cup_h) / float(disc_h)

    # F) Viz (with CDR text)
    cdr_text = f"CDR: {cd_ratio:.3f}" if cd_ratio is not None else "CDR: N/A"
    viz = _overlay_masks_and_boxes(img_bgr, pred_disc_mask, pred_cup_mask, dxyxy, cup_xyxy, cdr_text=cdr_text)
    viz_path = (out_viz_dir / f"{img.image_path.stem}_viz.jpg") if save_viz else None
    if viz_path:
        _save_viz(viz_path, viz)

    return cd_ratio, seg_disc_path, seg_cup_path, viz_path


# --- CSV writer ---

def _write_summary_csv(rows: List[dict], csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    fields = ["image_path", "label_path", "disc_mask_path", "cup_mask_path", "viz_path", "cd_ratio"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ======================================================================
# CLI + main
# ======================================================================

def _derive_outputs(out_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    """
    Returns: (labels_dir, disc_dir, cup_dir, viz_dir, csv_path)
    """
    labels = out_dir / "labels"
    disc   = out_dir / "disc"
    cup    = out_dir / "cup"
    viz    = out_dir / "viz"
    csvp   = out_dir / "summary.csv"
    for p in (labels, disc, cup, viz, csvp.parent):
        ensure_dir(p)
    return labels, disc, cup, viz, csvp

@dataclass
class _CLI:
    images_root: Path
    weights: Path
    medsam_ckpt: Path
    out_dir: Path
    include: Tuple[str, ...]
    exclude: Tuple[str, ...]
    recursive: bool
    subset_n: int
    subset_seed: int
    conf: float
    iou: float
    device: Optional[str]
    overwrite: bool
    save_viz: bool

def _parse_args() -> _CLI:
    ap = argparse.ArgumentParser(
        description="YOLO bbox → MedSAM disc/cup → CDR, with a single output root."
    )
    ap.add_argument("--images-root", required=True, help="Root of input images.")
    ap.add_argument("--weights", required=True, help="YOLO weights (single-stage detector).")
    ap.add_argument("--medsam-ckpt", required=True, help="MedSAM checkpoint (e.g., medsam_vit_b.pth).")

    # SINGLE output root
    ap.add_argument("--out-dir", required=True, help="Output root (creates labels/, disc/, cup/, viz/, summary.csv).")

    # Optional toggles (kept minimal)
    ap.add_argument("--include", nargs="*", default=[], help="Include substrings (case-insensitive).")
    ap.add_argument("--exclude", nargs="*", default=[], help="Exclude substrings (case-insensitive).")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    ap.add_argument("--subset-n", type=int, default=0, help="Optional patient-wise subset size.")
    ap.add_argument("--subset-seed", type=int, default=43, help="Subset RNG seed.")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    ap.add_argument("--iou",  type=float, default=0.50, help="YOLO IoU threshold.")
    ap.add_argument("--device", default=None, help="Device for YOLO & MedSAM (e.g., '0', 'cpu', 'mps').")
    ap.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing label files.")
    ap.add_argument("--no-viz", action="store_true", help="Do not save visualization images.")

    a = ap.parse_args()
    return _CLI(
        images_root=_expand(a.images_root),
        weights=_expand(a.weights),
        medsam_ckpt=_expand(a.medsam_ckpt),
        out_dir=_expand(a.out_dir),
        include=tuple(a.include or []),
        exclude=tuple(a.exclude or []),
        recursive=bool(a.recursive),
        subset_n=int(a.subset_n),
        subset_seed=int(a.subset_seed),
        conf=float(a.conf),
        iou=float(a.iou),
        device=a.device,
        overwrite=not bool(a.no_overwrite),
        save_viz=not bool(a.no_viz),
    )

def main() -> None:
    cli = _parse_args()

    # Derive subpaths from single out root
    out_labels, out_disc, out_cup, out_viz, csv_path = _derive_outputs(cli.out_dir)

    # Assemble Args with derived outputs
    args = Args(
        images_root=cli.images_root,
        weights=cli.weights,
        medsam_ckpt=cli.medsam_ckpt,
        out_dir=cli.out_dir,
        out_labels=out_labels,
        out_disc=out_disc,
        out_cup=out_cup,
        out_viz=out_viz,
        csv_path=csv_path,
        include=cli.include,
        exclude=cli.exclude,
        recursive=cli.recursive,
        subset_n=cli.subset_n,
        subset_seed=cli.subset_seed,
        conf=cli.conf,
        iou=cli.iou,
        device=cli.device,
        overwrite=cli.overwrite,
        save_viz=cli.save_viz,
    )

    # Collector
    images = collect_images(args)
    if not images:
        raise SystemExit("[ERR] No images collected.")

    # YOLO predictor + label writer
    bb_pred = BoundingBoxPredictor(
        weights=args.weights, conf=args.conf, iou=args.iou, device=args.device
    )
    lbl_writer = LabelWriter(args.out_labels, args.images_root, overwrite=args.overwrite)

    # MedSAM
    dev = _pick_device(args.device)
    msam = load_medsam(args.medsam_ckpt, dev, variant="vit_b")

    # Process
    rows: List[dict] = []
    for img in images:
        cd_ratio, seg_disc_path, seg_cup_path, viz_path = _process_one_image(
            img, bb_pred, msam, lbl_writer, args.out_disc, args.out_cup, args.out_viz, save_viz=args.save_viz
        )
        rel = img.image_path.resolve().relative_to(args.images_root.resolve())
        label_path = (args.out_labels / rel).with_suffix(".txt")

        rows.append({
            "image_path": str(img.image_path),
            "label_path": str(label_path) if label_path.exists() else "",
            "disc_mask_path": str(seg_disc_path) if seg_disc_path else "",
            "cup_mask_path": str(seg_cup_path) if seg_cup_path else "",
            "viz_path": str(viz_path) if viz_path else "",
            "cd_ratio": f"{cd_ratio:.6f}" if cd_ratio is not None else "",
        })

    _write_summary_csv(rows, args.csv_path)
    print(f"[OK] Processed {len(rows)} images.")
    print(f"[OK] Outputs → {args.out_dir}")
    print(f"  labels/  → {args.out_labels}")
    print(f"  disc/    → {args.out_disc}")
    print(f"  cup/     → {args.out_cup}")
    print(f"  viz/     → {args.out_viz}")
    print(f"  summary  → {args.csv_path}")


if __name__ == "__main__":
    main()