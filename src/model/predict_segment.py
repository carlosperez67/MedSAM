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
from src.imgpipe.utils import ensure_dir, stem_map_by_first_match

# NOTE: adjust this import path if needed
from src.model.predict_bounding_box import BoundingBoxPredictor, LabelWriter


# ======================================================================
# Small helpers (device, I/O, geometry, metrics)
# ======================================================================

def _expand(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()

def _load_image_bgr(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return im

def _save_mask_png(path: Path, mask_bool: np.ndarray) -> None:
    ensure_dir(path.parent)
    skio.imsave(str(path), (mask_bool.astype(np.uint8) * 255), check_contrast=False)

def _save_viz(path: Path, viz_bgr: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), viz_bgr)

def _mask_vertical_height(mask: np.ndarray) -> int:
    ys = np.where(mask > 0)[0]
    if ys.size == 0:
        return 0
    return int(ys.max() - ys.min() + 1)

def _cdr_from_masks(disc_mask: Optional[np.ndarray], cup_mask: Optional[np.ndarray]) -> Optional[float]:
    if disc_mask is None or cup_mask is None:
        return None
    dh = _mask_vertical_height(disc_mask)
    if dh <= 0:
        return None
    ch = _mask_vertical_height(cup_mask)
    return float(ch) / float(dh)

def _corners_inside(mask: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box_xyxy
    H, W = mask.shape[:2]
    x1 = np.clip(x1, 0, W - 1); x2 = np.clip(x2, 0, W - 1)
    y1 = np.clip(y1, 0, H - 1); y2 = np.clip(y2, 0, H - 1)
    pts = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    for (x, y) in pts:
        if mask[int(y), int(x)] == 0:
            return False
    return True

def _tight_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    return (x1, y1, x2 + 1, y2 + 1)  # half-open

def _shrink_box_to_fit_mask(mask: np.ndarray,
                            base_box: Tuple[int, int, int, int],
                            step_frac: float = 0.02,
                            max_iter: int = 200,
                            min_side_px: int = 8) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = map(float, base_box)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    H, W = mask.shape[:2]
    for _ in range(max_iter):
        bx1, by1, bx2, by2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        bx1 = max(0, min(bx1, W - 1)); bx2 = max(1, min(bx2, W))
        by1 = max(0, min(by1, H - 1)); by2 = max(1, min(by2, H))
        if (bx2 - bx1) < min_side_px or (by2 - by1) < min_side_px:
            return None
        if _corners_inside(mask, (bx1, by1, bx2, by2)):
            return (bx1, by1, bx2, by2)
        w = (x2 - x1) * (1.0 - step_frac)
        h = (y2 - y1) * (1.0 - step_frac)
        x1 = cx - w / 2.0; x2 = cx + w / 2.0
        y1 = cy - h / 2.0; y2 = cy + h / 2.0
    return None

def _overlay_masks_and_boxes(
    img_bgr: np.ndarray,
    disc_mask: Optional[np.ndarray],
    cup_mask: Optional[np.ndarray],
    disc_box: Optional[Tuple[int,int,int,int]],
    cup_box: Optional[Tuple[int,int,int,int]],
    cdr_text: Optional[str] = None,
    alpha: float = 0.4
) -> np.ndarray:
    out = img_bgr.copy()
    if disc_mask is not None:
        overlay = out.copy()
        overlay[disc_mask > 0] = (0, 255, 255)  # yellow
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    if cup_mask is not None:
        overlay = out.copy()
        overlay[cup_mask > 0] = (255, 0, 255)  # magenta
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    if disc_box is not None:
        x1, y1, x2, y2 = disc_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
    if cup_box is not None:
        x1, y1, x2, y2 = cup_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 200), 2)
    if cdr_text:
        cv2.putText(out, cdr_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(out, cdr_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def _make_side_by_side(
    img_bgr: np.ndarray,
    pred_disc: Optional[np.ndarray], pred_cup: Optional[np.ndarray],
    gt_disc: Optional[np.ndarray],   gt_cup: Optional[np.ndarray],
    pred_text: str, gt_text: str
) -> np.ndarray:
    left  = _overlay_masks_and_boxes(img_bgr, pred_disc, pred_cup, None, None, cdr_text=pred_text)
    right = _overlay_masks_and_boxes(img_bgr, gt_disc,   gt_cup,   None, None, cdr_text=gt_text)
    return np.hstack([left, right])

def _dice(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    if pred is None or gt is None:
        return None
    predb = (pred > 0).astype(np.uint8)
    gtb = (gt > 0).astype(np.uint8)
    inter = (predb & gtb).sum()
    s = predb.sum() + gtb.sum()
    if s == 0:
        return None
    return 2.0 * inter / float(s)


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
    weights: Path
    medsam_ckpt: Path
    out_dir: Path

    # derived subpaths
    out_labels: Path
    out_disc: Path
    out_cup: Path
    out_viz: Path
    out_viz_compare: Path
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

    # evaluation (optional)
    eval_enabled: bool
    gt_disc_root: Optional[Path]
    gt_cup_root: Optional[Path]
    viz_compare: bool  # <--- NEW FLAG


@dataclass
class _CollectCfg:
    project_dir: Path
    images_root: Path
    disc_masks: Optional[Path] = None
    cup_masks: Optional[Path] = None
    include_name_contains: Optional[List[str]] = None
    exclude_name_contains: Optional[List[str]] = None
    recursive: bool = True

def _attach_gt_masks(images: List[Image],
                     gt_disc_root: Optional[Path],
                     gt_cup_root: Optional[Path]) -> None:
    disc_map = stem_map_by_first_match(gt_disc_root) if gt_disc_root else {}
    cup_map  = stem_map_by_first_match(gt_cup_root)  if gt_cup_root  else {}
    for img in images:
        stem = img.image_path.stem
        if stem in disc_map:
            img.set_mask(Structure.DISC, LabelType.GT, BinaryMaskRef(path=disc_map[stem]))
        if stem in cup_map:
            img.set_mask(Structure.CUP, LabelType.GT, BinaryMaskRef(path=cup_map[stem]))
        # ensure GT boxes exist if masks are present (helps IoU)
        img.ensure_boxes_from_masks()

def collect_images(args: Args) -> List[Image]:
    cfg = _CollectCfg(
        project_dir=args.images_root.parent if args.images_root.parent else Path("."),
        images_root=args.images_root,
        disc_masks=args.gt_disc_root if args.eval_enabled else None,
        cup_masks=args.gt_cup_root  if args.eval_enabled else None,
        include_name_contains=list(args.include) if args.include else None,
        exclude_name_contains=list(args.exclude) if args.exclude else None,
        recursive=args.recursive,
    )
    coll = DatasetCollector(cfg)  # type: ignore[arg-type]
    ds = coll.collect()

    # Optional patient-wise subset
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

    if args.eval_enabled:
        _attach_gt_masks(ds.images, args.gt_disc_root, args.gt_cup_root)

    return ds.images


def build_cup_box_from_disc_mask(disc_mask: np.ndarray,
                                 fallback_box: Tuple[int,int,int,int]) -> Optional[Tuple[int,int,int,int]]:
    tight = _tight_bbox_from_mask(disc_mask)
    if tight is None:
        return None
    box = _shrink_box_to_fit_mask(disc_mask, tight, step_frac=0.02, max_iter=300)
    if box is None:
        box = _shrink_box_to_fit_mask(disc_mask, fallback_box, step_frac=0.02, max_iter=300)
    return box


def _process_one_image(
    img: Image,
    bb_pred: BoundingBoxPredictor,
    msam: MedSAMModel,
    lbl_writer: LabelWriter,
    out_disc_dir: Path,
    out_cup_dir: Path,
    out_viz_dir: Path,
    out_viz_compare_dir: Path,
    save_viz: bool = True,
    eval_enabled: bool = False,
    viz_compare: bool = False,
) -> Tuple[
    Optional[float], Optional[float],  # pred_cd_ratio, gt_cd_ratio
    Optional[Path], Optional[Path], Optional[Path], Optional[Path],  # disc_mask_path, cup_mask_path, viz_path, viz_compare_path
    Optional[float], Optional[float],  # disc_box_iou, cup_box_iou
    Optional[float], Optional[float],  # disc_dice, cup_dice
]:
    # A) Predict disc bounding box (YOLO)
    disc_box = bb_pred.predict_one_image_to_box(img)
    if disc_box is None:
        return None, None, None, None, None, None, None, None, None, None

    img.set_box(Structure.DISC, LabelType.PRED, disc_box)
    lbl_writer.write(img, require_both=False)

    # Image → embedding
    img_bgr = _load_image_bgr(img.image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    emb, H, W, _ = _embed_image_1024(msam, img_rgb)

    # B) MedSAM disc → mask
    dxyxy = (int(round(disc_box.x1)), int(round(disc_box.y1)),
             int(round(disc_box.x2)), int(round(disc_box.y2)))
    pred_disc_mask = medsam_infer(msam, emb, dxyxy, H, W)
    img.set_mask(Structure.DISC, LabelType.PRED, BinaryMaskRef(array=pred_disc_mask))
    seg_disc_path = out_disc_dir / f"{img.image_path.stem}.png"
    _save_mask_png(seg_disc_path, pred_disc_mask)

    # C) Cup box inside disc mask
    cup_xyxy = build_cup_box_from_disc_mask(pred_disc_mask, dxyxy)
    if cup_xyxy is None:
        pred_cd_ratio = _cdr_from_masks(pred_disc_mask, None)  # -> None
        gt_cd_ratio = None
        if eval_enabled and img.gt_disc_mask and img.gt_cup_mask:
            gt_cd_ratio = _cdr_from_masks(img.gt_disc_mask.load(), img.gt_cup_mask.load())
        viz_path = (out_viz_dir / f"{img.image_path.stem}_viz.jpg") if save_viz else None
        if viz_path:
            viz = _overlay_masks_and_boxes(img_bgr, pred_disc_mask, None, dxyxy, None,
                                           cdr_text=("CDR: N/A" if pred_cd_ratio is None else f"CDR: {pred_cd_ratio:.3f}"))
            _save_viz(viz_path, viz)

        # Optional side-by-side compare (requires GT)
        viz_compare_path = None
        if viz_compare and eval_enabled and img.gt_disc_mask and img.gt_cup_mask:
            gt_disc = img.gt_disc_mask.load()
            gt_cup  = img.gt_cup_mask.load()
            left_text = "Pred CDR: N/A"
            right_text = f"GT CDR: {(_cdr_from_masks(gt_disc, gt_cup) or 0):.3f}"
            comp = _make_side_by_side(img_bgr, pred_disc_mask, None, gt_disc, gt_cup, left_text, right_text)
            viz_compare_path = out_viz_compare_dir / f"{img.image_path.stem}_compare.jpg"
            _save_viz(viz_compare_path, comp)

        disc_box_iou = img.disc_iou() if eval_enabled else None
        disc_dice = _dice(pred_disc_mask, img.gt_disc_mask.load()) if (eval_enabled and img.gt_disc_mask) else None
        return pred_cd_ratio, gt_cd_ratio, seg_disc_path, None, viz_path, viz_compare_path, disc_box_iou, None, disc_dice, None

    # D) MedSAM cup → mask
    cup_box = BoundingBox(*map(float, cup_xyxy))
    img.set_box(Structure.CUP, LabelType.PRED, cup_box)
    pred_cup_mask = medsam_infer(msam, emb, cup_xyxy, H, W)
    img.set_mask(Structure.CUP, LabelType.PRED, BinaryMaskRef(array=pred_cup_mask))
    seg_cup_path = out_cup_dir / f"{img.image_path.stem}.png"
    _save_mask_png(seg_cup_path, pred_cup_mask)

    # E) CDRs
    pred_cd_ratio = _cdr_from_masks(pred_disc_mask, pred_cup_mask)
    gt_cd_ratio = None
    gt_disc = gt_cup = None
    if eval_enabled and img.gt_disc_mask and img.gt_cup_mask:
        gt_disc = img.gt_disc_mask.load()
        gt_cup  = img.gt_cup_mask.load()
        gt_cd_ratio = _cdr_from_masks(gt_disc, gt_cup)

    # F) Viz
    cdr_text = f"CDR: {pred_cd_ratio:.3f}" if pred_cd_ratio is not None else "CDR: N/A"
    viz = _overlay_masks_and_boxes(img_bgr, pred_disc_mask, pred_cup_mask, dxyxy, cup_xyxy, cdr_text=cdr_text)
    viz_path = (out_viz_dir / f"{img.image_path.stem}_viz.jpg") if save_viz else None
    if viz_path:
        _save_viz(viz_path, viz)

    # F2) Optional side-by-side compare
    viz_compare_path = None
    if viz_compare and eval_enabled and gt_disc is not None and gt_cup is not None:
        left_text  = f"Pred CDR: {pred_cd_ratio:.3f}" if pred_cd_ratio is not None else "Pred CDR: N/A"
        right_text = f"GT CDR: {gt_cd_ratio:.3f}"      if gt_cd_ratio is not None else "GT CDR: N/A"
        comp = _make_side_by_side(img_bgr, pred_disc_mask, pred_cup_mask, gt_disc, gt_cup, left_text, right_text)
        viz_compare_path = out_viz_compare_dir / f"{img.image_path.stem}_compare.jpg"
        _save_viz(viz_compare_path, comp)

    # G) Optional mask/box metrics
    disc_box_iou = img.disc_iou() if eval_enabled else None
    cup_box_iou  = img.cup_iou()  if eval_enabled else None
    disc_dice = _dice(pred_disc_mask, img.gt_disc_mask.load()) if (eval_enabled and img.gt_disc_mask) else None
    cup_dice  = _dice(pred_cup_mask,  img.gt_cup_mask.load())  if (eval_enabled and img.gt_cup_mask)  else None

    return pred_cd_ratio, gt_cd_ratio, seg_disc_path, seg_cup_path, viz_path, viz_compare_path, disc_box_iou, cup_box_iou, disc_dice, cup_dice


def _write_summary_csv(rows: List[dict], csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    fields = [
        "image_path", "label_path",
        "disc_mask_path", "cup_mask_path",
        "viz_path", "viz_compare_path",   # <--- NEW COLUMN
        "pred_cd_ratio", "gt_cd_ratio", "cdr_error", "cdr_abs_error", "cdr_ape",
        "disc_box_iou", "cup_box_iou",
        "disc_dice", "cup_dice",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ======================================================================
# CLI + main
# ======================================================================

def _derive_outputs(out_dir: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    labels = out_dir / "labels"
    disc   = out_dir / "disc"
    cup    = out_dir / "cup"
    viz    = out_dir / "viz"
    vizc   = out_dir / "viz_compare"  # <--- NEW DIR
    csvp   = out_dir / "summary.csv"
    for p in (labels, disc, cup, viz, vizc, csvp.parent):
        ensure_dir(p)
    return labels, disc, cup, viz, vizc, csvp

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
    eval_enabled: bool
    gt_disc_root: Optional[Path]
    gt_cup_root: Optional[Path]
    viz_compare: bool

def _parse_args() -> _CLI:
    ap = argparse.ArgumentParser(
        description="YOLO bbox → MedSAM disc/cup → CDR (single out dir). Optional comparison to GT."
    )
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--medsam-ckpt", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--include", nargs="*", default=[])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--subset-n", type=int, default=0)
    ap.add_argument("--subset-seed", type=int, default=43)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou",  type=float, default=0.50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-overwrite", action="store_true")
    ap.add_argument("--no-viz", action="store_true")

    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--gt-disc-masks", default="")
    ap.add_argument("--gt-cup-masks",  default="")
    ap.add_argument("--viz-compare", action="store_true", help="Also save side-by-side Pred vs GT overlays (requires --eval)")

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
        eval_enabled=bool(a.eval),
        gt_disc_root=(_expand(a.gt_disc_masks) if a.gt_disc_masks else None),
        gt_cup_root=(_expand(a.gt_cup_masks) if a.gt_cup_masks else None),
        viz_compare=bool(a.viz_compare),
    )

def main() -> None:
    cli = _parse_args()

    out_labels, out_disc, out_cup, out_viz, out_viz_compare, csv_path = _derive_outputs(cli.out_dir)

    args = Args(
        images_root=cli.images_root,
        weights=cli.weights,
        medsam_ckpt=cli.medsam_ckpt,
        out_dir=cli.out_dir,
        out_labels=out_labels,
        out_disc=out_disc,
        out_cup=out_cup,
        out_viz=out_viz,
        out_viz_compare=out_viz_compare,
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
        eval_enabled=cli.eval_enabled,
        gt_disc_root=cli.gt_disc_root,
        gt_cup_root=cli.gt_cup_root,
        viz_compare=cli.viz_compare,
    )

    images = collect_images(args)
    if not images:
        raise SystemExit("[ERR] No images collected.")

    bb_pred = BoundingBoxPredictor(weights=args.weights, conf=args.conf, iou=args.iou, device=args.device)
    lbl_writer = LabelWriter(args.out_labels, args.images_root, overwrite=args.overwrite)

    dev = _pick_device(args.device)
    msam = load_medsam(args.medsam_ckpt, dev, variant="vit_b")

    # Aggregates for evaluation
    disc_iou_vals: List[float] = []
    cup_iou_vals: List[float]  = []
    disc_dice_vals: List[float] = []
    cup_dice_vals: List[float]  = []
    cdr_pred_vals: List[float]  = []
    cdr_gt_vals:   List[float]  = []

    rows: List[dict] = []
    for img in images:
        (pred_cd_ratio, gt_cd_ratio, seg_disc_path, seg_cup_path, viz_path, viz_compare_path,
         disc_box_iou, cup_box_iou, disc_dice, cup_dice) = _process_one_image(
            img, bb_pred, msam, lbl_writer,
            args.out_disc, args.out_cup, args.out_viz, args.out_viz_compare,
            save_viz=args.save_viz, eval_enabled=args.eval_enabled, viz_compare=args.viz_compare
        )

        rel = img.image_path.resolve().relative_to(args.images_root.resolve())
        label_path = (args.out_labels / rel).with_suffix(".txt")

        if disc_box_iou is not None: disc_iou_vals.append(disc_box_iou)
        if cup_box_iou  is not None: cup_iou_vals.append(cup_box_iou)
        if disc_dice    is not None: disc_dice_vals.append(disc_dice)
        if cup_dice     is not None: cup_dice_vals.append(cup_dice)
        if (pred_cd_ratio is not None) and (gt_cd_ratio is not None):
            cdr_pred_vals.append(pred_cd_ratio)
            cdr_gt_vals.append(gt_cd_ratio)

        err = abs_err = ape = ""
        if (pred_cd_ratio is not None) and (gt_cd_ratio is not None):
            e = pred_cd_ratio - gt_cd_ratio
            err = f"{e:.6f}"
            abs_err = f"{abs(e):.6f}"
            denom = max(1e-6, abs(gt_cd_ratio))
            ape = f"{abs(e)/denom:.6f}"

        rows.append({
            "image_path": str(img.image_path),
            "label_path": str(label_path) if label_path.exists() else "",
            "disc_mask_path": str(seg_disc_path) if seg_disc_path else "",
            "cup_mask_path":  str(seg_cup_path)  if seg_cup_path  else "",
            "viz_path":       str(viz_path)      if viz_path      else "",
            "viz_compare_path": str(viz_compare_path) if viz_compare_path else "",
            "pred_cd_ratio": f"{pred_cd_ratio:.6f}" if pred_cd_ratio is not None else "",
            "gt_cd_ratio":   f"{gt_cd_ratio:.6f}"   if gt_cd_ratio   is not None else "",
            "cdr_error":     err,
            "cdr_abs_error": abs_err,
            "cdr_ape":       ape,
            "disc_box_iou": f"{disc_box_iou:.6f}" if disc_box_iou is not None else "",
            "cup_box_iou":  f"{cup_box_iou:.6f}"  if cup_box_iou  is not None else "",
            "disc_dice":    f"{disc_dice:.6f}"    if disc_dice    is not None else "",
            "cup_dice":     f"{cup_dice:.6f}"     if cup_dice     is not None else "",
        })

    _write_summary_csv(rows, args.csv_path)

    print(f"[OK] Processed {len(rows)} images.")
    print(f"[OK] Outputs → {args.out_dir}")
    print(f"  labels/       → {args.out_labels}")
    print(f"  disc/         → {args.out_disc}")
    print(f"  cup/          → {args.out_cup}")
    print(f"  viz/          → {args.out_viz}")
    if args.viz_compare:
        print(f"  viz_compare/  → {args.out_viz_compare}")
    print(f"  summary       → {args.csv_path}")

    # ---------- CDR metrics ----------
    if args.eval_enabled and cdr_pred_vals and cdr_gt_vals:
        pred = np.asarray(cdr_pred_vals, dtype=np.float64)
        gt   = np.asarray(cdr_gt_vals,   dtype=np.float64)
        diff = pred - gt
        mae  = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        bias = float(np.mean(diff))
        mape = float(np.mean(np.abs(diff) / np.maximum(1e-6, np.abs(gt))))
        r    = float(np.corrcoef(pred, gt)[0, 1]) if pred.size >= 2 else float("nan")
        r2   = float(r * r) if np.isfinite(r) else float("nan")
        md   = float(np.mean(diff))
        sd   = float(np.std(diff, ddof=1)) if diff.size >= 2 else float("nan")
        loa_lo = md - 1.96 * sd if np.isfinite(sd) else float("nan")
        loa_hi = md + 1.96 * sd if np.isfinite(sd) else float("nan")

        print("\n[CDR] Accuracy over items with GT present")
        print(f"  N pairs        : {pred.size}")
        print(f"  MAE            : {mae:.4f}")
        print(f"  RMSE           : {rmse:.4f}")
        print(f"  Bias (mean err): {bias:.4f}")
        print(f"  MAPE           : {mape:.4%}")
        print(f"  Pearson r      : {r:.4f}")
        print(f"  R^2            : {r2:.4f}")
        print(f"  Bland–Altman   : mean={md:.4f}, LoA=({loa_lo:.4f}, {loa_hi:.4f})")
    elif args.eval_enabled:
        print("[CDR] --eval set, but no valid GT/pred CDR pairs were found. "
              "Make sure both --gt-disc-masks and --gt-cup-masks are provided and match image stems.")


if __name__ == "__main__":
    main()