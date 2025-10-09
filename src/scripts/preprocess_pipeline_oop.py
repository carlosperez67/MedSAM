#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

from src.imgpipe.config import PipelineConfig
from src.imgpipe.image import Image
from src.imgpipe.image_factory import ImageFactory
from src.imgpipe.dataset import Dataset
from src.imgpipe.augment import Augmentor
from src.imgpipe.roi import ROIDatasetBuilder
from src.imgpipe.disc_only import DiscOnlyDatasetBuilder
from src.imgpipe.utils import ensure_dir


# -------------------------- small helpers --------------------------

def _normalize_steps(steps_arg: str | None) -> List[str]:
    if not steps_arg:
        return ["collect", "split", "save", "augment", "disc_only", "roi"]
    return [s.strip() for s in steps_arg.split(",") if s.strip()]

def _has_multi_datasets(cfg: PipelineConfig) -> bool:
    raw = getattr(cfg, "raw", None)
    return isinstance(raw, dict) and isinstance(raw.get("datasets"), list) and len(raw["datasets"]) > 0

def _group_by_subject(images: List[Image]) -> dict[str, List[Image]]:
    by_subj: dict[str, List[Image]] = defaultdict(list)
    for im in images:
        sid = im.subject_id or im.uid  # fallback to uid if subject_id missing
        by_subj[sid].append(im)
    return by_subj

def _subset_roots(base_project_dir: Path, n: int, seed: int) -> dict[str, Path]:
    root = base_project_dir / "data" / "_subset" / f"N{n}_seed{seed}"
    return {
        "root": root,
        "yolo_split":      root / "yolo_split",
        "yolo_aug":        root / "yolo_split_aug",
        "yolo_roi":        root / "yolo_split_cupROI",
        "yolo_disc_only":  root / "yolo_split_disc_only",
        "viz_out":         root / "viz_labels",
    }


# -------------------------- step functions --------------------------

def step_collect_single(
    dataset_name: str,
    images_root: Path,
    disc_masks_root: Path | None,
    cup_masks_root: Path | None,
    include_name_contains: Iterable[str] | None,
    exclude_name_contains: Iterable[str] | None,
    recursive: bool,
) -> List[Image]:
    factory = ImageFactory(
        dataset_name=dataset_name,
        images_root=images_root,
        disc_masks_root=disc_masks_root,
        cup_masks_root=cup_masks_root,
        include_name_contains=include_name_contains,
        exclude_name_contains=exclude_name_contains,
        recursive=recursive,
    )
    return factory.collect()

def step_collect(cfg: PipelineConfig) -> Dataset:
    if _has_multi_datasets(cfg):
        images: List[Image] = []
        for block in cfg.raw["datasets"]:  # type: ignore[index]
            tag = str(block.get("tag", "DS"))
            imgs_root = Path(block["images_root"]).expanduser().resolve()
            disc_root = Path(block["disc_masks"]).expanduser().resolve() if block.get("disc_masks") else None
            cup_root  = Path(block["cup_masks"]).expanduser().resolve()  if block.get("cup_masks")  else None
            include   = block.get("include_name_contains")
            exclude   = block.get("exclude_name_contains")
            recursive = bool(block.get("recursive", False))

            part = step_collect_single(
                dataset_name=tag,
                images_root=imgs_root,
                disc_masks_root=disc_root,
                cup_masks_root=cup_root,
                include_name_contains=include,
                exclude_name_contains=exclude,
                recursive=recursive,
            )
            print(f"[COLLECT] {tag}: {len(part)} images")
            images.extend(part)
        return Dataset(images)

    # single-dataset path
    images = step_collect_single(
        dataset_name="MDSET",
        images_root=cfg.images_root,
        disc_masks_root=cfg.disc_masks,
        cup_masks_root=cfg.cup_masks,
        include_name_contains=cfg.include_name_contains,
        exclude_name_contains=cfg.exclude_name_contains,
        recursive=cfg.recursive,
    )
    print(f"[COLLECT] MDSET: {len(images)} images")
    return Dataset(images)

def step_subset_if_enabled(cfg: PipelineConfig, ds: Dataset) -> tuple[Dataset, dict[str, Path]]:
    # Default outputs (no subset)
    outs = {
        "yolo_split":      cfg.yolo_split,
        "yolo_aug":        cfg.yolo_aug,
        "yolo_roi":        cfg.yolo_roi,
        "yolo_disc_only":  cfg.yolo_disc_only,
        "viz_out":         cfg.viz_out,
    }

    if not getattr(cfg, "subset_n", 0) or cfg.subset_n <= 0:
        return ds, outs

    # Patient-wise subset
    n_items = cfg.subset_n
    rng = random.Random(cfg.subset_seed)
    by_subj = _group_by_subject(ds.images)
    subjects = list(by_subj.keys())
    rng.shuffle(subjects)

    selected: List[Image] = []
    used_subjects = 0
    for sid in subjects:
        if len(selected) >= n_items:
            break
        selected.extend(by_subj[sid])
        used_subjects += 1
    if len(selected) > n_items:
        selected = selected[:n_items]

    ds_sub = Dataset(selected)

    # Redirect outputs to a subset sandbox
    subset = _subset_roots(cfg.project_dir, cfg.subset_n, cfg.subset_seed)
    for p in subset.values():
        ensure_dir(p)

    print(f"[SUBSET] patient-wise → {len(selected)} items from {used_subjects} subjects → {subset['root']}")

    return ds_sub, {
        "yolo_split":      subset["yolo_split"],
        "yolo_aug":        subset["yolo_aug"],
        "yolo_roi":        subset["yolo_roi"],
        "yolo_disc_only":  subset["yolo_disc_only"],
        "viz_out":         subset["viz_out"],
    }

def step_split(cfg: PipelineConfig, ds: Dataset) -> Dataset:
    train, val, test = ds.split_by_patient(
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        seed=cfg.seed,
        patient_regex=cfg.patient_regex,
    )
    print(f"[SPLIT] train={len(train.images)} | val={len(val.images)} | test={len(test.images)}")
    return Dataset(train.images + val.images + test.images)

def step_save_yolo(ds: Dataset, out_yolo_split: Path, prefer_copy: bool = False) -> None:
    ensure_dir(out_yolo_split)
    ds.save_as_yolo(out_yolo_split, write_yaml=True, prefer_copy=prefer_copy)
    print(f"[OK] YOLO split written → {out_yolo_split}")

def step_augment(cfg: PipelineConfig, ds: Dataset, in_split_root: Path, out_aug_root: Path) -> None:
    # Where to read hyperparams:
    aug_yaml = Path(cfg.raw.get("aug_yaml", ""))  # same key as your old config
    if not aug_yaml or not aug_yaml.exists():
        raise SystemExit("[ERR] 'aug_yaml' missing or not found in your pipeline config.")

    ensure_dir(out_aug_root)
    splits = cfg.augment_splits if cfg.augment_splits else ["train"]

    from src.imgpipe.augment import Augmentor
    aug = Augmentor(aug_yaml=aug_yaml)
    aug.run_from_yolo_root(
        data_root=in_split_root,
        out_root=out_aug_root,
        splits=splits,
        prefer_copy=getattr(cfg, "prefer_copy", False),
    )

def step_disc_only(cfg: PipelineConfig, in_split_root: Path, out_disc_root: Path, in_aug_root: Path) -> None:
    builder = DiscOnlyDatasetBuilder(
        drop_empty=cfg.disc_only_drop_empty,
        prefer_copy=cfg.disc_only_copy_images,
    )
    if cfg.disc_only_from_aug_train and in_aug_root.exists():
        # Pass 1: filter clean split
        ensure_dir(out_disc_root)
        builder.build(src_root=in_split_root, out_root=out_disc_root)
        # Pass 2: overwrite train with filtered augmented train
        tmp_out = out_disc_root.parent / (out_disc_root.name + "_TMP")
        builder.build(src_root=in_aug_root, out_root=tmp_out)
        for sub in ("images", "labels"):
            (out_disc_root / sub / "train").mkdir(parents=True, exist_ok=True)
            src = tmp_out / sub / "train"
            if src.exists():
                for p in src.iterdir():
                    p.rename(out_disc_root / sub / "train" / p.name)
        # cleanup best-effort
        try:
            import shutil
            shutil.rmtree(tmp_out)
        except Exception:
            pass
    else:
        builder.build(src_root=in_split_root, out_root=out_disc_root)

    print(f"[OK] Disc-only dataset written → {out_disc_root}")

def step_roi(cfg: PipelineConfig, in_split_root: Path, in_aug_root: Path, out_roi_root: Path) -> None:
    roi_src = in_aug_root if (cfg.roi_from_aug and in_aug_root.exists()) else in_split_root
    roi_builder = ROIDatasetBuilder(pad_pct=cfg.roi_pad_pct, keep_negatives=cfg.keep_roi_negatives)
    ensure_dir(out_roi_root)
    roi_builder.build(roi_src, out_roi_root)
    print(f"[OK] ROI dataset written → {out_roi_root}")


# ------------------------------- CLI -------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OOP Preprocess Pipeline")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--steps", default=None, help="Comma list (subset of: collect,split,save,augment,disc_only,roi)")
    return ap.parse_args()


# ------------------------------ main -------------------------------

def main():
    args = parse_args()
    cfg = PipelineConfig.load(Path(args.config))
    steps = _normalize_steps(args.steps)

    # 1) Collect
    if "collect" not in steps:
        raise SystemExit("[ERR] The 'collect' step is required in this OOP pipeline.")
    ds = step_collect(cfg)

    # 1.5) Optional subset & output redirection
    ds, outs = step_subset_if_enabled(cfg, ds)
    out_yolo_split = outs["yolo_split"]
    out_yolo_aug   = outs["yolo_aug"]
    out_yolo_roi   = outs["yolo_roi"]
    out_yolo_disc  = outs["yolo_disc_only"]

    # 2) Split
    if "split" in steps:
        ds = step_split(cfg, ds)

    # 3) Save YOLO (split)
    if "save" in steps:
        step_save_yolo(ds, out_yolo_split, prefer_copy=getattr(cfg, "prefer_copy", False))

    # 4) Augment (optional)
    if "augment" in steps and getattr(cfg, "augment_enabled", True):
        step_augment(cfg, ds, in_split_root=out_yolo_split, out_aug_root=out_yolo_aug)

    # 5) Disc-only derived dataset
    if "disc_only" in steps:
        step_disc_only(cfg, in_split_root=out_yolo_split, out_disc_root=out_yolo_disc, in_aug_root=out_yolo_aug)

    # 6) Cup-ROI dataset
    if "roi" in steps:
        step_roi(cfg, in_split_root=out_yolo_split, in_aug_root=out_yolo_aug, out_roi_root=out_yolo_roi)


if __name__ == "__main__":
    main()