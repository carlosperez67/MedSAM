#!/usr/bin/env python3
# preprocess_pipeline.py

from __future__ import annotations

import argparse
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Any, Union
import re

import yaml  # PyYAML

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ------------------------- tiny utils -------------------------

def _expand(p: Union[str, Path]) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _ensure_paths(paths: Iterable[Path]) -> None:
    """Create a list of paths; if an entry looks like a file (has suffix), create its parent."""
    for p in paths:
        _ensure_dir(p.parent if isinstance(p, Path) and p.suffix else p)

def require_dir(p: Path, desc: str) -> None:
    if not p.exists():
        raise SystemExit(f"[ERR] {desc} not found: {p}")

def run_cmd(cmd: Sequence[str], dry_run: bool = False) -> None:
    print("[CMD]", " ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)

def safe_link(src: Path, dst: Path, copy: bool = False) -> None:
    """Symlink if possible; otherwise copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy:
        import shutil
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except Exception:
            import shutil
            shutil.copy2(src, dst)

def list_files_with_ext(root: Path, exts: Iterable[str], recursive: bool = True) -> List[Path]:
    exts = tuple(e.lower() for e in exts)
    if recursive:
        return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    return sorted([p for p in root.iterdir() if p.suffix.lower() in exts])

def stem_map_by_first_match(root: Path, exts: Iterable[str]) -> Dict[str, Path]:
    """Map file stem → first matching path under root (recursive)."""
    out: Dict[str, Path] = {}
    for p in list_files_with_ext(root, exts, recursive=True):
        out.setdefault(p.stem, p)
    return out

def _name_matches_filters(name: str, include: list | None, exclude: list | None) -> bool:
    n = name.lower()
    if include and not any(str(s).lower() in n for s in include):
        return False
    if exclude and any(str(s).lower() in n for s in exclude):
        return False
    return True

def _to_float_list(v: Union[str, List[Any], None]) -> List[float]:
    if v is None:
        return []
    if isinstance(v, list):
        vals: List[float] = []
        for x in v:
            try:
                vals.append(float(x))
            except Exception:
                pass
        return vals
    parts = [p.strip() for p in str(v).split(",") if p.strip()]
    vals: List[float] = []
    for p in parts:
        try:
            vals.append(float(p))
        except Exception:
            pass
    return vals


# ------------------------- defaults from project -------------------------

def resolve_defaults(project_dir: Path) -> dict:
    script_dir = project_dir / "bounding_box" / "preprocess"
    data_dir   = project_dir / "bounding_box" / "data"
    defaults = {
        "script_dir": script_dir,
        "labels_dir": data_dir / "labels",
        "yolo_split": data_dir / "yolo_split",
        "yolo_aug":   data_dir / "yolo_split_aug",
        "yolo_roi":   data_dir / "yolo_split_cupROI",
        "viz_out":    data_dir / "viz_labels",
        "images_root_guess": data_dir / "images",  # only a guess if user doesn't pass images_root
        "subset_base": data_dir / "_subset",       # where we create small test subsets
    }
    return defaults


# ------------------------- subset builders (single-dataset) -------------------------

def build_subset_for_masks_stage_multi(
    subset_base: Path,
    datasets: List[dict],
    n: int,
    seed: int,
    copy: bool = False,
    require_both_default: bool = False,
) -> Tuple[List[dict], List[tuple]]:
    """
    Build a tiny per-dataset filesystem with only N sampled items total.
    Returns (new_datasets_list_with_overridden_paths, sampled_entries).

    We shuffle ALL candidates across all datasets using the seed, then take the first N.
    """
    rng = random.Random(seed)

    # Gather all candidates from every dataset
    choices: List[tuple] = []
    for ds in datasets:
        choices.extend(_collect_candidates_for_dataset(ds, require_both_default=require_both_default))

    if not choices:
        raise SystemExit("[ERR] No candidate images with masks found across datasets.")

    # Shuffle globally across datasets, then slice
    rng.shuffle(choices)
    sampled = choices[: min(n, len(choices))] if n > 0 else choices

    # Prepare per-dataset subset roots
    new_datasets: List[dict] = []
    for ds in datasets:
        tag = ds["tag"]
        s_img  = subset_base / "images"     / tag
        s_disc = subset_base / "disc_masks" / tag
        s_cup  = subset_base / "cup_masks"  / tag
        for d in (s_img, s_disc, s_cup):
            _ensure_dir(d)
        new_datasets.append({
            **ds,
            "images_root": str(s_img),
            "disc_masks":  str(s_disc),
            "cup_masks":   str(s_cup),
        })

    # Link/copy the sampled files
    for tag, stem, ip, dmp, cmp in sampled:
        safe_link(ip,  subset_base / "images"     / tag / ip.name,  copy=copy)
        if dmp: safe_link(dmp, subset_base / "disc_masks" / tag / dmp.name, copy=copy)
        if cmp: safe_link(cmp, subset_base / "cup_masks"  / tag / cmp.name, copy=copy)

    return new_datasets, sampled


def build_subset_for_split_stage(
    subset_base: Path,
    labels_dir: Path,
    images_root: Path,
    n: int,
    seed: int,
    copy: bool = False,
    patient_regex: str | None = None,
) -> Tuple[Path, Path, List[str]]:
    """
    Prepare a small subset for 'split' stage:
      subset_labels_dir/ with ~N labels, subset_images_root/ with matching images.
    We *sample patients*, not individual labels, so a patient never gets split
    across the subset. We then include all stems for the selected patients
    until we reach ~N items (or we include all if n==0).

    Returns (labels_dir', images_root', stems)
    """
    rng = random.Random(seed)
    _ensure_dir(subset_base)
    s_labels = subset_base / "labels"
    s_images = subset_base / "images"
    for d in (s_labels, s_images):
        _ensure_dir(d)

    # Build image map
    img_map = stem_map_by_first_match(images_root, IMG_EXTS)

    # Collect labels
    all_label_paths = sorted(labels_dir.glob("*.txt"))
    if not all_label_paths:
        raise SystemExit("[ERR] No label files in labels_dir; cannot build subset for split stage.")

    # Patient id derivation (regex first, fallback: keep before last '-')
    rx = re.compile(patient_regex, re.I) if patient_regex else None

    def pid_from_stem(stem: str) -> str:
        if rx:
            m = rx.search(stem)
            if m:
                return m.group(1)
        if "-" in stem:
            return stem[: stem.rfind("-")]
        return stem

    # Group label paths by patient id (and keep only those with matching image)
    by_pid: Dict[str, List[Path]] = {}
    for lp in all_label_paths:
        st = lp.stem
        if st not in img_map:
            continue
        by_pid.setdefault(pid_from_stem(st), []).append(lp)

    if not by_pid:
        raise SystemExit("[ERR] After matching labels to images, nothing remained for subset.")

    # Shuffle patients and select until ~N items
    pids = list(by_pid.keys())
    rng.shuffle(pids)

    selected_lbls: List[Path] = []
    for pid in pids:
        k = len(selected_lbls)
        if 0 < n <= k:
            break
        selected_lbls.extend(by_pid[pid])

    if 0 < n < len(selected_lbls):
        selected_lbls = selected_lbls[:n]

    # Link files
    stems: List[str] = []
    for lp in selected_lbls:
        st = lp.stem
        ip = img_map.get(st)
        if not ip:
            continue
        stems.append(st)
        safe_link(lp, s_labels / lp.name, copy=copy)
        safe_link(ip, s_images / ip.name, copy=copy)

    if not stems:
        raise SystemExit("[ERR] No (image,label) pairs were linked into the subset.")

    print(f"[SUBSET] split-stage subset: {len(stems)} items across {len(pids)} patients (sampled) → {subset_base}")
    return s_labels, s_images, stems


# ------------------------- subset builders (multi-dataset) -------------------------

def _first_match_map(root: Path) -> Dict[str, Path]:
    return stem_map_by_first_match(root, IMG_EXTS)

def _collect_candidates_for_dataset(ds: dict, require_both_default: bool=False) -> List[tuple]:
    """
    Returns [(tag, stem, img_path, disc_path_or_None, cup_path_or_None), ...]
    Only stems that have at least one mask (or both if require_both) and pass name filters.
    """
    tag = ds["tag"]
    img_root  = _expand(ds["images_root"])
    disc_root = _expand(ds["disc_masks"])
    cup_root  = _expand(ds["cup_masks"])

    img_map  = _first_match_map(img_root)
    disc_map = _first_match_map(disc_root)
    cup_map  = _first_match_map(cup_root)

    include = ds.get("include_name_contains")
    exclude = ds.get("exclude_name_contains")
    require_both = bool(ds.get("require_both", require_both_default))

    out = []
    for stem, ip in img_map.items():
        if not _name_matches_filters(ip.name, include, exclude):
            continue
        dmp = disc_map.get(stem)
        cmp = cup_map.get(stem)
        if require_both and (dmp is None or cmp is None):
            continue
        if dmp is None and cmp is None:
            continue
        out.append((tag, stem, ip, dmp, cmp))
    return out


# ------------------------- stage runners -------------------------

def stage_masks_to_boxes(args, defaults) -> None:
    script = defaults["script_dir"] / "masks_to_boxes.py"
    require_dir(script.parent, "preprocess script dir")
    cmd = [
        sys.executable, str(script),
        "--images",     str(args.images_root),
        "--disc_masks", str(args.disc_masks),
        "--cup_masks",  str(args.cup_masks),
        "--out_labels", str(args.labels_dir),
        "--out_csv",    str(args.labels_dir.parent / "labels_summary.csv"),
    ]
    if args.pad_pct is not None:     cmd += ["--pad_pct", str(args.pad_pct)]
    if args.min_area_px is not None: cmd += ["--min_area_px", str(args.min_area_px)]
    if args.largest_only:            cmd += ["--largest_only"]
    if args.require_both:            cmd += ["--require_both"]
    if args.recursive:               cmd += ["--recursive"]
    if args.verbose:                 cmd += ["--verbose"]
    if args.masks_extra:             cmd += shlex.split(args.masks_extra)
    # pass-through of name-based filters (if provided)
    if getattr(args, "masks_exclude_name_contains", ""):
        cmd += ["--exclude_name_contains", args.masks_exclude_name_contains]
    if getattr(args, "masks_include_name_contains", ""):
        cmd += ["--include_name_contains", args.masks_include_name_contains]
    run_cmd(cmd, args.dry_run)


def stage_masks_to_boxes_one_dataset(
    script_path: Path,
    out_labels_root: Path,
    ds: dict,
    verbose: bool,
    dry_run: bool,
):
    """Call masks_to_boxes.py for a single dataset block into a per-dataset labels folder."""
    def _csv(v):
        return v if isinstance(v, str) else ",".join(map(str, v))

    tag = ds["tag"]
    out_labels = out_labels_root / tag
    out_csv    = out_labels_root / f"labels_summary_{tag}.csv"
    args = [
        sys.executable, str(script_path),
        "--images",     str(_expand(ds["images_root"])),
        "--disc_masks", str(_expand(ds["disc_masks"])),
        "--cup_masks",  str(_expand(ds["cup_masks"])),
        "--out_labels", str(out_labels),
        "--out_csv",    str(out_csv),
    ]
    # dataset-specific switches
    if ds.get("pad_pct")        is not None: args += ["--pad_pct", str(ds["pad_pct"])]
    if ds.get("min_area_px")    is not None: args += ["--min_area_px", str(ds["min_area_px"])]
    if ds.get("largest_only"):                 args += ["--largest_only"]
    if ds.get("require_both"):                args += ["--require_both"]
    if ds.get("recursive"):                   args += ["--recursive"]
    if verbose:                                args += ["--verbose"]
    if ds.get("exclude_name_contains"):
        args += ["--exclude_name_contains", _csv(ds["exclude_name_contains"])]
    if ds.get("include_name_contains"):
        args += ["--include_name_contains", _csv(ds["include_name_contains"])]
    run_cmd(args, dry_run=dry_run)


def _safe_link_or_copy(src: Path, dst: Path, do_copy: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        if do_copy:
            import shutil; shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve())
    except Exception:
        import shutil; shutil.copy2(src, dst)


def _log(msg: str):
    print(f"[MDSET] {msg}")


def build_union_images_and_labels(
    datasets: List[dict],
    per_dataset_labels_root: Path,
    union_images_root: Path,
    union_labels_dir: Path,
    prefer_copy: bool = False,
) -> None:
    """
    Create a unified images root and labels dir by symlinking (or copying) from each dataset.
    If a label stem collides, prefix the stem with '<tag>___' in BOTH the label and the image.
    """
    _ensure_dir(union_images_root)
    _ensure_dir(union_labels_dir)

    stem_seen: set[str] = set()
    collisions = 0
    added = 0

    for ds in datasets:
        tag = ds["tag"]
        img_root = _expand(ds["images_root"])
        lbl_dir  = per_dataset_labels_root / tag
        if not lbl_dir.exists():
            _log(f"labels for dataset '{tag}' not found; skipping")
            continue

        # map image stems under this dataset
        img_map = stem_map_by_first_match(img_root, IMG_EXTS)

        for lp in sorted(lbl_dir.glob("*.txt")):
            st = lp.stem
            ip = img_map.get(st)
            if ip is None:
                # label without matching image → skip
                continue

            final_stem = st
            if final_stem in stem_seen:
                final_stem = f"{tag}___{st}"
                collisions += 1
            stem_seen.add(final_stem)

            # link/copy image and label into union (keep original image extension)
            _safe_link_or_copy(ip, union_images_root / f"{final_stem}{ip.suffix}", do_copy=prefer_copy)
            _safe_link_or_copy(lp, union_labels_dir / f"{final_stem}.txt", do_copy=prefer_copy)
            added += 1

    _log(f"union built: {added} pairs | collisions (prefixed) = {collisions}")


def stage_masks_multi(args, defaults, cfg_dict: dict):
    """Run masks_to_boxes for each dataset in cfg['datasets'] and build union I/O for split."""
    script = defaults["script_dir"] / "masks_to_boxes.py"
    require_dir(script.parent, "preprocess script dir")

    # Per-dataset labels will be placed under labels_dir/<TAG>
    per_ds_labels_root = _expand(args.labels_dir)
    _ensure_dir(per_ds_labels_root)

    datasets: List[dict] = cfg_dict["datasets"]
    for ds in datasets:
        stage_masks_to_boxes_one_dataset(
            script_path=script,
            out_labels_root=per_ds_labels_root,
            ds=ds,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )

    # Build union images + labels for the split stage
    union_images_root = _expand(cfg_dict.get("union_images_root", defaults["images_root_guess"]))
    union_labels_dir  = _expand(cfg_dict.get("union_labels_dir",  per_ds_labels_root / "__ALL__"))

    build_union_images_and_labels(
        datasets=datasets,
        per_dataset_labels_root=per_ds_labels_root,
        union_images_root=union_images_root,
        union_labels_dir=union_labels_dir,
        prefer_copy=False,  # set True if your system forbids symlinks
    )

    # Redirect subsequent stages to use the union
    args.images_root = union_images_root
    args.labels_dir  = union_labels_dir


# ------------------------- augment YAML auto-builder -------------------------

def _build_auto_aug_yaml_dict(args) -> Dict[str, Any]:
    """Create an augment config dict compatible with augment_yolo_ds.py --aug_yaml."""
    # Defaults for the Albumentations transform (match script defaults)
    transform = {
        "hflip_p": 0.5,
        "vflip_p": 0.5,
        "affine": {
            "p": 0.7,
            "scale": [0.9, 1.1],
            "translate_percent": [-0.05, 0.05],
            "rotate": [-15, 15],
            "shear": [-5, 5],
        },
        "random_resized_crop": {
            "p": 0.5,
            "scale": [0.9, 1.0],
            "ratio": [0.9, 1.1],
        },
        "color_jitter": {
            "p": 0.5,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        },
    }

    zoom_scales = _to_float_list(getattr(args, "zoom_scales", None)) or [0.5, 0.7, 0.85]
    sweep_scales = _to_float_list(getattr(args, "zoom_sweep_scales", None)) or [0.35, 0.5, 0.7]

    cfg = {
        "out_ext": getattr(args, "out_ext", ".jpg"),
        "multiplier": int(getattr(args, "multiplier", 2)),
        "include_images_without_labels": bool(getattr(args, "include_images_without_labels", False)),
        "seed": int(getattr(args, "seed", 1337)),
        "write_yaml": bool(getattr(args, "write_yaml", True)),
        "transform": transform,
        "tiling": {
            "enable": bool(getattr(args, "enable_tiling", False)),
            "tile_size": int(getattr(args, "tile_size", 512)),
            "tile_overlap": float(getattr(args, "tile_overlap", 0.2)),
            "min_tile_vis": float(getattr(args, "min_tile_vis", 0.2)),
            "keep_empty_tiles": bool(getattr(args, "keep_empty_tiles", False)),
            "from_aug": bool(getattr(args, "tile_from_aug", False)),
        },
        "zoom_crops": {
            "enable": bool(getattr(args, "enable_zoom_crops", False)),
            "scales": zoom_scales,
            "per_obj": int(getattr(args, "zoom_per_obj", 1)),
            "on": getattr(args, "zoom_on", "both"),
            "out_size": int(getattr(args, "zoom_out_size", 640)),
            "jitter": float(getattr(args, "zoom_jitter", 0.05)),
            "min_vis": float(getattr(args, "zoom_min_vis", 0.2)),
            "from_aug": bool(getattr(args, "zoom_from_aug", False)),
            "keep_empty": bool(getattr(args, "zoom_keep_empty", False)),
        },
        "zoom_sweep": {
            "enable": bool(getattr(args, "enable_zoom_sweep", False)),
            "scales": sweep_scales,
            "overlap": float(getattr(args, "zoom_sweep_overlap", 0.25)),
            "min_vis": float(getattr(args, "zoom_sweep_min_vis", 0.2)),
            "keep_empty": bool(getattr(args, "zoom_sweep_keep_empty", False)),
            "out_size": int(getattr(args, "zoom_sweep_out_size", 640)),
            "from_aug": bool(getattr(args, "zoom_sweep_from_aug", False)),
        },
    }
    return cfg

def _write_auto_aug_yaml(args, out_root: Path) -> Path:
    """Write the auto-generated augment YAML into out_root and return the path."""
    _ensure_dir(out_root)
    cfg = _build_auto_aug_yaml_dict(args)
    aug_yaml = out_root / "_auto_augment.yaml"
    with open(aug_yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[INFO] Auto-generated augment config → {aug_yaml}")
    return aug_yaml


# ------------------------- stage: split / augment / roi / viz / disc_only -------------------------

def stage_disc_only(args, defaults) -> None:
    """Build disc-only dataset from 2-class YOLO labels.
       Train splits can be taken from yolo_split_aug; val/test from clean yolo_split."""
    script = defaults["script_dir"] / "build_disc_only_dataset.py"
    require_dir(script.parent, "preprocess script dir")
    base_root = args.yolo_split               # clean split must exist
    aug_root  = args.yolo_aug if args.disc_only_from_aug_train else None

    # normalize train_splits
    splits = args.disc_only_train_splits
    if isinstance(splits, str):
        splits = [s.strip() for s in splits.split(",") if s.strip()]

    cmd = [
        sys.executable, str(script),
        "--base_root",  str(base_root),
        "--out_root",   str(args.yolo_disc_only),
        "--train_splits", ",".join(splits),
    ]
    if aug_root: cmd += ["--aug_root", str(aug_root)]
    if args.disc_only_copy_images: cmd += ["--copy_images"]
    if args.disc_only_drop_empty:  cmd += ["--drop_empty"]

    run_cmd(cmd, args.dry_run)


def stage_split_yolo(args, defaults) -> None:
    script = defaults["script_dir"] / "split_yolo.py"
    require_dir(script.parent, "preprocess script dir")
    cmd = [
        sys.executable, str(script),
        "--project_dir", str(args.project_dir),
        "--images_root", str(args.images_root),
        "--labels_dir",  str(args.labels_dir),
        "--out_root",    str(args.yolo_split),
        "--val_frac",    str(args.val_frac),
        "--test_frac",   str(args.test_frac),
        "--seed",        str(args.seed),
    ]
    if args.copy:        cmd += ["--copy"]
    if args.patient_regex: cmd += ["--patient_regex", args.patient_regex]
    if args.write_yaml:  cmd += ["--write_yaml"]
    if args.split_extra: cmd += shlex.split(args.split_extra)
    run_cmd(cmd, args.dry_run)


def stage_augment(args, defaults) -> None:
    """
    - If args.aug_yaml is provided, pass it through to augment_yolo_ds.py.
    - Otherwise, synthesize an augment YAML from pipeline args and pass that.
    """
    script = defaults["script_dir"] / "augment_yolo_ds.py"
    require_dir(script.parent, "preprocess script dir")

    # Ensure output root exists so we have a place to put the auto YAML
    _ensure_dir(args.yolo_aug)

    aug_yaml_path: Path
    if getattr(args, "aug_yaml", None):
        aug_yaml_path = _expand(args.aug_yaml)
    else:
        aug_yaml_path = _write_auto_aug_yaml(args, _expand(args.yolo_aug))

    cmd = [
        sys.executable, str(script),
        "--project_dir", str(args.project_dir),
        "--data_root",   str(args.yolo_split),
        "--out_root",    str(args.yolo_aug),
        "--aug_yaml",    str(aug_yaml_path),
    ]

    # splits
    if args.augment_splits:
        splits = args.augment_splits if isinstance(args.augment_splits, list) else normalize_list(args.augment_splits)
        if splits:
            cmd += ["--splits"] + splits

    run_cmd(cmd, args.dry_run)


def stage_build_roi(args, defaults) -> None:
    script = defaults["script_dir"] / "build_cup_roi_dataset.py"
    require_dir(script.parent, "preprocess script dir")

    # choose source
    src_root = args.yolo_aug if getattr(args, "roi_from_aug", False) else args.yolo_split
    require_dir(src_root, "ROI source root")

    cmd = [
        sys.executable, str(script),
        "--project_dir", str(args.project_dir),
        "--data_root",   str(src_root),
        "--out_root",    str(args.yolo_roi),
        "--pad_pct",     str(args.roi_pad_pct),
    ]
    if args.keep_roi_negatives:
        cmd += ["--keep_negatives"]
    if args.roi_extra:
        cmd += shlex.split(args.roi_extra)
    run_cmd(cmd, args.dry_run)


def stage_viz(args, defaults) -> None:
    script = defaults["script_dir"] / "viz_yolo_two_boxes.py"
    require_dir(script.parent, "preprocess script dir")
    cmd = [
        sys.executable, str(script),
        "--labels_dir", str(args.labels_dir),
        "--images_dir", str(args.images_root),
        "--out_dir",    str(args.viz_out),
        "--sample",     str(args.viz_sample),
        "--alpha",      str(args.viz_alpha),
    ]
    if args.viz_save_crops:
        cmd += ["--save_crops"]
    if args.viz_make_montage:
        cmd += ["--make_montage"]
    if args.verbose:
        cmd += ["--verbose"]
    if args.viz_extra:
        cmd += shlex.split(args.viz_extra)
    run_cmd(cmd, args.dry_run)


# ------------------------- config helpers -------------------------

def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"[ERR] YAML at {path} must be a mapping of key: value")
    return cfg

def apply_config_defaults(ap: argparse.ArgumentParser, cfg: Dict[str, Any]) -> None:
    """Set parser defaults from YAML config (unknown keys are ignored)."""
    ap.set_defaults(**cfg)

def normalize_steps(steps_value: Union[str, List[str], None]) -> List[str]:
    if steps_value is None:
        return []
    if isinstance(steps_value, list):
        return [s.strip() for s in steps_value if s and s.strip()]
    return [s.strip() for s in str(steps_value).split(",") if s.strip()]

def normalize_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [s.strip() for s in str(value).split(",") if s.strip()]

def build_config_template(project_dir: Path) -> str:
    d = resolve_defaults(project_dir)
    # NOTE: YAML comments are static; values below are pre-filled with sensible defaults.
    return f"""# === Preprocessing Pipeline Config (YAML) ===
# Edit this file and pass it via:  --config /path/to/config.yaml
# CLI flags still override anything here.

# ---- Core project paths ----
project_dir: "{project_dir}"     # Root of your MedSAM project
images_root: "{d['images_root_guess']}"  # Fundus images root (recursive)
labels_dir:  "{d['labels_dir']}"         # Where YOLO .txt labels live (or will be written)
yolo_split:  "{d['yolo_split']}"         # Output split root (images/ + labels/)
yolo_aug:    "{d['yolo_aug']}"           # Output augmented dataset
yolo_roi:    "{d['yolo_roi']}"           # Cup-ROI dataset output
viz_out:     "{d['viz_out']}"            # Visualizations output folder

# ---- Which stages to run ----
all: false                     # If true, runs: [masks, split, augment, disc_only, roi, viz]
steps: ["masks", "split", "augment", "disc_only", "roi", "viz"]

# ---- Small subset mode (E2E sanity check) ----
subset_n: 0                    # If >0, run on a random subset of N items
subset_seed: 1234              # RNG seed for subset sampling
subset_copy: false             # true = copy files; false = symlink
# Subset is applied only if the FIRST stage is 'masks' or 'split'

# ---- Common flags ----
write_yaml: true               # Write data.yaml for split/augment where supported
dry_run: false                 # Print the commands instead of executing them
verbose: false                 # Verbose mode for underlying tools

# ---- masks_to_boxes.py ----
disc_masks: ""                 # Optic disc mask directory
cup_masks: ""                  # Optic cup mask directory
pad_pct: 0.0                   # Expand boxes by this fraction of max(W,H)
min_area_px: 25                # Min CC area in mask (pixels)
largest_only: false            # Keep only the largest connected component
require_both: false            # Keep only samples that have BOTH disc and cup
recursive: false               # Recurse under images_root when searching images
masks_extra: ""                # Extra raw args to append

# ---- split_yolo.py ----
val_frac: 0.15                 # Validation fraction (by patient)
test_frac: 0.15                # Test fraction (by patient)
seed: 1337                     # RNG for patient split AND augment default
copy: false                    # Copy instead of symlink into split folders
patient_regex: ""              # Regex with ONE capturing group for patient id
split_extra: ""                # Extra raw args to append

# ---- augment_yolo_ds.py ----
# Provide an external augment YAML, OR leave blank to auto-generate from keys below.
aug_yaml: ""                   # Path to augment config YAML (if provided, overrides keys below)
multiplier: 2                  # Augmented copies per original (if auto-generating)
out_ext: ".jpg"                # Output extension for augmented images (if auto-generating)
augment_splits: ["train","val","test"]  # Which splits to process (CLI)
include_images_without_labels: false     # If auto-generating YAML

# Tiling (grid sweep) -- used ONLY if aug_yaml is empty and we auto-generate
enable_tiling: false
tile_size: 512
tile_overlap: 0.40
min_tile_vis: 0.05
keep_empty_tiles: true
tile_from_aug: false

# Zoom crops (object-centric) -- used ONLY if aug_yaml is empty and we auto-generate
enable_zoom_crops: false
zoom_scales: [0.35, 0.5, 0.7]
zoom_per_obj: 2
zoom_on: "both"
zoom_out_size: 640
zoom_jitter: 0.05
zoom_min_vis: 0.10
zoom_from_aug: false
zoom_keep_empty: true

# Multi-scale sliding zoom sweep (covers whole image) -- only if auto-generating
enable_zoom_sweep: false
zoom_sweep_scales: [0.35, 0.5, 0.7]
zoom_sweep_overlap: 0.25
zoom_sweep_min_vis: 0.10
zoom_sweep_keep_empty: true
zoom_sweep_out_size: 640
zoom_sweep_from_aug: false

augment_extra: ""              # (deprecated) no longer used

# ---- build_cup_roi_dataset.py ----
roi_pad_pct: 0.10              # Padding fraction around disc to form ROI (cup-only dataset)
keep_roi_negatives: false      # Keep ROI crops even if cup not visible
roi_extra: ""                  # Extra raw args to append
roi_from_aug: false            # Use yolo_aug as ROI input instead of yolo_split

# ---- disc-only dataset (derived from yolo_split / yolo_split_aug) ----
yolo_disc_only: "{d['yolo_split'].parent}/yolo_split_disc_only"
disc_only_from_aug_train: true
disc_only_train_splits: ["train"]
disc_only_copy_images: false
disc_only_drop_empty: false

# ---- viz_yolo_two_boxes.py ----
viz_sample: 12                 # Number of random samples to visualize
viz_alpha: 0.25                # Box fill transparency
viz_save_crops: false          # Also save OD/OC crops
viz_make_montage: false        # Save side-by-side [original | annotated]
viz_extra: ""                  # Extra raw args to append
"""


# ---------- MULTI-DATASET SUPPORT ----------

def cfg_has_datasets(cfg: dict) -> bool:
    return isinstance(cfg.get("datasets"), list) and len(cfg["datasets"]) > 0


# ------------------------- CLI -------------------------

def parse_args() -> argparse.Namespace:
    # Step 1: lightweight parser to get --config / --write_config_template
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", default=None, help="Path to YAML config with all arguments.")
    base.add_argument("--write_config_template", default=None,
                      help="Write a commented YAML template to this path and exit.")
    base.add_argument("--project_dir", default=".", help="Project root (MedSAM), used for the template defaults too.")
    prelim, _ = base.parse_known_args()

    if prelim.write_config_template:
        proj = _expand(prelim.project_dir)
        tpl = build_config_template(proj)
        outp = _expand(prelim.write_config_template)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(tpl)
        print(f"[OK] Wrote config template → {outp}")
        sys.exit(0)

    # Step 2: main parser with all flags
    ap = argparse.ArgumentParser(
        parents=[base],
        description="Wrapper to run the full preprocessing pipeline, with optional YAML config and small-subset testing.",
    )

    # which steps
    ap.add_argument("--all", action="store_true", help="Run all stages (masks, split, augment, disc_only, roi, viz).")
    ap.add_argument("--steps", default=None,
                    help="Comma list OR YAML list of stages to run, e.g., 'masks,split,augment,disc_only,roi,viz'.")

    # small-subset test mode
    ap.add_argument("--subset_n", type=int, default=0,
                    help="If >0, run on a random subset of N items (applied when first stage is 'masks' or 'split').")
    ap.add_argument("--subset_seed", type=int, default=1234, help="RNG seed for subset sampling.")
    ap.add_argument("--subset_copy", action="store_true", help="Copy files into subset instead of symlink.")

    # common paths
    ap.add_argument("--images_root", default=None, help="Fundus image root (recursively searched).")
    ap.add_argument("--labels_dir",  default=None, help="Directory for YOLO labels.")
    ap.add_argument("--yolo_split",  default=None, help="Output split root.")
    ap.add_argument("--yolo_aug",    default=None, help="Output augmented root.")
    ap.add_argument("--yolo_roi",    default=None, help="Output ROI root.")
    ap.add_argument("--viz_out",     default=None, help="Output folder for label visualizations.")
    ap.add_argument("--write_yaml",  action="store_true", help="Write data.yaml for split/augment where applicable.")
    ap.add_argument("--dry_run",     action="store_true", help="Print commands only, don't execute.")
    ap.add_argument("--verbose",     action="store_true", help="Verbose flags for underlying tools.")

    # masks_to_boxes args
    ap.add_argument("--disc_masks", default=None, help="Optic disc mask directory.")
    ap.add_argument("--cup_masks",  default=None, help="Optic cup mask directory.")
    ap.add_argument("--pad_pct",    type=float, default=None, help="Expand boxes by this fraction of max(W,H).")
    ap.add_argument("--min_area_px",type=int,   default=None, help="Min connected component area in mask.")
    ap.add_argument("--largest_only", action="store_true", help="Keep only largest connected component.")
    ap.add_argument("--require_both", action="store_true", help="Keep only samples that have BOTH disc and cup.")
    ap.add_argument("--recursive",    action="store_true", help="Recurse under images_root when searching images.")
    ap.add_argument("--masks_extra",  default="", help="Raw extra args to append to masks_to_boxes.py")
    ap.add_argument("--masks_exclude_name_contains", default="",
                    help="Comma-separated substrings; skip any image/mask whose filename contains any of these (case-insensitive).")
    ap.add_argument("--masks_include_name_contains", default="",
                    help="Comma-separated substrings; if set, keep only files whose filename contains any of these (case-insensitive).")

    # split_yolo args
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed",     type=int,   default=1337)
    ap.add_argument("--copy",     action="store_true", help="Copy files instead of symlinking.")
    ap.add_argument("--patient_regex", default="", help="Regex with ONE capturing group for patient id.")
    ap.add_argument("--split_extra", default="", help="Raw extra args to append to split_yolo.py")

    # augment_yolo_ds args (paths/splits + aug_yaml only; per-aug knobs live in YAML)
    ap.add_argument("--augment_splits", default=None,
                    help="Comma list OR YAML list of splits to process, e.g. 'train,val' (defaults to train,val,test).")
    ap.add_argument("--aug_yaml", default=None,
                    help="Path to augment config YAML. If omitted, one is auto-generated from pipeline params.")

    # Back-compat knobs (used only if we auto-generate an augment YAML)
    ap.add_argument("--multiplier", type=int,   default=2,    help="[auto-aug] Augmented copies per original.")
    ap.add_argument("--out_ext",    default=".jpg",           help="[auto-aug] Output extension for augmented images.")
    ap.add_argument("--include_images_without_labels", action="store_true",
                    help="[auto-aug] Include images that have no label file (create negatives).")
    # tiling
    ap.add_argument("--enable_tiling",   action="store_true", help="[auto-aug] Enable tiling.")
    ap.add_argument("--tile_size",       type=int,   default=512, help="[auto-aug]")
    ap.add_argument("--tile_overlap",    type=float, default=0.2, help="[auto-aug]")
    ap.add_argument("--min_tile_vis",    type=float, default=0.2, help="[auto-aug]")
    ap.add_argument("--keep_empty_tiles",action="store_true", help="[auto-aug]")
    ap.add_argument("--tile_from_aug",   action="store_true", help="[auto-aug]")
    # zoom crops
    ap.add_argument("--enable_zoom_crops", action="store_true", help="[auto-aug]")
    ap.add_argument("--zoom_scales",       default="0.5,0.7,0.85", help="[auto-aug]")
    ap.add_argument("--zoom_per_obj",      type=int, default=1, help="[auto-aug]")
    ap.add_argument("--zoom_on",           default="both", choices=["disc","cup","both","any"], help="[auto-aug]")
    ap.add_argument("--zoom_out_size",     type=int, default=640, help="[auto-aug]")
    ap.add_argument("--zoom_jitter",       type=float, default=0.05, help="[auto-aug]")
    ap.add_argument("--zoom_min_vis",      type=float, default=0.2, help="[auto-aug]")
    ap.add_argument("--zoom_from_aug",     action="store_true", help="[auto-aug]")
    ap.add_argument("--zoom_keep_empty",   action="store_true", help="[auto-aug]")
    # zoom sweep (whole-image)
    ap.add_argument("--enable_zoom_sweep", action="store_true", help="[auto-aug]")
    ap.add_argument("--zoom_sweep_scales", default="0.35,0.5,0.7", help="[auto-aug]")
    ap.add_argument("--zoom_sweep_overlap", type=float, default=0.25, help="[auto-aug]")
    ap.add_argument("--zoom_sweep_min_vis", type=float, default=0.2, help="[auto-aug]")
    ap.add_argument("--zoom_sweep_keep_empty", action="store_true", help="[auto-aug]")
    ap.add_argument("--zoom_sweep_out_size", type=int, default=640, help="[auto-aug]")
    ap.add_argument("--zoom_sweep_from_aug", action="store_true", help="[auto-aug]")

    # build_cup_roi_dataset args
    ap.add_argument("--roi_pad_pct",      type=float, default=0.10)
    ap.add_argument("--keep_roi_negatives", action="store_true", help="Keep ROI crops with no cup visible.")
    ap.add_argument("--roi_extra",        default="", help="Raw extra args to append to build_cup_roi_dataset.py")
    ap.add_argument("--roi_from_aug", action="store_true", help="Use yolo_aug as ROI input instead of yolo_split.")

    # viz args
    ap.add_argument("--viz_sample",      type=int,   default=12)
    ap.add_argument("--viz_alpha",       type=float, default=0.25)
    ap.add_argument("--viz_save_crops",  action="store_true")
    ap.add_argument("--viz_make_montage",action="store_true")
    ap.add_argument("--viz_extra",       default="", help="Raw extra args to append to viz_yolo_two_boxes.py")

    # disc-only stage args
    ap.add_argument("--yolo_disc_only", default=None, help="Output root for disc-only dataset.")
    ap.add_argument("--disc_only_from_aug_train", action="store_true",
                    help="Use yolo_split_aug for train splits when building disc-only.")
    ap.add_argument("--disc_only_train_splits", default="train",
                    help="Comma or YAML list of splits to source from aug root.")
    ap.add_argument("--disc_only_copy_images", action="store_true", help="Copy instead of symlink.")
    ap.add_argument("--disc_only_drop_empty", action="store_true", help="Drop images with empty filtered labels.")

    # Step 3: apply YAML defaults (if provided), then parse CLI to override
    if prelim.config:
        cfg = load_yaml_config(_expand(prelim.config))
        apply_config_defaults(ap, cfg)

    args = ap.parse_args()

    # expose raw YAML so main() can access datasets
    if prelim.config:
        args._raw_config = cfg

    # Normalize steps
    args.steps = normalize_steps(args.steps)
    return args


# ------------------------- main -------------------------

def main():
    args = parse_args()
    PROJECT_DIR = _expand(args.project_dir)
    defaults = resolve_defaults(PROJECT_DIR)

    # Resolve common paths (with project defaults)
    args.labels_dir = _expand(args.labels_dir) if args.labels_dir else defaults["labels_dir"]
    args.yolo_split = _expand(args.yolo_split) if args.yolo_split else defaults["yolo_split"]
    args.yolo_aug   = _expand(args.yolo_aug)   if args.yolo_aug   else defaults["yolo_aug"]
    args.yolo_roi   = _expand(args.yolo_roi)   if args.yolo_roi   else defaults["yolo_roi"]
    args.viz_out    = _expand(args.viz_out)    if args.viz_out    else defaults["viz_out"]
    args.yolo_disc_only = _expand(args.yolo_disc_only) if args.yolo_disc_only else (
        defaults["yolo_split"].parent / "yolo_split_disc_only"
    )

    # Early create output roots (safe)
    _ensure_paths([args.labels_dir, args.yolo_split, args.yolo_aug, args.yolo_roi, args.viz_out, args.yolo_disc_only])

    # Images root: provided or guess
    args.images_root = _expand(args.images_root) if args.images_root else defaults["images_root_guess"]

    # Stages to run
    if args.all:
        stages = ["masks", "split", "augment", "disc_only", "roi", "viz"]
    else:
        stages = args.steps
        if not stages:
            print("[INFO] No --steps given; nothing to do. Use --all or --steps masks,split,...")
            return

    # ---------- optional subset mode ----------
    subset_used = False
    if args.subset_n and args.subset_n > 0:
        first_stage = stages[0]
        subset_root = defaults["subset_base"] / f"subsetN{args.subset_n}_seed{args.subset_seed}"
        _ensure_dir(subset_root)

        if first_stage == "masks":
            # MULTI-DATASET path
            if hasattr(args, "_raw_config") and cfg_has_datasets(args._raw_config):
                # Build a tiny per-dataset FS with only N total samples across all datasets
                ds_subset, _sampled = build_subset_for_masks_stage_multi(
                    subset_root,
                    args._raw_config["datasets"],
                    n=args.subset_n,
                    seed=args.subset_seed,
                    copy=args.subset_copy,
                    require_both_default=bool(args.require_both),
                )
                # Override the YAML config for this run to point at the subset
                args._raw_config = {
                    **args._raw_config,
                    "datasets": ds_subset,
                    # Ensure union (for split stage) also lives under the subset root
                    "union_images_root": str(subset_root / "union_images"),
                    "union_labels_dir":  str(subset_root / "labels" / "__ALL__"),
                }
                # All downstream outputs live under subset root
                args.labels_dir = subset_root / "labels"
                args.yolo_split = subset_root / "yolo_split"
                args.yolo_aug   = subset_root / "yolo_split_aug"
                args.yolo_roi   = subset_root / "yolo_split_cupROI"
                args.viz_out    = subset_root / "viz_labels"
                subset_used = True

            # SINGLE-DATASET path
            else:
                if not args.disc_masks or not args.cup_masks:
                    raise SystemExit("[ERR] Stage 'masks' needs --disc_masks and --cup_masks when using --subset_n.")
                s_labels, s_images, _stems = build_subset_for_split_stage(
                    subset_root,
                    _expand(args.labels_dir),
                    _expand(args.images_root),
                    n=args.subset_n,
                    seed=args.subset_seed,
                    copy=args.subset_copy,
                    patient_regex=args.patient_regex,
                )
                args.images_root = s_images
                args.labels_dir  = subset_root / "labels"
                args.yolo_split  = subset_root / "yolo_split"
                args.yolo_aug    = subset_root / "yolo_split_aug"
                args.yolo_roi    = subset_root / "yolo_split_cupROI"
                args.viz_out     = subset_root / "viz_labels"
                subset_used = True

        elif first_stage == "split":
            # Works for both single- and multi-dataset as long as args.labels_dir points at a flat dir of .txt
            s_labels, s_images, _stems = build_subset_for_split_stage(
                subset_root, _expand(args.labels_dir), _expand(args.images_root),
                n=args.subset_n, seed=args.subset_seed, copy=args.subset_copy
            )
            args.labels_dir  = s_labels
            args.images_root = s_images
            args.yolo_split  = subset_root / "yolo_split"
            args.yolo_aug    = subset_root / "yolo_split_aug"
            args.yolo_roi    = subset_root / "yolo_split_cupROI"
            args.viz_out     = subset_root / "viz_labels"
            subset_used = True
        else:
            print(f"[WARN] --subset_n is applied only when the first stage is 'masks' or 'split'. "
                  f"First stage here is '{first_stage}', so subset mode is ignored.")

    # ---------- Dispatch stages ----------
    for s in stages:
        print(f"\n========== [STAGE: {s}{' (subset)' if subset_used else ''}] ==========")
        if s == "masks":
            # If datasets: run multi-dataset path; else fall back to single
            if hasattr(args, "_raw_config") and cfg_has_datasets(args._raw_config):
                stage_masks_multi(args, defaults, args._raw_config)
            else:
                require_dir(args.images_root, "images_root")
                require_dir(_expand(args.disc_masks), "disc_masks")
                require_dir(_expand(args.cup_masks),  "cup_masks")
                stage_masks_to_boxes(args, defaults)

        elif s == "split":
            require_dir(args.images_root, "images_root")   # union if multi-dataset
            require_dir(args.labels_dir,  "labels_dir")    # union if multi-dataset
            stage_split_yolo(args, defaults)

        elif s == "augment":
            require_dir(args.yolo_split, "yolo_split")
            stage_augment(args, defaults)

        elif s == "disc_only":
            require_dir(args.yolo_split, "yolo_split")
            if args.disc_only_from_aug_train:
                require_dir(args.yolo_aug, "yolo_split_aug")
            stage_disc_only(args, defaults)

        elif s == "roi":
            require_dir(args.yolo_split, "yolo_split")
            stage_build_roi(args, defaults)

        elif s == "viz":
            require_dir(args.labels_dir,  "labels_dir")
            require_dir(args.images_root, "images_root")
            stage_viz(args, defaults)

        else:
            raise SystemExit(f("[ERR] Unknown stage: {s}"))

    print("\n[OK] Pipeline finished.")


if __name__ == "__main__":
    main()