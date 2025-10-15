def expand(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()

def load_image_bgr(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return im

def save_mask_png(path: Path, mask_bool: np.ndarray) -> None:
    ensure_dir(path.parent)
    skio.imsave(str(path), (mask_bool.astype(np.uint8) * 255), check_contrast=False)

def save_viz(path: Path, viz_bgr: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), viz_bgr)

def mask_vertical_height(mask: np.ndarray) -> int:
    ys = np.where(mask > 0)[0]
    if ys.size == 0:
        return 0
    return int(ys.max() - ys.min() + 1)

def corners_inside(mask: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box_xyxy
    H, W = mask.shape[:2]
    x1 = np.clip(x1, 0, W - 1); x2 = np.clip(x2, 0, W - 1)
    y1 = np.clip(y1, 0, H - 1); y2 = np.clip(y2, 0, H - 1)
    pts = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    for (x, y) in pts:
        if mask[int(y), int(x)] == 0:
            return False
    return True

def tight_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    return (x1, y1, x2 + 1, y2 + 1)  # half-open

def shrink_box_to_fit_mask(mask: np.ndarray,
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

def overlay_masks_and_boxes(
    img_bgr: np.ndarray,
    disc_mask: Optional[np.ndarray],
    cup_mask: Optional[np.ndarray],
    disc_box: Optional[Tuple[int,int,int,int]],
    cup_box: Optional[Tuple[int,int,int,int]],
    cdr_text: Optional[str] = None,
    alpha: float = 0.4
) -> np.ndarray:
    out = img_bgr.copy()

    # color overlays (BGR): disc=yellowish, cup=magenta-ish
    if disc_mask is not None:
        overlay = out.copy()
        overlay[disc_mask > 0] = (0, 255, 255)  # yellow
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    if cup_mask is not None:
        overlay = out.copy()
        overlay[cup_mask > 0] = (255, 0, 255)  # magenta
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    # boxes
    if disc_box is not None:
        x1, y1, x2, y2 = disc_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)  # yellow
    if cup_box is not None:
        x1, y1, x2, y2 = cup_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 200), 2)  # magenta

    # CDR text
    if cdr_text:
        cv2.putText(out, cdr_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(out, cdr_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return out