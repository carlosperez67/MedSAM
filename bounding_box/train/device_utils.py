# device_utils.py

# device_utils.py
import os
from torch.cuda import is_available, device_count


def ultralytics_device_arg() -> str:
    """
    Returns a device argument for Ultralytics .train():
      - If YOLO_DEVICES is set (e.g. "0,1" or "0,1,2,3"), return it (enables DDP).
      - Else if CUDA is visible, return "0".
      - Else return "cpu".
    """
    env = os.getenv("YOLO_DEVICES")
    if env:
        return env  # Ultralytics treats "0,1" as multi-GPU (DDP)
    # simple fallback
    if is_available() and device_count() > 0:
        return "0"
    return "cpu"