# device_utils.py
import torch

def ultralytics_device_arg() -> str:
    """
    Returns a string Ultralytics accepts for its 'device' argument.
    - CUDA available -> '0' (first GPU)
    - otherwise      -> 'cpu'
    """
    return "0" if torch.cuda.is_available() else "cpu"