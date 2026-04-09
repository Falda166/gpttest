import warnings

import torch


def configure_runtime(quiet_warnings: bool = True, enable_tf32: bool = True):
    if enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if quiet_warnings:
        warnings.filterwarnings(
            "ignore",
            message=r"TensorFloat-32 \(TF32\) has been disabled.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"std\(\): degrees of freedom is <= 0.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Found keys that are not in the model state dict.*",
            category=UserWarning,
        )


def resolve_device():
    has_cuda = torch.cuda.is_available()
    device_torch = torch.device("cuda" if has_cuda else "cpu")
    device_str = "cuda" if has_cuda else "cpu"
    compute_type = "float16" if has_cuda else "int8"
    return device_torch, device_str, compute_type
