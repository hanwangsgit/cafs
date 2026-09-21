import math
import torch
import torch.nn.functional as F


def _gaussian_window(
    size: int, sigma: float, channels: int, device, dtype
) -> torch.Tensor:
    coordinate = (
        torch.arange(size, device=device, dtype=dtype) - size // 2
    )
    one = torch.exp(-coordinate.square() / (2 * sigma**2))
    one = one / one.sum()
    two = torch.outer(one, one).reshape(1, 1, size, size)
    return two.expand(channels, 1, size, size).contiguous()


def local_ssim(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Wang-style local Gaussian SSIM with an explicit data range."""
    if (
        reconstruction.shape != target.shape
        or reconstruction.ndim != 4
        or reconstruction.shape[-2] < window_size
        or reconstruction.shape[-1] < window_size
    ):
        raise ValueError("SSIM inputs must be matching BCHW images")
    if data_range <= 0:
        raise ValueError("SSIM data range must be positive")
    reconstruction = reconstruction.float()
    target = target.to(reconstruction.device, dtype=reconstruction.dtype)
    channels = reconstruction.shape[1]
    window = _gaussian_window(
        window_size,
        sigma,
        channels,
        reconstruction.device,
        reconstruction.dtype,
    )
    mean_r = F.conv2d(reconstruction, window, groups=channels)
    mean_t = F.conv2d(target, window, groups=channels)
    var_r = (
        F.conv2d(reconstruction.square(), window, groups=channels)
        - mean_r.square()
    ).clamp_min(0)
    var_t = (
        F.conv2d(target.square(), window, groups=channels)
        - mean_t.square()
    ).clamp_min(0)
    covariance = (
        F.conv2d(reconstruction * target, window, groups=channels)
        - mean_r * mean_t
    )
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    value = (
        (2 * mean_r * mean_t + c1) * (2 * covariance + c2)
        / (
            (mean_r.square() + mean_t.square() + c1)
            * (var_r + var_t + c2)
        )
    )
    return float(value.mean())


def _psnr(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float,
) -> float:
    mse = float((reconstruction - target).square().mean())
    if mse == 0:
        return float("inf")
    return 10 * math.log10(data_range**2 / mse)


def _magnitude(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4 or value.shape[1] != 2:
        raise ValueError("MRI metrics require two-channel complex BCHW tensors")
    return value.square().sum(dim=1, keepdim=True).sqrt()


def _central_crop(value: torch.Tensor, size: int = 320) -> torch.Tensor:
    height, width = value.shape[-2:]
    if height < size or width < size:
        raise ValueError("MRI tensor is smaller than the central metric crop")
    top = (height - size) // 2
    left = (width - size) // 2
    return value[..., top : top + size, left : left + size]


def mri_metrics(
    reconstruction: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    """Unclamped fastMRI magnitude metrics on the central 320-by-320 crop."""
    if reconstruction.shape != target.shape:
        raise ValueError("MRI reconstruction and target shapes differ")
    truth = _central_crop(_magnitude(target))
    estimate = _central_crop(_magnitude(reconstruction))
    scale = float(truth.max())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("MRI target maximum must be finite and positive")
    truth = truth / scale
    estimate = estimate / scale
    denominator = float(truth.square().sum())
    if denominator <= 0:
        raise ValueError("MRI NMSE target energy must be positive")
    return {
        "psnr": _psnr(estimate, truth, data_range=1.0),
        "ssim": local_ssim(estimate, truth, data_range=1.0),
        "nmse": float((estimate - truth).square().sum()) / denominator,
    }


def face_quick_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Return the frozen face PSNR/SSIM pair without learned metric assets."""
    if (
        not isinstance(reconstruction, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or reconstruction.shape != target.shape
        or reconstruction.ndim != 4
        or reconstruction.shape[1] != 3
    ):
        raise ValueError(
            "quick face metrics require matching RGB BCHW tensors"
        )
    if not reconstruction.is_floating_point() or not target.is_floating_point():
        raise ValueError("quick face metrics require floating-point tensors")
    target = target.to(
        device=reconstruction.device,
        dtype=reconstruction.dtype,
    )
    mse = (reconstruction - target).square().mean()
    dynamic_range = torch.as_tensor(
        2.0, dtype=mse.dtype, device=mse.device
    )
    psnr = float(
        20 * torch.log10(dynamic_range)
        - 10 * torch.log10(mse.clamp_min(1e-12))
    )
    return {
        "psnr": psnr,
        "ssim": float(
            local_ssim(reconstruction, target, data_range=2.0)
        ),
    }
