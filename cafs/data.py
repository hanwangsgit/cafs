import hashlib
import json
import torch
MRI_KSPACE_NORM = 7.072103529760345e-07
MRI_NORMALIZING_FACTOR = 0.3040714


def mri_target_from_kspace(kspace) -> torch.Tensor:
    """Transform one fastMRI slice into float32 real/imaginary channels.

    Uses NumPy inverse FFT in double precision with the recorded shift and
    normalization convention. The data loader checks the resulting tensor
    against the sample manifest's hash."""
    import numpy as np

    stacked = np.stack([kspace.real, kspace.imag], axis=-1)
    shifted = np.fft.ifftshift(stacked, axes=(-3, -2))
    spectrum = shifted[..., 0] + 1j * shifted[..., 1]
    inverse = np.fft.ifftn(spectrum, axes=(-2, -1), norm="backward")
    target = np.stack([inverse.real, inverse.imag], axis=-1)
    target = np.fft.ifftshift(target, axes=(-3, -2))
    target = target / MRI_KSPACE_NORM / MRI_NORMALIZING_FACTOR
    return torch.from_numpy(
        np.ascontiguousarray(target)
    ).permute(2, 0, 1).float()


def load_mri_slice_deterministic(path, slice_index: int) -> torch.Tensor:
    """Read one slice from an HDF5 container and transform it reproducibly."""
    import h5py

    with h5py.File(path, "r") as data:
        if "kspace" not in data:
            raise ValueError(f"fastMRI volume has no kspace: {path}")
        kspace = data["kspace"][int(slice_index)]
    return mri_target_from_kspace(kspace)


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and exact contiguous CPU bytes."""
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    raw = value.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(header + raw).hexdigest()
