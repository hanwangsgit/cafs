import torch
MRI_KSPACE_NORM = 7.072103529760345e-07
MRI_NORMALIZING_FACTOR = 0.3040714


def mri_target_from_kspace(kspace) -> torch.Tensor:
    """Reconstruct one fastMRI slice reproducibly on any CPU.

    `fastmri_data._load_slice` runs the inverse transform through
    `torch.fft`, which on CPU calls MKL kernels selected from the host's
    vector extensions. The same slice therefore comes out bit-different on
    different machines -- measured at 2.6e-7 relative between a login node and
    a compute node, which is float32 rounding and scientifically inert, but
    enough to change a SHA-256 completely. That made a bound MRI tensor
    identity verifiable only on the machine that wrote it, and it failed every
    MRI shard of the first array.

    NumPy's pocketfft has no such dispatch and computes in double precision,
    so it is bit-identical across hosts. The controlled protocol loads through
    it; `fastmri_data` is left alone so training and the earlier benchmarks
    still reproduce their own bytes exactly.
    """
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
