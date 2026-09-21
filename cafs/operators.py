from __future__ import annotations
from abc import ABC, abstractmethod
from functools import lru_cache
import torch
from .groups import real_fourier_orbits


class LinearOp(ABC):
    """Abstract linear operator A : R^{B×C×H×W} -> measurement space.

    Subclasses must populate self.C, self.H, self.W in __init__.
    """

    C: int
    H: int
    W: int

    @property
    @abstractmethod
    def m(self) -> int:
        """Number of scalar real measurements."""

    @property
    def signal_shape(self) -> tuple:
        return (self.C, self.H, self.W)

    @property
    def achieved_m(self) -> int:
        return self.m

    @property
    def achieved_ratio(self) -> float:
        return self.m / (self.C * self.H * self.W)

    @property
    def meas_per_dict_row(self) -> int:
        """Real measurements contributed per selected dictionary index.

        Approximate for Hermitian-symmetric ops, where score maps are mirror-
        symmetric so top-k indices arrive in conjugate pairs. Used to trim the
        final acquisition round to the measurement budget.
        """
        return 1

    def candidate_measurements(self, x0: torch.Tensor) -> torch.Tensor:
        """Simulate the FULL-dictionary measurement of image batch x0.

        Returns (B, C', L) with L indexed in dictionary order — the per-
        candidate view that ADS entropy scoring compares across particles.
        """
        return x0.reshape(x0.shape[0], x0.shape[1], -1)

    @abstractmethod
    def measure(self, x: torch.Tensor) -> torch.Tensor:
        """Apply A. Return value lives in the operator's native measurement
        space (image-shaped for PixelMaskOp, complex spectrum for FourierOp)."""

    @abstractmethod
    def map_correction(self, x: torch.Tensor, y: torch.Tensor,
                       zeta: float = 1.0) -> torch.Tensor:
        """Return x - zeta * Aᵀ(A x - y), in image space (B, C, H, W)."""

    @abstractmethod
    def score_candidates(self, x_samples: torch.Tensor) -> torch.Tensor:
        """Posterior variance of each dictionary element from K MC samples.

        Args:
            x_samples: (K, C, H, W) — K Monte Carlo posterior samples of x_hat.

        Returns:
            scores: 1-D tensor of length |D|, indexed in dictionary order.
                    Higher = pick this row next.
        """

    @abstractmethod
    def add_rows(self, indices: torch.Tensor) -> "LinearOp":
        """Return a NEW operator with rows from D[indices] appended.
        `indices` is a 1-D LongTensor of dictionary-element ids."""

    @abstractmethod
    def unused_dictionary_indices(self) -> torch.Tensor:
        """Indices of dictionary elements NOT yet in A."""

    @property
    def group_ids(self) -> torch.Tensor:
        """Stable action IDs; legacy operators treat each row as one group."""
        return torch.arange(self.H * self.W, device=self.device)

    @property
    def group_costs(self) -> torch.Tensor:
        return torch.full((self.H * self.W,), self.meas_per_dict_row,
                          dtype=torch.long, device=self.device)

    @property
    def selected_group_ids(self) -> torch.Tensor:
        unused = torch.zeros(self.H * self.W, dtype=torch.bool,
                             device=self.device)
        unused[self.unused_group_ids()] = True
        return self.group_ids[~unused]

    def unused_group_ids(self) -> torch.Tensor:
        return self.unused_dictionary_indices()

    def score_groups(self, samples: torch.Tensor) -> torch.Tensor:
        return self.score_candidates(samples) / self.group_costs

    def add_groups(self, ids) -> "LinearOp":
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        return self.add_rows(ids)

    def serialize_actions(self) -> dict:
        selected = self.selected_group_ids.tolist()
        return {
            "group_ids": selected,
            "group_costs": self.group_costs[self.selected_group_ids].tolist(),
            "requested_m": self.m,
            "achieved_m": self.achieved_m,
            "achieved_ratio": self.achieved_ratio,
        }


class PixelMaskOp(LinearOp):
    """A x = mask ⊙ x. Dictionary D = unit one-hots per (h, w) location.

    A single dictionary row measures one *pixel location* (all C channels
    together) — so |D| = H*W and m = C * |mask=1|.
    """

    def __init__(self, mask: torch.Tensor, n_channels: int = 3):
        assert mask.dim() == 4 and mask.shape[:2] == (1, 1), \
            f"mask must be (1,1,H,W), got {tuple(mask.shape)}"
        assert ((mask == 0) | (mask == 1)).all(), "mask must be binary"
        self.mask = mask.float()
        self.C = n_channels
        self.H, self.W = mask.shape[-2:]

    @property
    def m(self) -> int:
        return self.C * int(self.mask.sum().item())

    @property
    def device(self):
        return self.mask.device

    @property
    def meas_per_dict_row(self) -> int:
        return self.C                    # one pixel location, all C channels

    def measure(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask * x

    def map_correction(self, x, y, zeta=1.0):
        # Aᵀ(A x - y) = mask·(mask·x - y) = mask·x - y  (since mask·y == y)
        return x - zeta * (self.mask * x - y)

    def score_candidates(self, x_samples):
        # var of pixel intensity across K samples, averaged across channels
        var = x_samples.var(dim=0)              # (C, H, W)
        score_map = var.mean(dim=0)             # (H, W)
        return score_map.view(-1)               # (H*W,)

    def add_rows(self, indices):
        new_mask = self.mask.clone()
        h_idx = indices // self.W
        w_idx = indices %  self.W
        new_mask[0, 0, h_idx, w_idx] = 1.0
        return PixelMaskOp(new_mask, self.C)

    def unused_dictionary_indices(self):
        return (1 - self.mask.view(-1)).bool().nonzero(as_tuple=True)[0]


def hermitian_symmetrize(freq_mask: torch.Tensor) -> torch.Tensor:
    """Make a 2-D freq mask Hermitian symmetric: M[k] selected ⇔ M[-k] selected.

    Ensures Aᵀ A acts purely on the real-image subspace (no spurious imag).
    """
    # M(k_y, k_x) and M(-k_y, -k_x) — torch.fft uses standard ordering so
    # negation corresponds to row/col flip with index 0 fixed.
    flipped = torch.roll(freq_mask.flip(dims=(0, 1)), shifts=(1, 1), dims=(0, 1))
    return freq_mask | flipped


@lru_cache(maxsize=16)
def _fourier_orbit_tables(height: int, width: int, channels: int, device_key: str):
    """Orbit lookup tables for an ``H x W`` real-image FFT, cached per shape/device.

    Rebuilding these dominated oracle runs: every ``FourierOp`` construction walked
    all ~32k orbits in Python, and the oracle constructs one operator per candidate,
    so this cost roughly ten times the GPU reconstruction it fed. The tables depend
    only on the shape and channel count, never on the mask.

    Returns the orbit tuple, an id lookup, the orbit id owning each flat bin, the
    number of bins per orbit, and the per-orbit cost.
    """
    groups = real_fourier_orbits(height, width, channels)
    orbit_of_bin = torch.empty(height * width, dtype=torch.long)
    orbit_size = torch.zeros(len(groups), dtype=torch.long)
    for group in groups:
        indices = torch.as_tensor(group.indices, dtype=torch.long)
        orbit_of_bin[indices] = group.id
        orbit_size[group.id] = indices.numel()
    costs = torch.tensor([group.cost for group in groups], dtype=torch.long)
    device = torch.device(device_key)
    return (
        groups,
        {group.id: group for group in groups},
        orbit_of_bin.to(device),
        orbit_size.to(device),
        costs.to(device),
    )


class FourierOp(LinearOp):
    """A x = M_f ⊙ FFT2(x). Dictionary D = all (k_y, k_x) frequencies.

    The freq_mask is Hermitian-symmetric so the real-image projection is exact.
    Each *unique* selected frequency pair contributes 2 real measurements
    (real + imag); the DC and Nyquist bins are purely real (1 each).
    """

    def __init__(self, freq_mask: torch.Tensor, n_channels: int = 3,
                 symmetrize: bool = True, requested_m: int | None = None):
        assert freq_mask.dim() == 2 and freq_mask.dtype == torch.bool, \
            f"freq_mask must be (H,W) bool, got {freq_mask.shape}/{freq_mask.dtype}"
        if symmetrize:
            freq_mask = hermitian_symmetrize(freq_mask)
        self.freq_mask = freq_mask
        self.C = n_channels
        self.H, self.W = freq_mask.shape
        (
            self.groups,
            self._group_by_id,
            self._orbit_of_bin,
            self._orbit_size,
            self._group_costs,
        ) = _fourier_orbit_tables(self.H, self.W, self.C, str(self.freq_mask.device))

        # Per-orbit selected-bin counts, vectorized. An orbit is fully selected
        # when its count equals its size and partially selected when the count is
        # strictly between zero and its size, which is exactly the condition the
        # per-group `any() and not all()` loop used to test one orbit at a time.
        # freq_mask is never mutated after construction, so this is reused by
        # selected_group_ids rather than recomputed.
        flat_mask = self.freq_mask.view(-1)
        counts = torch.zeros_like(self._orbit_size)
        counts.scatter_add_(0, self._orbit_of_bin, flat_mask.long())
        if bool(((counts > 0) & (counts < self._orbit_size)).any()):
            raise ValueError(
                "Fourier masks must contain complete conjugate orbits"
            )
        self._orbit_counts = counts
        self.requested_m = self.m if requested_m is None else int(requested_m)

    @property
    def m(self) -> int:
        # A complete conjugate pair stores two bins but contains two independent
        # real coefficients per channel; self-conjugate bins contain one.
        return self.C * int(self.freq_mask.sum().item())

    @property
    def device(self):
        return self.freq_mask.device

    @property
    def meas_per_dict_row(self) -> int:
        # The legacy row adapter exposes one representative per orbit below;
        # generic representatives therefore add a complete 2C-cost pair.
        return 2 * self.C

    def candidate_measurements(self, x0: torch.Tensor) -> torch.Tensor:
        X = torch.fft.fft2(x0, norm="ortho")
        return X.reshape(x0.shape[0], self.C, -1)

    def measure(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.fft.fft2(x, norm="ortho")
        return X * self.freq_mask                  # (B, C, H, W) complex

    def map_correction(self, x, y, zeta=1.0):
        X = torch.fft.fft2(x, norm="ortho")
        residual_freq  = self.freq_mask * (X - y)
        residual_image = torch.fft.ifft2(residual_freq, norm="ortho").real
        return x - zeta * residual_image

    def score_candidates(self, x_samples):
        # For each frequency d_j, Var_i[⟨d_j, x_i⟩] = Var_i[FFT(x_i)[k_j]]
        # Compute FFT of each sample, take per-bin complex variance.
        X = torch.fft.fft2(x_samples, norm="ortho")           # (K, C, H, W) complex
        # Variance of a complex random variable Z is E|Z - EZ|², a real scalar.
        mean = X.mean(dim=0, keepdim=True)
        var  = (X - mean).abs().pow(2).mean(dim=0)            # (C, H, W) real
        score_map = var.mean(dim=0)                            # (H, W)
        return score_map.view(-1)                              # (H*W,)

    @property
    def group_ids(self) -> torch.Tensor:
        return torch.arange(len(self.groups), device=self.device)

    @property
    def group_costs(self) -> torch.Tensor:
        # Cloned: the cached table is shared by every operator of this shape, and
        # the previous implementation built a fresh tensor per call. Returning the
        # cache directly would let a caller alias shared state.
        return self._group_costs.clone()

    @property
    def selected_group_ids(self) -> torch.Tensor:
        return torch.nonzero(
            self._orbit_counts == self._orbit_size, as_tuple=False
        ).view(-1)

    def unused_group_ids(self) -> torch.Tensor:
        selected = torch.zeros(len(self.groups), dtype=torch.bool,
                               device=self.device)
        selected[self.selected_group_ids] = True
        return self.group_ids[~selected]

    def score_groups(self, samples: torch.Tensor) -> torch.Tensor:
        coefficient_scores = self.score_candidates(samples)
        totals = torch.zeros(
            len(self.groups),
            dtype=coefficient_scores.dtype,
            device=coefficient_scores.device,
        )
        totals.scatter_add_(
            0, self._orbit_of_bin.to(coefficient_scores.device), coefficient_scores
        )
        return totals / self._group_costs.to(coefficient_scores.device)

    def add_groups(self, ids) -> "FourierOp":
        requested_ids = torch.as_tensor(ids, dtype=torch.long).view(-1).tolist()
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("duplicate group id")

        selected_ids = set(self.selected_group_ids.tolist())
        new_mask = self.freq_mask.clone().view(-1)
        for group_id in requested_ids:
            if group_id not in self._group_by_id:
                raise ValueError(f"unknown group id {group_id}")
            if group_id in selected_ids:
                raise ValueError(f"group id {group_id} is already selected")
            new_mask[list(self._group_by_id[group_id].indices)] = True

        new_mask = new_mask.view(self.H, self.W)
        achieved_m = self.C * int(new_mask.sum().item())
        return FourierOp(new_mask, self.C, symmetrize=False,
                         requested_m=max(self.requested_m, achieved_m))

    def serialize_actions(self) -> dict:
        selected = self.selected_group_ids.tolist()
        groups = [self._group_by_id[group_id] for group_id in selected]
        return {
            "group_ids": selected,
            "flattened_fft_indices": [
                index for group in groups for index in group.indices
            ],
            "group_costs": [group.cost for group in groups],
            "requested_m": self.requested_m,
            "achieved_m": self.achieved_m,
            "achieved_ratio": self.achieved_ratio,
        }

    def add_rows(self, indices):
        new_mask = self.freq_mask.clone()
        ky = indices // self.W
        kx = indices %  self.W
        new_mask[ky, kx] = True
        # Symmetrize newly added rows too — keeps the operator self-adjoint on
        # the real-image subspace.
        return FourierOp(new_mask, self.C, symmetrize=True)

    def unused_dictionary_indices(self):
        flat_mask = self.freq_mask.view(-1)
        representatives = [
            group.indices[0]
            for group in self.groups
            if not flat_mask[list(group.indices)].all()
        ]
        return torch.tensor(representatives, dtype=torch.long,
                            device=self.device)


class PhaseEncodingLineOp(LinearOp):
    """Complex MRI FFT sampling by columns along the 368-line phase axis."""

    def __init__(
        self,
        line_mask: torch.Tensor,
        *,
        calibration_lines: tuple[int, ...],
        readout_size: int = 640,
        requested_m: int | None = None,
    ):
        if line_mask.ndim != 1 or line_mask.dtype != torch.bool:
            raise ValueError("line_mask must be a one-dimensional bool tensor")
        if readout_size <= 0:
            raise ValueError("readout_size must be positive")
        self.line_mask = line_mask.clone()
        self.C = 2
        self.H = int(readout_size)
        self.W = len(line_mask)
        self.freq_mask = self.line_mask.view(1, self.W).expand(
            self.H, self.W
        ).clone()
        self._calibration_lines = tuple(calibration_lines)
        if len(self._calibration_lines) != len(set(self._calibration_lines)):
            raise ValueError("calibration lines must be unique")
        if any(line < 0 or line >= self.W for line in self._calibration_lines):
            raise ValueError("calibration line is out of range")
        if self._calibration_lines and not self.line_mask[
            list(self._calibration_lines)
        ].all():
            raise ValueError("calibration lines must be selected")
        self.requested_m = self.m if requested_m is None else int(requested_m)

    @property
    def calibration_lines(self) -> tuple[int, ...]:
        return self._calibration_lines

    @property
    def device(self):
        return self.line_mask.device

    @property
    def m(self) -> int:
        return 2 * self.H * int(self.line_mask.sum().item())

    @property
    def group_ids(self) -> torch.Tensor:
        return torch.arange(self.W, device=self.device)

    @property
    def group_costs(self) -> torch.Tensor:
        return torch.full(
            (self.W,), 2 * self.H, dtype=torch.long, device=self.device
        )

    @property
    def selected_group_ids(self) -> torch.Tensor:
        return self.line_mask.nonzero(as_tuple=True)[0]

    def unused_group_ids(self) -> torch.Tensor:
        return (~self.line_mask).nonzero(as_tuple=True)[0]

    @staticmethod
    def _as_complex(value: torch.Tensor) -> torch.Tensor:
        return torch.complex(value[:, 0], value[:, 1])

    @staticmethod
    def _as_channels(value: torch.Tensor) -> torch.Tensor:
        return torch.stack((value.real, value.imag), dim=1)

    def candidate_measurements(self, x0: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft2(self._as_complex(x0), norm="ortho")
        return spectrum.reshape(x0.shape[0], 1, -1)

    def measure(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft2(self._as_complex(x), norm="ortho")
        return spectrum * self.freq_mask

    def map_correction(
        self, x: torch.Tensor, y: torch.Tensor, zeta: float = 1.0
    ) -> torch.Tensor:
        spectrum = torch.fft.fft2(self._as_complex(x), norm="ortho")
        residual = torch.fft.ifft2(
            self.freq_mask * (spectrum - y), norm="ortho"
        )
        return x - zeta * self._as_channels(residual)

    def score_candidates(self, x_samples: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft2(
            self._as_complex(x_samples), norm="ortho"
        )
        centered = spectrum - spectrum.mean(dim=0, keepdim=True)
        return centered.abs().square().mean(dim=0).reshape(-1)

    def score_groups(self, samples: torch.Tensor) -> torch.Tensor:
        coefficient = self.score_candidates(samples).view(self.H, self.W)
        return coefficient.sum(dim=0) / self.group_costs

    def add_groups(self, ids) -> "PhaseEncodingLineOp":
        requested = torch.as_tensor(ids, dtype=torch.long).view(-1).tolist()
        if len(requested) != len(set(requested)):
            raise ValueError("duplicate phase-encoding action")
        new_mask = self.line_mask.clone()
        for line in requested:
            if line < 0 or line >= self.W:
                raise ValueError(f"unknown phase-encoding line {line}")
            if new_mask[line]:
                raise ValueError(f"phase-encoding line {line} is already selected")
            new_mask[line] = True
        achieved = 2 * self.H * int(new_mask.sum().item())
        return PhaseEncodingLineOp(
            new_mask,
            calibration_lines=self.calibration_lines,
            readout_size=self.H,
            requested_m=max(self.requested_m, achieved),
        )

    def remove_groups(self, ids) -> "PhaseEncodingLineOp":
        requested = torch.as_tensor(ids, dtype=torch.long).view(-1).tolist()
        if any(line in self.calibration_lines for line in requested):
            raise ValueError("calibration lines cannot be removed")
        new_mask = self.line_mask.clone()
        new_mask[requested] = False
        return PhaseEncodingLineOp(
            new_mask,
            calibration_lines=self.calibration_lines,
            readout_size=self.H,
            requested_m=self.requested_m,
        )

    def add_rows(self, indices):
        raise NotImplementedError(
            "PhaseEncodingLineOp accepts phase-encoding actions, not rows"
        )

    def unused_dictionary_indices(self):
        raise NotImplementedError(
            "PhaseEncodingLineOp accepts phase-encoding actions, not rows"
        )

    def serialize_actions(self) -> dict:
        selected = self.selected_group_ids.tolist()
        return {
            "group_ids": selected,
            "phase_encoding_lines": selected,
            "group_costs": [2 * self.H] * len(selected),
            "calibration_lines": list(self.calibration_lines),
            "requested_m": self.requested_m,
            "achieved_m": self.achieved_m,
            "achieved_ratio": self.achieved_ratio,
        }
