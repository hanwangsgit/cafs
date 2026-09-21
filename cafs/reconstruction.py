from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable
import torch
from .operators import LinearOp, PixelMaskOp
from .acquisition import SelectionAccounting


def cm_reconstruct(
    cm_unet,
    alphas_cumprod: torch.Tensor,
    A_or_mask,
    y: torch.Tensor,
    k: int = 10,
    t_start: int = 800,
    t_end: int = 50,
    zeta: float = 1.0,
    verbose: bool = False,
    clamp: tuple | None = (-1.0, 1.0),
    generator: torch.Generator | None = None,
    trace_callback=None,
) -> torch.Tensor:
    """Reconstruct with ``k`` consistency-model predictions and corrections."""
    A = A_or_mask if isinstance(A_or_mask, LinearOp) else PixelMaskOp(A_or_mask)
    device = y.device if torch.is_tensor(y) else "cpu"
    zero_fill = A.map_correction(
        torch.zeros((1, *A.signal_shape), device=device), y, zeta=1.0
    )
    batch_size = zero_fill.shape[0]
    reconstruction = torch.zeros_like(zero_fill)

    alpha_start = alphas_cumprod[t_start].to(device)
    noise = torch.randn(
        zero_fill.shape,
        device=zero_fill.device,
        dtype=zero_fill.dtype,
        generator=generator,
    )
    x = (alpha_start.sqrt() * zero_fill
         + (1 - alpha_start).sqrt() * noise)
    timesteps = torch.linspace(t_start, t_end, k + 1).long()

    for index in range(k):
        t_current = int(timesteps[index].item())
        t_next = int(timesteps[index + 1].item())
        alpha_current = alphas_cumprod[t_current].to(device)
        alpha_next = alphas_cumprod[t_next].to(device)
        t_batch = torch.full(
            (batch_size,), t_current, device=device, dtype=torch.long
        )

        with torch.no_grad():
            epsilon = cm_unet(x, t_batch).sample
        prediction = (
            x - (1 - alpha_current).sqrt() * epsilon
        ) / alpha_current.sqrt()
        if clamp is not None:
            prediction = prediction.clamp(*clamp)

        corrected = A.map_correction(prediction, y, zeta=zeta)
        reconstruction = corrected
        if clamp is not None:
            reconstruction = reconstruction.clamp(*clamp)

        if trace_callback is not None:
            trace_callback(
                {
                    "index": index,
                    "timestep": t_current,
                    "next_timestep": t_next,
                    "prediction": prediction,
                    "corrected": corrected,
                    "reconstruction": reconstruction,
                }
            )

        if index < k - 1:
            noise = torch.randn(
                reconstruction.shape,
                device=reconstruction.device,
                dtype=reconstruction.dtype,
                generator=generator,
            )
            x = (alpha_next.sqrt() * reconstruction
                 + (1 - alpha_next).sqrt() * noise)
        else:
            x = reconstruction

        if verbose:
            print(f"  CM reconstruction {index + 1:2d}/{k}  t={t_current:4d}")

    return (reconstruction.clamp(*clamp)
            if clamp is not None else reconstruction)


def _tweedie(eps, x, ab):
    return (x - (1 - ab).sqrt() * eps) / ab.sqrt()


def ddrm_posterior_sample(unet, alphas_cumprod, A: LinearOp, y,
                          n_samples: int = 8, steps: int = 20,
                          t_start: int = 999, zeta: float = 1.0,
                          max_batch: int = 8, device: str = "cpu",
                          clamp: tuple | None = (-1.0, 1.0)):
    """Draw n_samples DDRM posterior samples. Returns (samples, nfe).

    For orthonormal-row A (FourierOp, PixelMaskOp, orthonormal DenseOp) the
    DDRM step is exactly map_correction(x0, y, zeta=1). With A.m == 0 this
    reduces to unconditional ancestral sampling (needed by the PCA arm's
    first round).
    """
    t_sched = torch.linspace(t_start, 0, steps + 1).long()
    out, nfe = [], 0
    for lo in range(0, n_samples, max_batch):
        b = min(max_batch, n_samples - lo)
        x = torch.randn(b, *A.signal_shape, device=device)
        for i in range(steps):
            t_cur = int(t_sched[i])
            ab_cur = alphas_cumprod[t_cur].to(device)
            t_batch = torch.full((b,), t_cur, device=device, dtype=torch.long)
            with torch.no_grad():
                eps = unet(x, t_batch).sample
            nfe += b
            x0 = _tweedie(eps, x, ab_cur)
            if clamp is not None:
                x0 = x0.clamp(*clamp)
            x0 = A.map_correction(x0, y, zeta=zeta)
            if i < steps - 1:
                ab_next = alphas_cumprod[int(t_sched[i + 1])].to(device)
                x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * torch.randn_like(x0)
            else:
                x = x0
        out.append(x)
    return torch.cat(out, dim=0), nfe


@dataclass(frozen=True)
class CMStepTrace:
    index: int
    timestep: int
    next_timestep: int
    residual_before_correction: float | None
    residual_after_correction: float | None
    residual_after_clipping: float | None


@dataclass(frozen=True)
class CMTrace:
    timesteps: tuple[int, ...]
    steps: tuple[CMStepTrace, ...]
    wall_seconds: float
    diagnostic_seconds: float
    peak_memory_bytes: int


class CMReconstructor:
    """Call the historical corrected CM while recording its actual schedule."""

    def __init__(
        self,
        *,
        unet,
        alphas_cumprod: torch.Tensor,
        steps: int,
        t_start: int,
        t_end: int,
        zeta: float,
        clamp: tuple[float, float] | None,
        synchronize: Callable[[], None] | None = None,
    ):
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("CM steps must be positive")
        self.unet = unet
        self.alphas_cumprod = alphas_cumprod
        self.steps = steps
        self.t_start = t_start
        self.t_end = t_end
        self.zeta = zeta
        self.clamp = clamp
        self.synchronize = synchronize or (lambda: None)
        self.traces: list[CMTrace] = []

    def __call__(self, operator, measurements, generator):
        rows: list[CMStepTrace] = []
        diagnostic_seconds = 0.0

        def capture(raw):
            nonlocal diagnostic_seconds
            residuals = {
                "residual_before_correction": None,
                "residual_after_correction": None,
                "residual_after_clipping": None,
            }
            if raw["index"] == self.steps - 1:
                self.synchronize()
                diagnostic_started = time.perf_counter()

                def residual(value):
                    return float(
                        (operator.measure(value) - measurements)
                        .abs().square().sum().sqrt()
                    )

                residuals = {
                    "residual_before_correction": residual(raw["prediction"]),
                    "residual_after_correction": residual(raw["corrected"]),
                    "residual_after_clipping": residual(raw["reconstruction"]),
                }
                self.synchronize()
                diagnostic_seconds += time.perf_counter() - diagnostic_started
            rows.append(CMStepTrace(
                index=raw["index"],
                timestep=raw["timestep"],
                next_timestep=raw["next_timestep"],
                **residuals,
            ))

        device = measurements.device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        self.synchronize()
        started = time.perf_counter()
        output = cm_reconstruct(
            self.unet,
            self.alphas_cumprod,
            operator,
            measurements,
            k=self.steps,
            t_start=self.t_start,
            t_end=self.t_end,
            zeta=self.zeta,
            clamp=self.clamp,
            generator=generator,
            trace_callback=capture,
        )
        self.synchronize()
        elapsed = max(0.0, time.perf_counter() - started - diagnostic_seconds)
        peak = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        trace = CMTrace(
            timesteps=tuple(row.timestep for row in rows),
            steps=tuple(rows),
            wall_seconds=elapsed,
            diagnostic_seconds=diagnostic_seconds,
            peak_memory_bytes=peak,
        )
        self.traces.append(trace)
        return output, SelectionAccounting(
            intermediate_reconstruction_nfe=self.steps,
            forward_calls=self.steps,
            forward_sample_evaluations=self.steps,
        )
