from __future__ import annotations
import math, re
from dataclasses import dataclass
from typing import Callable, Sequence
import torch
from .operators import LinearOp
from .reconstruction import _tweedie
from .acquisition import PolicyContext, PolicyResult, SelectionAccounting, run_feedback_policy
from .actions import ads_event_steps
from .policies import _select_event
OFFICIAL_SIGMA = 50.0
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_MRI_ADS_SETTINGS = (16,50.0,0.85,10000,(0.005,0.25),True,"positive","whole_action_before_kernel")
_MRI_CONTROLLED_ADAPTATION = "shared_640x368_prior_action_space_and_cadence"
_FACE_CONTROLLED_ADAPTATION = "nonofficial_face_entropy_adaptation"
PosteriorSampler = Callable[..., tuple[torch.Tensor, SelectionAccounting]]
ADSAdvance = Callable[..., tuple[torch.Tensor, SelectionAccounting]]


def group_squared_distances(operator: LinearOp, particles: torch.Tensor,
                            chunk: int = 65536) -> torch.Tensor:
    """Pairwise per-group squared measurement distance.

    Returns (Np, Np, G) with
        d2[i, j, g] = sum_{l in group g} sum_c |y_i[c, l] - y_j[c, l]|^2

    This is the official aggregation order: distances are summed over the whole
    action *before* any kernel is applied (ActiveSampler.py:249).

    Uses d2 = |y_i|^2 + |y_j|^2 - 2 Re<y_i, y_j> so the pairwise tensor is only
    materialised at group resolution, not at coefficient resolution.
    """
    y = operator.candidate_measurements(particles)          # (Np, C', L)
    n_particles, _, n_coefficients = y.shape
    groups = _group_index(operator, n_coefficients, y.device)
    n_groups = int(groups.max().item()) + 1

    energy = torch.zeros(n_particles, n_groups, device=y.device)
    gram = torch.zeros(n_particles, n_particles, n_groups, device=y.device)

    for lo in range(0, n_coefficients, chunk):
        segment = y[:, :, lo:lo + chunk]                    # (Np, C', l)
        index = groups[lo:lo + chunk]
        energy.index_add_(
            1, index, segment.abs().pow(2).sum(dim=1)
        )
        # Re<y_i, y_j> summed over channels, per coefficient, then per group.
        cross = torch.einsum(
            "icl,jcl->ijl", segment.conj(), segment
        ).real                                              # (Np, Np, l)
        gram.index_add_(2, index, cross)

    return (energy.unsqueeze(1) + energy.unsqueeze(0) - 2.0 * gram).clamp_min(0)


def _group_index(operator: LinearOp, n_coefficients: int,
                 device) -> torch.Tensor:
    """Map each flat coefficient index to its group id."""
    if (
        hasattr(operator, "line_mask")
        and n_coefficients == operator.H * operator.W
        and len(operator.group_ids) == operator.W
    ):
        # Publication MRI actions are columns along the 368-wide phase axis.
        # Flattened FFT storage is row-major, so the column ID repeats once
        # per readout row.
        return torch.arange(operator.W, device=device).repeat(operator.H)
    if not hasattr(operator, "groups"):
        raise ValueError("operator must expose grouped actions")
    index = torch.full((n_coefficients,), -1, dtype=torch.long, device=device)
    for group in operator.groups:
        index[list(group.indices)] = group.id
    if int(index.min().item()) < 0:
        raise ValueError("operator groups do not cover the candidate dictionary")
    return index


def official_entropy_group_scores(operator: LinearOp, particles: torch.Tensor,
                                  sigma: float = OFFICIAL_SIGMA,
                                  zero_floor: bool = True,
                                  cost_normalize: bool = True
                                  ) -> torch.Tensor:
    """ADS Eq. 10 as implemented in ActiveSampler.select_column_entropy.

        g_ij  = exp(+ d2_ij / (2 sigma^2))
        E_i   = sum_j g_ij
        score = sum_i log E_i                       -> argmax

    Computed as sum_i logsumexp_j(d2_ij / 2 sigma^2) for stability: the
    official positive exponent overflows fp32 for d2 > ~1.8e6 at sigma=50.

    zero_floor subtracts the Np*log(Np) floor attained at zero disagreement.
    The official code does not (it can rely on argmax over equal-cost lines),
    but without it the score has a positive offset that makes division by a
    non-uniform action cost meaningless. For uniform-cost actions -- every
    Cartesian line -- it cannot change the ranking.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    d2 = group_squared_distances(operator, particles)        # (Np, Np, G)
    n_particles = d2.shape[0]
    scores = torch.logsumexp(d2 / (2.0 * sigma ** 2), dim=1).sum(dim=0)
    if zero_floor:
        scores = scores - n_particles * math.log(n_particles)
    if not cost_normalize:
        return scores
    return scores / operator.group_costs.to(scores.dtype)


@dataclass(frozen=True)
class ADSConfig:
    """Domain-bound ADS settings transcribed from one pinned source config."""

    domain: str
    source_domain: str
    source_config_path: str
    source_config_sha256: str
    source_commit: str
    n_particles: int
    sigma: float
    guidance: float
    n_steps: int
    window: tuple[float, float]
    hard_consistency: bool
    entropy_sign: str = "positive"
    aggregation: str = "whole_action_before_kernel"
    controlled_adaptation: str | None = None

    def __post_init__(self) -> None:
        if self.domain not in {"face", "mri"}:
            raise ValueError(f"unknown ADS domain: {self.domain!r}")
        if self.source_domain not in {"face", "mri"}:
            raise ValueError(
                "ADS source domain does not match the experiment domain"
            )
        if not self.source_config_path:
            raise ValueError("ADS source config path is required")
        if not _SHA256.fullmatch(self.source_config_sha256):
            raise ValueError("ADS source config requires an exact SHA-256")
        if not _GIT_SHA.fullmatch(self.source_commit):
            raise ValueError("ADS source commit requires a full Git SHA")
        if (
            not isinstance(self.n_particles, int)
            or self.n_particles < 2
            or not isinstance(self.n_steps, int)
            or self.n_steps <= 1
            or self.sigma <= 0
            or self.guidance < 0
        ):
            raise ValueError("ADS numeric settings are invalid")
        if (
            len(self.window) != 2
            or not 0 <= self.window[0] < self.window[1] <= 1
        ):
            raise ValueError("ADS sampling window must be an ordered fraction")

        settings = (
            self.n_particles,
            float(self.sigma),
            float(self.guidance),
            self.n_steps,
            tuple(self.window),
            self.hard_consistency,
            self.entropy_sign,
            self.aggregation,
        )
        if self.domain == "mri":
            if (
                self.source_domain != "mri"
                or self.controlled_adaptation
                not in {None, _MRI_CONTROLLED_ADAPTATION}
                or settings != _MRI_ADS_SETTINGS
            ):
                raise ValueError(
                    "official MRI ADS settings must be 16 particles, sigma "
                    "50, guidance 0.85, 10,000 steps, window (0.005, 0.25), "
                    "hard consistency, positive entropy, and whole-action "
                    "aggregation"
                )
            return
        if self.source_domain == "mri":
            if (
                self.controlled_adaptation != _FACE_CONTROLLED_ADAPTATION
                or settings != _MRI_ADS_SETTINGS
            ):
                raise ValueError(
                    "face ADS adaptation source does not match the exact "
                    "disclosed MRI-source settings"
                )
            return
        if self.controlled_adaptation is not None:
            raise ValueError(
                "source-aligned face ADS cannot declare an adaptation"
            )
        if (
            self.entropy_sign != "positive"
            or self.aggregation != "whole_action_before_kernel"
        ):
            raise ValueError(
                "official face ADS requires positive whole-action entropy"
            )


def _validate_event_costs(
    context: PolicyContext, event_costs: Sequence[int]
) -> tuple[int, ...]:
    declared = tuple(int(value) for value in event_costs)
    if declared != context.event_costs:
        raise ValueError("matched baseline event costs differ from the runner")
    return declared


def _validate_phase_accounting(
    accounting: SelectionAccounting,
) -> SelectionAccounting:
    if not isinstance(accounting, SelectionAccounting):
        raise ValueError("matched selector must return SelectionAccounting")
    if (
        accounting.event_count
        or accounting.action_count
        or accounting.wall_seconds
        or accounting.native_reconstruction_nfe
    ):
        raise ValueError(
            "matched selection cannot claim runner events, actions, wall "
            "time, or native reconstruction"
        )
    return accounting


def adasense_select(
    context: PolicyContext,
    *,
    posterior_sampler: PosteriorSampler,
    s: int = 8,
    ddrm_steps: int = 25,
    event_costs: Sequence[int],
) -> PolicyResult:
    """Run constrained AdaSense mask selection and no final DDRM estimate."""
    _validate_event_costs(context, event_costs)
    if s <= 1 or ddrm_steps <= 0:
        raise ValueError("AdaSense requires multiple samples and positive steps")

    def select(current: PolicyContext, event_cost: int):
        samples, accounting = posterior_sampler(
            operator=current.operator,
            measurements=current.measurements,
            n_samples=s,
            steps=ddrm_steps,
            generator=current.selector_generator,
        )
        accounting = _validate_phase_accounting(accounting)
        if not isinstance(samples, torch.Tensor) or len(samples) != s:
            raise ValueError("AdaSense sampler returned the wrong sample count")
        scores = current.operator.score_groups(samples)
        return _select_event(current, scores, event_cost), accounting

    return run_feedback_policy(context, select_event=select)


def ads_select(
    context: PolicyContext,
    *,
    particles: int,
    steps: int,
    window: tuple[float, float],
    event_costs: Sequence[int],
    config: ADSConfig,
    advance_to_event: ADSAdvance,
    entropy_scorer: Callable = official_entropy_group_scores,
) -> PolicyResult:
    """Run ADS through the final common acquisition and discard its suffix."""
    costs = _validate_event_costs(context, event_costs)
    if (
        particles != config.n_particles
        or steps != config.n_steps
        or tuple(window) != config.window
    ):
        raise ValueError("ADS runtime settings differ from the source config")
    schedule = ads_event_steps(
        n_steps=steps, n_events=len(costs), window=tuple(window)
    )

    def select(current: PolicyContext, event_cost: int):
        target_step = schedule[current.event_index]
        samples, accounting = advance_to_event(
            current,
            target_step=target_step,
            config=config,
        )
        accounting = _validate_phase_accounting(accounting)
        if (
            not isinstance(samples, torch.Tensor)
            or len(samples) != config.n_particles
        ):
            raise ValueError("ADS trajectory returned the wrong particle count")
        scores = entropy_scorer(
            current.operator,
            samples,
            sigma=config.sigma,
            zero_floor=True,
            cost_normalize=True,
        )
        return _select_event(current, scores, event_cost), accounting

    # run_feedback_policy returns immediately after the final callback. No
    # post-acquisition diffusion step or particle-mean reconstruction exists.
    return run_feedback_policy(context, select_event=select)


@dataclass
class DDRMPosteriorSampler:
    """Generator-local constrained DDRM samples used only for selection."""

    unet: object
    alphas_cumprod: torch.Tensor
    device: str | torch.device
    t_start: int = 999
    max_batch: int = 8
    clamp: tuple[float, float] | None = None
    zeta: float = 1.0

    def __post_init__(self) -> None:
        if self.clamp is not None:
            raise ValueError(
                "AdaSense efficient_generalized_steps does not clamp x0"
            )
        if self.zeta != 1.0:
            raise ValueError(
                "AdaSense efficient_generalized_steps requires exact "
                "orthogonal measurement replacement"
            )

    def __call__(
        self,
        *,
        operator,
        measurements: torch.Tensor,
        n_samples: int,
        steps: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, SelectionAccounting]:
        if n_samples <= 0 or steps <= 0 or self.max_batch <= 0:
            raise ValueError("DDRM sample, step, and batch counts must be positive")
        if self.t_start + 1 != len(self.alphas_cumprod):
            raise ValueError(
                "AdaSense t_start must identify the final training timestep"
            )
        skip = (self.t_start + 1) // steps
        if skip <= 0:
            raise ValueError("AdaSense steps exceed the training schedule")
        source_timesteps = tuple(range(0, self.t_start + 1, skip))
        if len(source_timesteps) != steps:
            raise ValueError(
                "AdaSense source timestep rule does not yield exactly the "
                "requested number of steps"
            )
        t_schedule = tuple(reversed(source_timesteps))
        outputs = []
        forward_calls = 0
        for lower in range(0, n_samples, self.max_batch):
            batch = min(self.max_batch, n_samples - lower)
            initial_noise = torch.randn(
                (batch, *operator.signal_shape),
                device=self.device,
                generator=generator,
            )
            initial_alpha = self.alphas_cumprod[t_schedule[0]].to(
                self.device
            )
            pinv_measurements = operator.map_correction(
                torch.zeros_like(initial_noise),
                measurements,
                zeta=self.zeta,
            )
            x = (
                initial_alpha.sqrt() * pinv_measurements
                + (1 - initial_alpha).sqrt() * initial_noise
            )
            for index, timestep in enumerate(t_schedule):
                alpha = self.alphas_cumprod[timestep].to(self.device)
                t_batch = torch.full(
                    (batch,),
                    timestep,
                    device=self.device,
                    dtype=torch.long,
                )
                with torch.no_grad():
                    epsilon = self.unet(x, t_batch).sample
                forward_calls += 1
                x0 = _tweedie(epsilon, x, alpha)
                next_alpha = (
                    torch.ones_like(alpha)
                    if index == len(t_schedule) - 1
                    else self.alphas_cumprod[t_schedule[index + 1]].to(
                        self.device
                    )
                )
                transition_noise = torch.randn(
                    x0.shape,
                    device=self.device,
                    dtype=x0.dtype,
                    generator=generator,
                )
                proposal = (
                    next_alpha.sqrt() * x0
                    + (1 - next_alpha).sqrt() * transition_noise
                )
                measurement_noise = torch.randn(
                    x0.shape,
                    device=self.device,
                    dtype=x0.dtype,
                    generator=generator,
                )
                noisy_measurements = (
                    next_alpha.sqrt() * measurements
                    + operator.measure(
                        (1 - next_alpha).sqrt() * measurement_noise
                    )
                )
                x = operator.map_correction(
                    proposal, noisy_measurements, zeta=self.zeta
                )
            outputs.append(x)
        return torch.cat(outputs), SelectionAccounting(
            probe_nfe=n_samples * steps,
            forward_calls=forward_calls,
            forward_sample_evaluations=n_samples * steps,
            max_ensemble_size=n_samples,
        )


class ADSDPSTrajectory:
    """Stateful, target-free DPS trajectory advanced only to ADS events."""

    def __init__(
        self,
        *,
        unet,
        alphas_cumprod: torch.Tensor,
        config: ADSConfig,
        signal_shape: tuple[int, ...],
        generator: torch.Generator,
        device: str | torch.device,
        t_start: int = 999,
        max_batch: int | None = None,
        clamp: tuple[float, float] | None = None,
        time_schedule: str = "truncated",
    ):
        if max_batch is not None and max_batch < config.n_particles:
            raise ValueError(
                "ADS guidance uses one joint particle-batch L2 norm; "
                "publication batching cannot split the source particle set"
            )
        self.unet = unet
        self.alphas_cumprod = alphas_cumprod
        self.config = config
        self.generator = generator
        self.device = torch.device(device)
        self.clamp = clamp
        self.time_schedule = time_schedule
        positions = torch.linspace(t_start, 0, config.n_steps + 1)
        # The network is a discrete-time DDPM with one trained embedding per
        # integer level, so it is always conditioned on an integer regardless
        # of schedule. Only the alpha-bar algebra can be refined.
        self._time = positions.long()
        if time_schedule == "truncated":
            # Source parity for every sealed record. With n_steps > len(
            # alphas_cumprod) this truncates many steps onto the same level,
            # and where alpha_next == alpha the DDIM update below collapses
            # to x <- x - gradient: guidance with no denoising transition.
            self._alpha_bar = None
        elif time_schedule == "interpolated":
            # One distinct noise level per step, by linear interpolation
            # between the trained levels. Upstream reaches 10,000 distinct
            # levels through a continuous cosine schedule, which a
            # discrete-time prior cannot reproduce exactly; this recovers the
            # distinct *transitions* without claiming to recover their model.
            floor = positions.floor()
            last = alphas_cumprod.numel() - 1
            # `positions` is a CPU index grid while the trained levels sit
            # wherever the prior was loaded, so the blend weight has to be
            # moved before the algebra. The gather tolerates a CPU index; the
            # multiply does not.
            weight = (positions - floor).to(
                device=alphas_cumprod.device, dtype=alphas_cumprod.dtype
            )
            self._alpha_bar = (
                alphas_cumprod[floor.long().clamp(0, last)] * (1 - weight)
                + alphas_cumprod[positions.ceil().long().clamp(0, last)]
                * weight
            )
        else:
            raise ValueError(
                f"unknown ADS time schedule: {time_schedule!r}"
            )
        self._next_step = 0
        self._x = torch.randn(
            (config.n_particles, *signal_shape),
            device=self.device,
            generator=generator,
        )

    def _alpha_at(self, step: int) -> torch.Tensor:
        if self._alpha_bar is None:
            return self.alphas_cumprod[int(self._time[step])].to(self.device)
        return self._alpha_bar[step].to(self.device)

    def advance(
        self,
        context: PolicyContext,
        *,
        target_step: int,
        config: ADSConfig,
    ) -> tuple[torch.Tensor, SelectionAccounting]:
        if config != self.config:
            raise ValueError("ADS trajectory config changed within a cell")
        if target_step < self._next_step or target_step >= config.n_steps:
            raise ValueError("ADS event step is stale or out of range")

        probe_nfe = 0
        forward_calls = 0
        backward_calls = 0
        predicted_images = self._x
        for step in range(self._next_step, target_step + 1):
            timestep = int(self._time[step])
            alpha = self._alpha_at(step)
            next_alpha = self._alpha_at(step + 1)
            xi = self._x.detach().requires_grad_(True)
            t_batch = torch.full(
                (config.n_particles,),
                timestep,
                device=self.device,
                dtype=torch.long,
            )
            epsilon = self.unet(xi, t_batch).sample
            estimate = _tweedie(epsilon, xi, alpha)
            residual = context.operator.measure(estimate) - context.measurements
            measurement_error = (
                config.guidance * residual.abs().square().sum().sqrt()
            )
            gradient = torch.autograd.grad(measurement_error, xi)[0]
            predicted_images = estimate.detach()
            if self.clamp is not None:
                predicted_images = predicted_images.clamp(*self.clamp)
            probe_nfe += 2 * config.n_particles
            forward_calls += 1
            backward_calls += 1
            with torch.no_grad():
                if step < config.n_steps - 1:
                    self._x = (
                        next_alpha.sqrt() * predicted_images
                        + (1 - next_alpha).sqrt() * epsilon.detach()
                        - gradient.detach()
                    )
                else:
                    self._x = predicted_images

        self._next_step = target_step + 1
        return predicted_images, SelectionAccounting(
            probe_nfe=probe_nfe,
            forward_calls=forward_calls,
            backward_calls=backward_calls,
            forward_sample_evaluations=(
                forward_calls * config.n_particles
            ),
            backward_sample_evaluations=(
                backward_calls * config.n_particles
            ),
            max_ensemble_size=config.n_particles,
        )
