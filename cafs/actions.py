from __future__ import annotations
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from typing import Sequence
import torch
from .groups import MeasurementGroup, largest_feasible_budget, real_fourier_orbits
FACE_HEIGHT = FACE_WIDTH = 256
FACE_CHANNELS = 3
MRI_READOUT = 640
MRI_PHASE_LINES = 368


@dataclass(frozen=True)
class BudgetTarget:
    domain: str
    requested_ratio: float
    requested_m: int
    achieved_m: int
    target_actions: int | None
    achieved_ratio: float


def _half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@lru_cache(maxsize=None)
def face_action_dictionary(
    height: int = FACE_HEIGHT,
    width: int = FACE_WIDTH,
    channels: int = FACE_CHANNELS,
) -> tuple[MeasurementGroup, ...]:
    """Return the frozen face orbit dictionary.

    Cached because record validation rebuilds it once per record: every
    element is an immutable ``MeasurementGroup`` inside a tuple, so callers
    cannot observe the sharing.
    """
    return real_fourier_orbits(height, width, channels)


def _mri_action_dictionary(
    phase_lines: int = MRI_PHASE_LINES,
    readout: int = MRI_READOUT,
) -> tuple[MeasurementGroup, ...]:
    return tuple(
        MeasurementGroup(line, (line,), 2 * readout)
        for line in range(phase_lines)
    )


@lru_cache(maxsize=None)
def target_budget(domain: str, ratio: float) -> BudgetTarget:
    if ratio not in {0.05, 0.10, 0.25}:
        raise ValueError("publication-v2 ratio must be 0.05, 0.10, or 0.25")
    if domain == "face":
        signal_size = FACE_CHANNELS * FACE_HEIGHT * FACE_WIDTH
        requested = math.floor(ratio * signal_size)
        achieved = largest_feasible_budget(
            requested, 0, face_action_dictionary()
        )
        return BudgetTarget(
            domain=domain,
            requested_ratio=ratio,
            requested_m=requested,
            achieved_m=achieved,
            target_actions=None,
            achieved_ratio=achieved / signal_size,
        )
    if domain == "mri":
        lines = _half_up(ratio * MRI_PHASE_LINES)
        achieved = lines * 2 * MRI_READOUT
        return BudgetTarget(
            domain=domain,
            requested_ratio=ratio,
            requested_m=achieved,
            achieved_m=achieved,
            target_actions=lines,
            achieved_ratio=lines / MRI_PHASE_LINES,
        )
    raise ValueError(f"unknown publication domain: {domain!r}")


def _ranked_by_cost(
    scores: torch.Tensor,
    actions: Sequence[MeasurementGroup],
) -> dict[int, tuple[MeasurementGroup, ...]]:
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not actions:
        return {}
    ids = [action.id for action in actions]
    if len(ids) != len(set(ids)) or min(ids) < 0 or max(ids) >= len(scores):
        raise ValueError("action IDs must be unique valid score indices")
    if not torch.isfinite(scores[torch.tensor(ids)]).all():
        raise ValueError("action scores must be finite")
    grouped: dict[int, list[MeasurementGroup]] = {}
    for action in actions:
        if action.cost <= 0:
            raise ValueError("action costs must be positive")
        grouped.setdefault(action.cost, []).append(action)
    return {
        cost: tuple(
            sorted(
                rows,
                key=lambda action: (-float(scores[action.id]), action.id),
            )
        )
        for cost, rows in grouped.items()
    }


def _count_combinations(
    grouped: dict[int, tuple[MeasurementGroup, ...]],
    total_cost: int,
    max_actions: int,
):
    costs = tuple(sorted(grouped))
    if not costs:
        return

    def visit(position: int, cost_left: int, count_left: int, counts: list[int]):
        cost = costs[position]
        available = min(len(grouped[cost]), count_left, cost_left // cost)
        if position == len(costs) - 1:
            if cost_left % cost:
                return
            count = cost_left // cost
            if count <= available:
                yield tuple(counts + [count])
            return
        for count in range(available + 1):
            yield from visit(
                position + 1,
                cost_left - count * cost,
                count_left - count,
                counts + [count],
            )

    yield from visit(0, total_cost, max_actions, [])


def select_exact_cost(
    scores: torch.Tensor,
    actions: Sequence[MeasurementGroup],
    *,
    remaining_cost: int,
    max_actions: int | None = None,
) -> tuple[int, ...]:
    """Select the maximum-score subset with exactly the requested cost."""
    if remaining_cost < 0:
        raise ValueError("remaining_cost must be non-negative")
    if remaining_cost == 0:
        return ()
    grouped = _ranked_by_cost(scores, actions)
    action_limit = len(actions) if max_actions is None else max_actions
    if action_limit < 0:
        raise ValueError("max_actions must be non-negative")
    costs = tuple(sorted(grouped))
    prefix = {
        cost: torch.cat(
            (
                torch.zeros(1, dtype=torch.float64),
                torch.tensor(
                    [float(scores[action.id]) for action in grouped[cost]],
                    dtype=torch.float64,
                ).cumsum(0),
            )
        )
        for cost in costs
    }
    best: tuple[float, tuple[int, ...]] | None = None
    for counts in _count_combinations(
        grouped, remaining_cost, action_limit
    ):
        selected = [
            action
            for cost, count in zip(costs, counts)
            for action in grouped[cost][:count]
        ]
        ordered = tuple(
            action.id
            for action in sorted(
                selected,
                key=lambda action: (-float(scores[action.id]), action.id),
            )
        )
        utility = sum(
            float(prefix[cost][count])
            for cost, count in zip(costs, counts)
        )
        candidate = (utility, tuple(-value for value in ordered))
        if best is None or candidate > (
            best[0], tuple(-value for value in best[1])
        ):
            best = (utility, ordered)
    if best is None:
        raise ValueError(f"no exact action subset reaches cost {remaining_cost}")
    return best[1]


def _gumbel_scores(base: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    uniform = torch.rand(base.shape, generator=generator).clamp_(1e-12, 1 - 1e-12)
    return base - torch.log(-torch.log(uniform))


def initial_actions(
    domain: str, target: BudgetTarget, *, repeat_seed: int
) -> tuple[int, ...]:
    """Return the image-independent variable-density initialization."""
    if target.domain != domain:
        raise ValueError("target domain differs from initialization domain")
    if domain == "face":
        actions = face_action_dictionary()
        seed_request = _half_up(0.2 * target.achieved_m)
        seed_target = largest_feasible_budget(seed_request, 0, actions)
        mandatory = (0,)
        remaining = seed_target - actions[0].cost
        base = torch.empty(len(actions), dtype=torch.float64)
        for action in actions:
            index = action.indices[0]
            ky, kx = divmod(index, FACE_WIDTH)
            fy = min(ky, FACE_HEIGHT - ky)
            fx = min(kx, FACE_WIDTH - kx)
            base[action.id] = -math.log1p(fy * fy + fx * fx)
        scores = _gumbel_scores(base, repeat_seed)
        unused = tuple(action for action in actions if action.id != 0)
        chosen = select_exact_cost(
            scores, unused, remaining_cost=remaining
        )
        return tuple(sorted(mandatory + chosen))
    if domain == "mri":
        actions = _mri_action_dictionary()
        target_lines = int(target.target_actions)
        seed_lines = max(4, _half_up(0.2 * target_lines))
        # `data.py` ifftshifts the loaded k-space, so DC is line 0 and the
        # Nyquist line is the middle of the index range. A centred convention
        # here makes the mandatory "calibration" lines the emptiest in k-space.
        central = (MRI_PHASE_LINES - 2, MRI_PHASE_LINES - 1, 0, 1)
        base = -torch.tensor(
            [
                min(line, MRI_PHASE_LINES - line)
                for line in range(MRI_PHASE_LINES)
            ],
            dtype=torch.float64,
        ) / 24
        scores = _gumbel_scores(base, repeat_seed)
        unused = tuple(
            action for action in actions if action.id not in central
        )
        chosen = select_exact_cost(
            scores,
            unused,
            remaining_cost=(seed_lines - 4) * 2 * MRI_READOUT,
        )
        return tuple(sorted(central + chosen))
    raise ValueError(f"unknown publication domain: {domain!r}")


def ads_event_steps(
    *, n_steps: int, n_events: int, window: tuple[float, float]
) -> tuple[int, ...]:
    """Place matched acquisition events across the ADS sampling window."""
    if n_steps <= 1 or n_events < 0:
        raise ValueError("positive n_steps and non-negative n_events required")
    if n_events == 0:
        return ()
    start = _half_up(window[0] * n_steps)
    stop = min(_half_up(window[1] * n_steps), n_steps - 1)
    if stop <= start or n_events > stop - start + 1:
        raise ValueError("ADS acquisition events do not fit the sampling window")
    if n_events == 1:
        return (stop,)
    span = Decimal(stop - start)
    positions = tuple(
        start
        + int(
            (span * Decimal(index) / Decimal(n_events - 1)).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
        )
        for index in range(n_events)
    )
    if len(set(positions)) != len(positions):
        raise ValueError("ADS event schedule contains duplicate steps")
    return positions
