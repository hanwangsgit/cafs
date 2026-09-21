from __future__ import annotations
import math
from dataclasses import dataclass, replace
import torch
from .groups import MeasurementGroup
from .operators import PhaseEncodingLineOp
from .actions import select_exact_cost
from .acquisition import PolicyContext, SelectionAccounting, run_feedback_policy, run_static_policy


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    cadence: str
    cm_steps: int


def spectral_energy_scores(
    operator, reconstruction: torch.Tensor
) -> torch.Tensor:
    """Return cost-normalized energy for every complete acquisition action."""
    values = operator.candidate_measurements(reconstruction)
    if values.ndim < 2:
        raise ValueError("candidate measurements must retain an action axis")
    coefficient_scores = values.abs().square().mean(
        dim=tuple(range(values.ndim - 1))
    )
    if coefficient_scores.ndim != 1:
        raise ValueError("candidate measurements produced invalid score shape")

    if hasattr(operator, "groups"):
        scores = torch.stack(
            [
                coefficient_scores[list(group.indices)].sum() / group.cost
                for group in operator.groups
            ]
        )
    elif isinstance(operator, PhaseEncodingLineOp):
        if coefficient_scores.numel() != operator.H * operator.W:
            raise ValueError("MRI spectrum does not match the audited geometry")
        scores = coefficient_scores.reshape(operator.H, operator.W).sum(dim=0)
        scores = scores / operator.group_costs.to(scores.device)
    elif coefficient_scores.numel() == len(operator.group_ids):
        ids = operator.group_ids.to(coefficient_scores.device)
        scores = coefficient_scores[ids]
        scores = scores / operator.group_costs.to(scores.device)
    else:
        raise ValueError(
            "operator does not expose a complete action-to-spectrum mapping"
        )

    if scores.ndim != 1 or len(scores) != len(operator.group_ids):
        raise ValueError("spectral scorer must return one score per action")
    if not torch.isfinite(scores).all():
        raise ValueError("spectral scores must be finite")
    return scores


def _cpu_scores(scores: torch.Tensor, operator) -> torch.Tensor:
    result = scores.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    ids = operator.group_ids.detach().to(device="cpu").tolist()
    if ids != list(range(len(ids))):
        raise ValueError("publication action IDs must be contiguous from zero")
    if len(result) != len(ids):
        raise ValueError("policy scorer must return one score per action")
    if not torch.isfinite(result).all():
        raise ValueError("policy scores must be finite")
    return result


def _available_actions(operator) -> tuple[MeasurementGroup, ...]:
    unused = set(operator.unused_group_ids().detach().cpu().tolist())
    costs = operator.group_costs.detach().cpu().tolist()
    return tuple(
        MeasurementGroup(action_id, (), int(costs[action_id]))
        for action_id in sorted(unused)
    )


def _select_event(
    context: PolicyContext, scores: torch.Tensor, event_cost: int
) -> tuple[int, ...]:
    scores = _cpu_scores(scores, context.operator)
    return select_exact_cost(
        scores,
        _available_actions(context.operator),
        remaining_cost=event_cost,
    )


def _density_logits(operator, domain: str) -> torch.Tensor:
    if domain == "mri":
        # Unshifted k-space: DC is line 0, Nyquist is the middle index. This
        # must match `initial_actions`, and the face branch below.
        width = operator.W
        return -torch.tensor(
            [min(line, width - line) / 24 for line in range(width)],
            dtype=torch.float64,
        )
    if domain == "face" and hasattr(operator, "groups"):
        logits = torch.empty(len(operator.groups), dtype=torch.float64)
        for group in operator.groups:
            index = group.indices[0]
            ky, kx = divmod(index, operator.W)
            fy = min(ky, operator.H - ky)
            fx = min(kx, operator.W - kx)
            logits[group.id] = -math.log1p(math.hypot(fy, fx))
        return logits
    raise ValueError("variable-density policy received the wrong operator")


def _available(operator) -> tuple[MeasurementGroup, ...]:
    unused = set(operator.unused_group_ids().detach().cpu().tolist())
    costs = operator.group_costs.detach().cpu().tolist()
    return tuple(
        MeasurementGroup(int(action_id), (), int(costs[action_id]))
        for action_id in sorted(unused)
    )


def _select(context: PolicyContext, scores: torch.Tensor, cost: int):
    values = scores.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    return select_exact_cost(values, _available(context.operator), remaining_cost=cost)


def _unpack_reconstruction(context: PolicyContext, steps: int):
    output = context.intermediate_reconstructor(
        context.operator, context.measurements, context.selector_generator
    )
    if isinstance(output, tuple) and len(output) == 2:
        reconstruction, accounting = output
        if not isinstance(accounting, SelectionAccounting):
            raise ValueError("intermediate accounting has the wrong type")
    else:
        reconstruction = output
        accounting = SelectionAccounting(
            intermediate_reconstruction_nfe=steps,
            forward_calls=steps,
            forward_sample_evaluations=steps,
        )
    if not isinstance(reconstruction, torch.Tensor):
        raise ValueError("intermediate reconstructor must return a tensor")
    return reconstruction, accounting


def _cm_policy(context: PolicyContext, spec: VariantSpec):
    frozen_scores = None
    first_accounting = None

    def select_event(current, event_cost):
        nonlocal frozen_scores, first_accounting
        if spec.cadence == "repeat" or frozen_scores is None:
            reconstruction, accounting = _unpack_reconstruction(
                current, spec.cm_steps
            )
            scores = spectral_energy_scores(current.operator, reconstruction)
            if spec.cadence == "once":
                frozen_scores = scores
                first_accounting = accounting
            else:
                return _select(current, scores, event_cost), accounting
        accounting = first_accounting or SelectionAccounting()
        first_accounting = None
        return _select(current, frozen_scores, event_cost), accounting

    return run_feedback_policy(context, select_event=select_event)


def _static_scores(context: PolicyContext, *, variable_density: bool):
    operator = context.operator
    if variable_density and hasattr(operator, "groups"):
        scores = torch.empty(len(operator.groups), dtype=torch.float64)
        for group in operator.groups:
            ky, kx = divmod(group.indices[0], operator.W)
            scores[group.id] = -math.log1p(
                math.hypot(min(ky, operator.H - ky), min(kx, operator.W - kx))
            )
    else:
        scores = torch.zeros(len(operator.group_ids), dtype=torch.float64)
    # The runner hands every policy a generator on the compute device, and
    # torch refuses to fill a tensor from a generator on another device. Draw
    # where the stream lives, then let `_select` move the result to the CPU
    # search. On CPU this is bit-identical to the unqualified call.
    generator = context.selector_generator
    uniforms = torch.rand(
        scores.shape,
        generator=generator,
        dtype=torch.float64,
        device=generator.device,
    ).clamp_(1e-12, 1 - 1e-12)
    return scores.to(uniforms.device) - torch.log(-torch.log(uniforms))


def _static_policy(context: PolicyContext, policy_id: str):
    scores = _static_scores(
        context, variable_density=(policy_id == "variable_density")
    )
    events = []
    planning = context
    for cost in context.event_costs:
        action_ids = _select(planning, scores, cost)
        events.append(action_ids)
        planning = replace(
            planning,
            operator=planning.operator.add_groups(action_ids),
            event_index=planning.event_index + 1,
        )
    return run_static_policy(context, planned_events=events)


def run_policy(context: PolicyContext, spec):
    """Run one new experiment variant without exposing target state."""
    if context.policy_id != spec.variant_id:
        raise ValueError("policy/context identifier mismatch")
    if spec.variant_id in {"uniform", "variable_density"}:
        return _static_policy(context, spec.variant_id)
    if not isinstance(spec, VariantSpec):
        raise ValueError("CM policy requires a VariantSpec")
    return _cm_policy(context, spec)
