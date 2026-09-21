from __future__ import annotations
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Protocol, Sequence
import torch
from .operators import LinearOp


@dataclass(frozen=True)
class SelectionAccounting:
    intermediate_reconstruction_nfe: int = 0
    probe_nfe: int = 0
    forward_calls: int = 0
    backward_calls: int = 0
    forward_sample_evaluations: int = 0
    backward_sample_evaluations: int = 0
    max_ensemble_size: int = 0
    event_count: int = 0
    action_count: int = 0
    wall_seconds: float = 0.0
    native_reconstruction_nfe: int = 0

    def __post_init__(self) -> None:
        counts = (
            self.intermediate_reconstruction_nfe,
            self.probe_nfe,
            self.forward_calls,
            self.backward_calls,
            self.forward_sample_evaluations,
            self.backward_sample_evaluations,
            self.max_ensemble_size,
            self.event_count,
            self.action_count,
            self.native_reconstruction_nfe,
        )
        if any(not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("selection accounting counts must be non-negative integers")
        if self.wall_seconds < 0:
            raise ValueError("selection wall time must be non-negative")

    def __add__(self, other: "SelectionAccounting") -> "SelectionAccounting":
        if not isinstance(other, SelectionAccounting):
            return NotImplemented
        return SelectionAccounting(
            intermediate_reconstruction_nfe=(
                self.intermediate_reconstruction_nfe
                + other.intermediate_reconstruction_nfe
            ),
            probe_nfe=self.probe_nfe + other.probe_nfe,
            forward_calls=self.forward_calls + other.forward_calls,
            backward_calls=self.backward_calls + other.backward_calls,
            forward_sample_evaluations=(
                self.forward_sample_evaluations
                + other.forward_sample_evaluations
            ),
            backward_sample_evaluations=(
                self.backward_sample_evaluations
                + other.backward_sample_evaluations
            ),
            max_ensemble_size=max(
                self.max_ensemble_size, other.max_ensemble_size
            ),
            event_count=self.event_count + other.event_count,
            action_count=self.action_count + other.action_count,
            wall_seconds=self.wall_seconds + other.wall_seconds,
            native_reconstruction_nfe=(
                self.native_reconstruction_nfe
                + other.native_reconstruction_nfe
            ),
        )


class _RequestActions(Protocol):
    def __call__(
        self, context: "PolicyContext", action_ids: tuple[int, ...]
    ) -> "PolicyContext":
        ...


@dataclass(frozen=True)
class _AcquisitionCapability:
    request: _RequestActions = field(repr=False, compare=False)

    def acquire(
        self, context: "PolicyContext", action_ids: Sequence[int]
    ) -> "PolicyContext":
        return self.request(context, tuple(int(value) for value in action_ids))


@dataclass(frozen=True)
class PolicyContext:
    policy_id: str
    operator: LinearOp
    measurements: torch.Tensor
    event_costs: tuple[int, ...]
    event_index: int
    selector_generator: torch.Generator
    intermediate_reconstructor: Callable
    _capability: _AcquisitionCapability = field(repr=False, compare=False)
    _selector_generators: tuple[torch.Generator, ...] = field(
        default=(), repr=False, compare=False
    )

    @property
    def next_event_cost(self) -> int | None:
        if self.event_index == len(self.event_costs):
            return None
        return self.event_costs[self.event_index]

    def acquire_actions(self, action_ids: Sequence[int]) -> "PolicyContext":
        return self._capability.acquire(self, action_ids)


class MeasurementOracle:
    """Runner-owned target capability; policy contexts never contain the target."""

    def __init__(
        self,
        *,
        target: torch.Tensor,
        initial_operator: LinearOp,
        event_costs: Sequence[int],
    ):
        if not isinstance(target, torch.Tensor):
            raise ValueError("measurement target must be a tensor")
        self._target = target
        self._initial_operator = initial_operator
        self._event_costs = tuple(int(value) for value in event_costs)
        if any(value <= 0 for value in self._event_costs):
            raise ValueError("event costs must be positive")

    def policy_context(
        self,
        *,
        policy_id: str,
        selector_generator: torch.Generator | None = None,
        selector_generators: Sequence[torch.Generator] | None = None,
        intermediate_reconstructor: Callable,
    ) -> PolicyContext:
        if not policy_id:
            raise ValueError("policy_id must be non-empty")
        if (selector_generator is None) == (selector_generators is None):
            raise ValueError(
                "provide exactly one selector generator or round-generator sequence"
            )
        if selector_generators is None:
            streams = (selector_generator,) * max(1, len(self._event_costs))
        else:
            streams = tuple(selector_generators)
            if len(streams) != max(1, len(self._event_costs)):
                raise ValueError(
                    "selector generator count must match acquisition events"
                )
        if any(not isinstance(generator, torch.Generator) for generator in streams):
            raise ValueError("selector streams must be torch generators")
        expected_event = 0
        expected_selected = tuple(
            self._initial_operator.selected_group_ids.tolist()
        )

        def request(
            context: PolicyContext, action_ids: tuple[int, ...]
        ) -> PolicyContext:
            nonlocal expected_event, expected_selected
            if context.event_index != expected_event:
                raise ValueError("stale or replayed acquisition context")
            current_selected = tuple(context.operator.selected_group_ids.tolist())
            if current_selected != expected_selected:
                raise ValueError("policy operator differs from oracle state")
            if expected_event >= len(self._event_costs):
                raise ValueError("acquisition schedule is already complete")
            if len(action_ids) != len(set(action_ids)):
                raise ValueError("duplicate action request")
            all_ids = set(context.operator.group_ids.tolist())
            unknown = sorted(set(action_ids) - all_ids)
            if unknown:
                raise ValueError(f"unknown action {unknown[0]}")
            already = sorted(set(action_ids) & set(current_selected))
            if already:
                raise ValueError(f"action {already[0]} is already selected")
            costs = context.operator.group_costs
            actual_cost = sum(int(costs[action_id]) for action_id in action_ids)
            expected_cost = self._event_costs[expected_event]
            if actual_cost != expected_cost:
                raise ValueError(
                    f"expected event cost {expected_cost}, got {actual_cost}"
                )
            operator = context.operator.add_groups(action_ids)
            measurements = operator.measure(self._target)
            expected_event += 1
            expected_selected = tuple(operator.selected_group_ids.tolist())
            return replace(
                context,
                operator=operator,
                measurements=measurements,
                event_index=expected_event,
                selector_generator=streams[
                    min(expected_event, len(streams) - 1)
                ],
            )

        capability = _AcquisitionCapability(request=request)
        return PolicyContext(
            policy_id=policy_id,
            operator=self._initial_operator,
            measurements=self._initial_operator.measure(self._target),
            event_costs=self._event_costs,
            event_index=0,
            selector_generator=streams[0],
            intermediate_reconstructor=intermediate_reconstructor,
            _capability=capability,
            _selector_generators=streams,
        )


@dataclass(frozen=True)
class PolicyResult:
    policy_id: str
    final_operator: LinearOp
    seed_action_ids: tuple[int, ...]
    action_history: tuple[tuple[int, ...], ...]
    event_costs: tuple[int, ...]
    accounting: SelectionAccounting

    def __post_init__(self) -> None:
        if len(self.action_history) != len(self.event_costs):
            raise ValueError("policy history must contain every acquisition event")
        flattened = tuple(
            action_id for event in self.action_history for action_id in event
        )
        if len(flattened) != len(set(flattened)):
            raise ValueError("policy action history contains duplicate actions")
        if self.accounting.event_count != len(self.action_history):
            raise ValueError("accounting event count differs from policy history")
        if self.accounting.action_count != len(flattened):
            raise ValueError("accounting action count differs from policy history")
        if self.accounting.native_reconstruction_nfe:
            raise ValueError("controlled policy result contains native reconstruction")


def _finish_result(
    initial: PolicyContext,
    final: PolicyContext,
    history: list[tuple[int, ...]],
    accounting: SelectionAccounting,
    elapsed: float,
) -> PolicyResult:
    if final.event_index != len(final.event_costs):
        raise ValueError("policy did not complete the runner-owned event schedule")
    completed = replace(
        accounting,
        event_count=len(history),
        action_count=sum(len(event) for event in history),
        wall_seconds=elapsed,
    )
    return PolicyResult(
        policy_id=initial.policy_id,
        final_operator=final.operator,
        seed_action_ids=tuple(initial.operator.selected_group_ids.tolist()),
        action_history=tuple(history),
        event_costs=initial.event_costs,
        accounting=completed,
    )


def run_static_policy(
    context: PolicyContext,
    *,
    planned_events: Sequence[Sequence[int]],
) -> PolicyResult:
    """Execute a precomputed image-independent action plan."""
    if len(planned_events) != len(context.event_costs):
        raise ValueError("static plan must contain every runner-owned event")
    started = time.perf_counter()
    initial = context
    history: list[tuple[int, ...]] = []
    for event in planned_events:
        action_ids = tuple(int(value) for value in event)
        context = context.acquire_actions(action_ids)
        history.append(action_ids)
    return _finish_result(
        initial,
        context,
        history,
        SelectionAccounting(),
        time.perf_counter() - started,
    )


def run_feedback_policy(
    context: PolicyContext,
    *,
    select_event: Callable[
        [PolicyContext, int],
        tuple[Sequence[int], SelectionAccounting],
    ],
) -> PolicyResult:
    """Execute one target-free selector callback at each common event."""
    started = time.perf_counter()
    initial = context
    history: list[tuple[int, ...]] = []
    accounting = SelectionAccounting()
    while context.next_event_cost is not None:
        action_ids, event_accounting = select_event(
            context, context.next_event_cost
        )
        action_ids = tuple(int(value) for value in action_ids)
        context = context.acquire_actions(action_ids)
        history.append(action_ids)
        accounting = accounting + event_accounting
    return _finish_result(
        initial,
        context,
        history,
        accounting,
        time.perf_counter() - started,
    )
