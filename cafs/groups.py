from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable


@dataclass(frozen=True)
class MeasurementGroup:
    """One indivisible acquisition action in flattened dictionary coordinates."""

    id: int
    indices: tuple[int, ...]
    cost: int


@lru_cache(maxsize=16)
def real_fourier_orbits(
    H: int,
    W: int,
    channels: int,
) -> tuple[MeasurementGroup, ...]:
    """Partition an ``H x W`` real-image FFT into conjugate orbits.

    Cached: the partition depends only on the shape and channel count, never on a
    mask, and rebuilding it walked H*W bins on every operator construction. The
    return value is a tuple of frozen dataclasses, so sharing it is safe.
    """
    if H <= 0 or W <= 0 or channels <= 0:
        raise ValueError("H, W, and channels must be positive")

    orbits: set[tuple[int, ...]] = set()
    for ky in range(H):
        for kx in range(W):
            index = ky * W + kx
            partner = ((-ky) % H) * W + ((-kx) % W)
            orbits.add(tuple(sorted({index, partner})))

    return tuple(
        MeasurementGroup(id=group_id, indices=indices,
                         cost=channels * len(indices))
        for group_id, indices in enumerate(sorted(orbits))
    )


def largest_feasible_budget(
    requested_m: int,
    current_m: int,
    groups: Iterable[MeasurementGroup],
) -> int:
    """Return the largest total cost reachable without exceeding a request."""
    if requested_m < 0 or current_m < 0:
        raise ValueError("requested_m and current_m must be non-negative")
    if current_m > requested_m:
        raise ValueError("current_m cannot exceed requested_m")

    capacity = requested_m - current_m
    counts = Counter(group.cost for group in groups)
    if any(cost <= 0 for cost in counts):
        raise ValueError("measurement-group costs must be positive")

    # Exact bounded subset-sum using a bitset. Binary decomposition keeps the
    # work proportional to the number of distinct cost/count bits, rather than
    # to the potentially large number of Fourier groups.
    reachable = 1
    mask = (1 << (capacity + 1)) - 1
    for cost, count in counts.items():
        chunk = 1
        while count:
            take = min(chunk, count)
            reachable |= reachable << (cost * take)
            reachable &= mask
            count -= take
            chunk <<= 1

    additional = reachable.bit_length() - 1
    return current_m + additional
