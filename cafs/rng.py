from __future__ import annotations
import hashlib, json, random
from contextlib import contextmanager
from typing import Iterable, Iterator
import numpy as np
import torch
_NAMESPACES = {"initial_mask", "measurement_noise", "selector", "final_reconstruction", "execution_order", "bootstrap"}


def semantic_seed(protocol_seed: int, namespace: str, *keys) -> int:
    """Derive a stable non-negative 63-bit seed from semantic experiment keys."""
    if not isinstance(protocol_seed, int) or protocol_seed < 0:
        raise ValueError("protocol_seed must be a non-negative integer")
    if namespace not in _NAMESPACES:
        raise ValueError(f"unknown RNG namespace: {namespace!r}")
    try:
        payload = json.dumps(
            [protocol_seed, namespace, *keys],
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except TypeError as error:
        raise ValueError("semantic RNG keys must be JSON serializable") from error
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _cuda_device_indices(devices: Iterable[int | str | torch.device]) -> tuple[int, ...]:
    indices: list[int] = []
    for device in devices:
        parsed = torch.device(device)
        if parsed.type != "cuda":
            continue
        index = torch.cuda.current_device() if parsed.index is None else parsed.index
        if index not in indices:
            indices.append(index)
    return tuple(indices)


@contextmanager
def preserve_global_rng(
    devices: Iterable[int | str | torch.device] = (),
) -> Iterator[None]:
    """Save and restore Python, NumPy, CPU Torch, and requested CUDA RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_indices = (
        _cuda_device_indices(devices) if torch.cuda.is_available() else ()
    )
    cuda_states = {
        index: torch.cuda.get_rng_state(index) for index in cuda_indices
    }
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        for index, state in cuda_states.items():
            torch.cuda.set_rng_state(state, index)
