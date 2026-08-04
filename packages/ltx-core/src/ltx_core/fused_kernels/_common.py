"""Shared helpers for the Triton fused kernels in this package.

Every kernel module in :mod:`ltx_core.fused_kernels` follows the same
"single-block-per-row" pattern (one Triton program normalizes/transforms one
full row of the feature dimension), so the block-size/num-warps heuristics and
the broadcast bookkeeping live here once instead of being copy-pasted three
times.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

try:
    import triton

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - triton missing (e.g. CPU-only / non-CUDA install)
    triton = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False


def triton_usable(*tensors: torch.Tensor) -> bool:
    """Whether the Triton fast path should be used for the given tensors.

    Triton kernels normally require a CUDA tensor. The one exception is
    ``TRITON_INTERPRET=1`` (Triton's CPU interpreter, used by this package's
    test suite to validate kernel *logic* on machines without a GPU) -- in
    that mode CPU tensors are accepted too.

    Also honors the global :func:`ltx_core.fused_kernels.set_triton_enabled`
    switch (implemented via an env var so it is visible from this leaf module
    without importing the package ``__init__`` and risking a cycle).
    """
    if not TRITON_AVAILABLE:
        return False
    if os.environ.get("_LTX_FUSED_KERNELS_FORCE_DISABLE") == "1":
        return False
    if os.environ.get("TRITON_INTERPRET") == "1":
        return True
    return all(t.is_cuda for t in tensors)


@lru_cache(maxsize=None)
def next_power_of_2(n: int) -> int:
    if TRITON_AVAILABLE:
        return triton.next_power_of_2(n)
    p = 1
    while p < n:
        p *= 2
    return p


@lru_cache(maxsize=None)
def block_size_for_row(num_cols: int, element_size: int, max_bytes: int = 65536) -> int:
    """Pick a power-of-2 Triton block size covering a whole row.

    All kernels here reduce over the last dimension inside a single program
    (no split-D / multi-block reduction), so the whole row must fit in one
    block. Raise instead of silently truncating: a truncated block would
    silently compute a wrong (partial) reduction rather than failing loudly.
    """
    max_fused_size = max_bytes // max(element_size, 1)
    block = min(max_fused_size, next_power_of_2(num_cols))
    if block < num_cols:
        raise RuntimeError(
            f"Row of {num_cols} columns ({element_size}B/elem) does not fit in a single "
            f"{max_bytes}B Triton block (max block={block}); this kernel only supports "
            "single-block reductions over the feature dimension."
        )
    return block


@lru_cache(maxsize=None)
def num_warps_for_block(block_size: int) -> int:
    return min(max(block_size // 256, 1), 8)


def broadcast_batch_stride(tensor: torch.Tensor, dim: int, target_size: int) -> int:
    """Element stride to use for `dim` when indexing `tensor` with an index that
    ranges over `target_size`, where `tensor.shape[dim]` is either `1`
    (broadcast -- always read index 0, so stride must be forced to 0) or
    exactly `target_size` (no broadcast -- use the real stride).
    """
    size = tensor.shape[dim]
    if size == target_size:
        return tensor.stride(dim)
    if size == 1:
        return 0
    raise ValueError(f"Cannot broadcast dim {dim} of size {size} to {target_size}")


def unbroadcast(grad: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Sum-reduce `grad` back down to `target_shape`.

    Replicates the reduction that autograd performs automatically for
    ordinary broadcasted ops (e.g. `x + bias`). Needed here because the fused
    kernels compute elementwise per-position gradients for tensors (scale,
    shift, ...) that may have been broadcast during the forward pass (e.g. a
    batch or sequence dim of size 1).
    """
    while grad.ndim > len(target_shape):
        grad = grad.sum(dim=0)
    for dim, (g_size, t_size) in enumerate(zip(grad.shape, target_shape, strict=True)):
        if t_size == 1 and g_size != 1:
            grad = grad.sum(dim=dim, keepdim=True)
    return grad.reshape(target_shape)


__all__ = [
    "TRITON_AVAILABLE",
    "block_size_for_row",
    "broadcast_batch_stride",
    "next_power_of_2",
    "num_warps_for_block",
    "triton_usable",
    "unbroadcast",
]
