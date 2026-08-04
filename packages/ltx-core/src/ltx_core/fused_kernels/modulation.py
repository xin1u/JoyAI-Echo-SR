"""Fused AdaLN-Zero "Modulation" Triton kernel.

Drop-in replacement for
:class:`ltx_core.model.transformer.ops.PytorchAdaZeroFunction` (the
``AdaZeroCallable`` implementation used as ``ada_zero_function`` throughout
:class:`ltx_core.model.transformer.transformer.BasicAVTransformerBlock`)::

    def __call__(self, x, eps, scale, shift):
        return rms_norm(x, eps=eps) * (1 + scale) + shift

i.e. an *unweighted* RMSNorm (no learnable gamma) immediately followed by the
AdaLN-Zero affine modulation. Fusing the two avoids materializing the
intermediate normalized tensor and the ``(1 + scale)`` broadcast tensor.

Math (row = last dimension, size ``N``; ``x``: ``(B, T, N)``, ``scale``/
``shift``: broadcastable to ``(B, T, N)``, e.g. ``(B, 1, N)``):

    rstd  = rsqrt(mean(x**2) + eps)
    x_hat = x * rstd
    out   = x_hat * (1 + scale) + shift

Backward. Because ``scale``/``shift`` come from a *different* input path (an
AdaLN timestep embedding through a ``Linear``, see
``AdaLayerNormSingle``/``get_ada_values``) rather than being derived from
``x``, they are independent variables from ``x``'s point of view. Treating
``w = (1 + scale)`` as a (per-position, not just per-channel) weight, the
``dx`` term takes exactly the same closed form as fused weighted-RMSNorm
backward, and ``dscale``/``dshift`` are the plain product-rule terms:

    dy_w   = dy * (1 + scale)
    c      = mean(dy_w * x_hat, dim=-1)
    dx     = rstd * (dy_w - x_hat * c)
    dscale = unbroadcast(dy * x_hat, scale.shape)
    dshift = unbroadcast(dy,         shift.shape)

``unbroadcast`` sum-reduces over any batch/sequence dim that was size-1 (and
therefore broadcast) in the forward pass -- see :func:`ltx_core.fused_kernels._common.unbroadcast`.
"""

from __future__ import annotations

import torch

from ._common import (
    TRITON_AVAILABLE,
    block_size_for_row,
    broadcast_batch_stride,
    num_warps_for_block,
    triton_usable,
    unbroadcast,
)

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    @triton.jit
    def _modulate_fwd_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        out_ptr,
        rstd_ptr,
        stride_x_row,
        T,
        stride_scale_b,
        stride_scale_t,
        stride_shift_b,
        stride_shift_t,
        n_cols,
        eps,
        BLOCK: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        b = row // T
        t = row % T
        x_off = row * stride_x_row
        s_off = b * stride_scale_b + t * stride_scale_t
        sh_off = b * stride_shift_b + t * stride_shift_t

        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols

        x = tl.load(x_ptr + x_off + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + s_off + cols, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + sh_off + cols, mask=mask, other=0.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / n_cols
        rstd = tl.rsqrt(var + eps)
        x_hat = x * rstd
        y = x_hat * (1.0 + scale) + shift

        tl.store(rstd_ptr + row, rstd)
        tl.store(out_ptr + x_off + cols, y, mask=mask)

    @triton.jit
    def _modulate_bwd_kernel(
        dy_ptr,
        x_ptr,
        scale_ptr,
        rstd_ptr,
        dx_ptr,
        dscale_elem_ptr,
        stride_x_row,
        T,
        stride_scale_b,
        stride_scale_t,
        n_cols,
        BLOCK: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        b = row // T
        t = row % T
        x_off = row * stride_x_row
        s_off = b * stride_scale_b + t * stride_scale_t

        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols

        dy = tl.load(dy_ptr + x_off + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + x_off + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + s_off + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row).to(tl.float32)

        x_hat = x * rstd
        dy_w = dy * (1.0 + scale)
        c = tl.sum(dy_w * x_hat, axis=0) / n_cols
        dx = rstd * (dy_w - x_hat * c)
        dscale_elem = dy * x_hat

        tl.store(dx_ptr + x_off + cols, dx, mask=mask)
        tl.store(dscale_elem_ptr + x_off + cols, dscale_elem, mask=mask)


def _modulate_forward(
    x: torch.Tensor, eps: float, scale: torch.Tensor, shift: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    b, t, n_cols = x.shape
    block = block_size_for_row(n_cols, x.element_size())

    stride_scale_b = broadcast_batch_stride(scale, 0, b)
    stride_scale_t = broadcast_batch_stride(scale, 1, t)
    stride_shift_b = broadcast_batch_stride(shift, 0, b)
    stride_shift_t = broadcast_batch_stride(shift, 1, t)

    out = torch.empty_like(x)
    rstd = torch.empty(b * t, dtype=torch.float32, device=x.device)

    _modulate_fwd_kernel[(b * t,)](
        x,
        scale,
        shift,
        out,
        rstd,
        x.stride(1),
        t,
        stride_scale_b,
        stride_scale_t,
        stride_shift_b,
        stride_shift_t,
        n_cols,
        eps,
        BLOCK=block,
        num_warps=num_warps_for_block(block),
        num_stages=1,
    )
    return out, rstd


def _modulate_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, t, n_cols = x.shape
    block = block_size_for_row(n_cols, x.element_size())

    stride_scale_b = broadcast_batch_stride(scale, 0, b)
    stride_scale_t = broadcast_batch_stride(scale, 1, t)

    dy = dy.contiguous()
    dx = torch.empty_like(x)
    dscale_elem = torch.empty_like(x)

    _modulate_bwd_kernel[(b * t,)](
        dy,
        x,
        scale,
        rstd,
        dx,
        dscale_elem,
        x.stride(1),
        t,
        stride_scale_b,
        stride_scale_t,
        n_cols,
        BLOCK=block,
        num_warps=num_warps_for_block(block),
        num_stages=1,
    )
    return dx, dscale_elem


class _ModulateFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        out, rstd = _modulate_forward(x, eps, scale, shift)
        ctx.save_for_backward(x, scale, rstd)
        ctx.scale_shape = scale.shape
        ctx.shift_shape = shift.shape
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor]:
        x, scale, rstd = ctx.saved_tensors
        dx, dscale_elem = _modulate_backward(grad_output, x, scale, rstd)
        dscale = unbroadcast(dscale_elem, ctx.scale_shape)
        dshift = unbroadcast(grad_output, ctx.shift_shape)
        return dx, None, dscale, dshift


def modulate(x: torch.Tensor, eps: float, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Fused ``rms_norm(x, eps=eps) * (1 + scale) + shift``.

    Functional Triton-fused drop-in for
    ``PytorchAdaZeroFunction()(x, eps, scale, shift)``. Falls back to the
    plain PyTorch formula when Triton/CUDA isn't usable, or when ``x`` isn't
    3D (``(B, T, N)`` -- the shape used at every ``ada_zero_function`` call
    site in this codebase).
    """
    can_fuse = (
        x.ndim == 3
        and scale.ndim == 3
        and shift.ndim == 3
        and scale.shape[-1] == x.shape[-1]
        and shift.shape[-1] == x.shape[-1]
        and scale.shape[0] in (1, x.shape[0])
        and scale.shape[1] in (1, x.shape[1])
        and shift.shape[0] in (1, x.shape[0])
        and shift.shape[1] in (1, x.shape[1])
        and triton_usable(x, scale, shift)
    )
    if not can_fuse:
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=None, eps=eps) * (1 + scale) + shift

    return _ModulateFunction.apply(x.contiguous(), eps, scale.contiguous(), shift.contiguous())


class TritonAdaZeroFunction:
    """Triton-fused implementation of
    :class:`ltx_core.model.transformer.ops.AdaZeroCallable`.

    Drop-in for :class:`ltx_core.model.transformer.ops.PytorchAdaZeroFunction`,
    e.g.::

        TransformerOpsConfig.from_functions(ada_zero=TritonAdaZeroFunction())
    """

    def __call__(
        self,
        x: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return modulate(x, eps, scale, shift)


__all__ = ["TritonAdaZeroFunction", "modulate"]
