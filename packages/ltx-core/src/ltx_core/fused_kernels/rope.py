"""Fused RoPE (rotary positional embedding) Triton kernels.

Drop-in replacement for :mod:`ltx_core.model.transformer.rope`'s
``apply_rotary_emb`` / ``apply_split_rotary_emb`` / ``apply_interleaved_rotary_emb``.
Both LTX RoPE conventions are supported:

* ``LTXRopeType.SPLIT`` (default, production path): the last dimension is cut
  into two *contiguous* halves ``x1, x2`` (this is the common "rotate_half"
  convention, as in GPT-NeoX/LLaMA)::

      out1 = x1 * cos - x2 * sin
      out2 = x2 * cos + x1 * sin

  ``cos``/``sin`` have shape ``(Bc, H, T, D // 2)`` with ``Bc`` broadcastable
  to the input batch. This mirrors
  :func:`ltx_core.model.transformer.rope.apply_split_rotary_emb`, including
  its "3D input + 4D freqs -> reshape to (B, H, T, D)" handling.

* ``LTXRopeType.INTERLEAVED`` (legacy): adjacent *pairs* ``(x[2i], x[2i+1])``
  are rotated, with ``cos``/``sin`` already ``repeat_interleave(2)``-expanded
  to the full last dimension (``cos[2i] == cos[2i+1]``)::

      out[2i]   = x[2i]   * cos[2i]   - x[2i+1] * sin[2i]
      out[2i+1] = x[2i+1] * cos[2i+1] + x[2i]   * sin[2i+1]

Both are pure rotations (2x2 orthogonal matrix per pair), so the backward is
the inverse rotation (transpose of an orthogonal matrix): swap the sign of
the ``sin`` term and feed the upstream gradient back through the *same*
kernel via a ``CONJUGATE`` compile-time flag. `cos`/`sin` are treated as
non-differentiable (they come from :func:`precompute_freqs_cis`, never from
an ``nn.Parameter`` in this codebase), so their gradient is ``None``.
"""

from __future__ import annotations

import torch

from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.rope import apply_interleaved_rotary_emb as _reference_apply_interleaved
from ltx_core.model.transformer.rope import apply_split_rotary_emb as _reference_apply_split

from ._common import TRITON_AVAILABLE, broadcast_batch_stride, next_power_of_2, num_warps_for_block, triton_usable

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    # ── SPLIT convention: contiguous halves ────────────────────────────────

    @triton.jit
    def _split_rope_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        stride_xb,
        stride_xh,
        stride_xt,
        stride_cb,
        stride_ch,
        stride_ct,
        H,
        T,
        half_dim,
        BLOCK: tl.constexpr,
        CONJUGATE: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        t = pid % T
        tmp = pid // T
        h = tmp % H
        b = tmp // H

        x_off = b * stride_xb + h * stride_xh + t * stride_xt
        c_off = b * stride_cb + h * stride_ch + t * stride_ct

        d = tl.arange(0, BLOCK)
        mask = d < half_dim

        x1 = tl.load(x_ptr + x_off + d, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + x_off + half_dim + d, mask=mask, other=0.0).to(tl.float32)
        cos = tl.load(cos_ptr + c_off + d, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + c_off + d, mask=mask, other=0.0).to(tl.float32)

        if CONJUGATE:
            out1 = x1 * cos + x2 * sin
            out2 = x2 * cos - x1 * sin
        else:
            out1 = x1 * cos - x2 * sin
            out2 = x2 * cos + x1 * sin

        tl.store(out_ptr + x_off + d, out1, mask=mask)
        tl.store(out_ptr + x_off + half_dim + d, out2, mask=mask)

    # ── INTERLEAVED convention: adjacent pairs, XOR-swap partner ───────────

    @triton.jit
    def _interleaved_rope_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        stride_x_row,
        stride_c_row,
        n_rows_c,
        n_cols,
        BLOCK: tl.constexpr,
        CONJUGATE: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        # `cos`/`sin` broadcast along dim 0 when they only have a single row.
        c_row = row % n_rows_c
        x_off = row * stride_x_row
        c_off = c_row * stride_c_row

        d = tl.arange(0, BLOCK)
        mask = d < n_cols
        partner = d ^ 1
        partner_mask = partner < n_cols

        x = tl.load(x_ptr + x_off + d, mask=mask, other=0.0).to(tl.float32)
        x_partner = tl.load(x_ptr + x_off + partner, mask=mask & partner_mask, other=0.0).to(tl.float32)
        cos = tl.load(cos_ptr + c_off + d, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + c_off + d, mask=mask, other=0.0).to(tl.float32)

        # Forward: even positions get -partner*sin, odd get +partner*sin.
        # Conjugate (backward) flips both signs -- transpose of the rotation.
        sign = tl.where((d & 1) == 0, 1.0, -1.0) if CONJUGATE else tl.where((d & 1) == 0, -1.0, 1.0)

        out = x * cos + sign * x_partner * sin
        tl.store(out_ptr + x_off + d, out, mask=mask)


def _launch_split_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, conjugate: bool) -> torch.Tensor:
    """`x`: (B, H, T, D) contiguous. `cos`/`sin`: (Bc, H, T, D // 2), Bc in {1, B}."""
    b, h, t, d = x.shape
    half_dim = d // 2
    out = torch.empty_like(x)
    block = next_power_of_2(half_dim)

    stride_cb = broadcast_batch_stride(cos, 0, b)

    grid = (b * h * t,)
    _split_rope_kernel[grid](
        x,
        cos,
        sin,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        stride_cb,
        cos.stride(1),
        cos.stride(2),
        h,
        t,
        half_dim,
        BLOCK=block,
        CONJUGATE=conjugate,
        num_warps=num_warps_for_block(block),
        num_stages=1,
    )
    return out


def _launch_interleaved_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, conjugate: bool) -> torch.Tensor:
    """`x`: (..., D) contiguous, flattened to (rows, D). `cos`/`sin` broadcastable
    to the same leading ("rows") shape, flattened to (rows_c, D) with
    `rows_c in {1, rows}`.
    """
    n_cols = x.shape[-1]
    x_flat = x.reshape(-1, n_cols)
    n_rows = x_flat.shape[0]

    cos_flat = cos.reshape(-1, n_cols)
    sin_flat = sin.reshape(-1, n_cols)
    n_rows_c = cos_flat.shape[0]
    if n_rows_c not in (1, n_rows):
        raise ValueError(
            f"apply_interleaved_rotary_emb (Triton): cos/sin leading size ({n_rows_c}) must be "
            f"1 (broadcast) or match the input's leading size ({n_rows})."
        )

    out = torch.empty_like(x_flat)
    block = next_power_of_2(n_cols)
    grid = (n_rows,)
    _interleaved_rope_kernel[grid](
        x_flat,
        cos_flat,
        sin_flat,
        out,
        x_flat.stride(0),
        cos_flat.stride(0),
        n_rows_c,
        n_cols,
        BLOCK=block,
        CONJUGATE=conjugate,
        num_warps=num_warps_for_block(block),
        num_stages=1,
    )
    return out.reshape(x.shape)


class _SplitRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(cos, sin)
        return _launch_split_rope(x, cos, sin, conjugate=False)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        cos, sin = ctx.saved_tensors
        grad_input = _launch_split_rope(grad_output.contiguous(), cos, sin, conjugate=True)
        return grad_input, None, None


class _InterleavedRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(cos, sin)
        return _launch_interleaved_rope(x, cos, sin, conjugate=False)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        cos, sin = ctx.saved_tensors
        grad_input = _launch_interleaved_rope(grad_output.contiguous(), cos, sin, conjugate=True)
        return grad_input, None, None


def apply_split_rotary_emb(
    input_tensor: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    """Triton-fused drop-in for
    :func:`ltx_core.model.transformer.rope.apply_split_rotary_emb`.

    Replicates the reference function's "(B, T, H*D) input + (Bc, H, T, D//2)
    freqs -> reshape to (B, H, T, D)" preprocessing exactly, then dispatches
    the rotate-half math to a fused Triton kernel. Falls back to the pure
    PyTorch reference when Triton/CUDA isn't usable, or for shapes the fast
    path doesn't cover (odd head dim, or an input that doesn't reduce to a
    4D (B, H, T, D) tensor with a matching freqs tensor).
    """
    if sin_freqs.shape != cos_freqs.shape:
        raise ValueError(
            f"apply_split_rotary_emb: sin_freqs.shape {tuple(sin_freqs.shape)} must equal "
            f"cos_freqs.shape {tuple(cos_freqs.shape)}."
        )

    needs_reshape = input_tensor.ndim != 4 and cos_freqs.ndim == 4
    reshaped = input_tensor
    if needs_reshape:
        b_freq = cos_freqs.shape[0]
        h = cos_freqs.shape[1]
        b_in = input_tensor.shape[0]
        if b_freq not in (1, b_in):
            raise ValueError(
                f"apply_split_rotary_emb: cos_freqs batch ({b_freq}) must be 1 "
                f"(broadcast) or equal input_tensor batch ({b_in})."
            )
        reshaped = input_tensor.unflatten(-1, (h, -1)).transpose(1, 2)

    can_fuse = (
        triton_usable(reshaped, cos_freqs, sin_freqs)
        and reshaped.ndim == 4
        and cos_freqs.ndim == 4
        and reshaped.shape[-1] % 2 == 0
        and reshaped.shape[-1] // 2 == cos_freqs.shape[-1]
        and cos_freqs.shape[0] in (1, reshaped.shape[0])
        and cos_freqs.shape[1] == reshaped.shape[1]
        and cos_freqs.shape[2] == reshaped.shape[2]
    )
    if not can_fuse:
        return _reference_apply_split(input_tensor, cos_freqs, sin_freqs)

    out = _SplitRoPEFunction.apply(reshaped.contiguous(), cos_freqs.contiguous(), sin_freqs.contiguous())
    if needs_reshape:
        out = out.transpose(1, 2).flatten(-2)
    return out


def apply_interleaved_rotary_emb(
    input_tensor: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    """Triton-fused drop-in for
    :func:`ltx_core.model.transformer.rope.apply_interleaved_rotary_emb`
    (legacy pair-interleaved RoPE convention).
    """
    if not triton_usable(input_tensor, cos_freqs, sin_freqs) or input_tensor.shape[-1] % 2 != 0:
        return _reference_apply_interleaved(input_tensor, cos_freqs, sin_freqs)
    return _InterleavedRoPEFunction.apply(input_tensor.contiguous(), cos_freqs.contiguous(), sin_freqs.contiguous())


def apply_rotary_emb(
    input_tensor: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
    rope_type: LTXRopeType = LTXRopeType.SPLIT,
) -> torch.Tensor:
    """Triton-fused drop-in for :func:`ltx_core.model.transformer.rope.apply_rotary_emb`."""
    if rope_type == LTXRopeType.INTERLEAVED:
        return apply_interleaved_rotary_emb(input_tensor, *freqs_cis)
    if rope_type == LTXRopeType.SPLIT:
        return apply_split_rotary_emb(input_tensor, *freqs_cis)
    raise ValueError(f"Invalid rope type: {rope_type}")


class TritonPreAttention:
    """Triton-fused implementation of
    :class:`ltx_core.model.transformer.ops.PreAttentionCallable`.

    Drop-in for :class:`ltx_core.model.transformer.ops.PytorchPreAttention`:
    identical to it (``q_norm``/``k_norm``, then RoPE) except the RoPE step
    dispatches to this module's fused kernel, e.g.::

        AttentionOps(preattention_function=TritonPreAttention())
        # or: TransformerOpsConfig.from_functions(preattention=TritonPreAttention())
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_module: torch.nn.Module,
        mask: torch.Tensor | None,  # noqa: ARG002
        pe: torch.Tensor | None,
        k_pe: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = attn_module.q_norm(q)
        k = attn_module.k_norm(k)
        if pe is not None:
            q = apply_rotary_emb(q, pe, attn_module.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, attn_module.rope_type)
        return q, k


__all__ = [
    "LTXRopeType",
    "TritonPreAttention",
    "apply_interleaved_rotary_emb",
    "apply_rotary_emb",
    "apply_split_rotary_emb",
]
