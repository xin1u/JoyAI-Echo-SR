"""Correctness + backward-precision + benchmark checks for
:mod:`ltx_core.fused_kernels.rope`, against the exact reference functions in
``ltx_core.model.transformer.rope`` (not a re-derived formula -- this is a
literal equivalence check against the production code being replaced).

Run directly: ``python -m ltx_core.fused_kernels.tests.test_rope``.
"""

from __future__ import annotations

import os
import sys

import torch

# On a machine with no CUDA GPU, re-exec under Triton's CPU interpreter
# (TRITON_INTERPRET=1) *before* `ltx_core.fused_kernels` is ever imported.
# Setting the env var further down in this file (even before the import
# below) would be too late: this file lives *inside*
# `ltx_core.fused_kernels.tests`, so merely importing it forces Python to
# first fully run the ancestor `ltx_core/fused_kernels/__init__.py`, which
# imports the `@triton.jit`-decorated kernel modules -- locking in real-GPU
# compilation before any code in this file gets a chance to run. Re-executing
# the process is the only way to make the env var visible before that
# ancestor import happens.
if __name__ == "__main__" and not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1":
    os.environ["TRITON_INTERPRET"] = "1"
    os.execv(sys.executable, [sys.executable, "-m", "ltx_core.fused_kernels.tests.test_rope", *sys.argv[1:]])

from ltx_core.fused_kernels import apply_interleaved_rotary_emb, apply_split_rotary_emb
from ltx_core.fused_kernels.tests._test_utils import (
    DEVICE,
    DTYPES,
    OpReport,
    benchmark_fwd_bwd,
    passes,
    print_footer,
    print_header,
)
from ltx_core.model.transformer.rope import apply_interleaved_rotary_emb as reference_interleaved
from ltx_core.model.transformer.rope import apply_split_rotary_emb as reference_split

# (B, H, T, D): small logic-correctness shapes -- D matches the head dims
# actually used in the LTX transformer config (video=128, audio=64), but
# H/T are kept small since correctness doesn't depend on grid size and
# `TRITON_INTERPRET`'s pure-Python per-program simulation is orders of
# magnitude slower than a real GPU launch. Large, production-scale shapes are
# exercised by `run_benchmark` (CUDA-only; skipped entirely under the
# interpreter).
SPLIT_4D_SHAPES = [
    (2, 3, 37, 32),  # odd seq len -> exercises grid edge, small D
    (1, 4, 9, 64),  # audio-like head dim
    (2, 2, 5, 128),  # video-like head dim
]
# (B, T, H*D) 3D input against a 4D freqs tensor -> exercises `needs_reshape`.
SPLIT_3D_SHAPES = [
    (2, 13, 4, 64),  # (B, T, H, D) -> inner_dim = H * D
    (1, 17, 2, 128),
]

INTERLEAVED_SHAPES = [
    (2, 8, 37, 64),
    (1, 21, 128),
]


def _run_split_case(shape_4d: tuple[int, ...] | None, shape_3d: tuple[int, ...] | None, batch_broadcast: bool):
    if shape_4d is not None:
        b, h, t, d = shape_4d
        x_shape = shape_4d
    else:
        b, t, h, d = shape_3d
        x_shape = (b, t, h * d)

    cb = 1 if batch_broadcast else b
    for dtype in DTYPES:
        x = torch.randn(x_shape, device=DEVICE, dtype=dtype, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        cos = torch.randn(cb, h, t, d // 2, device=DEVICE, dtype=dtype)
        sin = torch.randn(cb, h, t, d // 2, device=DEVICE, dtype=dtype)

        out = apply_split_rotary_emb(x, cos, sin)
        out_ref = reference_split(x_ref, cos, sin)

        grad = torch.randn_like(out)
        out.backward(grad)
        out_ref.backward(grad)

        yield dtype, x_shape, (cb, h, t, d // 2), out, out_ref, x, x_ref


def run_split() -> OpReport:
    report = OpReport("RoPE-SPLIT")
    for shape_4d in SPLIT_4D_SHAPES:
        for batch_broadcast in (False, True):
            for dtype, x_shape, c_shape, out, out_ref, x, x_ref in _run_split_case(shape_4d, None, batch_broadcast):
                ok_fwd, fwd_err, fwd_rl2 = passes(out, out_ref, dtype)
                ok_dx, dx_err, dx_rl2 = passes(x.grad, x_ref.grad, dtype)
                report.add(
                    layout="4D",
                    shape=x_shape,
                    cos_shape=c_shape,
                    dtype=dtype,
                    fwd_err=fwd_err,
                    fwd_rl2=fwd_rl2,
                    dx_err=dx_err,
                    dx_rl2=dx_rl2,
                    ok=ok_fwd and ok_dx,
                )
    for shape_3d in SPLIT_3D_SHAPES:
        for batch_broadcast in (False, True):
            for dtype, x_shape, c_shape, out, out_ref, x, x_ref in _run_split_case(None, shape_3d, batch_broadcast):
                ok_fwd, fwd_err, fwd_rl2 = passes(out, out_ref, dtype)
                ok_dx, dx_err, dx_rl2 = passes(x.grad, x_ref.grad, dtype)
                report.add(
                    layout="3D->4D",
                    shape=x_shape,
                    cos_shape=c_shape,
                    dtype=dtype,
                    fwd_err=fwd_err,
                    fwd_rl2=fwd_rl2,
                    dx_err=dx_err,
                    dx_rl2=dx_rl2,
                    ok=ok_fwd and ok_dx,
                )
    return report


def run_interleaved() -> OpReport:
    report = OpReport("RoPE-INTERLEAVED")
    for shape in INTERLEAVED_SHAPES:
        d = shape[-1]
        for dtype in DTYPES:
            x = torch.randn(shape, device=DEVICE, dtype=dtype, requires_grad=True)
            x_ref = x.detach().clone().requires_grad_(True)
            cos = torch.randn(*shape[:-1], d // 2, device=DEVICE, dtype=dtype).repeat_interleave(2, dim=-1)
            sin = torch.randn(*shape[:-1], d // 2, device=DEVICE, dtype=dtype).repeat_interleave(2, dim=-1)

            out = apply_interleaved_rotary_emb(x, cos, sin)
            out_ref = reference_interleaved(x_ref, cos, sin)

            grad = torch.randn_like(out)
            out.backward(grad)
            out_ref.backward(grad)

            ok_fwd, fwd_err, fwd_rl2 = passes(out, out_ref, dtype)
            ok_dx, dx_err, dx_rl2 = passes(x.grad, x_ref.grad, dtype)
            report.add(
                shape=shape,
                dtype=dtype,
                fwd_err=fwd_err,
                fwd_rl2=fwd_rl2,
                dx_err=dx_err,
                dx_rl2=dx_rl2,
                ok=ok_fwd and ok_dx,
            )
    return report


def print_report(report: OpReport) -> None:
    print_header(f"[{report.name}] correctness + backward precision (device={DEVICE})")
    for r in report.rows:
        tag = "PASS" if r["ok"] else "FAIL"
        layout = f"layout={r['layout']:8} " if "layout" in r else ""
        cshape = f"cos_shape={r['cos_shape']!s:18} " if "cos_shape" in r else ""
        print(
            f"  [{tag}] {layout}shape={r['shape']!s:20} {cshape}dtype={r['dtype']!s:16} "
            f"fwd_err={r['fwd_err']:.3e}(rl2={r['fwd_rl2']:.3e})  dx_err={r['dx_err']:.3e}(rl2={r['dx_rl2']:.3e})"
        )
    print(f"  Summary: {'ALL PASS' if report.all_passed else 'SOME FAILED'} ({len(report.rows)} cases)")
    print_footer()


def run_benchmark() -> OpReport:
    report = OpReport("RoPE-benchmark")
    if DEVICE != "cuda":
        print("[RoPE benchmark] skipped: no CUDA device available in this environment.")
        return report

    # (batch, heads, seq_len i.e. token count, head_dim) -- sweep token count
    # from the existing production scale up to tens-of-thousands. Several
    # values are deliberately *not* powers of 2 (24576 = 3*8192, 45067/60003
    # are arbitrary odd lengths) to exercise the grid-edge / block-mask tail
    # path at large scale, not just the small hand-picked shapes in
    # `SPLIT_4D_SHAPES` above. head_dim covers both video (128) and audio (64).
    cfgs = [
        (2, 32, 4096, 128),
        (2, 32, 8192, 128),
        (1, 32, 16384, 128),
        (1, 32, 24576, 128),  # 24k, not a power of 2
        (1, 32, 32768, 128),
        (1, 32, 45067, 128),  # ~45k, arbitrary non-power-of-2 length
        (1, 32, 65536, 128),
        (2, 32, 4096, 64),
        (1, 16, 32768, 64),
        (1, 16, 60003, 64),  # ~60k, arbitrary non-power-of-2 length, audio head_dim
    ]
    for b, h, t, d in cfgs:
        x = torch.randn(b, h, t, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        cos = torch.randn(1, h, t, d // 2, device=DEVICE, dtype=torch.bfloat16)
        sin = torch.randn(1, h, t, d // 2, device=DEVICE, dtype=torch.bfloat16)

        def grad_fn(out):  # noqa: ANN001
            return torch.randn_like(out)

        t_triton = benchmark_fwd_bwd(lambda x=x, cos=cos, sin=sin: apply_split_rotary_emb(x, cos, sin), grad_fn)
        t_ref = benchmark_fwd_bwd(lambda x=x, cos=cos, sin=sin: reference_split(x, cos, sin), grad_fn)
        report.add_bench(
            shape=(b, h, t, d),
            n_tokens=b * t,
            t_triton_ms=t_triton,
            t_ref_ms=t_ref,
            speedup=t_ref / t_triton,
        )
    return report


def main() -> bool:
    split_report = run_split()
    print_report(split_report)
    interleaved_report = run_interleaved()
    print_report(interleaved_report)
    bench = run_benchmark()
    if bench.bench_rows:
        print_header("[RoPE-SPLIT] benchmark (fwd+bwd, bf16)")
        for r in bench.bench_rows:
            print(
                f"  shape={r['shape']!s:20} tokens={r['n_tokens']:>6}  Triton={r['t_triton_ms']:.3f}ms  "
                f"Ref={r['t_ref_ms']:.3f}ms  speedup={r['speedup']:.2f}x"
            )
        print_footer()
    return split_report.all_passed and interleaved_report.all_passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
