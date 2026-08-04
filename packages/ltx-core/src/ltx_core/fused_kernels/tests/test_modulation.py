"""Correctness + backward-precision + benchmark checks for
:mod:`ltx_core.fused_kernels.modulation`, against
:class:`ltx_core.model.transformer.ops.PytorchAdaZeroFunction` (the exact
reference implementation being replaced).

Run directly: ``python -m ltx_core.fused_kernels.tests.test_modulation``.
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
    os.execv(sys.executable, [sys.executable, "-m", "ltx_core.fused_kernels.tests.test_modulation", *sys.argv[1:]])

from ltx_core.fused_kernels import modulate
from ltx_core.fused_kernels.tests._test_utils import (
    DEVICE,
    DTYPES,
    OpReport,
    benchmark_fwd_bwd,
    passes,
    passes_reduced,
    print_footer,
    print_header,
)
from ltx_core.model.transformer.ops import PytorchAdaZeroFunction

_reference = PytorchAdaZeroFunction()

# (B, T, D): D matches the LTX video/audio block hidden sizes; scale/shift
# below are given either the (B, 1, D) "one AdaLN vector per batch,
# broadcast over tokens" shape (the common case for `vshift_msa` etc. in
# `BasicAVTransformerBlock`) or the full (B, T, D) per-token shape.
SHAPES = [
    (2, 37, 64),  # odd seq len -> exercises grid edge, small D
    (2, 300, 2048),  # audio-like
    (1, 512, 4096),  # video-like
]


def run(eps: float = 1e-6) -> OpReport:
    torch.manual_seed(0)
    report = OpReport("Modulation")

    for shape in SHAPES:
        b, t, d = shape
        for scale_t in (1, t):
            for dtype in DTYPES:
                x = torch.randn(shape, device=DEVICE, dtype=dtype, requires_grad=True)
                scale = torch.randn(b, scale_t, d, device=DEVICE, dtype=dtype, requires_grad=True)
                shift = torch.randn(b, scale_t, d, device=DEVICE, dtype=dtype, requires_grad=True)
                x_ref = x.detach().clone().requires_grad_(True)
                scale_ref = scale.detach().clone().requires_grad_(True)
                shift_ref = shift.detach().clone().requires_grad_(True)

                out = modulate(x, eps, scale, shift)
                out_ref = _reference(x_ref, eps, scale_ref, shift_ref)

                grad = torch.randn_like(out)
                out.backward(grad)
                out_ref.backward(grad)

                # When scale/shift broadcast over T (scale_t == 1), their
                # gradients are a *sum* over all T positions -- use the
                # reduction-aware check (see `passes_reduced`'s docstring).
                # Per-token (scale_t == t) is a 1:1 elementwise gradient, so
                # the regular elementwise+rel_l2 check applies.
                grad_check = passes_reduced if scale_t == 1 else passes
                ok_fwd, fwd_err, fwd_rl2 = passes(out, out_ref, dtype)
                ok_dx, dx_err, dx_rl2 = passes(x.grad, x_ref.grad, dtype)
                ok_dscale, dscale_err, dscale_rl2 = grad_check(scale.grad, scale_ref.grad, dtype)
                ok_dshift, dshift_err, dshift_rl2 = grad_check(shift.grad, shift_ref.grad, dtype)
                report.add(
                    shape=shape,
                    scale_t=scale_t,
                    dtype=dtype,
                    fwd_err=fwd_err,
                    fwd_rl2=fwd_rl2,
                    dx_err=dx_err,
                    dx_rl2=dx_rl2,
                    dscale_err=dscale_err,
                    dscale_rl2=dscale_rl2,
                    dshift_err=dshift_err,
                    dshift_rl2=dshift_rl2,
                    ok=ok_fwd and ok_dx and ok_dscale and ok_dshift,
                )
    return report


def print_report(report: OpReport) -> None:
    print_header(f"[{report.name}] correctness + backward precision (device={DEVICE})")
    for r in report.rows:
        tag = "PASS" if r["ok"] else "FAIL"
        broadcast = "T-broadcast" if r["scale_t"] == 1 else "per-token"
        print(
            f"  [{tag}] shape={r['shape']!s:16} scale/shift={broadcast:12} dtype={r['dtype']!s:16}\n"
            f"         fwd_err={r['fwd_err']:.3e}(rl2={r['fwd_rl2']:.3e})  "
            f"dx_err={r['dx_err']:.3e}(rl2={r['dx_rl2']:.3e})  "
            f"dscale_err={r['dscale_err']:.3e}(rl2={r['dscale_rl2']:.3e})  "
            f"dshift_err={r['dshift_err']:.3e}(rl2={r['dshift_rl2']:.3e})"
        )
    print(f"  Summary: {'ALL PASS' if report.all_passed else 'SOME FAILED'} ({len(report.rows)} cases)")
    print_footer()


def run_benchmark() -> OpReport:
    report = OpReport("Modulation-benchmark")
    if DEVICE != "cuda":
        print("[Modulation benchmark] skipped: no CUDA device available in this environment.")
        return report

    # (batch, seq_len i.e. token count, feature dim) -- sweep token count from
    # the existing production scale up to tens-of-thousands. Several values
    # are deliberately *not* powers of 2 (24576 = 3*8192, 45067/60003 are
    # arbitrary odd lengths) to exercise the grid-edge / block-mask tail path
    # at large scale. Feature dim covers both video (4096) and audio (2048).
    cfgs = [
        (2, 4096, 4096),
        (2, 8192, 4096),
        (1, 16384, 4096),
        (1, 24576, 4096),  # 24k, not a power of 2
        (1, 32768, 4096),
        (1, 45067, 4096),  # ~45k, arbitrary non-power-of-2 length
        (1, 65536, 4096),
        (1, 4096, 2048),
        (1, 32768, 2048),
        (1, 60003, 2048),  # ~60k, arbitrary non-power-of-2 length, audio dim
    ]
    for b, t, d in cfgs:
        x = torch.randn(b, t, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        scale = torch.randn(b, 1, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        shift = torch.randn(b, 1, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

        def grad_fn(out):  # noqa: ANN001
            return torch.randn_like(out)

        t_triton = benchmark_fwd_bwd(lambda x=x, scale=scale, shift=shift: modulate(x, 1e-6, scale, shift), grad_fn)
        t_ref = benchmark_fwd_bwd(lambda x=x, scale=scale, shift=shift: _reference(x, 1e-6, scale, shift), grad_fn)
        report.add_bench(
            shape=(b, t, d),
            n_tokens=b * t,
            t_triton_ms=t_triton,
            t_ref_ms=t_ref,
            speedup=t_ref / t_triton,
        )
    return report


def main() -> bool:
    report = run()
    print_report(report)
    bench = run_benchmark()
    if bench.bench_rows:
        print_header("[Modulation] benchmark (fwd+bwd, bf16)")
        for r in bench.bench_rows:
            print(
                f"  shape={r['shape']!s:18} tokens={r['n_tokens']:>6}  Triton={r['t_triton_ms']:.3f}ms  "
                f"Ref={r['t_ref_ms']:.3f}ms  speedup={r['speedup']:.2f}x"
            )
        print_footer()
    return report.all_passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
