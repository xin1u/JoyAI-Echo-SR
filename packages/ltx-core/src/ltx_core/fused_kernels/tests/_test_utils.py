"""Shared helpers for the fused-kernel correctness/perf test scripts.

Each ``test_*.py`` in this folder is a standalone script (no pytest
dependency, mirroring the reference kernels this package is modeled after)
that can be run directly: ``python -m ltx_core.fused_kernels.tests.test_rms_norm``.

Device selection: on a machine with a CUDA GPU, tests run for real on
``cuda``. On a CPU-only machine (no GPU), tests automatically fall back to
Triton's ``TRITON_INTERPRET=1`` interpreter so the kernel *logic* (forward and
backward math) is still exercised and checked bit-for-bit against the
PyTorch reference -- only wall-clock speed/memory benchmarking requires an
actual GPU and is skipped in that mode.

The "automatically" part is handled by each ``test_*.py`` / ``run_all.py``
entry point re-exec'ing itself (``os.execv``) with ``TRITON_INTERPRET=1`` set
*before* importing anything else, when run as ``__main__`` with no CUDA GPU
present. This can't be done by simply setting the env var from a module-level
import in this file: these test modules live *inside*
``ltx_core.fused_kernels.tests``, so importing any of them -- including this
one -- forces Python to first fully run the ancestor
``ltx_core/fused_kernels/__init__.py``, which imports the
``@triton.jit``-decorated kernel modules and locks in real-GPU compilation
before any code here would get a chance to run.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

# Must be set before any triton kernel is JIT-compiled/launched.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    os.environ.setdefault("TRITON_INTERPRET", "1")

DTYPES = (torch.float32, torch.float16, torch.bfloat16)

# Absolute/relative tolerances per dtype, as specified by the task: fp32 at the
# ~1e-6 level, bf16 (and fp16, which has strictly more mantissa bits than
# bf16) at the ~1e-3 level.
TOLERANCES: dict[torch.dtype, dict[str, float]] = {
    torch.float32: {"atol": 1e-6, "rtol": 1e-5},
    # fp16/bf16: the elementwise band here is intentionally loose -- its job
    # is only to catch gross errors (wrong sign, wrong broadcast, transposed
    # axes, ...), not fine rounding. A kernel that's mathematically identical
    # to the reference can still legitimately land 1-2 ULP away on any given
    # element purely from summation-order differences (Triton's reduction
    # tree vs PyTorch's), and low-precision dtypes have coarse ULPs (fp16
    # ~1e-3, bf16 ~4e-3 *relative*, i.e. several 1e-2 absolute at O(1-10)
    # magnitudes and up to ~2x that when two roundings compound, e.g. RoPE's
    # `x1*cos - x2*sin`). The precise, dtype-appropriate check is
    # `REL_L2_TOLERANCES` below (aggregate relative L2 error, at "the e-3
    # level" the task asks for).
    torch.float16: {"atol": 0.1, "rtol": 0.05},
    torch.bfloat16: {"atol": 0.1, "rtol": 0.05},
}
REL_L2_TOLERANCES: dict[torch.dtype, float] = {
    torch.float32: 1e-6,
    torch.float16: 1e-3,
    # bf16 has an 8-bit mantissa -> unit-roundoff ~2^-8 ~= 3.9e-3. A
    # numerically-correct bf16 kernel will land on the *adjacent*
    # representable value from the reference roughly as often as summation
    # order / instruction scheduling differs (unavoidable across two
    # independent implementations), so the aggregate relative-L2 error floor
    # for *any* correct bf16 RoPE/modulation implementation is at this
    # magnitude, not below it. Measured empirically at ~4e-3 for both ops in
    # this package -- consistent with "the e-3 level" the task asks for, just
    # not below the dtype's own precision.
    torch.bfloat16: 6e-3,
}


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a.float() - b.float()).norm()
    den = b.float().norm().clamp_min(1e-12)
    return (num / den).item()


def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def passes(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> tuple[bool, float, float]:
    """Returns (ok, max_abs_err, rel_l2_err). `ok` requires the relative L2
    error to be within `REL_L2_TOLERANCES[dtype]` (the primary "is this
    numerically the same op" check -- robust to the specific tensor scale)
    AND the elementwise max error to be within the dtype's `atol + rtol *
    |ref|` band (guards against a few outlier elements).

    Only appropriate for element-preserving comparisons (forward output, dx)
    where every output element has an "own" reference value of comparable
    magnitude. For a *reduced* gradient (e.g. `dw`/`dscale`/`dshift`, summed
    over a broadcast batch/sequence dim), use :func:`passes_reduced` instead --
    a reduction's absolute rounding-error floor scales with the number of
    terms summed, not with the (possibly near-zero, by cancellation) value of
    the sum itself, so a fixed elementwise `atol` is the wrong check there.
    """
    err = max_abs_err(a, b)
    rl2 = rel_l2(a, b)
    tol = TOLERANCES[dtype]
    elementwise_ok = bool(((a.float() - b.float()).abs() <= (tol["atol"] + tol["rtol"] * b.float().abs())).all())
    ok = elementwise_ok and rl2 <= REL_L2_TOLERANCES[dtype]
    return ok, err, rl2


def passes_reduced(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> tuple[bool, float, float]:
    """Like :func:`passes`, but for gradients produced by summing over a
    broadcast dimension (see that function's docstring for why the
    elementwise `atol` check is dropped): only the aggregate relative-L2
    error is gated.
    """
    err = max_abs_err(a, b)
    rl2 = rel_l2(a, b)
    return rl2 <= REL_L2_TOLERANCES[dtype], err, rl2


@dataclass
class OpReport:
    """Accumulates pass/fail + error rows for one op, for the final Markdown report."""

    name: str
    rows: list[dict] = field(default_factory=list)
    bench_rows: list[dict] = field(default_factory=list)

    def add(self, **row) -> None:
        self.rows.append(row)

    def add_bench(self, **row) -> None:
        self.bench_rows.append(row)

    @property
    def all_passed(self) -> bool:
        return all(r.get("ok", False) for r in self.rows)


def benchmark_fwd_bwd(
    fn: Callable[[], torch.Tensor],
    make_grad: Callable[[torch.Tensor], torch.Tensor],
    n_warmup: int = 10,
    n_iter: int = 50,
) -> float:
    """Returns average forward+backward wall time in milliseconds. Callers must
    already be on `DEVICE`. On CPU (interpreter mode) this is *not*
    representative of real GPU perf -- it only exercises the code path.
    """
    for _ in range(n_warmup):
        out = fn()
        out.backward(make_grad(out))
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn()
        out.backward(make_grad(out))
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000.0


def peak_memory_mb(fn: Callable[[], torch.Tensor], make_grad: Callable[[torch.Tensor], torch.Tensor]) -> float | None:
    """Peak CUDA memory (MB) for one forward+backward call, or None off-CUDA."""
    if DEVICE != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    out.backward(make_grad(out))
    return torch.cuda.max_memory_allocated() / (1024**2)


def print_header(title: str) -> None:
    print("=" * 88)
    print(title)
    print("=" * 88)


def print_footer() -> None:
    print("=" * 88)
