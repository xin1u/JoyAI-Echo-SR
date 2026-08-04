"""Randomized-shape stress test for the fused kernels (RoPE, Modulation).

The other ``test_*.py`` scripts check a fixed, hand-picked set of shapes
(including a few deliberately awkward ones: odd sequence length, non-power-of-2
feature dim, ...). This script complements them by drawing many *random*
shapes per dtype -- arbitrary batch/seq/feature sizes, arbitrary broadcast
combinations, arbitrary (even) head dims for RoPE -- and aggregates the
resulting relative-L2 error distribution (min/mean/max) per dtype, so "does
this generalize to any shape, not just the ones we thought to test" has an
actual statistical answer instead of just a handful of anecdotes.

Run directly: ``python -m ltx_core.fused_kernels.tests.test_fuzz``.
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field

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
    os.execv(sys.executable, [sys.executable, "-m", "ltx_core.fused_kernels.tests.test_fuzz", *sys.argv[1:]])

from ltx_core.fused_kernels import apply_interleaved_rotary_emb, apply_split_rotary_emb, modulate
from ltx_core.fused_kernels.tests._test_utils import (
    DEVICE,
    DTYPES,
    REL_L2_TOLERANCES,
    print_footer,
    print_header,
    rel_l2,
)
from ltx_core.model.transformer.ops import PytorchAdaZeroFunction
from ltx_core.model.transformer.rope import apply_interleaved_rotary_emb as reference_interleaved
from ltx_core.model.transformer.rope import apply_split_rotary_emb as reference_split

_reference_modulate = PytorchAdaZeroFunction()

N_TRIALS_PER_DTYPE = 20


@dataclass
class FuzzStats:
    op: str
    dtype: torch.dtype
    rl2: list[float] = field(default_factory=list)
    n_fail: int = 0
    worst_shape: tuple | None = None
    _worst_err: float = -1.0

    def add(self, err: float, shape: tuple, tol: float) -> None:
        self.rl2.append(err)
        if err > self._worst_err:
            self._worst_err = err
            self.worst_shape = shape
        if err > tol:
            self.n_fail += 1

    @property
    def n(self) -> int:
        return len(self.rl2)

    @property
    def mean(self) -> float:
        return sum(self.rl2) / max(self.n, 1)

    @property
    def max(self) -> float:
        return max(self.rl2, default=0.0)

    @property
    def min(self) -> float:
        return min(self.rl2, default=0.0)


def _random_shape_rope(rng: random.Random) -> tuple[int, int, int, int]:
    b = rng.randint(1, 4)
    h = rng.randint(1, 6)
    t = rng.randint(1, 40)
    d = rng.randint(1, 64) * 2  # always even
    return b, h, t, d


def fuzz_rope_split(seed: int = 5678) -> dict[torch.dtype, FuzzStats]:
    rng = random.Random(seed)
    stats = {dtype: FuzzStats("RoPE-SPLIT", dtype) for dtype in DTYPES}
    for dtype in DTYPES:
        torch.manual_seed(seed)
        for _ in range(N_TRIALS_PER_DTYPE):
            b, h, t, d = _random_shape_rope(rng)
            cb = 1 if rng.random() < 0.5 else b
            x = torch.randn(b, h, t, d, device=DEVICE, dtype=dtype, requires_grad=True)
            x_ref = x.detach().clone().requires_grad_(True)
            cos = torch.randn(cb, h, t, d // 2, device=DEVICE, dtype=dtype)
            sin = torch.randn(cb, h, t, d // 2, device=DEVICE, dtype=dtype)

            out = apply_split_rotary_emb(x, cos, sin)
            out_ref = reference_split(x_ref, cos, sin)
            grad = torch.randn_like(out)
            out.backward(grad)
            out_ref.backward(grad)

            err = max(rel_l2(out, out_ref), rel_l2(x.grad, x_ref.grad))
            stats[dtype].add(err, (b, h, t, d, cb), REL_L2_TOLERANCES[dtype])
    return stats


def fuzz_rope_interleaved(seed: int = 9012) -> dict[torch.dtype, FuzzStats]:
    rng = random.Random(seed)
    stats = {dtype: FuzzStats("RoPE-INTERLEAVED", dtype) for dtype in DTYPES}
    for dtype in DTYPES:
        torch.manual_seed(seed)
        for _ in range(N_TRIALS_PER_DTYPE):
            b, h, t, d = _random_shape_rope(rng)
            shape = (b, h, t, d)
            x = torch.randn(shape, device=DEVICE, dtype=dtype, requires_grad=True)
            x_ref = x.detach().clone().requires_grad_(True)
            cos = torch.randn(*shape[:-1], d // 2, device=DEVICE, dtype=dtype).repeat_interleave(2, dim=-1)
            sin = torch.randn(*shape[:-1], d // 2, device=DEVICE, dtype=dtype).repeat_interleave(2, dim=-1)

            out = apply_interleaved_rotary_emb(x, cos, sin)
            out_ref = reference_interleaved(x_ref, cos, sin)
            grad = torch.randn_like(out)
            out.backward(grad)
            out_ref.backward(grad)

            err = max(rel_l2(out, out_ref), rel_l2(x.grad, x_ref.grad))
            stats[dtype].add(err, shape, REL_L2_TOLERANCES[dtype])
    return stats


def _random_shape_modulate(rng: random.Random) -> tuple[int, int, int]:
    b = rng.randint(1, 4)
    t = rng.randint(1, 64)
    d = rng.randint(1, 4096)
    return b, t, d


def fuzz_modulation(seed: int = 3456) -> dict[torch.dtype, FuzzStats]:
    rng = random.Random(seed)
    stats = {dtype: FuzzStats("Modulation", dtype) for dtype in DTYPES}
    for dtype in DTYPES:
        torch.manual_seed(seed)
        for _ in range(N_TRIALS_PER_DTYPE):
            b, t, d = _random_shape_modulate(rng)
            scale_t = 1 if rng.random() < 0.5 else t
            x = torch.randn(b, t, d, device=DEVICE, dtype=dtype, requires_grad=True)
            scale = torch.randn(b, scale_t, d, device=DEVICE, dtype=dtype, requires_grad=True)
            shift = torch.randn(b, scale_t, d, device=DEVICE, dtype=dtype, requires_grad=True)
            x_ref = x.detach().clone().requires_grad_(True)
            scale_ref = scale.detach().clone().requires_grad_(True)
            shift_ref = shift.detach().clone().requires_grad_(True)

            out = modulate(x, 1e-6, scale, shift)
            out_ref = _reference_modulate(x_ref, 1e-6, scale_ref, shift_ref)
            grad = torch.randn_like(out)
            out.backward(grad)
            out_ref.backward(grad)

            err = max(
                rel_l2(out, out_ref),
                rel_l2(x.grad, x_ref.grad),
                rel_l2(scale.grad, scale_ref.grad),
                rel_l2(shift.grad, shift_ref.grad),
            )
            stats[dtype].add(err, (b, t, d, scale_t), REL_L2_TOLERANCES[dtype])
    return stats


def print_stats(op: str, per_dtype: dict[torch.dtype, FuzzStats]) -> bool:
    print_header(f"[{op}] random-shape fuzz ({N_TRIALS_PER_DTYPE} trials/dtype, device={DEVICE})")
    all_ok = True
    for dtype, s in per_dtype.items():
        ok = s.n_fail == 0
        all_ok = all_ok and ok
        tag = "PASS" if ok else "FAIL"
        print(
            f"  [{tag}] dtype={dtype!s:16} n={s.n:3d}  rel_l2 min/mean/max="
            f"{s.min:.3e}/{s.mean:.3e}/{s.max:.3e}  tol={REL_L2_TOLERANCES[dtype]:.1e}  "
            f"fails={s.n_fail}  worst_shape={s.worst_shape}"
        )
    print_footer()
    return all_ok


def main() -> bool:
    ok = True
    ok &= print_stats("RoPE-SPLIT", fuzz_rope_split())
    ok &= print_stats("RoPE-INTERLEAVED", fuzz_rope_interleaved())
    ok &= print_stats("Modulation", fuzz_modulation())
    return ok


if __name__ == "__main__":
    passed = main()
    raise SystemExit(0 if passed else 1)
