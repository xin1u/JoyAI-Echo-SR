"""Run every fused-kernel test script and report a combined pass/fail summary.

Run: ``python -m ltx_core.fused_kernels.tests.run_all``
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
# compilation before any code in this file (or the `test_*` submodules it
# imports below) gets a chance to run. Re-executing the process is the only
# way to make the env var visible before that ancestor import happens.
if __name__ == "__main__" and not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1":
    os.environ["TRITON_INTERPRET"] = "1"
    os.execv(sys.executable, [sys.executable, "-m", "ltx_core.fused_kernels.tests.run_all", *sys.argv[1:]])

from ltx_core.fused_kernels.tests import test_fuzz, test_modulation, test_rope
from ltx_core.fused_kernels.tests._test_utils import DEVICE, print_footer, print_header


def main() -> bool:
    print_header(f"ltx_core.fused_kernels test suite (device={DEVICE})")
    results = {
        "RoPE": test_rope.main(),
        "Modulation": test_modulation.main(),
        "Fuzz (random shapes)": test_fuzz.main(),
    }
    print_header("FINAL SUMMARY")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print(f"\n{'ALL OPS PASSED' if all_ok else 'AT LEAST ONE OP FAILED'}")
    print_footer()
    return all_ok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
