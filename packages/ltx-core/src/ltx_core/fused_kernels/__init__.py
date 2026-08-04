"""Hand-written Triton fused kernels, providing drop-in replacements for the
pluggable ops in :mod:`ltx_core.model.transformer`:

* :mod:`ltx_core.fused_kernels.rope` -- ``apply_rotary_emb`` /
  ``apply_split_rotary_emb`` / ``apply_interleaved_rotary_emb`` (replace
  ``ltx_core.model.transformer.rope``'s functions of the same name).
* :mod:`ltx_core.fused_kernels.modulation` -- ``modulate`` (function) and
  ``TritonAdaZeroFunction`` (``AdaZeroCallable`` implementation, replaces
  ``ltx_core.model.transformer.ops.PytorchAdaZeroFunction``).

Both fall back automatically to the plain PyTorch reference formula when
Triton isn't installed or the tensors aren't on CUDA (see
:func:`ltx_core.fused_kernels._common.triton_usable`), so importing/using this
package is always safe -- the fast path only activates where it can.

Note: there used to be a third kernel here, ``rms_norm`` (a Triton
``RMSNorm``/``rms_norm`` drop-in). It was removed: since PyTorch 2.8,
``torch.nn.functional.rms_norm``/``torch.nn.RMSNorm`` dispatch to a native
fused CUDA kernel (``aten::_fused_rms_norm``/``_fused_rms_norm_backward``,
introduced in pytorch/pytorch#153666) that is itself already a hand-tuned,
bandwidth-bound single-pass implementation -- benchmarking showed our Triton
version no longer beats it by a meaningful margin on that PyTorch version (it
only still wins on pre-2.8 PyTorch, where the reference is an unfused,
multi-kernel decomposition). Just use ``torch.nn.RMSNorm`` /
``torch.nn.functional.rms_norm`` directly instead.

Global enable/disable switch (useful for A/B correctness or perf checks):

    from ltx_core.fused_kernels import set_triton_enabled
    set_triton_enabled(False)  # force every op below back to plain PyTorch
"""

from __future__ import annotations

import os

from ._common import TRITON_AVAILABLE as _TRITON_IMPORTABLE
from .modulation import TritonAdaZeroFunction, modulate
from .rope import (
    LTXRopeType,
    TritonPreAttention,
    apply_interleaved_rotary_emb,
    apply_rotary_emb,
    apply_split_rotary_emb,
)

_triton_enabled = True


def is_triton_enabled() -> bool:
    """Whether the fused Triton kernels are currently enabled.

    ``True`` only when triton is importable *and* :func:`set_triton_enabled`
    hasn't disabled it. Note this does not check for CUDA availability --
    that's handled per-call by each op's own fallback (also active under
    ``TRITON_INTERPRET=1`` for CPU-only testing).
    """
    return _TRITON_IMPORTABLE and _triton_enabled


def set_triton_enabled(enabled: bool) -> None:
    """Globally force every op in this package back to its plain PyTorch
    reference implementation (``enabled=False``), or restore the default
    Triton-when-possible behavior (``enabled=True``).
    """
    global _triton_enabled  # noqa: PLW0603
    _triton_enabled = enabled
    if enabled:
        os.environ.pop("_LTX_FUSED_KERNELS_FORCE_DISABLE", None)
    else:
        os.environ["_LTX_FUSED_KERNELS_FORCE_DISABLE"] = "1"


TRITON_AVAILABLE = _TRITON_IMPORTABLE

__all__ = [
    "TRITON_AVAILABLE",
    "LTXRopeType",
    "TritonAdaZeroFunction",
    "TritonPreAttention",
    "apply_interleaved_rotary_emb",
    "apply_rotary_emb",
    "apply_split_rotary_emb",
    "is_triton_enabled",
    "modulate",
    "set_triton_enabled",
]
