"""MergeableLoRA: unified LoRA with hook-based training + merge/unmerge switching.

Combines NativeLoRA (hook training, FSDP compatible) + merge/unmerge (branch switching, export).

Two modes of operation:
  1. ACTIVE mode (hook enabled): forward hook injects delta, gradients flow to A/B
  2. MERGED mode (hook disabled): delta baked into base weight, zero-cost inference

For DMD distillation with 3 branches (student/teacher/critic):
  - Student: ACTIVE mode (hook enabled, requires_grad=True) → gets gradients
  - Teacher: MERGED when needed for scoring, unmerged otherwise (frozen)
  - Critic: ACTIVE mode (hook enabled, requires_grad=True) → gets gradients

For export (open-source release):
  - merge() → save full base model state_dict → done

Compatible with official LTX LoRA format:
  - diffusion_model.{module}.lora_A.weight
  - diffusion_model.{module}.lora_B.weight (scaling baked in)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import Tensor


class MergeableLoRA(nn.Module):
    """LoRA branch with hook-based training and merge/unmerge for inference switching.

    Args:
        model: Base model containing target Linear layers.
        target_modules: Module name patterns (suffix matching).
        rank: LoRA rank.
        alpha: LoRA alpha (scaling = alpha / rank).
        dropout: Dropout probability.
        param_dtype: A/B parameter dtype.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        target_modules: list[str],
        rank: int = 384,
        alpha: int = 384,
        module_ranks: dict[str, int] | None = None,
        dropout: float = 0.0,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout_p = dropout
        self.param_dtype = param_dtype
        import weakref
        self._model_ref = weakref.ref(model)  # weak ref to avoid circular reference with FSDP
        self._module_ranks: dict[str, int] = {}
        self._module_scalings: dict[str, float] = {}

        # State
        self._hook_enabled = False
        self._merged = False
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        # Resolve target modules
        self._targets: dict[str, nn.Linear] = {}
        available = dict(model.named_modules())
        for target in target_modules:
            matches = [
                name for name, mod in available.items()
                if (name == target or name.endswith(f".{target}"))
                and isinstance(mod, nn.Linear)
            ]
            for name in matches:
                self._targets[name] = available[name]

        if not self._targets:
            raise ValueError(f"No Linear modules found for targets: {target_modules}")

        # Create A/B parameters (per-module rank support)
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self._dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for module_name, linear in self._targets.items():
            key = module_name.replace(".", "__")
            mod_rank = (module_ranks or {}).get(module_name, rank)
            mod_scaling = alpha / mod_rank
            self._module_ranks[module_name] = mod_rank
            self._module_scalings[module_name] = mod_scaling

            a = nn.Parameter(torch.zeros(mod_rank, linear.in_features, dtype=param_dtype))
            b = nn.Parameter(torch.zeros(linear.out_features, mod_rank, dtype=param_dtype))
            nn.init.kaiming_uniform_(a, a=5**0.5)
            nn.init.zeros_(b)
            self.lora_A[key] = a
            self.lora_B[key] = b

    def _get_key(self, module_name: str) -> str:
        return module_name.replace(".", "__")

    def _get_delta(self, module_name: str) -> Tensor:
        """Compute delta: (B @ A) * scaling."""
        key = self._get_key(module_name)
        scaling = self._module_scalings[module_name]
        return (self.lora_B[key] @ self.lora_A[key]) * scaling

    # ---- Hook-based training (ACTIVE mode) ----

    def enable_hooks(self) -> None:
        """Enable forward hooks for training. Gradients flow to A/B."""
        if self._hook_enabled:
            return
        if self._merged:
            self.unmerge()

        for module_name, linear in self._targets.items():
            handle = linear.register_forward_hook(self._make_hook(module_name))
            self._handles.append(handle)
        self._hook_enabled = True

    def disable_hooks(self) -> None:
        """Disable forward hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._hook_enabled = False

    def _make_hook(self, module_name: str):
        key = self._get_key(module_name)
        scaling = self._module_scalings[module_name]

        def hook(_module: nn.Linear, inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
            x = inputs[0]
            x = self._dropout(x)
            a = self.lora_A[key].to(dtype=x.dtype)
            b = self.lora_B[key].to(dtype=x.dtype)
            delta = F.linear(F.linear(x, a), b) * scaling
            return output + delta

        return hook

    # ---- Merge/Unmerge (for branch switching + inference) ----

    def merge(self) -> None:
        """Merge LoRA into base weights. Disables hooks if active."""
        if self._merged:
            return
        if self._hook_enabled:
            self.disable_hooks()
        for module_name, linear in self._targets.items():
            delta = self._get_delta(module_name).to(dtype=linear.weight.dtype, device=linear.weight.device)
            linear.weight.data.add_(delta)
        self._merged = True

    def unmerge(self) -> None:
        """Remove LoRA from base weights."""
        if not self._merged:
            return
        for module_name, linear in self._targets.items():
            delta = self._get_delta(module_name).to(dtype=linear.weight.dtype, device=linear.weight.device)
            linear.weight.data.sub_(delta)
        self._merged = False

    @staticmethod
    def infer_ranks_from_checkpoint(path: str | Path, target_modules: list[str], model: nn.Module) -> dict[str, int]:
        """Infer per-module ranks from a checkpoint file."""
        path = Path(path)
        if path.suffix == ".safetensors":
            sd = load_file(str(path))
        else:
            sd = torch.load(path, map_location="cpu", weights_only=False)

        available = dict(model.named_modules())
        ranks = {}
        for target in target_modules:
            matches = [
                name for name, mod in available.items()
                if (name == target or name.endswith(f".{target}"))
                and isinstance(mod, nn.Linear)
            ]
            for module_name in matches:
                a_key = f"diffusion_model.{module_name}.lora_A.weight"
                if a_key in sd:
                    ranks[module_name] = sd[a_key].shape[0]
        return ranks
        return self._merged

    @property
    def is_merged(self) -> bool:
        return self._merged

    @property
    def is_active(self) -> bool:
        return self._hook_enabled

    # ---- Parameter management ----

    def requires_grad_(self, requires_grad: bool = True) -> "MergeableLoRA":
        for param in self.lora_A.values():
            param.requires_grad_(requires_grad)
        for param in self.lora_B.values():
            param.requires_grad_(requires_grad)
        return self

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        yield from self.lora_A.values()
        yield from self.lora_B.values()

    def named_lora_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for module_name in self._targets:
            key = self._get_key(module_name)
            yield f"{module_name}.lora_A", self.lora_A[key]
            yield f"{module_name}.lora_B", self.lora_B[key]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ---- Save/Load (official LTX LoRA format) ----

    def save_weights(self, path: str | Path, metadata: dict[str, str] | None = None) -> None:
        """Save LoRA weights. Scaling baked into B for official format compatibility."""
        state = {}
        for module_name in self._targets:
            key = self._get_key(module_name)
            scaling = self._module_scalings[module_name]
            a = self.lora_A[key].detach().cpu()
            b = self.lora_B[key].detach().cpu() * scaling
            state[f"diffusion_model.{module_name}.lora_A.weight"] = a
            state[f"diffusion_model.{module_name}.lora_B.weight"] = b
        save_file(state, str(path), metadata=metadata)

    def load_weights(self, path: str | Path) -> None:
        """Load from official LTX LoRA format (scaling baked into B).
        Automatically infers per-module rank from checkpoint tensor shapes.
        """
        path = Path(path)
        if path.suffix == ".safetensors":
            sd = load_file(str(path))
        else:
            sd = torch.load(path, map_location="cpu", weights_only=False)

        loaded = 0
        for module_name in self._targets:
            key = self._get_key(module_name)
            a_key = f"diffusion_model.{module_name}.lora_A.weight"
            b_key = f"diffusion_model.{module_name}.lora_B.weight"
            if a_key in sd:
                a_tensor = sd[a_key]
                ckpt_rank = a_tensor.shape[0]
                current_rank = self._module_ranks[module_name]
                if ckpt_rank != current_rank:
                    # Resize A/B to match checkpoint rank
                    linear = self._targets[module_name]
                    self.lora_A[key] = nn.Parameter(torch.zeros(ckpt_rank, linear.in_features, dtype=self.param_dtype))
                    self.lora_B[key] = nn.Parameter(torch.zeros(linear.out_features, ckpt_rank, dtype=self.param_dtype))
                    self._module_ranks[module_name] = ckpt_rank
                    self._module_scalings[module_name] = self.alpha / ckpt_rank
                self.lora_A[key].data.copy_(a_tensor.to(dtype=self.param_dtype))
                loaded += 1
            if b_key in sd:
                scaling = self._module_scalings[module_name]
                self.lora_B[key].data.copy_(
                    sd[b_key].to(dtype=self.param_dtype) / scaling
                )
        if loaded == 0:
            # Try without "diffusion_model." prefix
            for module_name in self._targets:
                key = self._get_key(module_name)
                a_key = f"{module_name}.lora_A.weight"
                b_key = f"{module_name}.lora_B.weight"
                if a_key in sd:
                    a_tensor = sd[a_key]
                    ckpt_rank = a_tensor.shape[0]
                    current_rank = self._module_ranks[module_name]
                    if ckpt_rank != current_rank:
                        linear = self._targets[module_name]
                        self.lora_A[key] = nn.Parameter(torch.zeros(ckpt_rank, linear.in_features, dtype=self.param_dtype))
                        self.lora_B[key] = nn.Parameter(torch.zeros(linear.out_features, ckpt_rank, dtype=self.param_dtype))
                        self._module_ranks[module_name] = ckpt_rank
                        self._module_scalings[module_name] = self.alpha / ckpt_rank
                    self.lora_A[key].data.copy_(a_tensor.to(dtype=self.param_dtype))
                    loaded += 1
                if b_key in sd:
                    scaling = self._module_scalings[module_name]
                    self.lora_B[key].data.copy_(
                        sd[b_key].to(dtype=self.param_dtype) / scaling
                    )

    def export_merged_state_dict(self) -> dict[str, Tensor]:
        """Export full model weights with LoRA merged (for open-source release)."""
        model = self._model_ref()
        if model is None:
            raise RuntimeError("Base model has been garbage collected")
        was_merged = self._merged
        was_hooked = self._hook_enabled
        if was_hooked:
            self.disable_hooks()
        self.merge()
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if not was_merged:
            self.unmerge()
        if was_hooked:
            self.enable_hooks()
        return state


# ---------------------------------------------------------------------------
# Helper: extract per-module ranks from checkpoint
# ---------------------------------------------------------------------------

def infer_module_ranks_from_checkpoint(checkpoint_path: str | Path, target_modules: list[str], model: nn.Module) -> dict[str, int]:
    """Pre-scan checkpoint to get per-module LoRA rank. Returns {resolved_module_name: rank}."""
    path = Path(checkpoint_path)
    if path.suffix == ".safetensors":
        sd = load_file(str(path))
    else:
        import torch as _torch
        sd = _torch.load(path, map_location="cpu", weights_only=False)

    # Resolve target modules to full names
    available = dict(model.named_modules())
    resolved = []
    for target in target_modules:
        matches = [
            name for name, mod in available.items()
            if (name == target or name.endswith(f".{target}"))
            and isinstance(mod, nn.Linear)
        ]
        resolved.extend(matches)

    module_ranks = {}
    for module_name in resolved:
        a_key = f"diffusion_model.{module_name}.lora_A.weight"
        if a_key in sd:
            module_ranks[module_name] = sd[a_key].shape[0]
    return module_ranks


# ---------------------------------------------------------------------------
# Multi-branch manager for DMD distillation
# ---------------------------------------------------------------------------

class DMDBranchManager:
    """Manages student/teacher/critic LoRA branches for DMD distillation.

    Uses NativeLoRAManager (attach_to_targets=True) for FSDP compatibility.
    Branch switching via set_enabled() — only one branch active at a time.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        target_modules: list[str],
        rank: int = 384,
        alpha: int = 384,
        teacher_checkpoint: str | None = None,
        student_init: str = "from_teacher",
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from ltx_av_sr_trainer.native_lora import NativeLoRAManager

        # Pre-infer per-module ranks from teacher checkpoint
        module_ranks = None
        module_alphas = None
        if teacher_checkpoint:
            ranks = infer_module_ranks_from_checkpoint(teacher_checkpoint, target_modules, model)
            module_ranks = ranks
            module_alphas = {k: alpha for k in ranks}

        common = dict(
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            module_ranks=module_ranks,
            module_alphas=module_alphas,
            dropout=0.0,
            param_dtype=param_dtype,
            attach_to_targets=True,
        )

        self.student = NativeLoRAManager(model, namespace="student", **common)
        self.teacher = NativeLoRAManager(model, namespace="teacher", **common)
        self.critic = NativeLoRAManager(model, namespace="critic", **common)

        # Load weights
        if teacher_checkpoint:
            self.teacher.load_lora_weights(teacher_checkpoint)
            self.critic.load_lora_weights(teacher_checkpoint)
            if student_init == "from_teacher":
                self.student.load_lora_weights(teacher_checkpoint)

        # Set trainability
        self.teacher.requires_grad_(False)
        self.student.requires_grad_(True)
        self.critic.requires_grad_(True)

        # Default: student active
        self.student.set_enabled(True)
        self.teacher.set_enabled(False)
        self.critic.set_enabled(False)
        self._active = "student"

    def activate(self, branch: str) -> None:
        """Switch active branch. Only one at a time via hook enable/disable."""
        if branch == self._active:
            return
        self.get_branch(self._active).set_enabled(False)
        if branch not in ("student", "teacher", "critic"):
            raise ValueError(f"Unknown branch: {branch}")
        self.get_branch(branch).set_enabled(True)
        self._active = branch

    @property
    def active(self) -> str:
        return self._active

    def get_branch(self, name: str):
        return {"student": self.student, "teacher": self.teacher, "critic": self.critic}[name]
