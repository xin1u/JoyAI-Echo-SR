"""EchoLoRA: Hook-based LoRA training with merge-based export.

Training: forward hooks inject delta → gradients flow to A/B
Export:   weight.data += (B @ A) * scaling → save full model weights

FSDP compatible via attach_to_targets mode (A/B registered as submodules
of target Linear layers, sharded together with weights).

Checkpoint format compatible with official LTX LoRA:
  diffusion_model.{module}.lora_A.weight
  diffusion_model.{module}.lora_B.weight  (scaling baked in)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import Tensor

from echo_sr import logger


def _sanitize(name: str) -> str:
    return name.replace(".", "__")


def infer_ranks_from_checkpoint(
    checkpoint_path: str | Path,
    target_modules: list[str],
    model: nn.Module,
) -> dict[str, int]:
    """Pre-scan checkpoint to get per-module LoRA rank."""
    path = Path(checkpoint_path)
    sd = load_file(str(path)) if path.suffix == ".safetensors" else torch.load(path, map_location="cpu", weights_only=False)

    available = dict(model.named_modules())
    resolved = []
    for target in target_modules:
        matches = [
            name for name, mod in available.items()
            if (name == target or name.endswith(f".{target}")) and isinstance(mod, nn.Linear)
        ]
        resolved.extend(matches)

    ranks = {}
    for module_name in resolved:
        # Support both formats: xxx.lora_A.weight (official) and xxx.lora_A (FSDP state_dict)
        a_key = f"diffusion_model.{module_name}.lora_A.weight"
        a_key_alt = f"diffusion_model.{module_name}.lora_A"
        if a_key in sd:
            ranks[module_name] = sd[a_key].shape[0]
        elif a_key_alt in sd:
            ranks[module_name] = sd[a_key_alt].shape[0]
    return ranks


class _Adapter(nn.Module):
    """Lightweight A/B parameter pair attached to a target Linear."""

    def __init__(self, rank: int, in_features: int, out_features: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)


class EchoLoRA(nn.Module):
    """Hook-based LoRA with FSDP-compatible attach_to_targets.

    Each target Linear gets an _Adapter submodule containing A/B.
    Forward hooks inject delta = (B @ A @ x) * scaling.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        target_modules: list[str],
        rank: int = 384,
        alpha: int = 384,
        module_ranks: dict[str, int] | None = None,
        checkpoint: str | Path | None = None,
        lora_structure_from: str | Path | None = None,
        namespace: str = "default",
        param_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.namespace = namespace
        self._adapter_attr = f"_echo_lora_{namespace}"
        self.enabled = True
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        # Infer per-module ranks: checkpoint > lora_structure_from > default rank
        if module_ranks is None:
            ref = checkpoint or lora_structure_from
            if ref:
                module_ranks = infer_ranks_from_checkpoint(ref, target_modules, model)

        # Resolve targets
        available = dict(model.named_modules())
        self._targets: dict[str, nn.Linear] = {}
        for target in target_modules:
            for name, mod in available.items():
                if (name == target or name.endswith(f".{target}")) and isinstance(mod, nn.Linear):
                    self._targets[name] = mod

        if not self._targets:
            raise ValueError(f"No Linear modules found for: {target_modules}")

        # Create and attach adapters
        self._module_ranks: dict[str, int] = {}
        self._module_scalings: dict[str, float] = {}

        for module_name, linear in self._targets.items():
            mod_rank = (module_ranks or {}).get(module_name, rank)
            mod_scaling = alpha / mod_rank
            self._module_ranks[module_name] = mod_rank
            self._module_scalings[module_name] = mod_scaling

            adapter = _Adapter(mod_rank, linear.in_features, linear.out_features, param_dtype)
            linear.add_module(self._adapter_attr, adapter)
            self._handles.append(linear.register_forward_hook(self._make_hook(module_name)))

        # Load weights if provided
        if checkpoint:
            self.load_weights(checkpoint)

        logger.info(f"EchoLoRA({namespace}): {self.num_parameters():,} params, "
                    f"{len(self._targets)} modules, ranks={set(self._module_ranks.values())}")

    def _get_adapter(self, module_name: str) -> _Adapter:
        return getattr(self._targets[module_name], self._adapter_attr)

    def _make_hook(self, module_name: str):
        scaling = self._module_scalings[module_name]

        def hook(_module: nn.Linear, inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
            if not self.enabled:
                return output
            adapter = getattr(_module, self._adapter_attr)
            x = inputs[0]
            a = adapter.lora_A.to(dtype=x.dtype)
            b = adapter.lora_B.to(dtype=x.dtype)
            return output + F.linear(F.linear(x, a), b) * scaling

        return hook

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    # ---- Parameter access ----

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        for module_name in self._targets:
            adapter = self._get_adapter(module_name)
            yield adapter.lora_A
            yield adapter.lora_B

    def named_lora_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for module_name in self._targets:
            adapter = self._get_adapter(module_name)
            yield f"{module_name}.lora_A", adapter.lora_A
            yield f"{module_name}.lora_B", adapter.lora_B

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def requires_grad_(self, requires_grad: bool = True):
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    # ---- Save / Load ----

    def save_weights(self, path: str | Path, metadata: dict[str, str] | None = None) -> None:
        """Export LoRA weights (scaling baked into B for official format)."""
        state = {}
        for module_name in self._targets:
            adapter = self._get_adapter(module_name)
            scaling = self._module_scalings[module_name]
            state[f"diffusion_model.{module_name}.lora_A.weight"] = adapter.lora_A.detach().cpu()
            state[f"diffusion_model.{module_name}.lora_B.weight"] = (adapter.lora_B.detach().cpu() * scaling)
        save_file(state, str(path), metadata=metadata)
        logger.info(f"Saved LoRA weights: {path} ({len(state)} tensors)")

    def load_weights(self, path: str | Path) -> None:
        """Load from official LTX LoRA format."""
        path = Path(path)
        sd = load_file(str(path)) if path.suffix == ".safetensors" else torch.load(path, map_location="cpu", weights_only=False)
        self.load_weights_from_state_dict(sd)
        logger.info(f"Loaded LoRA weights from {path.name}")

    def load_weights_from_state_dict(self, sd: dict) -> None:
        """Load LoRA weights from a state dict."""
        loaded = 0
        for module_name in self._targets:
            adapter = self._get_adapter(module_name)
            scaling = self._module_scalings[module_name]
            a_key = f"diffusion_model.{module_name}.lora_A.weight"
            b_key = f"diffusion_model.{module_name}.lora_B.weight"
            if a_key not in sd:
                a_key = f"diffusion_model.{module_name}.lora_A"
            if b_key not in sd:
                b_key = f"diffusion_model.{module_name}.lora_B"
            if a_key in sd:
                adapter.lora_A.data.copy_(sd[a_key].to(dtype=adapter.lora_A.dtype))
                loaded += 1
            if b_key in sd:
                adapter.lora_B.data.copy_(sd[b_key].to(dtype=adapter.lora_B.dtype) / scaling)
        logger.info(f"Loaded LoRA weights: {loaded}/{len(self._targets)} modules")

    def export_merged_state_dict(self, model: nn.Module) -> dict[str, Tensor]:
        """Merge LoRA into base weights and return full state dict for open-source release."""
        # Temporarily merge
        for module_name, linear in self._targets.items():
            adapter = self._get_adapter(module_name)
            scaling = self._module_scalings[module_name]
            delta = (adapter.lora_B @ adapter.lora_A).to(dtype=linear.weight.dtype, device=linear.weight.device) * scaling
            linear.weight.data.add_(delta)

        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                 if self._adapter_attr not in k}

        # Unmerge
        for module_name, linear in self._targets.items():
            adapter = self._get_adapter(module_name)
            scaling = self._module_scalings[module_name]
            delta = (adapter.lora_B @ adapter.lora_A).to(dtype=linear.weight.dtype, device=linear.weight.device) * scaling
            linear.weight.data.sub_(delta)

        return state

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class EchoDMD:
    """Three-branch LoRA manager for DMD2 distillation.

    FSDP-compatible: single set of EchoLoRA adapters (student).
    Teacher and critic weights stored as registered buffers on each adapter,
    so they get sharded by FSDP together with student weights.
    Branch switching selects which weights the hook uses — no data copy needed.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        target_modules: list[str],
        rank: int = 384,
        alpha: int = 384,
        teacher_checkpoint: str | Path,
        student_init: str = "from_teacher",
        student_checkpoint: str | Path | None = None,
        lora_structure_from: str | Path | None = None,
        param_dtype: torch.dtype = torch.float32,
        teacher_state_dict: dict | None = None,
    ) -> None:
        # Determine student init checkpoint
        # Priority: student_checkpoint > from_teacher > zero
        if student_checkpoint:
            stu_ckpt = student_checkpoint
        elif student_init == "from_teacher":
            stu_ckpt = teacher_checkpoint
        else:
            stu_ckpt = None  # zero init

        # Create student LoRA (the only one FSDP manages)
        self.student = EchoLoRA(
            model,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            lora_structure_from=teacher_checkpoint,
            checkpoint=stu_ckpt,
            namespace="default",
            param_dtype=param_dtype,
        )

        # Load teacher weights and register as buffers on each adapter
        if teacher_state_dict is not None:
            sd = teacher_state_dict
        else:
            sd = load_file(str(teacher_checkpoint)) if str(teacher_checkpoint).endswith(".safetensors") else torch.load(str(teacher_checkpoint), map_location="cpu", weights_only=False)

        for module_name in self.student._targets:
            adapter = self.student._get_adapter(module_name)
            scaling = self.student._module_scalings[module_name]
            a_key = f"diffusion_model.{module_name}.lora_A.weight"
            b_key = f"diffusion_model.{module_name}.lora_B.weight"
            if a_key not in sd:
                a_key = f"diffusion_model.{module_name}.lora_A"
            if b_key not in sd:
                b_key = f"diffusion_model.{module_name}.lora_B"
            if a_key in sd:
                t_a = sd[a_key].to(dtype=param_dtype)
                t_b = sd[b_key].to(dtype=param_dtype) / scaling
                # Teacher: frozen buffers (no grad, sharded by FSDP)
                adapter.register_buffer("teacher_A", t_a)
                adapter.register_buffer("teacher_B", t_b)
                # Critic: trainable parameters (get gradients in critic update)
                adapter.critic_A = nn.Parameter(t_a.clone())
                adapter.critic_B = nn.Parameter(t_b.clone())

        # Replace the hook to support branch switching
        self._active = "student"
        self._replace_hooks(model)

        n = self.student.num_parameters()
        logger.info(f"EchoDMD: {n:,} params/branch (teacher+critic as buffers), init={student_init}")

    def _replace_hooks(self, model: nn.Module) -> None:
        """Replace EchoLoRA hooks with branch-aware hooks."""
        # Remove original hooks
        self.student.remove_hooks()

        # Register new hooks that check self._active
        for module_name, linear in self.student._targets.items():
            scaling = self.student._module_scalings[module_name]
            adapter_attr = self.student._adapter_attr
            handle = linear.register_forward_hook(self._make_branch_hook(module_name, scaling, adapter_attr))
            self.student._handles.append(handle)

    def _make_branch_hook(self, module_name: str, scaling: float, adapter_attr: str):
        dmd = self  # closure ref

        def hook(_module: nn.Linear, inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
            if not dmd.student.enabled:
                return output
            adapter = getattr(_module, adapter_attr)
            x = inputs[0]

            if dmd._active == "student":
                a = adapter.lora_A.to(dtype=x.dtype)
                b = adapter.lora_B.to(dtype=x.dtype)
            elif dmd._active == "teacher":
                a = adapter.teacher_A.to(dtype=x.dtype)
                b = adapter.teacher_B.to(dtype=x.dtype)
            elif dmd._active == "critic":
                a = adapter.critic_A.to(dtype=x.dtype)
                b = adapter.critic_B.to(dtype=x.dtype)
            else:
                return output

            return output + F.linear(F.linear(x, a), b) * scaling

        return hook

    def activate(self, branch: str) -> None:
        """Switch active branch (zero-cost, just changes a flag)."""
        self._active = branch

    @property
    def active(self) -> str:
        return self._active

    def critic_parameters(self):
        """Return critic buffer parameters for optimizer."""
        params = []
        for module_name in self.student._targets:
            adapter = self.student._get_adapter(module_name)
            if hasattr(adapter, "critic_A"):
                params.append(adapter.critic_A)
                params.append(adapter.critic_B)
        return params
