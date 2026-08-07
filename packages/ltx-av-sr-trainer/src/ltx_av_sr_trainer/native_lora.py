from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn


def _sanitize_module_name(name: str) -> str:
    return name.replace(".", "__")


def _get_tensor_from_state_dict_by_suffix(state_dict: dict[str, Tensor], suffix: str) -> Tensor:
    if suffix in state_dict:
        return state_dict[suffix]

    matches = [key for key in state_dict.keys() if key.endswith(suffix)]
    if len(matches) != 1:
        preview = matches[:5]
        raise KeyError(
            f"Could not uniquely resolve state_dict suffix {suffix!r}. "
            f"matches={preview}, count={len(matches)}"
        )
    return state_dict[matches[0]]


def _resolve_target_modules(model: nn.Module, target_modules: list[str]) -> list[str]:
    """Resolve PEFT-style target module patterns to concrete full module paths.

    Official trainer configs commonly use suffix-style names such as
    ``attn1.to_k`` or ``ff.net.0.proj`` instead of full paths like
    ``transformer_blocks.0.attn1.to_k``. For export compatibility we must
    ultimately save concrete full module names, so we resolve them here.
    """
    available = dict(model.named_modules())
    available_names = list(available.keys())

    resolved: list[str] = []
    missing: list[str] = []

    for target in target_modules:
        if target in available:
            resolved.append(target)
            continue

        matches = [name for name in available_names if name == target or name.endswith(f".{target}")]
        if not matches:
            missing.append(target)
            continue
        resolved.extend(matches)

    if missing:
        raise ValueError(f"Target modules not found in model: {missing}")

    # Stable de-duplication while preserving traversal order.
    deduped = list(dict.fromkeys(resolved))
    return deduped


class NativeLoRAManager(nn.Module):
    """Native LoRA sidecar for exact delta-style save/load compatibility.

    This does **not** replace target layers with PEFT adapter modules.
    Instead, it keeps the original model structure intact and injects low-rank
    deltas through forward hooks on existing modules.

    Exported checkpoint keys exactly follow the official loader convention:

    - ``diffusion_model.<module>.lora_A.weight``
    - ``diffusion_model.<module>.lora_B.weight``
    """

    class _AttachedLoRAAdapter(nn.Module):
        def __init__(
            self,
            *,
            in_features: int,
            out_features: int,
            rank: int,
            param_dtype: torch.dtype,
        ) -> None:
            super().__init__()
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=param_dtype))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=param_dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)

    def __init__(
        self,
        model: nn.Module,
        *,
        target_modules: list[str],
        rank: int,
        alpha: int,
        module_ranks: dict[str, int] | None = None,
        module_alphas: dict[str, int] | None = None,
        dropout: float = 0.0,
        param_dtype: torch.dtype = torch.float32,
        attach_to_targets: bool = False,
        namespace: str | None = None,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.target_modules = list(target_modules)
        self.dropout_p = dropout
        self.module_ranks = dict(module_ranks or {})
        self.module_alphas = dict(module_alphas or {})
        self.enabled = True
        self.param_dtype = param_dtype
        self.attach_to_targets = attach_to_targets
        self.namespace = _sanitize_module_name(namespace or "default")
        self.adapter_attr_name = f"_native_lora_{self.namespace}"

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self._module_name_map: dict[str, str] = {}
        self._module_rank_map: dict[str, int] = {}
        self._module_alpha_map: dict[str, int] = {}
        self._module_scaling_map: dict[str, float] = {}
        self._module_to_linear: dict[str, nn.Linear] = {}
        self._attached_adapters: dict[str, NativeLoRAManager._AttachedLoRAAdapter] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        available = dict(model.named_modules())
        resolved_target_modules = _resolve_target_modules(model, self.target_modules)
        self.resolved_target_modules = resolved_target_modules

        for module_name in resolved_target_modules:
            module = available[module_name]
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"NativeLoRAManager currently supports nn.Linear only. "
                    f"Module '{module_name}' is {type(module).__name__}."
                )

            key = _sanitize_module_name(module_name)
            self._module_name_map[module_name] = key
            self._module_to_linear[module_name] = module
            module_rank = int(self.module_ranks.get(module_name, self.rank))
            module_alpha = int(self.module_alphas.get(module_name, self.alpha if self.alpha is not None else module_rank))
            module_scaling = module_alpha / module_rank
            self._module_rank_map[module_name] = module_rank
            self._module_alpha_map[module_name] = module_alpha
            self._module_scaling_map[module_name] = module_scaling

            if self.attach_to_targets:
                if hasattr(module, self.adapter_attr_name):
                    raise ValueError(
                        f"Target module '{module_name}' already has adapter attr '{self.adapter_attr_name}'. "
                        "Use a unique namespace per NativeLoRAManager."
                    )
                adapter = self._AttachedLoRAAdapter(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=module_rank,
                    param_dtype=self.param_dtype,
                )
                module.add_module(self.adapter_attr_name, adapter)
                self._attached_adapters[module_name] = adapter
            else:
                a = nn.Parameter(torch.zeros(module_rank, module.in_features, dtype=self.param_dtype))
                b = nn.Parameter(torch.zeros(module.out_features, module_rank, dtype=self.param_dtype))
                nn.init.kaiming_uniform_(a, a=5**0.5)
                nn.init.zeros_(b)

                self.lora_A[key] = a
                self.lora_B[key] = b
            self._handles.append(module.register_forward_hook(self._make_hook(module_name)))

    def _get_adapter(self, module_name: str) -> tuple[nn.Parameter, nn.Parameter]:
        if self.attach_to_targets:
            adapter = self._attached_adapters[module_name]
            return adapter.lora_A, adapter.lora_B

        key = self._module_name_map[module_name]
        return self.lora_A[key], self.lora_B[key]

    def _make_hook(self, module_name: str):
        scaling = self._module_scaling_map[module_name]

        def hook(_module: nn.Linear, inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
            if not self.enabled:
                return output
            x = inputs[0]
            x = self._dropout(x)
            a_param, b_param = self._get_adapter(module_name)
            a = a_param.to(dtype=x.dtype)
            b = b_param.to(dtype=x.dtype)
            delta = F.linear(F.linear(x, a), b) * scaling
            return output + delta

        return hook

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable this LoRA branch without removing its hooks.

        This is useful for DMD training where one frozen base transformer is
        shared by several LoRA branches (generator / fake score / real score).
        Inactive branches still keep their parameters registered but their
        hooks become exact no-ops.
        """
        self.enabled = enabled

    def export_state_dict(self, model_state_dict: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        state: dict[str, Tensor] = {}
        for module_name in self._module_name_map:
            # Official LTX loader fuses raw B @ A without alpha metadata, so we
            # bake the training-time scaling into the exported weights.
            if model_state_dict is not None:
                if self.attach_to_targets:
                    suffix_prefix = f"{module_name}.{self.adapter_attr_name}"
                    a_weight = (
                        _get_tensor_from_state_dict_by_suffix(model_state_dict, f"{suffix_prefix}.lora_A")
                        .detach()
                        .cpu()
                        .contiguous()
                    )
                    b_weight = (
                        _get_tensor_from_state_dict_by_suffix(model_state_dict, f"{suffix_prefix}.lora_B")
                        .detach()
                        .cpu()
                        .mul(self._module_scaling_map[module_name])
                        .contiguous()
                    )
                else:
                    key = self._module_name_map[module_name]
                    a_weight = _get_tensor_from_state_dict_by_suffix(model_state_dict, f"lora_A.{key}").detach().cpu().contiguous()
                    b_weight = (
                        _get_tensor_from_state_dict_by_suffix(model_state_dict, f"lora_B.{key}")
                        .detach()
                        .cpu()
                        .mul(self._module_scaling_map[module_name])
                        .contiguous()
                    )
            else:
                a_param, b_param = self._get_adapter(module_name)
                a_weight = a_param.detach().cpu().contiguous()
                b_weight = b_param.detach().cpu().mul(self._module_scaling_map[module_name]).contiguous()
            state[f"diffusion_model.{module_name}.lora_A.weight"] = a_weight
            state[f"diffusion_model.{module_name}.lora_B.weight"] = b_weight
        return state

    def load_lora_weights(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix == ".safetensors":
            sd = load_file(str(path))
        else:
            sd = torch.load(path, map_location="cpu", weights_only=False)

        for module_name in self._module_name_map:
            a_key = f"diffusion_model.{module_name}.lora_A.weight"
            b_key = f"diffusion_model.{module_name}.lora_B.weight"
            a_param, b_param = self._get_adapter(module_name)
            if a_key in sd:
                a_param.data.copy_(sd[a_key].to(dtype=a_param.dtype))
            if b_key in sd:
                b_param.data.copy_(sd[b_key].to(dtype=b_param.dtype) / self._module_scaling_map[module_name])

    def named_lora_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for module_name in self._module_name_map:
            a_param, b_param = self._get_adapter(module_name)
            yield f"{module_name}.lora_A.weight", a_param
            yield f"{module_name}.lora_B.weight", b_param

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:  # type: ignore[override]
        if self.attach_to_targets:
            for _name, param in self.named_lora_parameters():
                yield param
            return
        yield from super().parameters(recurse=recurse)

    def requires_grad_(self, requires_grad: bool = True):  # type: ignore[override]
        for _name, param in self.named_lora_parameters():
            param.requires_grad_(requires_grad)
        return self

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
