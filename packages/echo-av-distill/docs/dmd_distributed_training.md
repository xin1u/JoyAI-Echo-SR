# Echo SR DMD2 蒸馏：多分支分布式训练架构

## 概述

DMD2 蒸馏需要三个"模型分支"（Student / Teacher / Critic）共享同一个 22B base model，在 8×H200 FSDP 分布式环境下高效切换。核心挑战是 FSDP 的参数分片机制与多分支 LoRA 的兼容性。

---

## 架构设计

### 问题分析

FSDP `FULL_SHARD` 模式下，每个 `nn.Module` 的参数被 flatten 成一个 `FlatParameter` 并分片到各 rank。这意味着：

1. **不能在 forward 外访问完整参数**（只有分片 shard）
2. **不能动态添加/删除子模块**（FSDP 启动时固化了参数结构）
3. **所有 rank 必须执行相同的 forward 次数和顺序**

### 方案对比

| 方案 | 问题 |
|------|------|
| ❌ 三个独立 EchoLoRA (三套 hook) | 三倍参数，FSDP flatten 后通信量 3×，hook 互相干扰 |
| ❌ 单 adapter + 权重 swap | FSDP 分片后 `.data` shape=0，无法 copy |
| ✅ **单 adapter + 分支选择 hook** | 一组 adapter params + teacher/critic 作为同模块的 buffer/param，FSDP 统一管理 |

### 最终方案：`EchoDMD`

```
每个 target Linear module:
├── _echo_lora_default (Adapter submodule, sharded by FSDP)
│   ├── lora_A: Parameter [rank, in]      ← Student (trainable)
│   ├── lora_B: Parameter [out, rank]      ← Student (trainable)
│   ├── teacher_A: Buffer [rank, in]       ← Teacher (frozen, no grad)
│   ├── teacher_B: Buffer [out, rank]      ← Teacher (frozen, no grad)
│   ├── critic_A: Parameter [rank, in]     ← Critic (trainable)
│   └── critic_B: Parameter [out, rank]    ← Critic (trainable)
└── forward_hook: 根据 dmd._active flag 选择哪组 A/B
```

---

## 核心代码（`echo_sr/model/lora.py`）

### EchoDMD 初始化

```python
class EchoDMD:
    def __init__(self, model, target_modules, rank, alpha, teacher_checkpoint, ...):
        # 1. 创建单个 EchoLoRA（student）
        self.student = EchoLoRA(model, namespace="default", checkpoint=teacher_ckpt, ...)
        
        # 2. 在每个 adapter 上注册 teacher/critic 权重
        for module_name in self.student._targets:
            adapter = self.student._get_adapter(module_name)
            # Teacher: frozen buffer（FSDP 会 shard 但不算 grad）
            adapter.register_buffer("teacher_A", teacher_A_weights)
            adapter.register_buffer("teacher_B", teacher_B_weights)
            # Critic: trainable Parameter（FSDP shard + 有 grad）
            adapter.critic_A = nn.Parameter(critic_A_weights)
            adapter.critic_B = nn.Parameter(critic_B_weights)
        
        # 3. 替换 hook 为分支感知版本
        self._replace_hooks(model)
```

### 分支感知 Forward Hook

```python
def _make_branch_hook(self, module_name, scaling, adapter_attr):
    dmd = self  # closure

    def hook(_module, inputs, output):
        adapter = getattr(_module, adapter_attr)
        x = inputs[0]
        
        # 根据当前活跃分支选择权重
        if dmd._active == "student":
            a, b = adapter.lora_A, adapter.lora_B        # Parameters (有梯度)
        elif dmd._active == "teacher":
            a, b = adapter.teacher_A, adapter.teacher_B  # Buffers (无梯度)
        elif dmd._active == "critic":
            a, b = adapter.critic_A, adapter.critic_B    # Parameters (有梯度)
        
        return output + F.linear(F.linear(x, a), b) * scaling
    
    return hook
```

### 分支切换（零开销）

```python
def activate(self, branch: str):
    """切换只改一个 flag，不涉及数据拷贝或通信"""
    self._active = branch  # "student" | "teacher" | "critic"
```

---

## FSDP 兼容性

### 参数分片

FSDP 把每个 `_Adapter` 子模块的所有 tensor 一起 flatten：

```
FlatParameter = [lora_A | lora_B | teacher_A | teacher_B | critic_A | critic_B]
                  ← student →    ← teacher (buffer) →   ← critic →
```

每个 rank 持有 `1/N` 的 shard。Forward 时 all-gather 还原完整参数，hook 里可以正常读取任何分支的权重。

### 为什么不会死锁

| 条件 | 满足 |
|------|------|
| 所有 rank 执行相同的 forward 次数 | ✅ 每步固定 4 次 forward (student + teacher + critic_score + critic_update) |
| 每次 forward 涉及的参数集合相同 | ✅ 同一个 FlatParameter，只是 hook 里读不同 offset |
| backward 次数一致 | ✅ student loss backward 1次 + critic loss backward 1次，所有 rank 相同 |

### 关键约束

- `activate()` 必须**所有 rank 同步调用**（都在同一个代码路径上）
- Teacher forward 在 `torch.no_grad()` 内（buffer 本身也无 grad）
- Critic update 在 `accumulate()` 外独立做 backward

---

## 训练循环

```python
for step in range(steps):
    # ═══ Student forward (σ=1.0) ═══
    dmd.activate("student")
    v_pred, a_pred = transformer(video=..., audio=...)
    loss_student = compute_loss(v_pred, target)
    x0_student = noisy - v_pred  # 一步去噪结果
    
    # ═══ DMD Loss: Teacher score ═══
    dmd.activate("teacher")
    with torch.no_grad():
        pred_real = transformer(video=noisy_x0, ...)
        x0_real = noisy - pred_real * sigma
    
    # ═══ DMD Loss: Critic score ═══
    dmd.activate("critic")
    with torch.no_grad():
        pred_fake = transformer(video=noisy_x0, ...)
        x0_fake = noisy - pred_fake * sigma
    
    # ═══ DMD gradient ═══
    dmd.activate("student")
    dm_update = x0_real - x0_fake
    loss_dm = MSE(x0_student, x0_student + dm_update)
    
    # ═══ Pixel/LPIPS loss ═══
    x0_decoded = TinyDecoder(x0_student)
    loss_pixel = L1(x0_decoded, GT_pixel)
    loss_lpips = LPIPS(x0_decoded, GT_pixel)
    
    # ═══ Student backward ═══
    total_loss = loss_dm + loss_pixel + loss_lpips
    optimizer.zero_grad()
    accelerator.backward(total_loss)
    optimizer.step()
    
    # ═══ Critic update (独立) ═══
    dmd.activate("critic")
    pred_critic = transformer(video=noisy_x0, ...)
    loss_critic = MSE(pred_critic, target_velocity)
    critic_optimizer.zero_grad()
    accelerator.backward(loss_critic)
    critic_optimizer.step()
    
    dmd.activate("student")
```

---

## 显存分析（8×H200 140GB）

| 组件 | 总大小 | 每卡 (FSDP /8) |
|------|--------|----------------|
| Base model (22B fp32) | 88 GB | 11 GB |
| Student LoRA A+B (3.8B fp32) | 15.2 GB | 1.9 GB |
| Teacher A+B (buffers, fp32) | 15.2 GB | 1.9 GB |
| Critic A+B (params, fp32) | 15.2 GB | 1.9 GB |
| Student optimizer (AdamW) | 30.4 GB | 3.8 GB |
| Critic optimizer (AdamW) | 30.4 GB | 3.8 GB |
| LPIPS VGG | 0.5 GB | 0.5 GB |
| TinyDecoder (trainable) | 0.12 GB | 0.12 GB |
| VAE encoder | 2 GB | 2 GB |
| Activations (grad ckpt, batch=1) | ~20 GB | ~20 GB |
| **Total/card** | | **~47 GB** |

实际测量：39.1 GB（gradient checkpointing 大幅压缩 activation）。

---

## 与 v3 Distill 的对比

| 对比 | v3 (`train_distill_v3.py`) | echo-sr (`distiller.py`) |
|------|---------------------------|--------------------------|
| LoRA 管理 | `DMDBranchManager` (3个 NativeLoRA 实例) | `EchoDMD` (单 adapter + buffer/param) |
| 分支切换 | `set_enabled()` 控制 3 个 hook | `_active` flag 在 1 个 hook 内选择 |
| FSDP 兼容 | 依赖 `attach_to_targets=True` + 禁用 peft | 原生兼容（所有权重在同一个子模块） |
| 训练循环 | monkey-patch `_training_step` 闭包 | 独立 `EchoDistiller.train()` 方法 |
| Accumulate | 不用（手动管理） | 不用（手动管理，因为多次 forward） |
| Critic optimizer | 独立 AdamW | 独立 AdamW |
| 显存占比 | 3× adapter 参数 | 3× adapter 参数（相同） |

---

## 注意事项

1. **不能用 `accelerator.accumulate()`**：DMD 每步做 4 次 forward，FSDP 的 accumulate wrapper 追踪 forward 次数会失配
2. **Teacher buffer 不参与 backward**：`register_buffer` 自动排除 grad，即使在 forward 中被读取
3. **Critic 梯度不会污染 student**：critic forward 时 `_active="critic"` → hook 用 `critic_A/B` → 梯度流向 critic params → student params 的 grad 为 None
4. **Checkpoint 只保存 student**：`_save_checkpoint()` 沿用 EchoTrainer 的逻辑，只保存 `lora_A/B`（即 student 权重）
