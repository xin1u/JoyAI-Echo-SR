import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from einops import rearrange

# -----------------------------------------
# 1. 基础工具与常量
# -----------------------------------------
MAX_TENSOR_INDEX = 2**31 - 1

def _build_gaussian_kernel(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """构建高斯核，修复了 sigma=0 导致的 NaN 问题"""
    if sigma <= 0:
        # 如果 sigma 为 0，退化为单位冲激函数（不模糊）
        kernel = torch.zeros((kernel_size, kernel_size), device=device, dtype=torch.float32)
        kernel[kernel_size // 2, kernel_size // 2] = 1.0
    else:
        coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2.0
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
        # 增加 1e-9 防止除零
        kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2 + 1e-9))

    kernel = kernel / (kernel.sum() + 1e-9)
    return kernel.to(dtype=dtype).view(1, 1, kernel_size, kernel_size)

def _apply_gaussian_blur(frames: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """通用的 2D 卷积模糊实现，支持大 Tensor 分块处理"""
    padding = kernel_size // 2
    kernel = _build_gaussian_kernel(kernel_size, sigma, frames.device, frames.dtype)
    kernel = kernel.repeat(frames.shape[1], 1, 1, 1)

    # 分块处理逻辑，防止超过 MAX_TENSOR_INDEX 导致的计算错误
    b, c, h, w = frames.shape
    padded_numel = b * c * (h + 2 * padding) * (w + 2 * padding)
    if padded_numel > MAX_TENSOR_INDEX:
        max_b = max(1, MAX_TENSOR_INDEX // (c * (h + 2 * padding) * (w + 2 * padding)))
        outputs = []
        for i in range(0, b, max_b):
            chunk = frames[i:i + max_b]
            chunk = F.pad(chunk, (padding, padding, padding, padding), mode="reflect")
            outputs.append(F.conv2d(chunk, kernel, groups=c))
        return torch.cat(outputs, dim=0)
    frames = F.pad(frames, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(frames, kernel, groups=c)

# def _apply_gaussian_blur(frames: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
#     """通用的 2D 卷积模糊实现，支持大 Tensor 分块处理防止索引溢出"""
#     device, dtype = frames.device, frames.dtype
#     C = frames.shape[1]
#     padding = kernel_size // 2

#     # 1. 构建高斯核
#     kernel = _build_gaussian_kernel(kernel_size, sigma, device, dtype)
#     kernel = kernel.repeat(C, 1, 1, 1)

#     # 2. 边界填充
#     # 注意：填充会使 Tensor 变得更大，更容易触发溢出，所以先填充再判断
#     frames = F.pad(frames, (padding, padding, padding, padding), mode="reflect")

#     # 3. 分块处理逻辑
#     # 设定安全阈值：10^9 个元素 (约 1GB 浮点数据单次计算)
#     max_elements_per_op = 10**9
#     num_elements = frames.numel()

#     if num_elements > max_elements_per_op:
#         # 计算每一帧所占的元素数，据此决定每批处理多少帧
#         elements_per_frame = frames[0].numel()
#         # 至少处理 1 帧，或者根据阈值计算 chunk_size
#         chunk_size = max(1, max_elements_per_op // elements_per_frame)

#         output_chunks = []
#         for i in range(0, frames.shape[0], chunk_size):
#             chunk = frames[i : i + chunk_size]
#             # 执行卷积
#             out = F.conv2d(chunk, kernel, groups=C)
#             output_chunks.append(out)

#         # 将结果拼接回去
#         return torch.cat(output_chunks, dim=0)
#     else:
#         # 小于阈值，直接全量计算，保持高性能
#         return F.conv2d(frames, kernel, groups=C)
# -----------------------------------------
# 2. Real-ESRGAN 核心算子 (JPEG & Sinc)
# -----------------------------------------
# class FastJPEGSimulator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         K = 8
#         n = torch.arange(K).view(1, K)
#         k = torch.arange(K).view(K, 1)
#         basis = torch.cos(math.pi / K * (n + 0.5) * k) * math.sqrt(2.0 / K)
#         basis[0, :] /= math.sqrt(2.0)
#         kernels = []
#         for i in range(K):
#             for j in range(K):
#                 kernels.append(basis[i:i+1, :].t() @ basis[j:j+1, :])

#         self.register_buffer("weight", torch.stack(kernels).unsqueeze(1))
#         self.register_buffer("q_table_base", torch.tensor([
#             16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
#             14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
#             18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
#             49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99
#         ], dtype=torch.float32).view(64, 1, 1))

#     def forward(self, x, quality):
#         # x: (B*T, C, H, W)
#         bt, c, h, w = x.shape
#         device = x.device
#         dtype = x.dtype

#         q_factor = (5000 / quality) if quality < 50 else (200 - quality * 2)
#         table = (self.q_table_base.to(device) * q_factor / 100.0).clamp(1, 255)
#         weight = self.weight.to(device, dtype)

#         # 设置一个安全的 Chunk Size (例如每次处理 8 帧)
#         chunk_size = 8
#         outputs = []

#         for i in range(0, bt, chunk_size):
#             x_chunk = x[i : i + chunk_size] # 形状: (chunk, c, h, w)
#             curr_chunk_size = x_chunk.shape[0]

#             # 1. 重组并 Padding
#             x_chunk = x_chunk.view(curr_chunk_size * c, 1, h, w)
#             pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
#             # 在这里做 F.pad 就不会报错了，因为 chunk 变小了
#             x_chunk = F.pad(x_chunk, (0, pad_w, 0, pad_h), mode='reflect')

#             # 2. DCT -> 量化 -> IDCT
#             freq = F.conv2d(x_chunk, weight, stride=8)
#             freq = torch.round(freq / table) * table
#             out_chunk = F.conv_transpose2d(freq, weight, stride=8)

#             # 3. 裁剪回原大小并恢复维度
#             out_chunk = out_chunk[:, :, :h, :w].view(curr_chunk_size, c, h, w)
#             outputs.append(out_chunk)

#         return torch.cat(outputs, dim=0).clamp(0.0, 1.0)
# _jpeg_simulator = FastJPEGSimulator()

# class RealESRGAN_FastJPEG(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 1. 预计算 DCT 基函数 (8x8)
#         K = 8
#         n = torch.arange(K).view(1, K)
#         k = torch.arange(K).view(K, 1)
#         # 计算 1D DCT 变换矩阵
#         basis = torch.cos(math.pi / K * (n + 0.5) * k) * math.sqrt(2.0 / K)
#         basis[0, :] /= math.sqrt(2.0)

#         # 2. 生成 64 个 2D DCT 卷积核 (8x8)
#         kernels = []
#         for i in range(K):
#             for j in range(K):
#                 # 外积得到 2D 卷积核
#                 kernels.append(basis[i:i+1, :].t() @ basis[j:j+1, :])

#         # 权重形状为 [64, 1, 8, 8]
#         self.register_buffer("weight", torch.stack(kernels).unsqueeze(1))

#         # 3. 注册标准 JPEG 亮度量化表 (64 维)
#         self.register_buffer("q_table_raw", torch.tensor([
#             16, 11, 10, 16, 24, 40, 51, 61,
#             12, 12, 14, 19, 26, 58, 60, 55,
#             14, 13, 16, 24, 40, 57, 69, 56,
#             14, 17, 22, 29, 51, 87, 80, 62,
#             18, 22, 37, 56, 68, 109, 103, 77,
#             24, 35, 55, 64, 81, 104, 113, 92,
#             49, 64, 78, 87, 103, 121, 120, 101,
#             72, 92, 95, 98, 112, 100, 103, 99
#         ], dtype=torch.float32).view(64, 1, 1))

#     def forward(self, x, quality):
#         # x 形状: (B, C, H, W)，取值 [0, 1]
#         b, c, h, w = x.shape
#         device = x.device

#         # 根据质量采样计算量化表
#         quality = float(quality)
#         q_factor = 5000 / quality if quality < 50 else 200 - quality * 2
#         table = (self.q_table_raw.to(device) * q_factor / 100.0).clamp(1, 255)

#         # 预处理：合并通道，并 Pad 图像以满足 8 像素对齐
#         x = x.view(b * c, 1, h, w)
#         pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
#         x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

#         # 4. 正向 DCT (使用卷积，步长为 8)
#         # 输出形状: (B*C, 64, H/8, W/8)
#         freq = F.conv2d(x, self.weight.to(device, x.dtype), stride=8)

#         # 5. 量化 (关键：频域取整)
#         freq = torch.round(freq / table) * table

#         # 6. 逆向 DCT (使用转置卷积，步长为 8)
#         out = F.conv_transpose2d(freq, self.weight.to(device, x.dtype), stride=8)

#         # 恢复原始尺寸
#         out = out[:, :, :h, :w].view(b, c, h, w)
#         return out.clamp(0.0, 1.0)

# # 初始化实例
# _jpeg_simulator = RealESRGAN_FastJPEG()

# class JPEGArtifactsSimulator(nn.Module):
#     """纯 Torch 实现的 8x8 DCT 块量化，模拟 JPEG 压缩伪影"""
#     def __init__(self):
#         super().__init__()
#         self.register_buffer("q_table", torch.tensor([
#             [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
#             [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
#             [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
#             [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]
#         ], dtype=torch.float32))

#     def forward(self, x, quality):

#         b, c, h, w = x.shape
#         # 根据质量计算缩放因子
#         quality = float(quality)
#         q_factor = (5000 / quality) if quality < 50 else (200 - quality * 2)
#         # table = (self.q_table * q_factor / 100.0).clamp(1, 255)
#         # 将 q_table 移动到输入相同的设备
#         table = (self.q_table.to(x.device) * q_factor / 100.0).clamp(1, 255)

#         # 8x8 DCT 矩阵
#         K = 8
#         n = torch.arange(K, device=x.device, dtype=x.dtype).view(1, K)
#         k = torch.arange(K, device=x.device, dtype=x.dtype).view(K, 1)
#         dct_mat = torch.cos(math.pi / K * (n + 0.5) * k) * math.sqrt(2.0 / K)
#         dct_mat[0, :] /= math.sqrt(2.0)

#         # 对齐到 8 像素块
#         pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
#         x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
#         H, W = x.shape[-2:]
#         blocks = x.view(b, c, H // 8, 8, W // 8, 8).permute(0, 1, 2, 4, 3, 5)

#         # 频域操作
#         dct_blocks = dct_mat @ blocks @ dct_mat.t()
#         quantized = torch.round(dct_blocks / table) * table
#         idct_blocks = dct_mat.t() @ quantized @ dct_mat

#         out = idct_blocks.permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, H, W)
#         return out[..., :h, :w].clamp(0.0, 1.0)

# _jpeg_simulator = JPEGArtifactsSimulator()

def _jpeg_simulator(frames, quality):
    """
    最快的方法：通过随机的‘块状化’来模拟 JPEG 效果
    """
#     quality = rng.randint(50, 95)
    quality = quality
    # 模拟 JPEG 的 8x8 块效应，随机缩放到原图的 1/8 到 1/2
    # scale = max(0.125, quality / 100.0 * 0.5)
    scale = max(0.125, quality / 100.0 * 0.5)
    orig_size = frames.shape[-2:]
    # 下采样使用 Area，上采样使用 Nearest 产生方块感
    small = F.interpolate(frames, scale_factor=scale, mode='bicubic')
    return F.interpolate(small, size=orig_size, mode='bicubic')

# def _apply_sinc_filter(frames, kernel_size=13, rng=None):
#     """产生振铃效应，Real-ESRGAN 的灵魂算子"""
#     device = frames.device
#     C = frames.shape[1]
#     omega = rng.uniform(math.pi / 3, math.pi) if rng else math.pi/2
#     p = (kernel_size - 1) / 2
#     i = torch.arange(-p, p + 1, device=device)
#     grid_j, grid_i = torch.meshgrid(i, i, indexing='ij')
#     dist = torch.sqrt(grid_i**2 + grid_j**2)

#     kernel = torch.sin(omega * dist) / (math.pi * dist + 1e-9)
#     kernel[dist == 0] = omega / math.pi
#     kernel = kernel / (kernel.sum() + 1e-9)

#     # 准备卷积核
#     kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)

#     # 填充边界
#     frames = F.pad(frames, (kernel_size//2,)*4, mode='reflect')

#     # --- 关键修改：分块卷积防止 int32 索引溢出 ---
#     # 设定一个阈值（元素个数），通常超过 10^9 就容易溢出
#     max_elements_per_op = 2**30  # 约 10 亿个元素
#     num_elements = frames.numel()

#     if num_elements > max_elements_per_op:
#         # 计算每一帧的大小，决定每次处理多少帧
#         elements_per_frame = frames[0].numel()
#         chunk_size = max(1, max_elements_per_op // elements_per_frame)

#         output_chunks = []
#         # 按 N (Batch*Time) 维度分块处理
#         for i in range(0, frames.shape[0], chunk_size):
#             chunk = frames[i : i + chunk_size]
#             out = F.conv2d(chunk, kernel, groups=C)
#             output_chunks.append(out)
#         return torch.cat(output_chunks, dim=0)
#     else:
#         # 如果张量不大，直接计算以保持最高性能
#         return F.conv2d(frames, kernel, groups=C)

def _apply_sinc_filter(frames, kernel_size=13, rng=None):
    """产生振铃效应，Real-ESRGAN 的灵魂算子"""
    device = frames.device
    omega = rng.uniform(math.pi / 3, math.pi) if rng else math.pi/2
    p = (kernel_size - 1) / 2
    i = torch.arange(-p, p + 1, device=device)
    grid_j, grid_i = torch.meshgrid(i, i, indexing='ij')
    dist = torch.sqrt(grid_i**2 + grid_j**2)

    kernel = torch.sin(omega * dist) / (math.pi * dist + 1e-9)
    kernel[dist == 0] = omega / math.pi
    kernel = kernel / (kernel.sum() + 1e-9)

    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(frames.shape[1], 1, 1, 1)
    padding = kernel_size // 2
    b, c, h, w = frames.shape
    padded_numel = b * c * (h + 2 * padding) * (w + 2 * padding)
    if padded_numel > MAX_TENSOR_INDEX:
        max_b = max(1, MAX_TENSOR_INDEX // (c * (h + 2 * padding) * (w + 2 * padding)))
        outputs = []
        for i in range(0, b, max_b):
            chunk = frames[i:i + max_b]
            chunk = F.pad(chunk, (padding, padding, padding, padding), mode='reflect')
            outputs.append(F.conv2d(chunk, kernel, groups=c))
        return torch.cat(outputs, dim=0)
    frames = F.pad(frames, (padding, padding, padding, padding), mode='reflect')
    return F.conv2d(frames, kernel, groups=c)

# -----------------------------------------
# 3. 辅助功能 (Sharpening, Resize, Noise)
# -----------------------------------------
@torch.no_grad()
def _apply_enh_sharpening(frames: torch.Tensor, weight=0.6, kernel_size = 3, threshold=2/255.0):
    """锐化增强边缘"""
    blur = _apply_gaussian_blur(frames, kernel_size=kernel_size, sigma=1.0)
    residual = frames - blur
    enhanced = frames + weight * residual
    mask = residual.abs() > threshold
    return torch.where(mask, enhanced, frames).clamp(0.0, 1.0)


def _apply_usm_sharpening(frames: torch.Tensor, weight=0.5, threshold=10/255.0):
    """锐化增强边缘"""
    blur = _apply_gaussian_blur(frames, kernel_size=51, sigma=1.5)
    residual = frames - blur
    mask = residual.abs() > threshold
    sharpened = frames + weight * residual
    return torch.where(mask, sharpened, frames).clamp(0.0, 1.0)

def _smart_resize(frames: torch.Tensor, scale: float, rng: random.Random):
    mode = rng.choice(['bilinear', 'bicubic', 'area'])
    h_new = max(16, int(frames.shape[-2] * scale))
    w_new = max(16, int(frames.shape[-1] * scale))
    # 保持偶数尺寸
    h_new = h_new if h_new % 2 == 0 else h_new + 1
    w_new = w_new if w_new % 2 == 0 else w_new + 1
    return F.interpolate(frames, size=(h_new, w_new), mode=mode, align_corners=False if mode != 'area' else None)

def _add_random_noise(frames, sigma_range, rng, generator):
    """修复了 randn_like 的错误"""
    sigma = rng.uniform(*sigma_range)
    if rng.random() < 0.4: # 灰度噪声
        noise = torch.randn((frames.shape[0], 1, frames.shape[2], frames.shape[3]),
                            dtype=frames.dtype, device=frames.device, generator=generator) * sigma
    else: # 彩色噪声
        noise = torch.randn(frames.shape, dtype=frames.dtype, device=frames.device, generator=generator) * sigma
    return (frames + noise).clamp(0.0, 1.0)

# -----------------------------------------
# 4. 主函数: 二阶退化全流程
# -----------------------------------------
def apply_realbasicvsr_degradation(
    video: torch.Tensor, # (b, c, t, h, w), 范围 [-1, 1]
    rng: random.Random,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Real-ESRGAN 二阶退化逻辑 (视频版)
    输入尺寸: (B, C, T, 256, 256)
    输出尺寸: (B, C, T, 64, 64) -> 完成 1/4 下采样
    """
    if generator is None:
        seed = rng.randint(0, 2**31 - 1)
        generator = torch.Generator(device=video.device).manual_seed(seed)

    b, c, t, h, w = video.shape
    # 预处理: 将 [-1, 1] 转为 [0, 1] 并合并 B*T 维度
    frames = rearrange(video, "b c t h w -> (b t) c h w").contiguous()
    frames = ((frames + 1.0) / 2.0).clamp(0.0, 1.0).to(torch.float32)

    # 第一步: USM 锐化
    frames = _apply_usm_sharpening(frames)

    # --- 第一阶退化 ---
    # 1. 模糊
    k1 = rng.choice([7, 9, 11, 13, 15, 17, 19, 21])
    s1 = rng.uniform(0.25, 3.0)
    frames = _apply_gaussian_blur(frames, k1, s1)
    # 2. 缩放
    frames = _smart_resize(frames, rng.uniform(0.25, 1.5), rng)
    # 3. 噪声
    frames = _add_random_noise(frames, (1/255., 30/255.), rng, generator)
    # 4. JPEG
    if rng.random() < 0.9:
        frames = _jpeg_simulator(frames, rng.randint(30, 95))

    # --- 第二阶退化 ---
    # 1. 模糊 (80% 概率)
    if rng.random() < 0.8:
        k2 = rng.choice([7, 9, 11, 13, 15, 17, 19, 21])
        s2 = rng.uniform(0.2, 1.5)
        frames = _apply_gaussian_blur(frames, k2, s2)
    # 2. 缩放
    frames = _smart_resize(frames, rng.uniform(0.3, 1.2), rng)
    # 3. 噪声
    frames = _add_random_noise(frames, (1/255., 25/255.), rng, generator)
    # 4. JPEG 或 Sinc (Shuffle 阶段)
    if rng.random() < 0.8:
        if rng.random() < 0.5:
            frames = _jpeg_simulator(frames, rng.randint(30, 95))
        else:
            frames = _apply_sinc_filter(frames, kernel_size=13, rng=rng)

    # --- 关键: 强制 1/4 尺寸下采样 ---
    # 无论中间 resize 到了多少，最后这一步确保输入 LQ 是 HR 的 1/4
    # frames = F.interpolate(frames, size=(h // 4, w // 4), mode="bicubic", align_corners=False)
    # 两倍超分训练
    frames = F.interpolate(frames, size=(h // 2, w // 2), mode="bicubic", align_corners=False)
    # 对于我们的训练，还是先resize回到空原间尺度 h w
    frames = F.interpolate(frames, size=(h , w), mode="bicubic", align_corners=False)
    # 后处理: 转回 [-1, 1] 并恢复维度
    frames = (frames * 2.0 - 1.0).clamp(-1.0, 1.0).to(video.dtype)
    return rearrange(frames, "(b t) c h w -> b c t h w", b=b, t=t)

@torch.no_grad()
def cutmix_fixed_ratio(
    A: torch.Tensor,          # (B*T,C,H,W) 例如 frames_aigcdeg
    B: torch.Tensor,          # (B*T,C,H,W) 例如 frames_realesrdeg
    *,
    Bsz: int,                 # batch size = B
    T: int,                   # frames per sample = T（确保每个样本帧数一致）
    ratio_a: float = 0.4,     # AIGC 占比 0.4，realesr 占比 0.6
    p: float = 1.0,           # 每个 video 做 cutmix 的概率
    return_mask: bool = False # 返回 (B*T,1,H,W) mask（同 video 帧一致）
):
    assert A is not None and B is not None
    assert A.shape == B.shape and A.dim() == 4
    N, C, H, W = A.shape
    assert N == Bsz * T, f"N={N} must equal B*T={Bsz*T}"

    device, dtype = A.device, A.dtype
    out = B.clone()

    ratio_a = float(max(0.0, min(1.0, ratio_a)))
    cut_rat = math.sqrt(ratio_a)          # 近似正方形 patch
    cut_w = max(1, int(round(W * cut_rat)))
    cut_h = max(1, int(round(H * cut_rat)))

    # 是否对每个 video 应用
    apply_vid = (torch.rand(Bsz, device=device) < p)

    mask = None
    if return_mask:
        mask = torch.zeros((N, 1, H, W), device=device, dtype=dtype)

    for b in range(Bsz):
        if not apply_vid[b]:
            continue

        # 每个 video 采一次 bbox
        cx = int(torch.randint(0, W, (1,), device=device).item())
        cy = int(torch.randint(0, H, (1,), device=device).item())
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, x1 + cut_w)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, y1 + cut_h)

        # 边界修正：保证 bbox 尽量接近目标大小
        x1 = max(0, x2 - cut_w)
        y1 = max(0, y2 - cut_h)

        sl = slice(b * T, (b + 1) * T)   # 这个 video 的所有帧

        out[sl, :, y1:y2, x1:x2] = A[sl, :, y1:y2, x1:x2]
        if return_mask:
            mask[sl, :, y1:y2, x1:x2] = 1.0

    return (out, mask) if return_mask else out
