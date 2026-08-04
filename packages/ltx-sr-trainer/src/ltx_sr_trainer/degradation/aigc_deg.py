import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_video, write_video

# ===========================
# 配置
# ===========================

@dataclass
class DegradeConfig:
    # ---- 结构保持时空下采样 ----
    s_spat: int = 2          # 空间下采样倍率
    s_temp: int = 2          # 时间下采样倍率（低FPS）
    spat_mode: str = "bilinear"
    temp_mode: str = "linear"

    # ---- Temporal morphing ----
    alpha_min: float = 0.2
    alpha_max: float = 0.9

    # ---- 随机丢帧 ----
    p_drop: float = 0.12
    max_drop_run: int = 2    # 最大连续丢帧长度（约束）
    keep_first_last: bool = True

    # ---- 方向运动模糊 ----
    blur_len_min: int = 3
    blur_len_max: int = 21
    blur_bins: int = 12      # 方向离散数（速度/质量折中）
    blur_strength: float = 0.8
    # _min: float = 0.5  # 模糊混合强度（1=全用卷积结果；<1会和原图混合）
    # blur_strength_max: float = 0.8  # 模糊混合强度（1=全用卷积结果；<1会和原图混合）

    # ---- 网格形变 ----
    warp_max_px: float = 6.0     # 位移最大像素（越大越“抖”）
    warp_noise_res: int = 32     # 低分辨率噪声分辨率（越小越低频）
    warp_smooth_ks: int = 9      # 对位移场做空间平滑的核大小（奇数）
    roi_frac: float = 0.5

    # ---- 参数随时间平滑 ----
    traj_basis: str = "sin"      # "sin" 或 "noise"
    traj_smooth_win: int = 9     # 1D 平滑窗口（奇数，越大越平滑）
    traj_cycles_min: float = 0.3 # 低频正弦周期范围（单位：每T帧的 cycles）
    traj_cycles_max: float = 1.2

    # ---- 随机种子（可选）----
    seed: Optional[int] = None


# ===========================
# 工具：时间序列平滑 + 低频轨迹
# ===========================

def _box_smooth_1d(x: torch.Tensor, win: int) -> torch.Tensor:
    """
    x: (..., T)
    简单 1D box filter 平滑，避免参数跳变导致 flicker
    """
    if win <= 1:
        return x
    assert win % 2 == 1
    pad = win // 2
    # 视作 1D conv：把最后一维当作长度
    orig_shape = x.shape
    T = orig_shape[-1]
    y = x.reshape(-1, 1, T)  # (N,1,T)
    k = torch.ones(1, 1, win, device=x.device, dtype=x.dtype) / win
    y = F.pad(y, (pad, pad), mode="replicate")
    y = F.conv1d(y, k)
    return y.reshape(orig_shape)

def _lowfreq_traj(
    T: int,
    device: torch.device,
    dtype: torch.dtype,
    vmin: float,
    vmax: float,
    basis: str = "sin",
    smooth_win: int = 9,
    cycles_min: float = 0.3,
    cycles_max: float = 1.2,
) -> torch.Tensor:
    """
    生成随时间平滑变化的参数轨迹：shape (T,)
    """
    if basis == "sin":
        # 随机低频正弦叠加
        t = torch.linspace(0, 1, T, device=device, dtype=dtype)
        cycles = float(torch.empty(1).uniform_(cycles_min, cycles_max).item())
        phase = float(torch.empty(1).uniform_(0, 2 * math.pi).item())
        y = torch.sin(2 * math.pi * cycles * t + phase)

        # 再叠加一条更低频的正弦，增加自然变化
        cycles2 = float(torch.empty(1).uniform_(cycles_min * 0.5, cycles_max * 0.5).item())
        phase2 = float(torch.empty(1).uniform_(0, 2 * math.pi).item())
        y2 = 0.5 * torch.sin(2 * math.pi * cycles2 * t + phase2)
        traj = y + y2
        # 归一化到 [0,1]
        traj = (traj - traj.min()) / (traj.max() - traj.min() + 1e-6)
    else:
        # 低频噪声：先生成白噪声，再 1D 平滑
        traj = torch.rand(T, device=device, dtype=dtype)
        traj = _box_smooth_1d(traj[None, ...], smooth_win=max(3, smooth_win)).squeeze(0)
        traj = (traj - traj.min()) / (traj.max() - traj.min() + 1e-6)

    # 映射到 [vmin, vmax]
    return traj * (vmax - vmin) + vmin



def _make_soft_ellipse_mask(H, W, cx, cy, ax, ay, device, dtype, softness=2.0):
    """
    生成一个软边椭圆掩码 m∈[0,1]，越靠近中心越接近1，边缘平滑衰减
    """
    yy = torch.arange(H, device=device, dtype=dtype)
    xx = torch.arange(W, device=device, dtype=dtype)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")

    nx = (X - cx) / max(ax, 1e-6)
    ny = (Y - cy) / max(ay, 1e-6)
    r2 = nx * nx + ny * ny  # 椭圆归一化半径平方

    # r2<=1 是椭圆内部；用 exp(-r2*softness) 做软衰减，再截断到[0,1]
    m = torch.exp(-r2 * softness)
    m = m * (r2 <= 1.2).to(dtype)  # 稍微放宽一点范围，避免边缘太薄
    return m.clamp(0, 1)  # (H,W)


def _make_roi_mask_traj(T, H, W, device, dtype,
                        roi_frac=0.35,   # ROI 尺寸占画面比例（越小越局部）
                        drift=True,      # True: ROI 中心随时间漂移；False: 固定
                        smooth_win=9,    # 时间平滑窗口
                        seed=123):
    """
    生成时间一致的局部 ROI 掩码序列：m_t (T,1,H,W)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # ROI 半轴大小（像素）
    ax0 = (W * roi_frac) * (0.6 + 0.4 * torch.rand(1, device=device, dtype=dtype)).item()
    ay0 = (H * roi_frac) * (0.6 + 0.4 * torch.rand(1, device=device, dtype=dtype)).item()

    # ROI 初始中心（避免贴边）
    margin_x = int(ax0 * 0.8)
    margin_y = int(ay0 * 0.8)
    cx0 = torch.randint(low=margin_x, high=max(margin_x + 1, W - margin_x),
                        size=(1,), device=device).item()
    cy0 = torch.randint(low=margin_y, high=max(margin_y + 1, H - margin_y),
                        size=(1,), device=device).item()

    # 中心漂移轨迹（低频、时间平滑）
    if drift and T > 1:
        # 用低频噪声生成漂移（像素单位）
        dx = torch.randn(T, device=device, dtype=dtype)
        dy = torch.randn(T, device=device, dtype=dtype)

        # 1D box smooth（复用你已有的 _box_smooth_1d）
        dx = _box_smooth_1d(dx[None, :], smooth_win).squeeze(0)
        dy = _box_smooth_1d(dy[None, :], smooth_win).squeeze(0)

        # 控制漂移幅度：大概 ROI 尺寸的 10%~20%
        dx = dx / (dx.abs().max() + 1e-6) * (0.18 * ax0)
        dy = dy / (dy.abs().max() + 1e-6) * (0.18 * ay0)

        cxs = torch.clamp(torch.tensor(cx0, device=device, dtype=dtype) + dx,
                          margin_x, W - margin_x - 1)
        cys = torch.clamp(torch.tensor(cy0, device=device, dtype=dtype) + dy,
                          margin_y, H - margin_y - 1)
    else:
        cxs = torch.full((T,), float(cx0), device=device, dtype=dtype)
        cys = torch.full((T,), float(cy0), device=device, dtype=dtype)

    # 生成每帧掩码
    masks = []
    for t in range(T):
        m = _make_soft_ellipse_mask(
            H, W,
            cx=float(cxs[t].item()),
            cy=float(cys[t].item()),
            ax=float(ax0),
            ay=float(ay0),
            device=device, dtype=dtype,
            softness=2.2
        )
        masks.append(m)
    m = torch.stack(masks, dim=0)  # (T,H,W)

    return m[:, None, :, :]  # (T,1,H,W)


# ===========================
# 工具：帧丢弃（带最大连续丢帧约束） + 插值补帧
# ===========================

def _sample_drop_mask(T: int, p_drop: float, max_run: int, keep_first_last: bool = True) -> torch.Tensor:
    """
    返回 keep_mask: (T,) bool
    约束：连续被丢弃的帧 run-length <= max_run
    """
    if T <= 2:
        return torch.ones(T, dtype=torch.bool)

    keep = [True] * T
    run = 0
    for t in range(T):
        if keep_first_last and (t == 0 or t == T - 1):
            keep[t] = True
            run = 0
            continue

        drop = (random.random() < p_drop)
        if drop:
            run += 1
            if run > max_run:
                drop = False
                run = 0
        else:
            run = 0

        keep[t] = (not drop)

    return torch.tensor(keep, dtype=torch.bool)

def _interp_missing_linear(x: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """
    x: (T,C,H,W)
    keep_mask: (T,) bool，True 表示保留帧
    对 missing 帧用最近的左右邻居做线性插值
    """
    T, C, H, W = x.shape
    if keep_mask.all():
        return x

    idx_keep = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
    if idx_keep.numel() == 0:
        # 极端情况：全丢了，直接返回原
        return x

    y = x.clone()
    # 对每个 missing t，找 left/right 最近保留索引
    for t in range(T):
        if keep_mask[t]:
            continue
        # left
        left = idx_keep[idx_keep < t]
        right = idx_keep[idx_keep > t]
        if left.numel() == 0:
            y[t] = x[right.min()]
            continue
        if right.numel() == 0:
            y[t] = x[left.max()]
            continue
        l = int(left.max().item())
        r = int(right.min().item())
        if r == l:
            y[t] = x[l]
        else:
            w = (t - l) / float(r - l)
            y[t] = (1 - w) * x[l] + w * x[r]
    return y


# ===========================
# 工具：方向运动模糊核（线核）+ 卷积
# ===========================

def _make_line_kernel(length: int, angle_rad: float, device, dtype) -> torch.Tensor:
    """
    生成 (1,1,L,L) 线性核，归一化
    """
    if length % 2 == 0:
        length += 1
    L = length
    k = torch.zeros((L, L), device=device, dtype=dtype)
    c = L // 2
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    half = (L - 1) / 2.0
    x0 = int(round(c - dx * half))
    y0 = int(round(c - dy * half))
    x1 = int(round(c + dx * half))
    y1 = int(round(c + dy * half))

    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        if 0 <= x < L and 0 <= y < L:
            k[y, x] = 1.0
    k = k / (k.sum() + 1e-6)
    return k.view(1, 1, L, L)

def _apply_motion_blur_frame(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    x: (C,H,W)
    kernel: (1,1,L,L)
    对 C 通道做 depthwise 卷积
    """
    C, H, W = x.shape
    L = kernel.shape[-1]
    pad = L // 2
    w = kernel.repeat(C, 1, 1, 1)  # (C,1,L,L)
    y = F.conv2d(x.unsqueeze(0), w, padding=pad, groups=C).squeeze(0)
    return y


# ===========================
# 工具：网格形变（低频噪声位移场 + backward warp）
# ===========================

def _make_base_grid(H: int, W: int, device, dtype) -> torch.Tensor:
    """
    返回 grid: (1,H,W,2) in [-1,1]
    """
    ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).unsqueeze(0)

def _spatial_smooth_2d(field: torch.Tensor, ks: int) -> torch.Tensor:
    """
    field: (1,2,H,W) 位移场
    用 box blur 平滑（简单稳定）
    """
    if ks <= 1:
        return field
    assert ks % 2 == 1
    pad = ks // 2
    # depthwise 平滑：2通道分别平滑
    k = torch.ones(2, 1, ks, ks, device=field.device, dtype=field.dtype) / (ks * ks)
    y = F.conv2d(F.pad(field, (pad, pad, pad, pad), mode="replicate"), k, groups=2)
    return y

def _make_warp_field(T: int, H: int, W: int, cfg: DegradeConfig, device, dtype) -> torch.Tensor:
    """
    生成平滑随时间变化的位移场 dt: (T,2,H,W)，单位：像素
    方法：每帧低分辨率噪声 -> 上采样 -> 空间平滑
    再对时间维做 1D 平滑（避免 flicker）
    """
    res = int(cfg.warp_noise_res)
    # low-res random noise (T,2,res,res)
    low = torch.randn(T, 2, res, res, device=device, dtype=dtype)
    # upsample to (H,W)
    dt = F.interpolate(low, size=(H, W), mode="bicubic", align_corners=False)
    dt = _spatial_smooth_2d(dt, cfg.warp_smooth_ks)

    # 归一到 [-1,1] 再映射到像素幅度
    dt = dt / (dt.std(dim=(2, 3), keepdim=True) + 1e-6)
    dt = dt.tanh()  # 稍微限制极值
    dt = dt * float(cfg.warp_max_px)

    # 时间平滑（对每个像素的 dx/dy 轨迹做 box smooth）
    win = cfg.traj_smooth_win
    if win > 1:
        # (T,2,H,W) -> (2*H*W, T)
        tmp = dt.permute(1, 2, 3, 0).reshape(-1, T)
        tmp = _box_smooth_1d(tmp, win)
        dt = tmp.reshape(2, H, W, T).permute(3, 0, 1, 2)

    return dt


# ===========================
# 退化算子 Module
# ===========================

class SyntheticDegradation(nn.Module):
    """
    D(x; η) = composition of temporally coherent augmentations:
      - spatiotemporal downsample + upsample
      - temporal morphing
      - stochastic frame dropping + interpolation
      - directional motion blur
      - grid-based spatial warping
    """
    def __init__(self, cfg: DegradeConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.seed is not None:
            random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

    def _ensure_shape(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        统一到 (B,T,C,H,W)
        """
        if x.dim() == 4:
            return x.unsqueeze(0), True
        if x.dim() == 5:
            return x, False
        raise ValueError("输入必须是 (T,C,H,W) 或 (B,T,C,H,W)")

    def _spatiotemporal_downsample(self, x: torch.Tensor) -> torch.Tensor:
        """
        结构保持时空下采样：先降采样时空，再插值回原 (模拟低FPS/低分辨率)
        x: (B,T,C,H,W)
        """
        B, T, C, H, W = x.shape
        cfg = self.cfg

        # ---- 时间下采样：取每 s_temp 帧 ----
        if cfg.s_temp > 1 and T > 1:
            idx = torch.arange(0, T, cfg.s_temp, device=x.device)
            x_t = x.index_select(1, idx)  # (B,T2,C,H,W)
            T2 = x_t.shape[1]
            # 插值回 T：对每个像素/通道在时间上做线性插值
            # 变形为 (B*C*H*W, 1, T2) -> interpolate -> (B,C,H,W,T)
            xt = x_t.permute(0, 2, 3, 4, 1).reshape(B * C * H * W, 1, T2)
            xt_up = F.interpolate(xt, size=T, mode="linear", align_corners=False)
            x = xt_up.reshape(B, C, H, W, T).permute(0, 4, 1, 2, 3)  # (B,T,C,H,W)

        # ---- 空间下采样：降到 H/ssp, W/ssp 再插值回去 ----
        if cfg.s_spat > 1:
            h2 = max(2, H // cfg.s_spat)
            w2 = max(2, W // cfg.s_spat)
            # (B*T,C,H,W)
            xs = x.reshape(B * T, C, H, W)
            xs_d = F.interpolate(xs, size=(h2, w2), mode=cfg.spat_mode, align_corners=False)
            xs_u = F.interpolate(xs_d, size=(H, W), mode=cfg.spat_mode, align_corners=False)
            x = xs_u.reshape(B, T, C, H, W)

        return x

    def _temporal_morphing(self, x: torch.Tensor) -> torch.Tensor:
        """
        Yt = αt Xt + (1-αt) X_{t+1}
        αt 随时间平滑变化
        x: (B,T,C,H,W)
        """
        B, T, C, H, W = x.shape
        if T <= 1:
            return x
        cfg = self.cfg
        alpha = _lowfreq_traj(
            T=T,
            device=x.device,
            dtype=x.dtype,
            vmin=cfg.alpha_min,
            vmax=cfg.alpha_max,
            basis=cfg.traj_basis,
            smooth_win=cfg.traj_smooth_win,
            cycles_min=cfg.traj_cycles_min,
            cycles_max=cfg.traj_cycles_max,
        )  # (T,)
        alpha = alpha.view(1, T, 1, 1, 1)

        x_next = torch.roll(x, shifts=-1, dims=1)
        # 最后一帧用自己，避免越界影响
        x_next[:, -1] = x[:, -1]
        y = alpha * x + (1.0 - alpha) * x_next
        return y

    def _frame_drop_and_recon(self, x: torch.Tensor) -> torch.Tensor:
        """
        随机丢帧 + 线性插值补帧
        x: (B,T,C,H,W)
        """
        B, T, C, H, W = x.shape
        cfg = self.cfg
        if T <= 2 or cfg.p_drop <= 0:
            return x

        out = []
        for b in range(B):
            keep = _sample_drop_mask(T, cfg.p_drop, cfg.max_drop_run, cfg.keep_first_last).to(x.device)
            xb = x[b]  # (T,C,H,W)
            # 先把丢帧位置置空（这里直接用原 xb，但只通过插值函数覆盖 missing）
            yb = _interp_missing_linear(xb, keep)
            out.append(yb)
        return torch.stack(out, dim=0)

    def _directional_motion_blur(self, x: torch.Tensor) -> torch.Tensor:
        """
        方向运动模糊：Yt = Kt(θt, lt) * Xt
        θt, lt 随时间平滑变化
        x: (B,T,C,H,W)
        """
        B, T, C, H, W = x.shape
        cfg = self.cfg
        if T <= 0:
            return x

        # θt: [-pi, pi), lt: [len_min, len_max]
        theta = _lowfreq_traj(
            T=T, device=x.device, dtype=x.dtype,
            vmin=-math.pi, vmax=math.pi,
            basis=cfg.traj_basis,
            smooth_win=cfg.traj_smooth_win,
            cycles_min=cfg.traj_cycles_min,
            cycles_max=cfg.traj_cycles_max,
        )
        length = _lowfreq_traj(
            T=T, device=x.device, dtype=x.dtype,
            vmin=float(cfg.blur_len_min), vmax=float(cfg.blur_len_max),
            basis=cfg.traj_basis,
            smooth_win=cfg.traj_smooth_win,
            cycles_min=cfg.traj_cycles_min,
            cycles_max=cfg.traj_cycles_max,
        )

        # 为了速度：把方向离散到 blur_bins 档（仍然随时间平滑变化）
        # 这样核种类不会太多，适合训练时在线退化
        theta_bins = torch.round((theta + math.pi) / (2 * math.pi) * cfg.blur_bins) % cfg.blur_bins
        theta_q = (theta_bins / cfg.blur_bins) * (2 * math.pi) - math.pi

        # length 取整并变为奇数
        len_i = torch.round(length).clamp(cfg.blur_len_min, cfg.blur_len_max).to(torch.int64)
        len_i = len_i + (len_i % 2 == 0).to(torch.int64)

        # 对每帧做卷积（按 batch 循环，简单稳定；如果你要更极致速度，可以缓存核）
        y = x.clone()
        for t in range(T):
            L = int(len_i[t].item())
            th = float(theta_q[t].item())
            k = _make_line_kernel(L, th, device=x.device, dtype=x.dtype)  # (1,1,L,L)
            for b in range(B):
                xt = x[b, t]  # (C,H,W)
                bt = _apply_motion_blur_frame(xt, k)
                if cfg.blur_strength < 1.0:
                    y[b, t] = xt * (1.0 - cfg.blur_strength) + bt * cfg.blur_strength
                else:
                    y[b, t] = bt
        return y

    def _grid_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        局部网格形变：只对 ROI 区域做 wobble
        x: (B,T,C,H,W)
        """
        B, T, C, H, W = x.shape
        cfg = self.cfg
        if cfg.warp_max_px <= 0 or T == 0:
            return x

        dtype = x.dtype
        device = x.device

        # 1) 生成全帧位移场 dt: (T,2,H,W)，像素单位（你之前已有的实现）
        dt = _make_warp_field(T, H, W, cfg, device=device, dtype=dtype)  # (T,2,H,W)

        # 2) 生成 ROI 掩码（时间一致、边缘平滑）
        #    roi_frac 越小越局部；drift=True 表示 ROI 会缓慢漂移
        roi_mask = _make_roi_mask_traj(
            T=T, H=H, W=W,
            device=device, dtype=dtype,
            roi_frac=cfg.roi_frac,     # 你可以调 0.15~0.45
            drift=True,
            smooth_win=cfg.traj_smooth_win,
            seed=cfg.seed if hasattr(cfg, "seed") else 123
        )  # (T,1,H,W)

        # 3) 只在 ROI 内生效位移：dt' = mask * dt
        dt_local = dt * roi_mask  # 广播到 (T,2,H,W)

        # 4) backward warp（grid_sample）
        base = _make_base_grid(H, W, device=device, dtype=dtype)  # (1,H,W,2)

        y = x.clone()
        for t in range(T):
            dx = dt_local[t, 0]  # (H,W)
            dy = dt_local[t, 1]

            # 像素位移 -> 归一化位移
            nx = dx / ((W - 1) / 2.0)
            ny = dy / ((H - 1) / 2.0)

            grid = base + torch.stack([nx, ny], dim=-1).unsqueeze(0)  # (1,H,W,2)

            xt = x[:, t]  # (B,C,H,W)
            warped = F.grid_sample(
                xt, grid.repeat(B, 1, 1, 1),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )  # (B,C,H,W)

            # 5) 只在 ROI 内融合（避免 ROI 外也变化）
            mt = roi_mask[t].repeat(B, 1, 1, 1)  # (B,1,H,W)
            y[:, t] = xt * (1.0 - mt) + warped * mt

        return y


    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入 x: (T,C,H,W) 或 (B,T,C,H,W)，float, [0,1]
        输出同形状
        """
        xb, squeeze_back = self._ensure_shape(x)
        xb = xb.clamp(0, 1)

        # 1) 结构保持时空下采样
        xb = self._spatiotemporal_downsample(xb)

        # 2) Temporal morphing（曝光融合）
        xb = self._temporal_morphing(xb)

        # 3) 随机丢帧 + 插值补帧
        xb = self._frame_drop_and_recon(xb)

        # 4) 方向运动模糊
        xb = self._directional_motion_blur(xb)

        # 5) 网格形变（rolling shutter wobble / 低频几何畸变）
        xb = self._grid_warp(xb)

        xb = xb.clamp(0, 1)
        return xb.squeeze(0) if squeeze_back else xb



# DegradeConfig / SyntheticDegradation，以及它依赖的所有函数
# -------------------------------------------------------------


def list_videos(in_dir: str, exts: List[str]) -> List[str]:
    """递归扫描视频文件"""
    exts = [e.lower().lstrip(".") for e in exts]
    paths = []
    for root, _, files in os.walk(in_dir):
        for fn in files:
            suf = fn.lower().split(".")[-1]
            if suf in exts:
                paths.append(os.path.join(root, fn))
    paths.sort()
    return paths


def make_out_path(in_path: str, in_dir: str, out_dir: str, suffix: str = "_degraded") -> str:
    """
    保持相对目录结构：out_dir/相对路径/文件名_suffix.mp4
    默认统一输出 mp4（write_video 更稳定）
    """
    rel = os.path.relpath(in_path, in_dir)
    base, _ext = os.path.splitext(rel)
    out_path = os.path.join(out_dir, base + suffix + ".mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


@torch.no_grad()
def degrade_one_video(
    in_path: str,
    out_path: str,
    degrader: "SyntheticDegradation",
    device: torch.device,
):
    """
    读取单个视频 -> 退化 -> 保存
    fps 从输入视频 info 获取并原样保存
    """
    # 1) 读取视频
    video, _audio, info = read_video(in_path, pts_unit="sec")  # video: (T,H,W,C) uint8

    from fractions import Fraction

    def normalize_fps(fps_raw, default=30.0):
        # 1) 强制转成 Python float（避免 numpy.float64 传进去）
        try:
            fps = float(fps_raw)
        except Exception:
            fps = float(default)

        # 2) 防御：异常/非法 fps
        if not (fps > 0 and fps < 1000):
            fps = float(default)

        # 3) 有些 torchvision/av 版本对 Fraction 更稳
        #    如果你不想用 Fraction，也可以直接 return fps
        return Fraction(fps).limit_denominator(1000)

    # 使用：
    fps = normalize_fps(info.get("video_fps", 30.0))

    if video.numel() == 0 or video.shape[0] <= 1:
        # 空视频或只有1帧，直接原样写出
        write_video(out_path, video, fps=fps)
        return fps, int(video.shape[0])

    # 2) 转为 (T,C,H,W) float [0,1]
    x = video.permute(0, 3, 1, 2).contiguous().float() / 255.0

    # 3) 放到 device
    x = x.to(device)
    degrader = degrader.to(device).eval()

    # 4) 退化
    y = degrader(x)  # (T,C,H,W)

    # 5) 转回 uint8 (T,H,W,C) 并保存（fps=输入fps）
    y_u8 = (y.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).cpu()
    write_video(out_path, y_u8, fps=fps)

    return fps, int(video.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="输入视频文件夹（递归读取）")
    ap.add_argument("--out_dir", required=True, help="输出视频文件夹（保持目录结构）")
    ap.add_argument("--ext", default="mp4,mov,mkv,avi,webm", help="扩展名，逗号分隔")
    ap.add_argument("--suffix", default="_degraded", help="输出文件名后缀")
    ap.add_argument("--device", default="cuda", help="cuda 或 cpu")
    ap.add_argument("--fp32", action="store_true", help="强制用 fp32（默认按 torch 设备类型运行）")

    # 你可以把下面参数暴露成命令行，我这里先给一套“通用真实拍摄退化”默认值
    args = ap.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    exts = [x.strip() for x in args.ext.split(",") if x.strip()]

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device = {device}")

    # ====== 配置退化强度（按需改）======
    cfg = DegradeConfig(
        # 时空下采样（低FPS+低分辨率）
        s_spat=1,
        s_temp=1,

        # 曝光融合/时间混合
        alpha_min=0.8,
        alpha_max=0.95,

        # 丢帧 + 插值恢复
        p_drop=0.1,
        max_drop_run=1,

        # 方向运动模糊
        blur_len_min=3,
        blur_len_max=11,
        blur_bins=12,
        blur_strength=0.45,

        # 网格形变（rolling-shutter wobble）
        warp_max_px=3.5,
        warp_noise_res=48,
        warp_smooth_ks=11,
        roi_frac = 0.4,

        # 时间参数平滑（避免 flicker）
        traj_basis="sin",
        traj_smooth_win=11,

        seed=123,
    )
    degrader = SyntheticDegradation(cfg)

    videos = list_videos(in_dir, exts)
    if not videos:
        raise RuntimeError(f"在 {in_dir} 下未找到视频（ext={exts}）")

    print(f"[INFO] found {len(videos)} videos")
    ok = 0

    for i, vp in enumerate(videos, 1):
        outp = make_out_path(vp, in_dir, out_dir, suffix=args.suffix)
        try:
            fps, nframes = degrade_one_video(vp, outp, degrader, device=device)
            ok += 1
            print(f"[OK] {i}/{len(videos)} fps={fps:.3f} frames={nframes}  {vp} -> {outp}")
        except Exception as e:
            print(f"[ERR] {i}/{len(videos)} {vp}: {e}")

    print(f"[DONE] success {ok}/{len(videos)}")


if __name__ == "__main__":
    main()
