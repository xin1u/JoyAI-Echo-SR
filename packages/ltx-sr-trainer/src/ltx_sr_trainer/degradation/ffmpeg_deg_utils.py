import torch
import numpy as np
import cv2
import tempfile
import os
import subprocess
import random

def tensor_to_mp4_tc(video, path, fps=30):
    """
    video: (T, C, H, W), float32, [0,1]
    """
    video = (
        video.clamp(0, 1)
        .mul(255)
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )  # (T,H,W,C)

    T, H, W, _ = video.shape
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )
    for i in range(T):
        writer.write(cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR))
    writer.release()


def mp4_to_tensor_tc(path, device):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    video = np.stack(frames)  # (T,H,W,C)
    video = (
        torch.from_numpy(video)
        .float()
        .div(255.0)
        .permute(0, 3, 1, 2)
        .to(device)
    )
    return video


def ffmpeg_compress_video(
    in_mp4,
    out_mp4,
    codec="h264",
    crf=28,
    gop=30,
    preset="veryfast",
):
    if codec == "h264":
        cmd = [
            "ffmpeg", "-y",
            "-threads", "1",
            "-fflags", "+genpts",
            "-vsync", "cfr",
            "-i", in_mp4,
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-x264-params", f"keyint={gop}:min-keyint=5:scenecut=40",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            out_mp4,
         ]

    elif codec == "h265":
        cmd = [
            "ffmpeg", "-y",
            "-threads", "1",
            "-fflags", "+genpts",
            "-vsync", "cfr",
            "-i", in_mp4,
            "-c:v", "libx265",
            "-preset", preset,
            "-x265-params", f"crf={crf}:keyint={gop}",
            "-pix_fmt", "yuv420p",
            out_mp4
        ]
    else:
        raise ValueError(codec)

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


class FFmpegCompressionDegradeBTCHW:
    def __init__(
        self,
        crf_range=(23, 35),
        gop_range=(10, 60),
        codec_choices=("h264", "h265"),
        preset_choices=("ultrafast", "fast", "veryfast"),
        fps=30,
        prob=1.0,
    ):
        self.crf_range = crf_range
        self.gop_range = gop_range
        self.codec_choices = codec_choices
        self.preset_choices = preset_choices
        self.fps = fps
        self.prob = prob

    def __call__(self, video: torch.Tensor):
        """
        video: (B, T, C, H, W), float32, [0,1]
        """
        if random.random() > self.prob:
            return video

        B, T, C, H, W = video.shape
        device = video.device
        out = []

        with tempfile.TemporaryDirectory() as tmp:
            for b in range(B):
                in_mp4 = os.path.join(tmp, f"in_{b}.mp4")
                out_mp4 = os.path.join(tmp, f"out_{b}.mp4")

                tensor_to_mp4_tc(video[b], in_mp4, fps=self.fps)

                codec = random.choice(self.codec_choices)
                crf = random.randint(*self.crf_range)
                gop = random.randint(*self.gop_range)
                preset = random.choice(self.preset_choices)

                ffmpeg_compress_video(
                    in_mp4,
                    out_mp4,
                    codec=codec,
                    crf=crf,
                    gop=gop,
                    preset=preset,
                )

                video_lq = mp4_to_tensor_tc(out_mp4, device)

                # 安全对齐帧数（防止 ffmpeg 掉帧）
                if video_lq.shape[0] != T:
                    video_lq = video_lq[:T]

                out.append(video_lq)

        return torch.stack(out, dim=0)  # (B,T,C,H,W)
