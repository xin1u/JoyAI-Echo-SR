"""Echo-SR one-step stage-2 inference for LTX-2.3 22B and LTX-2 19B.

Input LQ video -> VAE encode -> spatial latent upsampler -> 3-step teacher and/or
1-step distilled student Refiner -> decoded MP4.

This entry decodes the video stream and defaults to the x2 training resolution.
The released DMD checkpoints carry the full audio branch (inherited from the
official distilled LoRA), so they remain audio-video capable — for joint
audio-video output use the long-video launchers (`scripts/infer_av_*.sh`).
Modified for the portable Echo-SR release in 2026.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from collections.abc import Iterator
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _pkg in ("ltx-core", "ltx-pipelines", "ltx-trainer", "ltx-sr-trainer"):
    _src = _REPO_ROOT / "packages" / _pkg / "src"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

import torch
from torch import Tensor

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.model.upsampler import LatentUpsamplerConfigurator, upsample_video
from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig, get_video_chunks_number
from ltx_pipelines.utils.blocks import DiffusionStage
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.media_io import encode_video, normalize_latent, resize_and_center_crop
from ltx_pipelines.utils.types import ModalitySpec
from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder, load_video_vae_decoder, load_video_vae_encoder
from ltx_trainer.video_utils import read_video


TEACHER_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]
STUDENT_SIGMAS = [0.909375, 0.0]


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def cuda_mem(prefix: str, device: torch.device) -> None:
    if device.type != "cuda":
        return
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    print(f"[mem] {prefix}: allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Echo-SR stage-2 inference on one LQ video.")
    parser.add_argument("--input-video", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-path", default=None, help="LTX-2.3 22B or LTX-2 19B base checkpoint.")
    parser.add_argument(
        "--teacher-lora-path",
        default="",
        help="3-step teacher Refiner LoRA. Pass empty string to skip teacher.",
    )
    parser.add_argument(
        "--student-lora-path",
        default="",
        help="1-step distilled student/generator LoRA. Empty by default; pass a checkpoint to enable student output.",
    )
    parser.add_argument(
        "--spatial-upsampler-path",
        default=None,
        help="Spatial latent upsampler matching the selected model family and scale.",
    )
    parser.add_argument(
        "--gemma-root",
        default=None,
        help="Gemma-3-12B text encoder directory.",
    )
    parser.add_argument("--target-height", type=int, default=1024)
    parser.add_argument("--target-width", type=int, default=1536)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--save-upsampler-output", action="store_true")
    parser.add_argument("--decode-latent-path", default=None, help="Internal: decode this saved latent and exit.")
    parser.add_argument("--decode-output-path", default=None, help="Internal: MP4 output path for --decode-latent-path.")
    parser.add_argument("--tile-size-pixels", type=int, default=512)
    parser.add_argument("--tile-overlap-pixels", type=int, default=64)
    parser.add_argument("--tile-size-frames", type=int, default=64)
    parser.add_argument("--tile-overlap-frames", type=int, default=24)
    parser.add_argument("--decode-tile-size-pixels", type=int, default=512)
    parser.add_argument("--decode-tile-overlap-pixels", type=int, default=64)
    parser.add_argument("--decode-tile-size-frames", type=int, default=64)
    parser.add_argument("--decode-tile-overlap-frames", type=int, default=24)
    parser.add_argument("--teacher-sigmas", default=",".join(str(x) for x in TEACHER_SIGMAS))
    parser.add_argument("--student-sigmas", default=",".join(str(x) for x in STUDENT_SIGMAS))
    parser.add_argument("--lora-alpha", type=int, default=384, help="Fallback alpha when a LoRA has no metadata.")
    parser.add_argument(
        "--streaming-prefetch-count",
        type=int,
        default=None,
        help="Enable official layer streaming for Refiner when set, e.g. 1 or 2.",
    )
    parser.add_argument("--max-batch-size", type=int, default=1, help="BatchSplitAdapter max batch size for Refiner.")
    return parser.parse_args()


def parse_sigmas(value: str) -> list[float]:
    sigmas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(sigmas) < 2:
        raise ValueError(f"Expected at least two sigma values, got {value!r}")
    if sigmas[-1] != 0.0:
        raise ValueError(f"Sigma schedule must end with 0.0, got {sigmas}")
    return sigmas


def assert_video_shape_args(args: argparse.Namespace) -> tuple[int, int]:
    if args.num_frames < 1 or (args.num_frames - 1) % 8 != 0:
        raise ValueError(f"--num-frames must be 8*k+1 for LTX, got {args.num_frames}")
    if args.target_height % 64 != 0 or args.target_width % 64 != 0:
        raise ValueError(
            f"Target resolution {args.target_height}x{args.target_width} must be divisible by 64. "
            "Use the resolution from the training config or another aligned size."
        )
    lq_height = int(round(args.target_height / args.scale))
    lq_width = int(round(args.target_width / args.scale))
    if lq_height % 32 != 0 or lq_width % 32 != 0:
        raise ValueError(
            f"LQ resolution after /{args.scale:g} is {lq_height}x{lq_width}, not divisible by 32."
        )
    return lq_height, lq_width


def make_tiling_config(
    *,
    tile_size_pixels: int,
    tile_overlap_pixels: int,
    tile_size_frames: int,
    tile_overlap_frames: int,
) -> TilingConfig:
    return TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=tile_size_pixels,
            tile_overlap_in_pixels=tile_overlap_pixels,
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=tile_size_frames,
            tile_overlap_in_frames=tile_overlap_frames,
        ),
    )


def build_tiling_config(args: argparse.Namespace) -> TilingConfig:
    return make_tiling_config(
        tile_size_pixels=args.tile_size_pixels,
        tile_overlap_pixels=args.tile_overlap_pixels,
        tile_size_frames=args.tile_size_frames,
        tile_overlap_frames=args.tile_overlap_frames,
    )


def build_decode_tiling_config(args: argparse.Namespace) -> TilingConfig:
    return make_tiling_config(
        tile_size_pixels=args.decode_tile_size_pixels,
        tile_overlap_pixels=args.decode_tile_overlap_pixels,
        tile_size_frames=args.decode_tile_size_frames,
        tile_overlap_frames=args.decode_tile_overlap_frames,
    )


def load_lq_video_latent(
    *,
    input_video: Path,
    vae_encoder: Any,
    target_frames: int,
    lq_height: int,
    lq_width: int,
    device: torch.device,
    dtype: torch.dtype,
    tiling_config: TilingConfig,
) -> tuple[Tensor, float]:
    frames, input_fps = read_video(input_video, max_frames=target_frames)
    if frames.shape[0] < target_frames:
        raise ValueError(f"Input video has only {frames.shape[0]} frames; need {target_frames}")
    # read_video returns [F,C,H,W] in [0,1]. LTX preprocessing expects [F,H,W,C] in [0,255].
    frames_fhwc = frames[:target_frames].permute(0, 2, 3, 1).mul(255.0)
    video = resize_and_center_crop(frames_fhwc, lq_height, lq_width)
    # Keep the full preprocessed video on CPU. tiled_encode moves one tile at a time
    # to the model device, which avoids a large upfront GPU allocation.
    video = normalize_latent(video, device=torch.device("cpu"), dtype=dtype)
    latent = vae_encoder.tiled_encode(video, tiling_config=tiling_config)
    return latent, input_fps


def decoded_uint8_chunks(
    vae_decoder: Any,
    latent: Tensor,
    *,
    tiling_config: TilingConfig,
    generator: torch.Generator,
    device: torch.device,
) -> Iterator[Tensor]:
    # Match validation_sampler's direct tiled_decode path, but stream chunks to
    # CPU one at a time instead of materializing the full 1080p video tensor.
    for frames in vae_decoder.tiled_decode(latent, tiling_config=tiling_config, generator=generator):
        video_chunk = (((frames + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        video_chunk = video_chunk[0].permute(1, 2, 3, 0).contiguous().to("cpu")
        del frames
        cleanup_cuda()
        yield video_chunk
        del video_chunk
        cleanup_cuda()


def save_decoded_video(
    *,
    vae_decoder: Any,
    latent: Tensor,
    output_path: Path,
    fps: float,
    seed: int,
    device: torch.device,
    tiling_config: TilingConfig,
    num_frames: int,
) -> None:
    generator = torch.Generator(device=device).manual_seed(seed)
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    cuda_mem(f"before_decode:{output_path.name}", device)
    video_iter = decoded_uint8_chunks(
        vae_decoder,
        latent,
        tiling_config=tiling_config,
        generator=generator,
        device=device,
    )
    encode_video(
        video=video_iter,
        fps=fps,
        audio=None,
        output_path=str(output_path),
        video_chunks_number=video_chunks_number,
    )
    cleanup_cuda()
    cuda_mem(f"after_decode:{output_path.name}", device)


def save_latent_tensor(latent: Tensor, output_dir: Path, name: str) -> Path:
    latent_path = output_dir / f"{name}.latent.pt"
    torch.save(latent.detach().to("cpu"), latent_path)
    print(f"[stage] saved latent: {latent_path}", flush=True)
    return latent_path


def decode_saved_latent(args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype) -> None:
    if not args.decode_output_path:
        raise ValueError("--decode-output-path is required with --decode-latent-path")
    latent_path = Path(args.decode_latent_path).expanduser().resolve()
    output_path = Path(args.decode_output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decode_tiling_config = build_decode_tiling_config(args)
    print(f"[decode-subprocess] loading latent: {latent_path}", flush=True)
    latent = torch.load(latent_path, map_location="cpu").to(device=device, dtype=dtype)
    cuda_mem(f"decode_loaded:{output_path.name}", device)
    vae_decoder = load_video_vae_decoder(args.checkpoint_path, device=device, dtype=dtype).eval()
    vae_decoder.requires_grad_(False)
    save_decoded_video(
        vae_decoder=vae_decoder,
        latent=latent,
        output_path=output_path,
        fps=args.fps,
        seed=args.seed,
        device=device,
        tiling_config=decode_tiling_config,
        num_frames=args.num_frames,
    )
    del latent, vae_decoder
    cleanup_cuda()


def decode_latent_subprocess(
    *,
    args: argparse.Namespace,
    latent_path: Path,
    output_path: Path,
    device: torch.device,
) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--decode-latent-path",
        str(latent_path),
        "--decode-output-path",
        str(output_path),
        "--checkpoint-path",
        args.checkpoint_path,
        "--num-frames",
        str(args.num_frames),
        "--fps",
        str(args.fps),
        "--seed",
        str(args.seed),
        "--device",
        str(device),
        "--decode-tile-size-pixels",
        str(args.decode_tile_size_pixels),
        "--decode-tile-overlap-pixels",
        str(args.decode_tile_overlap_pixels),
        "--decode-tile-size-frames",
        str(args.decode_tile_size_frames),
        "--decode-tile-overlap-frames",
        str(args.decode_tile_overlap_frames),
    ]
    print(f"[stage] decode subprocess: {output_path}", flush=True)
    subprocess.run(cmd, check=True)


def lora_tuple(lora_path: str) -> tuple[LoraPathStrengthAndSDOps, ...]:
    if not lora_path:
        return ()
    resolved = str(Path(lora_path).expanduser().resolve())
    return (LoraPathStrengthAndSDOps(resolved, 1.0, LTXV_LORA_COMFY_RENAMING_MAP),)


def encode_prompt_context(
    *,
    checkpoint_path: str,
    gemma_root: str,
    prompt: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    text_encoder = load_text_encoder(gemma_root, device=device, dtype=dtype, load_in_8bit=False)
    embeddings_processor = load_embeddings_processor(checkpoint_path, device=device, dtype=dtype)
    text_encoder.requires_grad_(False).eval()
    embeddings_processor.requires_grad_(False).eval()
    with torch.no_grad():
        hidden_states, attention_mask = text_encoder.encode(prompt)
        out = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
        video_context = out.video_encoding.to(device=device, dtype=dtype)
    del text_encoder, embeddings_processor, hidden_states, attention_mask, out
    cleanup_cuda()
    return video_context


@torch.no_grad()
def run_diffusion_refiner(
    *,
    checkpoint_path: str,
    lora_path: str,
    branch: str,
    condition_latent: Tensor,
    video_context: Tensor,
    sigmas: list[float],
    target_height: int,
    target_width: int,
    num_frames: int,
    fps: float,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    streaming_prefetch_count: int | None,
    max_batch_size: int,
) -> Tensor:
    print(
        f"[refiner] branch={branch} official_diffusion_stage latent={tuple(condition_latent.shape)} "
        f"tokens={condition_latent.shape[2] * condition_latent.shape[3] * condition_latent.shape[4]} sigmas={sigmas} "
        f"lora={lora_path}",
        flush=True,
    )
    cuda_mem(f"{branch}:before_stage", device)
    generator = torch.Generator(device=device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    stage = DiffusionStage(
        checkpoint_path=checkpoint_path,
        dtype=dtype,
        device=device,
        loras=lora_tuple(lora_path),
    )
    video_state, _ = stage(
        denoiser=SimpleDenoiser(v_context=video_context, a_context=None),
        sigmas=torch.tensor(sigmas, device=device, dtype=torch.float32),
        noiser=noiser,
        width=target_width,
        height=target_height,
        frames=num_frames,
        fps=fps,
        video=ModalitySpec(
            context=video_context,
            conditionings=[],
            noise_scale=float(sigmas[0]),
            initial_latent=condition_latent,
        ),
        audio=None,
        streaming_prefetch_count=streaming_prefetch_count,
        max_batch_size=max_batch_size,
    )
    del stage
    cleanup_cuda()
    if video_state is None:
        raise RuntimeError(f"{branch} Refiner returned no video state")
    cuda_mem(f"{branch}:after_stage", device)
    return video_state.latent


def main() -> None:
    args = parse_args()
    started_at = datetime.now().isoformat(timespec="seconds")
    if not args.checkpoint_path:
        raise SystemExit("--checkpoint-path is required.")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("CUDA is not available. Re-run on a GPU node or pass --allow-cpu for a slow smoke test.")

    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    if args.decode_latent_path:
        decode_saved_latent(args, device=device, dtype=dtype)
        return

    missing = [
        name
        for name, value in (
            ("--input-video", args.input_video),
            ("--prompt", args.prompt),
            ("--output-dir", args.output_dir),
            ("--spatial-upsampler-path", args.spatial_upsampler_path),
            ("--gemma-root", args.gemma_root),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"Required for full inference mode: {', '.join(missing)}")

    lq_height, lq_width = assert_video_shape_args(args)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    input_video = Path(args.input_video).expanduser().resolve()
    teacher_sigmas = parse_sigmas(args.teacher_sigmas)
    student_sigmas = parse_sigmas(args.student_sigmas)
    tiling_config = build_tiling_config(args)

    started = time.perf_counter()

    # Stage A: VAE encode + spatial latent upsample. Keep the full video tensor on CPU;
    # tiled_encode moves one tile at a time to CUDA.
    vae_encoder = load_video_vae_encoder(args.checkpoint_path, device=device, dtype=dtype).eval()
    vae_encoder.requires_grad_(False)
    spatial_upsampler = SingleGPUModelBuilder(
        model_path=args.spatial_upsampler_path,
        model_class_configurator=LatentUpsamplerConfigurator,
    ).build(device=device, dtype=dtype)
    spatial_upsampler.requires_grad_(False)
    spatial_upsampler.eval()

    lq_latent_small, input_fps = load_lq_video_latent(
        input_video=input_video,
        vae_encoder=vae_encoder,
        target_frames=args.num_frames,
        lq_height=lq_height,
        lq_width=lq_width,
        device=device,
        dtype=dtype,
        tiling_config=tiling_config,
    )
    print(f"[stage] encoded LQ latent: {tuple(lq_latent_small.shape)} input_fps={input_fps}", flush=True)
    cuda_mem("after_vae_encode", device)
    upsampled_latent = upsample_video(latent=lq_latent_small, video_encoder=vae_encoder, upsampler=spatial_upsampler)
    print(f"[stage] x{args.scale:g} upsampled latent: {tuple(upsampled_latent.shape)}", flush=True)
    cuda_mem("after_latent_upsample", device)
    expected_h = args.target_height // 32
    expected_w = args.target_width // 32
    if upsampled_latent.shape[3] != expected_h or upsampled_latent.shape[4] != expected_w:
        raise RuntimeError(
            f"Upsampler produced latent shape {tuple(upsampled_latent.shape)}, "
            f"expected spatial latent {expected_h}x{expected_w} for {args.target_height}x{args.target_width}."
        )
    del lq_latent_small, vae_encoder, spatial_upsampler
    cleanup_cuda()

    # Stage B: prompt encoding through the official lifecycle block. Free Gemma before Refiner loads.
    video_context = encode_prompt_context(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        prompt=args.prompt,
        dtype=dtype,
        device=device,
    )
    cleanup_cuda()
    cuda_mem("after_prompt_encode", device)

    # Stage C: video-branch Refiner through the official DiffusionStage LoRA loader/fuser.
    if not args.teacher_lora_path and not args.student_lora_path:
        raise ValueError("At least one of --teacher-lora-path or --student-lora-path must be non-empty.")

    latent_paths: dict[str, Path] = {}

    if args.teacher_lora_path:
        teacher_latent = run_diffusion_refiner(
            checkpoint_path=args.checkpoint_path,
            lora_path=args.teacher_lora_path,
            branch="teacher_3step",
            condition_latent=upsampled_latent,
            video_context=video_context,
            sigmas=teacher_sigmas,
            target_height=args.target_height,
            target_width=args.target_width,
            num_frames=args.num_frames,
            fps=args.fps,
            seed=args.seed,
            dtype=dtype,
            device=device,
            streaming_prefetch_count=args.streaming_prefetch_count,
            max_batch_size=args.max_batch_size,
        )
        latent_paths["teacher_3step"] = save_latent_tensor(teacher_latent, output_dir, "teacher_3step_refined")
        del teacher_latent
        cleanup_cuda()

    if args.student_lora_path:
        student_latent = run_diffusion_refiner(
            checkpoint_path=args.checkpoint_path,
            lora_path=args.student_lora_path,
            branch="student_1step",
            condition_latent=upsampled_latent,
            video_context=video_context,
            sigmas=student_sigmas,
            target_height=args.target_height,
            target_width=args.target_width,
            num_frames=args.num_frames,
            fps=args.fps,
            seed=args.seed,
            dtype=dtype,
            device=device,
            streaming_prefetch_count=args.streaming_prefetch_count,
            max_batch_size=args.max_batch_size,
        )
        latent_paths["student_1step"] = save_latent_tensor(student_latent, output_dir, "student_1step_refined")
        del student_latent
        cleanup_cuda()

    del video_context
    cleanup_cuda()

    # Stage D: decode in fresh subprocesses so VAE never shares CUDA state with Gemma/Refiner.
    outputs: dict[str, str] = {}
    if args.save_upsampler_output:
        latent_paths["upsampler_only"] = save_latent_tensor(upsampled_latent, output_dir, "upsampler_only")
    del upsampled_latent
    cleanup_cuda()

    output_names = {
        "upsampler_only": "upsampler_only.mp4",
        "teacher_3step": "teacher_3step_refined.mp4",
        "student_1step": "student_1step_refined.mp4",
    }
    for key, latent_path in latent_paths.items():
        out_path = output_dir / output_names[key]
        decode_latent_subprocess(args=args, latent_path=latent_path, output_path=out_path, device=device)
        outputs[key] = str(out_path)
        cleanup_cuda()

    metadata = {
        "started_at": started_at,
        "host": platform.node(),
        "argv": sys.argv,
        "input_video": str(input_video),
        "input_fps": input_fps,
        "prompt": args.prompt,
        "checkpoint_path": args.checkpoint_path,
        "spatial_upsampler_path": args.spatial_upsampler_path,
        "teacher_lora_path": args.teacher_lora_path,
        "student_lora_path": args.student_lora_path,
        "target_resolution": [args.target_height, args.target_width],
        "lq_resolution": [lq_height, lq_width],
        "num_frames": args.num_frames,
        "fps": args.fps,
        "teacher_sigmas": teacher_sigmas,
        "student_sigmas": student_sigmas,
        "tiling": {
            "tile_size_pixels": args.tile_size_pixels,
            "tile_overlap_pixels": args.tile_overlap_pixels,
            "tile_size_frames": args.tile_size_frames,
            "tile_overlap_frames": args.tile_overlap_frames,
        },
        "decode_tiling": {
            "tile_size_pixels": args.decode_tile_size_pixels,
            "tile_overlap_pixels": args.decode_tile_overlap_pixels,
            "tile_size_frames": args.decode_tile_size_frames,
            "tile_overlap_frames": args.decode_tile_overlap_frames,
        },
        "outputs": outputs,
        "latent_paths": {key: str(path) for key, path in latent_paths.items()},
        "elapsed_s": time.perf_counter() - started,
        "torch": torch.__version__,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
