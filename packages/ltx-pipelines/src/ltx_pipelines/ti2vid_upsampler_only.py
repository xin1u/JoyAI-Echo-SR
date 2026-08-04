import logging
import time
from collections.abc import Iterator

import torch
from tqdm import tqdm

from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader.registry import Registry
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.utils.args import ImageConditioningInput, default_2_stage_arg_parser, detect_checkpoint_path
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    get_device,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import ModalitySpec

logger = logging.getLogger(__name__)


class TI2VidUpsamplerOnlyPipeline:
    """
    Two-stage pipeline variant that stops after the learned spatial upsampler.
    It runs stage-1 generation at half resolution, upsamples the latent to the
    target resolution, decodes it, and skips the stage-2 DiT refinement.

    CLI compatibility note:
    - accepts --distilled-lora for drop-in compatibility with ti2vid_two_stages
      but does not use it.
    """

    def __init__(
        self,
        checkpoint_path: str,
        spatial_upsampler_path: str,
        gemma_root: str,
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16

        self.prompt_encoder = PromptEncoder(checkpoint_path, gemma_root, self.dtype, self.device, registry=registry)
        self.image_conditioner = ImageConditioner(checkpoint_path, self.dtype, self.device, registry=registry)
        self.upsampler = VideoUpsampler(
            checkpoint_path, spatial_upsampler_path, self.dtype, self.device, registry=registry
        )
        self.video_decoder = VideoDecoder(checkpoint_path, self.dtype, self.device, registry=registry)
        self.audio_decoder = AudioDecoder(checkpoint_path, self.dtype, self.device, registry=registry)

        self.stage_1 = DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=(),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
        )

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        streaming_prefetch_count: int | None = None,
        max_batch_size: int = 1,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
            streaming_prefetch_count=streaming_prefetch_count,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )

        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        stage_1_start = time.perf_counter()
        video_state, audio_state = self.stage_1(
            denoiser=FactoryGuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider_factory=create_multimodal_guider_factory(
                    params=video_guider_params,
                    negative_context=v_context_n,
                ),
                audio_guider_factory=create_multimodal_guider_factory(
                    params=audio_guider_params,
                    negative_context=a_context_n,
                ),
            ),
            sigmas=sigmas,
            noiser=noiser,
            width=stage_1_output_shape.width,
            height=stage_1_output_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=v_context_p, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=a_context_p),
            streaming_prefetch_count=streaming_prefetch_count,
            max_batch_size=max_batch_size,
        )
        logger.info("Stage 1 diffusion finished in %.2fs", time.perf_counter() - stage_1_start)

        with tqdm(total=1, desc="Spatial upsampler", unit="stage") as upsampler_pbar:
            upsampler_start = time.perf_counter()
            upscaled_video_latent = self.upsampler(video_state.latent[:1])
            upsampler_elapsed = time.perf_counter() - upsampler_start
            upsampler_pbar.set_postfix_str(f"{upsampler_elapsed:.2f}s")
            upsampler_pbar.update(1)
        logger.info("Spatial upsampler finished in %.2fs", upsampler_elapsed)

        decoded_video = self.video_decoder(upscaled_video_latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    checkpoint_path = detect_checkpoint_path()
    params = detect_params(checkpoint_path)
    parser = default_2_stage_arg_parser(params=params)
    args = parser.parse_args()

    pipeline = TI2VidUpsamplerOnlyPipeline(
        checkpoint_path=args.checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        quantization=args.quantization,
        torch_compile=args.compile,
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
        streaming_prefetch_count=args.streaming_prefetch_count,
        max_batch_size=args.max_batch_size,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
