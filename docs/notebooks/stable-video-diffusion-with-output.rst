Image to Video Generation with Stable Video Diffusion
=====================================================

Stable Video Diffusion (SVD) Image-to-Video is a diffusion model that
takes in a still image as a conditioning frame, and generates a video
from it. In this tutorial we consider how to convert and run Stable
Video Diffusion using OpenVINO. We will use
`stable-video-diffusion-img2video-xt <https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt>`__
model as example. Additionally, to speedup video generation process we
apply `AnimateLCM <https://arxiv.org/abs/2402.00769>`__ LoRA weights.

Table of contents:
------------------

-  `Prerequisites <#prerequisites>`__
-  `Download PyTorch Model <#download-pytorch-model>`__
-  `Convert Model to OpenVINO Intermediate
   Representation <#convert-model-to-openvino-intermediate-representation>`__

   -  `Image Encoder <#image-encoder>`__
   -  `U-net <#u-net>`__
   -  `VAE Encoder and Decoder <#vae-encoder-and-decoder>`__

-  `Prepare Inference Pipeline <#prepare-inference-pipeline>`__
-  `Run Video Generation <#run-video-generation>`__

   -  `Select Inference Device <#select-inference-device>`__

-  `Interactive Demo <#interactive-demo>`__

Prerequisites
-------------



.. code:: ipython3

    %pip install -q "torch>=2.1" "diffusers>=0.25" "peft==0.6.2" "transformers" "openvino>=2024.1.0" Pillow opencv-python tqdm  "gradio>=4.19" safetensors --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    WARNING: Skipping openvino-dev as it is not installed.
    WARNING: Skipping openvino as it is not installed.
    Note: you may need to restart the kernel to use updated packages.
    DEPRECATION: torchsde 0.2.5 has a non-standard dependency specifier numpy>=1.19.*; python_version >= "3.7". pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of torchsde or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Note: you may need to restart the kernel to use updated packages.


Download PyTorch Model
----------------------



The code below load Stable Video Diffusion XT model using
`Diffusers <https://huggingface.co/docs/diffusers/index>`__ library and
apply Consistency Distilled AnimateLCM weights.

.. code:: ipython3

    import torch
    from pathlib import Path
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import load_image, export_to_video
    from diffusers.models.attention_processor import AttnProcessor
    from safetensors import safe_open
    import gc
    import requests
    
    lcm_scheduler_url = "https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/raw/main/lcm_scheduler.py"
    
    r = requests.get(lcm_scheduler_url)
    
    with open("lcm_scheduler.py", "w") as f:
        f.write(r.text)
    
    from lcm_scheduler import AnimateLCMSVDStochasticIterativeScheduler
    from huggingface_hub import hf_hub_download
    
    MODEL_DIR = Path("model")
    
    IMAGE_ENCODER_PATH = MODEL_DIR / "image_encoder.xml"
    VAE_ENCODER_PATH = MODEL_DIR / "vae_encoder.xml"
    VAE_DECODER_PATH = MODEL_DIR / "vae_decoder.xml"
    UNET_PATH = MODEL_DIR / "unet.xml"
    
    
    load_pt_pipeline = not (VAE_ENCODER_PATH.exists() and VAE_DECODER_PATH.exists() and UNET_PATH.exists() and IMAGE_ENCODER_PATH.exists())
    
    unet, vae, image_encoder = None, None, None
    if load_pt_pipeline:
        noise_scheduler = AnimateLCMSVDStochasticIterativeScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=700.0,
            sigma_data=1.0,
            s_noise=1.0,
            rho=7,
            clip_denoised=False,
        )
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            variant="fp16",
            scheduler=noise_scheduler,
        )
        pipe.unet.set_attn_processor(AttnProcessor())
        hf_hub_download(
            repo_id="wangfuyun/AnimateLCM-SVD-xt",
            filename="AnimateLCM-SVD-xt.safetensors",
            local_dir="./checkpoints",
        )
        state_dict = {}
        LCM_LORA_PATH = Path(
            "checkpoints/AnimateLCM-SVD-xt.safetensors",
        )
        with safe_open(LCM_LORA_PATH, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        missing, unexpected = pipe.unet.load_state_dict(state_dict, strict=True)
    
        pipe.scheduler.save_pretrained(MODEL_DIR / "scheduler")
        pipe.feature_extractor.save_pretrained(MODEL_DIR / "feature_extractor")
        unet = pipe.unet
        unet.eval()
        vae = pipe.vae
        vae.eval()
        image_encoder = pipe.image_encoder
        image_encoder.eval()
        del pipe
        gc.collect()
    
    # Load the conditioning image
    image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
    image = image.resize((512, 256))


.. parsed-literal::

    2024-04-22 20:18:36.486796: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-04-22 20:18:36.488610: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-04-22 20:18:36.524343: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-04-22 20:18:36.525356: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-04-22 20:18:37.277389: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
      warn("The installed version of bitsandbytes was compiled without GPU support. "


.. parsed-literal::

    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so: undefined symbol: cadam32bit_grad_fp32


.. parsed-literal::

    WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions. xFormers was built for:
        PyTorch 2.0.1+cu118 with CUDA 1108 (you have 2.1.2+cpu)
        Python  3.8.18 (you have 3.8.10)
      Please reinstall xformers (see https://github.com/facebookresearch/xformers#installing-xformers)
      Memory-efficient attention, SwiGLU, sparse and more won't be available.
      Set XFORMERS_MORE_DETAILS=1 for more details


Convert Model to OpenVINO Intermediate Representation
-----------------------------------------------------



OpenVINO supports PyTorch models via conversion into Intermediate
Representation (IR) format. We need to provide a model object, input
data for model tracing to ``ov.convert_model`` function to obtain
OpenVINO ``ov.Model`` object instance. Model can be saved on disk for
next deployment using ``ov.save_model`` function.

Stable Video Diffusion consists of 3 parts:

-  **Image Encoder** for extraction embeddings from the input image.
-  **U-Net** for step-by-step denoising video clip.
-  **VAE** for encoding input image into latent space and decoding
   generated video.

Let’s convert each part.

Image Encoder
~~~~~~~~~~~~~



.. code:: ipython3

    import openvino as ov
    
    
    def cleanup_torchscript_cache():
        """
        Helper for removing cached model representation
        """
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        torch.jit._state._clear_class_state()
    
    
    if not IMAGE_ENCODER_PATH.exists():
        with torch.no_grad():
            ov_model = ov.convert_model(
                image_encoder,
                example_input=torch.zeros((1, 3, 224, 224)),
                input=[-1, 3, 224, 224],
            )
        ov.save_model(ov_model, IMAGE_ENCODER_PATH)
        del ov_model
        cleanup_torchscript_cache()
        print(f"Image Encoder successfully converted to IR and saved to {IMAGE_ENCODER_PATH}")
    del image_encoder
    gc.collect();

U-net
~~~~~



.. code:: ipython3

    import openvino as ov
    
    if not UNET_PATH.exists():
        unet_inputs = {
            "sample": torch.ones([2, 2, 8, 32, 32]),
            "timestep": torch.tensor(1.256),
            "encoder_hidden_states": torch.zeros([2, 1, 1024]),
            "added_time_ids": torch.ones([2, 3]),
        }
        with torch.no_grad():
            ov_model = ov.convert_model(unet, example_input=unet_inputs)
        ov.save_model(ov_model, UNET_PATH)
        del ov_model
        cleanup_torchscript_cache()
        print(f"UNet successfully converted to IR and saved to {UNET_PATH}")
    
    del unet
    gc.collect();

VAE Encoder and Decoder
~~~~~~~~~~~~~~~~~~~~~~~



As discussed above VAE model used for encoding initial image and
decoding generated video. Encoding and Decoding happen on different
pipeline stages, so for convenient usage we separate VAE on 2 parts:
Encoder and Decoder.

.. code:: ipython3

    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
    
        def forward(self, image):
            return self.vae.encode(x=image)["latent_dist"].sample()
    
    
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
    
        def forward(self, latents, num_frames: int):
            return self.vae.decode(latents, num_frames=num_frames)
    
    
    if not VAE_ENCODER_PATH.exists():
        vae_encoder = VAEEncoderWrapper(vae)
        with torch.no_grad():
            ov_model = ov.convert_model(vae_encoder, example_input=torch.zeros((1, 3, 576, 1024)))
        ov.save_model(ov_model, VAE_ENCODER_PATH)
        cleanup_torchscript_cache()
        print(f"VAE Encoder successfully converted to IR and saved to {VAE_ENCODER_PATH}")
        del vae_encoder
        gc.collect()
    
    if not VAE_DECODER_PATH.exists():
        vae_decoder = VAEDecoderWrapper(vae)
        with torch.no_grad():
            ov_model = ov.convert_model(vae_decoder, example_input=(torch.zeros((8, 4, 72, 128)), torch.tensor(8)))
        ov.save_model(ov_model, VAE_DECODER_PATH)
        cleanup_torchscript_cache()
        print(f"VAE Decoder successfully converted to IR and saved to {VAE_ENCODER_PATH}")
        del vae_decoder
        gc.collect()
    
    del vae
    gc.collect();

Prepare Inference Pipeline
--------------------------



The code bellow implements ``OVStableVideoDiffusionPipeline`` class for
running video generation using OpenVINO. The pipeline accepts input
image and returns the sequence of generated frames The diagram below
represents a simplified pipeline workflow.

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/a5671c5b-415b-4ae0-be82-9bf36527d452
   :alt: svd

   svd

The pipeline is very similar to `Stable Diffusion Image to Image
Generation
pipeline <stable-diffusion-text-to-image-with-output.html>`__
with the only difference that Image Encoder is used instead of Text
Encoder. Model takes input image and random seed as initial prompt. Then
image encoded into embeddings space using Image Encoder and into latent
space using VAE Encoder and passed as input to U-Net model. Next, the
U-Net iteratively *denoises* the random latent video representations
while being conditioned on the image embeddings. The output of the
U-Net, being the noise residual, is used to compute a denoised latent
image representation via a scheduler algorithm for next iteration in
generation cycle. This process repeats the given number of times and,
finally, VAE decoder converts denoised latents into sequence of video
frames.

.. code:: ipython3

    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    import PIL.Image
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.utils.torch_utils import randn_tensor
    from typing import Callable, Dict, List, Optional, Union
    from diffusers.pipelines.stable_video_diffusion import (
        StableVideoDiffusionPipelineOutput,
    )
    
    
    def _append_dims(x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
        return x[(...,) + (None,) * dims_to_append]
    
    
    def tensor2vid(video: torch.Tensor, processor, output_type="np"):
        # Based on:
        # https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78
    
        batch_size, channels, num_frames, height, width = video.shape
        outputs = []
        for batch_idx in range(batch_size):
            batch_vid = video[batch_idx].permute(1, 0, 2, 3)
            batch_output = processor.postprocess(batch_vid, output_type)
    
            outputs.append(batch_output)
    
        return outputs
    
    
    class OVStableVideoDiffusionPipeline(DiffusionPipeline):
        r"""
        Pipeline to generate video from an input image using Stable Video Diffusion.
    
        This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
        implemented for all pipelines (downloading, saving, running on a particular device, etc.).
    
        Args:
            vae ([`AutoencoderKL`]):
                Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
            image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
                Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
            unet ([`UNetSpatioTemporalConditionModel`]):
                A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
            scheduler ([`EulerDiscreteScheduler`]):
                A scheduler to be used in combination with `unet` to denoise the encoded image latents.
            feature_extractor ([`~transformers.CLIPImageProcessor`]):
                A `CLIPImageProcessor` to extract features from generated images.
        """
    
        def __init__(
            self,
            vae_encoder,
            image_encoder,
            unet,
            vae_decoder,
            scheduler,
            feature_extractor,
        ):
            super().__init__()
            self.vae_encoder = vae_encoder
            self.vae_decoder = vae_decoder
            self.image_encoder = image_encoder
            self.unet = unet
            self.scheduler = scheduler
            self.feature_extractor = feature_extractor
            self.vae_scale_factor = 2 ** (4 - 1)
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    
        def _encode_image(self, image, device, num_videos_per_prompt, do_classifier_free_guidance):
            dtype = torch.float32
    
            if not isinstance(image, torch.Tensor):
                image = self.image_processor.pil_to_numpy(image)
                image = self.image_processor.numpy_to_pt(image)
    
                # We normalize the image before resizing to match with the original implementation.
                # Then we unnormalize it after resizing.
                image = image * 2.0 - 1.0
                image = _resize_with_antialiasing(image, (224, 224))
                image = (image + 1.0) / 2.0
    
                # Normalize the image with for CLIP input
                image = self.feature_extractor(
                    images=image,
                    do_normalize=True,
                    do_center_crop=False,
                    do_resize=False,
                    do_rescale=False,
                    return_tensors="pt",
                ).pixel_values
    
            image = image.to(device=device, dtype=dtype)
            image_embeddings = torch.from_numpy(self.image_encoder(image)[0])
            image_embeddings = image_embeddings.unsqueeze(1)
    
            # duplicate image embeddings for each generation per prompt, using mps friendly method
            bs_embed, seq_len, _ = image_embeddings.shape
            image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
            image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)
    
            if do_classifier_free_guidance:
                negative_image_embeddings = torch.zeros_like(image_embeddings)
    
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])
            return image_embeddings
    
        def _encode_vae_image(
            self,
            image: torch.Tensor,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        ):
            image_latents = torch.from_numpy(self.vae_encoder(image)[0])
    
            if do_classifier_free_guidance:
                negative_image_latents = torch.zeros_like(image_latents)
    
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                image_latents = torch.cat([negative_image_latents, image_latents])
    
            # duplicate image_latents for each generation per prompt, using mps friendly method
            image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)
    
            return image_latents
    
        def _get_add_time_ids(
            self,
            fps,
            motion_bucket_id,
            noise_aug_strength,
            dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        ):
            add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
    
            passed_add_embed_dim = 256 * len(add_time_ids)
            expected_add_embed_dim = 3 * 256
    
            if expected_add_embed_dim != passed_add_embed_dim:
                raise ValueError(
                    f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
                )
    
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
            add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)
    
            if do_classifier_free_guidance:
                add_time_ids = torch.cat([add_time_ids, add_time_ids])
    
            return add_time_ids
    
        def decode_latents(self, latents, num_frames, decode_chunk_size=14):
            # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
            latents = latents.flatten(0, 1)
    
            latents = 1 / 0.18215 * latents
    
            # decode decode_chunk_size frames at a time to avoid OOM
            frames = []
            for i in range(0, latents.shape[0], decode_chunk_size):
                frame = torch.from_numpy(self.vae_decoder([latents[i : i + decode_chunk_size], num_frames])[0])
                frames.append(frame)
            frames = torch.cat(frames, dim=0)
    
            # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
            frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
    
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
            frames = frames.float()
            return frames
    
        def check_inputs(self, image, height, width):
            if not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image) and not isinstance(image, list):
                raise ValueError("`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is" f" {type(image)}")
    
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
    
        def prepare_latents(
            self,
            batch_size,
            num_frames,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None,
        ):
            shape = (
                batch_size,
                num_frames,
                num_channels_latents // 2,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                latents = latents.to(device)
    
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            return latents
    
        @torch.no_grad()
        def __call__(
            self,
            image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
            height: int = 320,
            width: int = 512,
            num_frames: Optional[int] = 8,
            num_inference_steps: int = 4,
            min_guidance_scale: float = 1.0,
            max_guidance_scale: float = 1.2,
            fps: int = 7,
            motion_bucket_id: int = 80,
            noise_aug_strength: int = 0.01,
            decode_chunk_size: Optional[int] = None,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            return_dict: bool = True,
        ):
            r"""
            The call function to the pipeline for generation.
    
            Args:discussed
                image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                    Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                    [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
                height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                    The height in pixels of the generated image.
                width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                    The width in pixels of the generated image.
                num_frames (`int`, *optional*):
                    The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
                num_inference_steps (`int`, *optional*, defaults to 25):
    
    
                    The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                    expense of slower inference. This parameter is modulated by `strength`.
                min_guidance_scale (`float`, *optional*, defaults to 1.0):
                    The minimum guidance scale. Used for the classifier free guidance with first frame.
                max_guidance_scale (`float`, *optional*, defaults to 3.0):
                    The maximum guidance scale. Used for the classifier free guidance with last frame.
                fps (`int`, *optional*, defaults to 7):
                    Frames per second. The rate at which the generated images shall be exported to a video after generation.
                    Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
                motion_bucket_id (`int`, *optional*, defaults to 127):
                    The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
                noise_aug_strength (`int`, *optional*, defaults to 0.02):
                    The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
                decode_chunk_size (`int`, *optional*):
                    The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                    between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                    for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
                num_videos_per_prompt (`int`, *optional*, defaults to 1):
                    The number of images to generate per prompt.
                generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                    A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                    generation deterministic.
                latents (`torch.FloatTensor`, *optional*):
                    Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                    generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                    tensor is generated by sampling using the supplied random `generator`.
                output_type (`str`, *optional*, defaults to `"pil"`):
                    The output format of the generated image. Choose between `PIL.Image` or `np.array`.
                callback_on_step_end (`Callable`, *optional*):
                    A function that calls at the end of each denoising steps during the inference. The function is called
                    with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                    callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                    `callback_on_step_end_tensor_inputs`.
                callback_on_step_end_tensor_inputs (`List`, *optional*):
                    The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                    will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                    `._callback_tensor_inputs` attribute of your pipeline class.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                    plain tuple.
    
            Returns:
                [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                    If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                    otherwise a `tuple` is returned where the first element is a list of list with the generated frames.
    
            Examples:
    
            ```py
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import load_image, export_to_video
    
            pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
            pipe.to("cuda")
    
            image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
            image = image.resize((1024, 576))
    
            frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
            export_to_video(frames, "generated.mp4", fps=7)
            ```
            """
            # 0. Default height and width to unet
            height = height or 96 * self.vae_scale_factor
            width = width or 96 * self.vae_scale_factor
    
            num_frames = num_frames if num_frames is not None else 25
            decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
    
            # 1. Check inputs. Raise error if not correct
            self.check_inputs(image, height, width)
    
            # 2. Define call parameters
            if isinstance(image, PIL.Image.Image):
                batch_size = 1
            elif isinstance(image, list):
                batch_size = len(image)
            else:
                batch_size = image.shape[0]
            device = torch.device("cpu")
    
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = max_guidance_scale > 1.0
    
            # 3. Encode input image
            image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
    
            # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
            # is why it is reduced here.
            # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
            fps = fps - 1
    
            # 4. Encode input image using VAE
            image = self.image_processor.preprocess(image, height=height, width=width)
            noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
            image = image + noise_aug_strength * noise
    
            image_latents = self._encode_vae_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
            image_latents = image_latents.to(image_embeddings.dtype)
    
            # Repeat the image latents for each frame so we can concatenate them with the noise
            # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
            image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
    
            # 5. Get Added Time IDs
            added_time_ids = self._get_add_time_ids(
                fps,
                motion_bucket_id,
                noise_aug_strength,
                image_embeddings.dtype,
                batch_size,
                num_videos_per_prompt,
                do_classifier_free_guidance,
            )
            added_time_ids = added_time_ids
    
            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            # 5. Prepare latent variables
            num_channels_latents = 8
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_frames,
                num_channels_latents,
                height,
                width,
                image_embeddings.dtype,
                device,
                generator,
                latents,
            )
    
            # 7. Prepare guidance scale
            guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
            guidance_scale = guidance_scale.to(device, latents.dtype)
            guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
            guidance_scale = _append_dims(guidance_scale, latents.ndim)
    
            # 8. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            num_timesteps = len(timesteps)
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    
                    # Concatenate image_latents over channels dimention
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
                    # predict the noise residual
                    noise_pred = torch.from_numpy(
                        self.unet(
                            [
                                latent_model_input,
                                t,
                                image_embeddings,
                                added_time_ids,
                            ]
                        )[0]
                    )
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample
    
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
    
                        latents = callback_outputs.pop("latents", latents)
    
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
    
            if not output_type == "latent":
                frames = self.decode_latents(latents, num_frames, decode_chunk_size)
                frames = tensor2vid(frames, self.image_processor, output_type=output_type)
            else:
                frames = latents
    
            if not return_dict:
                return frames
    
            return StableVideoDiffusionPipelineOutput(frames=frames)
    
    
    # resizing utils
    def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
        h, w = input.shape[-2:]
        factors = (h / size[0], w / size[1])
    
        # First, we have to determine sigma
        # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
        sigmas = (
            max((factors[0] - 1.0) / 2.0, 0.001),
            max((factors[1] - 1.0) / 2.0, 0.001),
        )
        # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
        # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
        # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
        ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))
    
        # Make sure it is odd
        if (ks[0] % 2) == 0:
            ks = ks[0] + 1, ks[1]
    
        if (ks[1] % 2) == 0:
    
            ks = ks[0], ks[1] + 1
    
        input = _gaussian_blur2d(input, ks, sigmas)
    
        output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
        return output
    
    
    def _compute_padding(kernel_size):
        """Compute padding tuple."""
        # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
        # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
        if len(kernel_size) < 2:
            raise AssertionError(kernel_size)
        computed = [k - 1 for k in kernel_size]
    
        # for even kernels we need to do asymmetric padding :(
        out_padding = 2 * len(kernel_size) * [0]
    
        for i in range(len(kernel_size)):
            computed_tmp = computed[-(i + 1)]
    
            pad_front = computed_tmp // 2
            pad_rear = computed_tmp - pad_front
    
            out_padding[2 * i + 0] = pad_front
            out_padding[2 * i + 1] = pad_rear
    
        return out_padding
    
    
    def _filter2d(input, kernel):
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)
    
        tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    
        height, width = tmp_kernel.shape[-2:]
    
        padding_shape: list[int] = _compute_padding([height, width])
        input = torch.nn.functional.pad(input, padding_shape, mode="reflect")
    
        # kernel and input tensor reshape to align element-wise or batch-wise params
        tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
        input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))
    
        # convolve the tensor with the kernel.
        output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    
        out = output.view(b, c, h, w)
        return out
    
    
    def _gaussian(window_size: int, sigma):
        if isinstance(sigma, float):
            sigma = torch.tensor([[sigma]])
    
        batch_size = sigma.shape[0]
    
        x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)
    
        if window_size % 2 == 0:
    
            x = x + 0.5
    
        gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    
        return gauss / gauss.sum(-1, keepdim=True)
    
    
    def _gaussian_blur2d(input, kernel_size, sigma):
        if isinstance(sigma, tuple):
            sigma = torch.tensor([sigma], dtype=input.dtype)
        else:
            sigma = sigma.to(dtype=input.dtype)
    
        ky, kx = int(kernel_size[0]), int(kernel_size[1])
        bs = sigma.shape[0]
        kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
        kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
        out_x = _filter2d(input, kernel_x[..., None, :])
        out = _filter2d(out_x, kernel_y[..., None])
    
        return out

Run Video Generation
--------------------



Select Inference Device
~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    import ipywidgets as widgets
    
    core = ov.Core()
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="AUTO",
        description="Device:",
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=3, options=('CPU', 'GPU.0', 'GPU.1', 'AUTO'), value='AUTO')



.. code:: ipython3

    from transformers import CLIPImageProcessor
    
    
    vae_encoder = core.compile_model(VAE_ENCODER_PATH, device.value)
    image_encoder = core.compile_model(IMAGE_ENCODER_PATH, device.value)
    unet = core.compile_model(UNET_PATH, device.value)
    vae_decoder = core.compile_model(VAE_DECODER_PATH, device.value)
    scheduler = AnimateLCMSVDStochasticIterativeScheduler.from_pretrained(MODEL_DIR / "scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(MODEL_DIR / "feature_extractor")

Now, let’s see model in action. > Please, note, video generation is
memory and time consuming process. For reducing memory consumption, we
decreased input video resolution to 576x320 and number of generated
frames that may affect quality of generated video. You can change these
settings manually providing ``height``, ``width`` and ``num_frames``
parameters into pipeline.

.. code:: ipython3

    ov_pipe = OVStableVideoDiffusionPipeline(vae_encoder, image_encoder, unet, vae_decoder, scheduler, feature_extractor)

.. code:: ipython3

    frames = ov_pipe(
        image,
        num_inference_steps=4,
        motion_bucket_id=60,
        num_frames=8,
        height=320,
        width=512,
        generator=torch.manual_seed(12342),
    ).frames[0]



.. parsed-literal::

      0%|          | 0/4 [00:00<?, ?it/s]


.. parsed-literal::

    denoise currently
    tensor(128.5637)
    denoise currently
    tensor(13.6784)
    denoise currently
    tensor(0.4969)
    denoise currently
    tensor(0.)


.. code:: ipython3

    out_path = Path("generated.mp4")
    
    export_to_video(frames, str(out_path), fps=7)
    frames[0].save(
        "generated.gif",
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=120,
        loop=0,
    )

.. code:: ipython3

    from IPython.display import HTML
    
    HTML('<img src="generated.gif">')




.. raw:: html

    <img src="generated.gif">



Interactive Demo
----------------



.. code:: ipython3

    import gradio as gr
    import random
    
    max_64_bit_int = 2**63 - 1
    
    example_images_urls = [
        "https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/resolve/main/test_imgs/ship-7833921_1280.jpg?download=true",
        "https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/resolve/main/test_imgs/ai-generated-8476858_1280.png?download=true",
        "https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/resolve/main/test_imgs/ai-generated-8481641_1280.jpg?download=true",
        "https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/resolve/main/test_imgs/dog-7396912_1280.jpg?download=true",
        "https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/resolve/main/test_imgs/cupcakes-380178_1280.jpg?download=true",
    ]
    
    example_images_dir = Path("example_images")
    example_images_dir.mkdir(exist_ok=True)
    example_imgs = []
    
    for image_id, url in enumerate(example_images_urls):
        img = load_image(url)
        image_path = example_images_dir / f"{image_id}.png"
        img.save(image_path)
        example_imgs.append([image_path])
    
    
    def sample(
        image: PIL.Image,
        seed: Optional[int] = 42,
        randomize_seed: bool = True,
        motion_bucket_id: int = 127,
        fps_id: int = 6,
        num_inference_steps: int = 15,
        num_frames: int = 4,
        max_guidance_scale=1.0,
        min_guidance_scale=1.0,
        decoding_t: int = 8,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
        output_folder: str = "outputs",
        progress=gr.Progress(track_tqdm=True),
    ):
        if image.mode == "RGBA":
            image = image.convert("RGB")
    
        if randomize_seed:
            seed = random.randint(0, max_64_bit_int)
        generator = torch.manual_seed(seed)
    
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        base_count = len(list(output_folder.glob("*.mp4")))
        video_path = output_folder / f"{base_count:06d}.mp4"
    
        frames = ov_pipe(
            image,
            decode_chunk_size=decoding_t,
            generator=generator,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=0.1,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            max_guidance_scale=max_guidance_scale,
            min_guidance_scale=min_guidance_scale,
        ).frames[0]
        export_to_video(frames, str(video_path), fps=fps_id)
    
        return video_path, seed
    
    
    def resize_image(image, output_size=(512, 320)):
        # Calculate aspect ratios
        target_aspect = output_size[0] / output_size[1]  # Aspect ratio of the desired size
        image_aspect = image.width / image.height  # Aspect ratio of the original image
    
        # Resize then crop if the original image is larger
        if image_aspect > target_aspect:
            # Resize the image to match the target height, maintaining aspect ratio
            new_height = output_size[1]
            new_width = int(new_height * image_aspect)
            resized_image = image.resize((new_width, new_height), PIL.Image.LANCZOS)
            # Calculate coordinates for cropping
            left = (new_width - output_size[0]) / 2
            top = 0
            right = (new_width + output_size[0]) / 2
            bottom = output_size[1]
        else:
            # Resize the image to match the target width, maintaining aspect ratio
            new_width = output_size[0]
            new_height = int(new_width / image_aspect)
            resized_image = image.resize((new_width, new_height), PIL.Image.LANCZOS)
            # Calculate coordinates for cropping
            left = 0
            top = (new_height - output_size[1]) / 2
            right = output_size[0]
            bottom = (new_height + output_size[1]) / 2
    
        # Crop the image
        cropped_image = resized_image.crop((left, top, right, bottom))
        return cropped_image
    
    
    with gr.Blocks() as demo:
        gr.Markdown(
            """# Stable Video Diffusion: Image to Video Generation with OpenVINO.
      """
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(label="Upload your image", type="pil")
                generate_btn = gr.Button("Generate")
            video = gr.Video()
        with gr.Accordion("Advanced options", open=False):
            seed = gr.Slider(
                label="Seed",
                value=42,
                randomize=True,
                minimum=0,
                maximum=max_64_bit_int,
                step=1,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            motion_bucket_id = gr.Slider(
                label="Motion bucket id",
                info="Controls how much motion to add/remove from the image",
                value=127,
                minimum=1,
                maximum=255,
            )
            fps_id = gr.Slider(
                label="Frames per second",
                info="The length of your video in seconds will be num_frames / fps",
                value=6,
                minimum=5,
                maximum=30,
                step=1,
            )
            num_frames = gr.Slider(label="Number of Frames", value=8, minimum=2, maximum=25, step=1)
            num_steps = gr.Slider(label="Number of generation steps", value=4, minimum=1, maximum=8, step=1)
            max_guidance_scale = gr.Slider(
                label="Max guidance scale",
                info="classifier-free guidance strength",
                value=1.2,
                minimum=1,
                maximum=2,
            )
            min_guidance_scale = gr.Slider(
                label="Min guidance scale",
                info="classifier-free guidance strength",
                value=1,
                minimum=1,
                maximum=1.5,
            )
        examples = gr.Examples(
            examples=example_imgs,
            inputs=[image_in],
            outputs=[video, seed],
        )
    
        image_in.upload(fn=resize_image, inputs=image_in, outputs=image_in)
        generate_btn.click(
            fn=sample,
            inputs=[
                image_in,
                seed,
                randomize_seed,
                motion_bucket_id,
                fps_id,
                num_steps,
                num_frames,
                max_guidance_scale,
                min_guidance_scale,
            ],
            outputs=[video, seed],
            api_name="video",
        )
    
    
    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(debug=False, share=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    Rerunning server... use `close()` to stop if you need to change `launch()` parameters.
    ----

