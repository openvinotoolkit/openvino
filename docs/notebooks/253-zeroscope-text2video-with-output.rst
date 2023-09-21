Video generation with ZeroScope and OpenVINO
============================================



The ZeroScope model is a free and open-source text-to-video model that
can generate realistic and engaging videos from text descriptions. It is
based on the
`Modelscope <https://modelscope.cn/models/damo/text-to-video-synthesis/summary>`__
model, but it has been improved to produce higher-quality videos with a
16:9 aspect ratio and no Shutterstock watermark. The ZeroScope model is
available in two versions: ZeroScope_v2 576w, which is optimized for
rapid content creation at a resolution of 576x320 pixels, and
ZeroScope_v2 XL, which upscales videos to a high-definition resolution
of 1024x576.

The ZeroScope model is trained on a dataset of over 9,000 videos and
29,000 tagged frames. It uses a diffusion model to generate videos,
which means that it starts with a random noise image and gradually adds
detail to it until it matches the text description. The ZeroScope model
is still under development, but it has already been used to create some
impressive videos. For example, it has been used to create videos of
people dancing, playing sports, and even driving cars.

The ZeroScope model is a powerful tool that can be used to create
various videos, from simple animations to complex scenes. It is still
under development, but it has the potential to revolutionize the way we
create and consume video content.

Both versions of the ZeroScope model are available on Hugging Face:

- `ZeroScope_v2 576w <https://huggingface.co/cerspense/zeroscope_v2_576w>`__
- `ZeroScope_v2 XL <https://huggingface.co/cerspense/zeroscope_v2_XL>`__

We will use the first one.

.. _top:

**Table of contents**:

- `Install and import required packages <#install-and-import-required-packages>`__
- `Load the model <#load-the-model>`__
- `Convert the model <#convert-the-model>`__

  - `Define the conversion function <#define-the-conversion-function>`__
  - `UNet <#unet>`__ -
  - `VAE <#vae>`__
  - `Text encoder <#text-encoder>`__

- `Build a pipeline <#build-a-pipeline>`__
- `Inference with OpenVINO <#inference-with-openvino>`__

  - `Select inference device <#select-inference-device>`__
  - `Define a prompt <#define-a-prompt>`__
  - `Video generation <#video-generation>`__


.. important::

   This tutorial requires at least 24GB of free memory to generate a video with 
   a frame size of 432x240 and 16 frames. Increasing either of these values will 
   require more memory and take more time.


Install and import required packages `⇑ <#top>`__
###############################################################################################################################

To work with text-to-video synthesis model, we will use Hugging Face’s
`Diffusers <https://github.com/huggingface/diffusers>`__ library. It
provides already pretrained model from ``cerspense``.

.. code:: ipython3

    !pip install -q "diffusers[torch]>=0.15.0" transformers "openvino==2023.1.0.dev20230811" numpy gradio

.. code:: ipython3

    import gc
    from pathlib import Path
    from typing import Optional, Union, List, Callable
    import base64
    import tempfile
    import warnings
    
    import diffusers
    import transformers
    import numpy as np
    import IPython
    import ipywidgets as widgets
    import torch
    import PIL
    import gradio as gr
    
    import openvino as ov


.. parsed-literal::

    2023-08-16 21:15:40.145184: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-08-16 21:15:40.146998: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2023-08-16 21:15:40.179214: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2023-08-16 21:15:40.180050: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-08-16 21:15:40.750499: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    

Original 576x320 inference requires a lot of RAM (>100GB), so let’s run
our example on a smaller frame size, keeping the same aspect ratio. Try
reducing values below to reduce the memory consumption.

.. code:: ipython3

    WIDTH = 432  # must be divisible by 8
    HEIGHT = 240  # must be divisible by 8
    NUM_FRAMES = 16

Load the model `⇑ <#top>`__
###############################################################################################################################

The model is loaded from HuggingFace using ``.from_pretrained`` method
of ``diffusers.DiffusionPipeline``.

.. code:: ipython3

    pipe = diffusers.DiffusionPipeline.from_pretrained('cerspense/zeroscope_v2_576w')


.. parsed-literal::

    vae/diffusion_pytorch_model.safetensors not found
    


.. parsed-literal::

    Loading pipeline components...:   0%|          | 0/5 [00:00<?, ?it/s]


.. code:: ipython3

    unet = pipe.unet
    unet.eval()
    vae = pipe.vae
    vae.eval()
    text_encoder = pipe.text_encoder
    text_encoder.eval()
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler
    vae_scale_factor = pipe.vae_scale_factor
    unet_in_channels = pipe.unet.config.in_channels
    sample_width = WIDTH // vae_scale_factor
    sample_height = HEIGHT // vae_scale_factor
    del pipe
    gc.collect();

Convert the model `⇑ <#top>`__
###############################################################################################################################

The architecture for generating videos from text comprises three
distinct sub-networks: one for extracting text features, another for
translating text features into the video latent space using a diffusion
model, and a final one for mapping the video latent space to the visual
space. The collective parameters of the entire model amount to
approximately 1.7 billion. It’s capable of processing English input. The
diffusion model is built upon the Unet3D model and achieves video
generation by iteratively denoising a starting point of pure Gaussian
noise video.

.. image:: 253-zeroscope-text2video-with-output_files/253-zeroscope-text2video-with-output_01_02.png


Define the conversion function `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Model components are PyTorch modules, that can be converted with
``ov.convert_model`` function directly. We also use ``ov.save_model``
function to serialize the result of conversion.

.. code:: ipython3

    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

.. code:: ipython3

    def convert(model: torch.nn.Module, xml_path: str, **convert_kwargs) -> Path:
        xml_path = Path(xml_path)
        if not xml_path.exists():
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                converted_model = ov.convert_model(model, **convert_kwargs)
            ov.save_model(converted_model, xml_path)
            del converted model
            gc.collect()
            torch._C._jit_clear_class_registry()
            torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
            torch.jit._state._clear_class_state()
        return xml_path

UNet `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Text-to-video generation pipeline main component is a conditional 3D
UNet model that takes a noisy sample, conditional state, and a timestep
and returns a sample shaped output.

.. code:: ipython3

    unet_xml_path = convert(
        unet,
        "models/unet.xml",
        example_input={
            "sample": torch.randn(2, 4, 2, 32, 32),
            "timestep": torch.tensor(1),
            "encoder_hidden_states": torch.randn(2, 77, 1024),
        },
        input=[
            ("sample", (2, 4, NUM_FRAMES, sample_height, sample_width)),
            ("timestep", ()),
            ("encoder_hidden_states", (2, 77, 1024)),
        ],
    )
    del unet
    gc.collect();


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.
    

.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.
    

VAE `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Variational autoencoder (VAE) uses UNet output to decode latents to
visual representations. Our VAE model has KL loss for encoding images
into latents and decoding latent representations into images. For
inference, we need only decoder part.

.. code:: ipython3

    class VaeDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
            
        def forward(self, z: torch.FloatTensor):
            return self.vae.decode(z)

.. code:: ipython3

    vae_decoder_xml_path = convert(
        VaeDecoderWrapper(vae),
        "models/vae.xml",
        example_input=torch.randn(2, 4, 32, 32),
        input=((NUM_FRAMES, 4, sample_height, sample_width)),
    )
    del vae
    gc.collect();

Text encoder `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Text encoder is used to encode the input prompt to tensor. Default
tensor length is 77.

.. code:: ipython3

    text_encoder_xml = convert(
        text_encoder,
        "models/text_encoder.xml",
        example_input=torch.ones(1, 77, dtype=torch.int64),
        input=((1, 77), (ov.Type.i64,)),
    )
    del text_encoder
    gc.collect();

Build a pipeline `⇑ <#top>`__
###############################################################################################################################

.. code:: ipython3

    def tensor2vid(video: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> List[np.ndarray]:
        # This code is copied from https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78
        # reshape to ncfhw
        mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
        std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
        # unnormalize back to [0,1]
        video = video.mul_(std).add_(mean)
        video.clamp_(0, 1)
        # prepare the final outputs
        i, c, f, h, w = video.shape
        images = video.permute(2, 3, 0, 4, 1).reshape(
            f, h, i * w, c
        )  # 1st (frames, h, batch_size, w, c) 2nd (frames, h, batch_size * w, c)
        images = images.unbind(dim=0)  # prepare a list of indvidual (consecutive frames)
        images = [(image.cpu().numpy() * 255).astype("uint8") for image in images]  # f h w c
        return images

.. code:: ipython3

    class OVTextToVideoSDPipeline(diffusers.DiffusionPipeline):
        def __init__(
            self,
            vae_decoder: ov.CompiledModel,
            text_encoder: ov.CompiledModel,
            tokenizer: transformers.CLIPTokenizer,
            unet: ov.CompiledModel,
            scheduler: diffusers.schedulers.DDIMScheduler,
        ):
            super().__init__()
    
            self.vae_decoder = vae_decoder
            self.text_encoder = text_encoder
            self.tokenizer = tokenizer
            self.unet = unet
            self.scheduler = scheduler
            self.vae_scale_factor = vae_scale_factor
            self.unet_in_channels = unet_in_channels
            self.width = WIDTH
            self.height = HEIGHT
            self.num_frames = NUM_FRAMES
    
        def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 9.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "np",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
        ):
            r"""
            Function invoked when calling the pipeline for generation.
    
            Args:
                prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.
                    instead.
                num_inference_steps (`int`, *optional*, defaults to 50):
                    The number of denoising steps. More denoising steps usually lead to a higher quality videos at the
                    expense of slower inference.
                guidance_scale (`float`, *optional*, defaults to 7.5):
                    Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                    `guidance_scale` is defined as `w` of equation 2. of [Imagen
                    Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                    1`. Higher guidance scale encourages to generate videos that are closely linked to the text `prompt`,
                    usually at the expense of lower video quality.
                negative_prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts not to guide the video generation. If not defined, one has to pass
                    `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                    less than `1`).
                eta (`float`, *optional*, defaults to 0.0):
                    Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                    [`schedulers.DDIMScheduler`], will be ignored for others.
                generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                    One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                    to make generation deterministic.
                latents (`torch.FloatTensor`, *optional*):
                    Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
                    generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                    tensor will ge generated by sampling using the supplied random `generator`. Latents should be of shape
                    `(batch_size, num_channel, num_frames, height, width)`.
                prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                    provided, text embeddings will be generated from `prompt` input argument.
                negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                    weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                    argument.
                output_type (`str`, *optional*, defaults to `"np"`):
                    The output format of the generate video. Choose between `torch.FloatTensor` or `np.array`.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~pipelines.stable_diffusion.TextToVideoSDPipelineOutput`] instead of a
                    plain tuple.
                callback (`Callable`, *optional*):
                    A function that will be called every `callback_steps` steps during inference. The function will be
                    called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
                callback_steps (`int`, *optional*, defaults to 1):
                    The frequency at which the `callback` function will be called. If not specified, the callback will be
                    called at every step.
    
            Returns:
                `List[np.ndarray]`: generated video frames
            """
    
            num_images_per_prompt = 1
    
            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
            )
    
            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]
    
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0
    
            # 3. Encode input prompt
            prompt_embeds = self._encode_prompt(
                prompt,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )
    
            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
    
            # 5. Prepare latent variables
            num_channels_latents = self.unet_in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                prompt_embeds.dtype,
                generator,
                latents,
            )
    
            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = {"generator": generator, "eta": eta}
    
            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    
                    # predict the noise residual
                    noise_pred = self.unet(
                        {
                            "sample": latent_model_input,
                            "timestep": t,
                            "encoder_hidden_states": prompt_embeds,
                        }
                    )[0]
                    noise_pred = torch.tensor(noise_pred)
    
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
    
                    # reshape latents
                    bsz, channel, frames, width, height = latents.shape
                    latents = latents.permute(0, 2, 1, 3, 4).reshape(
                        bsz * frames, channel, width, height
                    )
                    noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(
                        bsz * frames, channel, width, height
                    )
    
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs
                    ).prev_sample
    
                    # reshape latents back
                    latents = (
                        latents[None, :]
                        .reshape(bsz, frames, channel, width, height)
                        .permute(0, 2, 1, 3, 4)
                    )
    
                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)
    
            video_tensor = self.decode_latents(latents)
    
            if output_type == "pt":
                video = video_tensor
            else:
                video = tensor2vid(video_tensor)
    
            if not return_dict:
                return (video,)
    
            return {"frames": video}
    
        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
        def _encode_prompt(
            self,
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ):
            r"""
            Encodes the prompt into text encoder hidden states.
    
            Args:
                 prompt (`str` or `List[str]`, *optional*):
                    prompt to be encoded
                num_images_per_prompt (`int`):
                    number of images that should be generated per prompt
                do_classifier_free_guidance (`bool`):
                    whether to use classifier free guidance or not
                negative_prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts not to guide the image generation. If not defined, one has to pass
                    `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                    less than `1`).
                prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                    provided, text embeddings will be generated from `prompt` input argument.
                negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                    weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                    argument.
            """
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]
    
            if prompt_embeds is None:
                text_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = self.tokenizer(
                    prompt, padding="longest", return_tensors="pt"
                ).input_ids
    
                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                    )
                    print(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    )
    
                prompt_embeds = self.text_encoder(text_input_ids)
                prompt_embeds = prompt_embeds[0]
                prompt_embeds = torch.tensor(prompt_embeds)
    
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
    
            # get unconditional embeddings for classifier free guidance
            if do_classifier_free_guidance and negative_prompt_embeds is None:
                uncond_tokens: List[str]
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt
    
                max_length = prompt_embeds.shape[1]
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
    
                negative_prompt_embeds = self.text_encoder(uncond_input.input_ids)
                negative_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = torch.tensor(negative_prompt_embeds)
    
            if do_classifier_free_guidance:
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = negative_prompt_embeds.shape[1]
    
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_images_per_prompt, seq_len, -1
                )
    
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    
            return prompt_embeds
    
        def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            dtype,
            generator,
            latents=None,
        ):
            shape = (
                batch_size,
                num_channels_latents,
                self.num_frames,
                self.height // self.vae_scale_factor,
                self.width // self.vae_scale_factor,
            )
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            if latents is None:
                latents = diffusers.utils.randn_tensor(shape, generator=generator, dtype=dtype)
    
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            return latents
    
        def check_inputs(
            self,
            prompt,
            callback_steps,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        ):
            if self.height % 8 != 0 or self.width % 8 != 0:
                raise ValueError(
                    f"`height` and `width` have to be divisible by 8 but are {self.height} and {self.width}."
                )
    
            if (callback_steps is None) or (
                callback_steps is not None
                and (not isinstance(callback_steps, int) or callback_steps <= 0)
            ):
                raise ValueError(
                    f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                    f" {type(callback_steps)}."
                )
    
            if prompt is not None and prompt_embeds is not None:
                raise ValueError(
                    f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                    " only forward one of the two."
                )
            elif prompt is None and prompt_embeds is None:
                raise ValueError(
                    "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
                )
            elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    
            if negative_prompt is not None and negative_prompt_embeds is not None:
                raise ValueError(
                    f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                    f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
                )
    
            if prompt_embeds is not None and negative_prompt_embeds is not None:
                if prompt_embeds.shape != negative_prompt_embeds.shape:
                    raise ValueError(
                        "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                        f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                        f" {negative_prompt_embeds.shape}."
                    )
    
        def decode_latents(self, latents):
            scale_factor = 0.18215
            latents = 1 / scale_factor * latents
    
            batch_size, channels, num_frames, height, width = latents.shape
            latents = latents.permute(0, 2, 1, 3, 4).reshape(
                batch_size * num_frames, channels, height, width
            )
            image = self.vae_decoder(latents)[0]
            image = torch.tensor(image)
            video = (
                image[None, :]
                .reshape(
                    (
                        batch_size,
                        num_frames,
                        -1,
                    )
                    + image.shape[2:]
                )
                .permute(0, 2, 1, 3, 4)
            )
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
            video = video.float()
            return video

Inference with OpenVINO `⇑ <#top>`__
###############################################################################################################################

.. code:: ipython3

    core = ov.Core()

Select inference device `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=4, options=('CPU', 'GPU.0', 'GPU.1', 'GPU.2', 'AUTO'), value='AUTO')



.. code:: ipython3

    %%time
    ov_unet = core.compile_model(unet_xml_path, device_name=device.value)


.. parsed-literal::

    CPU times: user 14.1 s, sys: 5.62 s, total: 19.7 s
    Wall time: 10.6 s
    

.. code:: ipython3

    %%time
    ov_vae_decoder = core.compile_model(vae_decoder_xml_path, device_name=device.value)


.. parsed-literal::

    CPU times: user 456 ms, sys: 320 ms, total: 776 ms
    Wall time: 328 ms
    

.. code:: ipython3

    %%time
    ov_text_encoder = core.compile_model(text_encoder_xml, device_name=device.value)


.. parsed-literal::

    CPU times: user 1.78 s, sys: 1.44 s, total: 3.22 s
    Wall time: 1.13 s
    

Here we replace the pipeline parts with versions converted to OpenVINO
IR and compiled to specific device. Note that we use original pipeline
tokenizer and scheduler.

.. code:: ipython3

    ov_pipe = OVTextToVideoSDPipeline(ov_vae_decoder, ov_text_encoder, tokenizer, ov_unet, scheduler)

Define a prompt `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    prompt = "A panda eating bamboo on a rock."

Let’s generate a video for our prompt. For full list of arguments, see
``__call__`` function definition of ``OVTextToVideoSDPipeline`` class in
`Build a pipeline <#Build-a-pipeline>`__ section.

Video generation `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    frames = ov_pipe(prompt, num_inference_steps=25)['frames']



.. parsed-literal::

      0%|          | 0/25 [00:00<?, ?it/s]


.. code:: ipython3

    images = [PIL.Image.fromarray(frame) for frame in frames]
    images[0].save("output.gif", save_all=True, append_images=images[1:], duration=125, loop=0)
    with open("output.gif", "rb") as gif_file:
        b64 = f'data:image/gif;base64,{base64.b64encode(gif_file.read()).decode()}'
    IPython.display.HTML(f"<img src=\"{b64}\" />")


.. image:: 253-zeroscope-text2video-with-output_files/253-zeroscope-text2video-with-output_01_03.gif


Interactive demo `⇑ <#top>`__
###############################################################################################################################

.. code:: ipython3

    def generate(
        prompt, seed, num_inference_steps, _=gr.Progress(track_tqdm=True)
    ):
        generator = torch.Generator().manual_seed(seed)
        frames = ov_pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )["frames"]
        out_file = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        images = [PIL.Image.fromarray(frame) for frame in frames]
        images[0].save(
            out_file, save_all=True, append_images=images[1:], duration=125, loop=0
        )
        return out_file.name


    demo = gr.Interface(
        generate,
        [
            gr.Textbox(label="Prompt"),
            gr.Slider(0, 1000000, value=42, label="Seed", step=1),
            gr.Slider(10, 50, value=25, label="Number of inference steps", step=1),
        ],
        gr.Image(label="Result"),
        examples=[
            ["An astronaut riding a horse.", 0, 25],
            ["A panda eating bamboo on a rock.", 0, 25],
            ["Spiderman is surfing.", 0, 25],
        ],
        allow_flagging="never"
    )

    try:
        demo.queue().launch(debug=True)
    except Exception:
        demo.queue().launch(share=True, debug=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
