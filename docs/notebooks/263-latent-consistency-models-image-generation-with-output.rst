Image generation with Latent Consistency Model and OpenVINO
===========================================================

LCMs: The next generation of generative models after Latent Diffusion
Models (LDMs). Latent Diffusion models (LDMs) have achieved remarkable
results in synthesizing high-resolution images. However, the iterative
sampling is computationally intensive and leads to slow generation.

Inspired by `Consistency Models <https://arxiv.org/abs/2303.01469>`__,
`Latent Consistency Models <https://arxiv.org/pdf/2310.04378.pdf>`__
(LCMs) were proposed, enabling swift inference with minimal steps on any
pre-trained LDMs, including Stable Diffusion. The `Consistency Model
(CM) (Song et al., 2023) <https://arxiv.org/abs/2303.01469>`__ is a new
family of generative models that enables one-step or few-step
generation. The core idea of the CM is to learn the function that maps
any points on a trajectory of the PF-ODE (probability flow of `ordinary
differential
equation <https://en.wikipedia.org/wiki/Ordinary_differential_equation>`__)
to that trajectory’s origin (i.e., the solution of the PF-ODE). By
learning consistency mappings that maintain point consistency on
ODE-trajectory, these models allow for single-step generation,
eliminating the need for computation-intensive iterations. However, CM
is constrained to pixel space image generation tasks, making it
unsuitable for synthesizing high-resolution images. LCMs adopt a
consistency model in the image latent space for generation
high-resolution images. Viewing the guided reverse diffusion process as
solving an augmented probability flow ODE (PF-ODE), LCMs are designed to
directly predict the solution of such ODE in latent space, mitigating
the need for numerous iterations and allowing rapid, high-fidelity
sampling. Utilizing image latent space in large-scale diffusion models
like Stable Diffusion (SD) has effectively enhanced image generation
quality and reduced computational load. The authors of LCMs provide a
simple and efficient one-stage guided consistency distillation method
named Latent Consistency Distillation (LCD) to distill SD for few-step
(2∼4) or even 1-step sampling and propose the SKIPPING-STEP technique to
further accelerate the convergence. More details about proposed approach
and models can be found in `project
page <https://latent-consistency-models.github.io/>`__,
`paper <https://arxiv.org/abs/2310.04378>`__ and `original
repository <https://github.com/luosiallen/latent-consistency-model>`__.

In this tutorial, we consider how to convert and run LCM using OpenVINO.

**Table of contents:**


- `Prerequisites <#prerequisites>`__
- `Prepare models for OpenVINO format conversion <#prepare-models-for-openvino-format-conversion>`__
- `Convert models to OpenVINO format <#convert-models-to-openvino-format>`__
- `Text Encoder <#text-encoder>`__
- `U-Net <#u-net>`__
- `VAE <#vae>`__
- `Prepare inference pipeline <#prepare-inference-pipeline>`__
- `Configure Inference Pipeline <#configure-inference-pipeline>`__
- `Text-to-image generation <#text-to-image-generation>`__
- `Interactive demo <#interactive-demo>`__

Prerequisites 
-------------------------------------------------------

.. code:: ipython3

    %pip install -q "torch" --index-url https://download.pytorch.org/whl/cpu
    %pip install -q "openvino>=2023.1.0" transformers "diffusers>=0.21.4" pillow gradio

Prepare models for OpenVINO format conversion 
---------------------------------------------------------------------------------------

In this tutorial we will use
`LCM_Dreamshaper_v7 <https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7>`__
from `HuggingFace hub <https://huggingface.co/>`__. This model distilled
from `Dreamshaper v7 <https://huggingface.co/Lykon/dreamshaper-7>`__
fine-tune of `Stable-Diffusion
v1-5 <https://huggingface.co/runwayml/stable-diffusion-v1-5>`__ using
Latent Consistency Distillation (LCD) approach discussed above. This
model is also integrated into
`Diffusers <https://huggingface.co/docs/diffusers/index>`__ library. 🤗
Diffusers is the go-to library for state-of-the-art pretrained diffusion
models for generating images, audio, and even 3D structures of
molecules. This allows us to compare running original Stable Diffusion
(from this `notebook <../225-stable-diffusion-text-to-image>`__) and
distilled using LCD. The distillation approach efficiently converts a
pre-trained guided diffusion model into a latent consistency model by
solving an augmented PF-ODE.

For starting work with LCM, we should instantiate generation pipeline
first. ``DiffusionPipeline.from_pretrained`` method download all
pipeline components for LCM and configure them. This model uses custom
inference pipeline stored as part of model repository, we also should
provide which module should be loaded for initialization using
``custom_pipeline`` argument and revision for it.

.. code:: ipython3

    import gc
    import warnings
    from pathlib import Path
    from diffusers import DiffusionPipeline
    
    
    warnings.filterwarnings("ignore")
    
    TEXT_ENCODER_OV_PATH = Path("model/text_encoder.xml")
    UNET_OV_PATH = Path("model/unet.xml")
    VAE_DECODER_OV_PATH = Path("model/vae_decoder.xml")
    
    
    def load_orginal_pytorch_pipeline_componets(skip_models=False):
        pipe = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            custom_pipeline="latent_consistency_txt2img",
            custom_revision="main",
        )
        scheduler = pipe.scheduler
        tokenizer = pipe.tokenizer
        feature_extractor = pipe.feature_extractor
        safety_checker = pipe.safety_checker
        text_encoder, unet, vae = None, None, None
        if not skip_models:
            text_encoder = pipe.text_encoder
            text_encoder.eval()
            unet = pipe.unet
            unet.eval()
            vae = pipe.vae
            vae.eval()
        del pipe
        gc.collect()
        return (
            scheduler,
            tokenizer,
            feature_extractor,
            safety_checker,
            text_encoder,
            unet,
            vae,
        )


.. parsed-literal::

    /home/ea/work/ov_venv/lib/python3.8/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
      warn("The installed version of bitsandbytes was compiled without GPU support. "


.. parsed-literal::

    /home/ea/work/ov_venv/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so: undefined symbol: cadam32bit_grad_fp32


.. parsed-literal::

    2023-10-25 13:59:59.802031: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-10-25 13:59:59.841632: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-10-25 14:00:00.487700: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. code:: ipython3

    skip_conversion = (
        TEXT_ENCODER_OV_PATH.exists()
        and UNET_OV_PATH.exists()
        and VAE_DECODER_OV_PATH.exists()
    )
    
    (
        scheduler,
        tokenizer,
        feature_extractor,
        safety_checker,
        text_encoder,
        unet,
        vae,
    ) = load_orginal_pytorch_pipeline_componets(skip_conversion)



.. parsed-literal::

    Loading pipeline components...:   0%|          | 0/6 [00:00<?, ?it/s]


Convert models to OpenVINO format 
---------------------------------------------------------------------------

Starting from 2023.0 release, OpenVINO supports PyTorch models directly
via Model Conversion API. ``ov.convert_model`` function accepts instance
of PyTorch model and example inputs for tracing and returns object of
``ov.Model`` class, ready to use or save on disk using ``ov.save_model``
function.

Like original Stable Diffusion pipeline, the LCM pipeline consists of
three important parts:

-  Text Encoder to create condition to generate an image from a text
   prompt.
-  U-Net for step-by-step denoising latent image representation.
-  Autoencoder (VAE) for decoding latent space to image.

Let us convert each part:

Text Encoder 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The text-encoder is responsible for transforming the input prompt, for
example, “a photo of an astronaut riding a horse” into an embedding
space that can be understood by the U-Net. It is usually a simple
transformer-based encoder that maps a sequence of input tokens to a
sequence of latent text embeddings.

Input of the text encoder is the tensor ``input_ids`` which contains
indexes of tokens from text processed by tokenizer and padded to maximum
length accepted by model. Model outputs are two tensors:
``last_hidden_state`` - hidden state from the last MultiHeadAttention
layer in the model and ``pooler_out`` - Pooled output for whole model
hidden states.

.. code:: ipython3

    import torch
    import openvino as ov
    
    
    def cleanup_torchscript_cache():
        """
        Helper for removing cached model representation
        """
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        torch.jit._state._clear_class_state()
    
    
    def convert_encoder(text_encoder: torch.nn.Module, ir_path: Path):
        """
        Convert Text Encoder mode.
        Function accepts text encoder model, and prepares example inputs for conversion,
        Parameters:
            text_encoder (torch.nn.Module): text_encoder model from Stable Diffusion pipeline
            ir_path (Path): File for storing model
        Returns:
            None
        """
        input_ids = torch.ones((1, 77), dtype=torch.long)
        # switch model to inference mode
        text_encoder.eval()
    
        # disable gradients calculation for reducing memory consumption
        with torch.no_grad():
            # Export model to IR format
            ov_model = ov.convert_model(
                text_encoder,
                example_input=input_ids,
                input=[
                    (-1, 77),
                ],
            )
        ov.save_model(ov_model, ir_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print(f"Text Encoder successfully converted to IR and saved to {ir_path}")
    
    
    if not TEXT_ENCODER_OV_PATH.exists():
        convert_encoder(text_encoder, TEXT_ENCODER_OV_PATH)
    else:
        print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")
    
    del text_encoder
    gc.collect()


.. parsed-literal::

    Text encoder will be loaded from model/text_encoder.xml




.. parsed-literal::

    0



U-Net 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

U-Net model, similar to Stable Diffusion UNet model, has four inputs:

-  ``sample`` - latent image sample from previous step. Generation
   process has not been started yet, so you will use random noise.
-  ``timestep`` - current scheduler step.
-  ``encoder_hidden_state`` - hidden state of text encoder.
-  ``timestep_cond`` - timestep condition for generation. This input is
   not present in original Stable Diffusion U-Net model and introduced
   by LCM for improving generation quality using Classifier-Free
   Guidance. `Classifier-free guidance
   (CFG) <https://arxiv.org/abs/2207.12598>`__ is crucial for
   synthesizing high-quality text-aligned images in Stable Diffusion,
   because it controls how similar the generated image will be to the
   prompt. In Latent Consistency Models, CFG serves as augmentation
   parameter for PF-ODE.

Model predicts the ``sample`` state for the next step.

.. code:: ipython3

    def convert_unet(unet: torch.nn.Module, ir_path: Path):
        """
        Convert U-net model to IR format.
        Function accepts unet model, prepares example inputs for conversion,
        Parameters:
            unet (StableDiffusionPipeline): unet from Stable Diffusion pipeline
            ir_path (Path): File for storing model
        Returns:
            None
        """
        # prepare inputs
        dummy_inputs = {
            "sample": torch.randn((1, 4, 64, 64)),
            "timestep": torch.ones([1]).to(torch.float32),
            "encoder_hidden_states": torch.randn((1, 77, 768)),
            "timestep_cond": torch.randn((1, 256)),
        }
        unet.eval()
        with torch.no_grad():
            ov_model = ov.convert_model(unet, example_input=dummy_inputs)
        ov.save_model(ov_model, ir_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print(f"Unet successfully converted to IR and saved to {ir_path}")
    
    
    if not UNET_OV_PATH.exists():
        convert_unet(unet, UNET_OV_PATH)
    else:
        print(f"Unet will be loaded from {UNET_OV_PATH}")
    del unet
    gc.collect()


.. parsed-literal::

    Unet will be loaded from model/unet.xml




.. parsed-literal::

    0



VAE 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The VAE model has two parts, an encoder and a decoder. The encoder is
used to convert the image into a low dimensional latent representation,
which will serve as the input to the U-Net model. The decoder,
conversely, transforms the latent representation back into an image.

During latent diffusion training, the encoder is used to get the latent
representations (latents) of the images for the forward diffusion
process, which applies more and more noise at each step. During
inference, the denoised latents generated by the reverse diffusion
process are converted back into images using the VAE decoder. When you
run inference for text-to-image, there is no initial image as a starting
point. You can skip this step and directly generate initial random
noise.

In our inference pipeline, we will not use VAE encoder part and skip its
conversion for reducing memory consumption. The process of conversion
VAE encoder, can be found in Stable Diffusion notebook.

.. code:: ipython3

    def convert_vae_decoder(vae: torch.nn.Module, ir_path: Path):
        """
        Convert VAE model for decoding to IR format.
        Function accepts vae model, creates wrapper class for export only necessary for inference part,
        prepares example inputs for conversion,
        Parameters:
            vae (torch.nn.Module): VAE model frm StableDiffusion pipeline
            ir_path (Path): File for storing model
        Returns:
            None
        """
    
        class VAEDecoderWrapper(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae
    
            def forward(self, latents):
                return self.vae.decode(latents)
    
        vae_decoder = VAEDecoderWrapper(vae)
        latents = torch.zeros((1, 4, 64, 64))
    
        vae_decoder.eval()
        with torch.no_grad():
            ov_model = ov.convert_model(vae_decoder, example_input=latents)
        ov.save_model(ov_model, ir_path)
        del ov_model
        cleanup_torchscript_cache()
        print(f"VAE decoder successfully converted to IR and saved to {ir_path}")
    
    
    if not VAE_DECODER_OV_PATH.exists():
        convert_vae_decoder(vae, VAE_DECODER_OV_PATH)
    else:
        print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")
    
    del vae
    gc.collect()


.. parsed-literal::

    VAE decoder will be loaded from model/vae_decoder.xml




.. parsed-literal::

    0



Prepare inference pipeline 
--------------------------------------------------------------------

Putting it all together, let us now take a closer look at how the model
works in inference by illustrating the logical flow.

.. figure:: https://user-images.githubusercontent.com/29454499/277402235-079bacfb-3b6d-424b-8d47-5ddf601e1639.png
   :alt: lcm-pipeline

   lcm-pipeline

The pipeline takes a latent image representation and a text prompt is
transformed to text embedding via CLIP’s text encoder as an input. The
initial latent image representation generated using random noise
generator. In difference, with original Stable Diffusion pipeline, LCM
also uses guidance scale for getting timestep conditional embeddings as
input for diffusion process, while in Stable Diffusion, it used for
scaling output latents.

Next, the U-Net iteratively *denoises* the random latent image
representations while being conditioned on the text embeddings. The
output of the U-Net, being the noise residual, is used to compute a
denoised latent image representation via a scheduler algorithm. LCM
introduces own scheduling algorithm that extends the denoising procedure
introduced in denoising diffusion probabilistic models (DDPMs) with
non-Markovian guidance. The *denoising* process is repeated given number
of times (by default 50 in original SD pipeline, but for LCM small
number of steps required ~2-8) to step-by-step retrieve better latent
image representations. When complete, the latent image representation is
decoded by the decoder part of the variational auto encoder.

.. code:: ipython3

    from typing import Union, Optional, Any, List, Dict
    from transformers import CLIPTokenizer, CLIPImageProcessor
    from diffusers.pipelines.stable_diffusion.safety_checker import (
        StableDiffusionSafetyChecker,
    )
    from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
    from diffusers.image_processor import VaeImageProcessor
    
    
    class LatentConsistencyModelPipeline(DiffusionPipeline):
        def __init__(
            self,
            vae_decoder: ov.Model,
            text_encoder: ov.Model,
            tokenizer: CLIPTokenizer,
            unet: ov.Model,
            scheduler: None,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            requires_safety_checker: bool = True,
        ):
            super().__init__()
            self.vae_decoder = vae_decoder
            self.text_encoder = text_encoder
            self.tokenizer = tokenizer
            self.unet = unet
            self.scheduler = scheduler
            self.safety_checker = safety_checker
            self.feature_extractor = feature_extractor
            self.vae_scale_factor = 2**3
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    
        def _encode_prompt(
            self,
            prompt,
            num_images_per_prompt,
            prompt_embeds: None,
        ):
            r"""
            Encodes the prompt into text encoder hidden states.
            Args:
                prompt (`str` or `List[str]`, *optional*):
                    prompt to be encoded
                num_images_per_prompt (`int`):
                    number of images that should be generated per prompt
                prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                    provided, text embeddings will be generated from `prompt` input argument.
            """
    
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
    
                if untruncated_ids.shape[-1] >= text_input_ids.shape[
                    -1
                ] and not torch.equal(text_input_ids, untruncated_ids):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    )
    
                prompt_embeds = self.text_encoder(text_input_ids, share_inputs=True, share_outputs=True)
                prompt_embeds = torch.from_numpy(prompt_embeds[0])
    
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )
    
            # Don't need to get uncond prompt embedding because of LCM Guided Distillation
            return prompt_embeds
    
        def run_safety_checker(self, image, dtype):
            if self.safety_checker is None:
                has_nsfw_concept = None
            else:
                if torch.is_tensor(image):
                    feature_extractor_input = self.image_processor.postprocess(
                        image, output_type="pil"
                    )
                else:
                    feature_extractor_input = self.image_processor.numpy_to_pil(image)
                safety_checker_input = self.feature_extractor(
                    feature_extractor_input, return_tensors="pt"
                )
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
                )
            return image, has_nsfw_concept
    
        def prepare_latents(
            self, batch_size, num_channels_latents, height, width, dtype, latents=None
        ):
            shape = (
                batch_size,
                num_channels_latents,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )
            if latents is None:
                latents = torch.randn(shape, dtype=dtype)
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            return latents
    
        def get_w_embedding(self, w, embedding_dim=512, dtype=torch.float32):
            """
            see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
            Args:
            timesteps: torch.Tensor: generate embedding vectors at these timesteps
            embedding_dim: int: dimension of the embeddings to generate
            dtype: data type of the generated embeddings
            Returns:
            embedding vectors with shape `(len(timesteps), embedding_dim)`
            """
            assert len(w.shape) == 1
            w = w * 1000.0
    
            half_dim = embedding_dim // 2
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
            emb = w.to(dtype)[:, None] * emb[None, :]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            if embedding_dim % 2 == 1:  # zero pad
                emb = torch.nn.functional.pad(emb, (0, 1))
            assert emb.shape == (w.shape[0], embedding_dim)
            return emb
    
        @torch.no_grad()
        def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            guidance_scale: float = 7.5,
            num_images_per_prompt: Optional[int] = 1,
            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 4,
            lcm_origin_steps: int = 50,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        ):
    
            # 1. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]
    
            # do_classifier_free_guidance = guidance_scale > 0.0
            # In LCM Implementation:  cfg_noise = noise_cond + cfg_scale * (noise_cond - noise_uncond) , (cfg_scale > 0.0 using CFG)
    
            # 2. Encode input prompt
            prompt_embeds = self._encode_prompt(
                prompt,
                num_images_per_prompt,
                prompt_embeds=prompt_embeds,
            )
    
            # 3. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, lcm_origin_steps)
            timesteps = self.scheduler.timesteps
    
            # 4. Prepare latent variable
            num_channels_latents = 4
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                latents,
            )
    
            bs = batch_size * num_images_per_prompt
    
            # 5. Get Guidance Scale Embedding
            w = torch.tensor(guidance_scale).repeat(bs)
            w_embedding = self.get_w_embedding(w, embedding_dim=256)
    
            # 6. LCM MultiStep Sampling Loop:
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
    
                    ts = torch.full((bs,), t, dtype=torch.long)
    
                    # model prediction (v-prediction, eps, x)
                    model_pred = self.unet([latents, ts, prompt_embeds, w_embedding],share_inputs=True, share_outputs=True)[0]
    
                    # compute the previous noisy sample x_t -> x_t-1
                    latents, denoised = self.scheduler.step(
                        torch.from_numpy(model_pred), i, t, latents, return_dict=False
                    )
                    progress_bar.update()
    
            if not output_type == "latent":
                image = torch.from_numpy(self.vae_decoder(denoised / 0.18215, share_inputs=True, share_outputs=True)[0])
                image, has_nsfw_concept = self.run_safety_checker(
                    image, prompt_embeds.dtype
                )
            else:
                image = denoised
                has_nsfw_concept = None
    
            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
    
            image = self.image_processor.postprocess(
                image, output_type=output_type, do_denormalize=do_denormalize
            )
    
            if not return_dict:
                return (image, has_nsfw_concept)
    
            return StableDiffusionPipelineOutput(
                images=image, nsfw_content_detected=has_nsfw_concept
            )

Configure Inference Pipeline 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, you should create instances of OpenVINO Model and compile it
using selected device. Select device from dropdown list for running
inference using OpenVINO.

.. code:: ipython3

    core = ov.Core()
    
    import ipywidgets as widgets
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="CPU",
        description="Device:",
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'GPU', 'AUTO'), value='CPU')



.. code:: ipython3

    text_enc = core.compile_model(TEXT_ENCODER_OV_PATH, device.value)
    unet_model = core.compile_model(UNET_OV_PATH, device.value)
    
    ov_config = {"INFERENCE_PRECISION_HINT": "f32"} if device.value != "CPU" else {}
    
    vae_decoder = core.compile_model(VAE_DECODER_OV_PATH, device.value, ov_config)

Model tokenizer and scheduler are also important parts of the pipeline.
This pipeline is also uses Safety Checker, the filter for detecting that
corresponding generated image contains “not-safe-for-work” (nsfw)
content. The process of nsfw content detection requires to obtain image
embeddings using CLIP model, so additionally feature extractor component
should be added in the pipeline. We reuse tokenizer, feature extractor,
scheduler and safety checker from original LCM pipeline.

.. code:: ipython3

    ov_pipe = LatentConsistencyModelPipeline(
        tokenizer=tokenizer,
        text_encoder=text_enc,
        unet=unet_model,
        vae_decoder=vae_decoder,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
        safety_checker=safety_checker,
    )

Text-to-image generation 
------------------------------------------------------------------

Now, let’s see model in action

.. code:: ipython3

    prompt = "a beautiful pink unicorn, 8k"
    num_inference_steps = 4
    torch.manual_seed(1234567)
    
    images = ov_pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=8.0,
        lcm_origin_steps=50,
        output_type="pil",
        height=512,
        width=512,
    ).images



.. parsed-literal::

      0%|          | 0/4 [00:00<?, ?it/s]


.. code:: ipython3

    images[0]




.. image:: 263-latent-consistency-models-image-generation-with-output_files/263-latent-consistency-models-image-generation-with-output_21_0.png



Nice. As you can see, the picture has quite a high definition 🔥.

Interactive demo 
----------------------------------------------------------

.. code:: ipython3

    import random
    import gradio as gr
    import numpy as np
    
    MAX_SEED = np.iinfo(np.int32).max
    
    examples = [
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour,"
        "style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
    ]
    
    
    def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        return seed
    
    
    MAX_IMAGE_SIZE = 768
    
    
    def generate(
        prompt: str,
        seed: int = 0,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 8.0,
        num_inference_steps: int = 4,
        num_images: int = 1,
        randomize_seed: bool = False,
        progress=gr.Progress(track_tqdm=True),
    ):
        seed = randomize_seed_fn(seed, randomize_seed)
        torch.manual_seed(seed)
        result = ov_pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images,
            lcm_origin_steps=50,
            output_type="pil",
        ).images[0]
        return result, seed
    
    
    with gr.Blocks() as demo:
        with gr.Group():
            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                run_button = gr.Button("Run", scale=0)
            result = gr.Image(label="Image", type="pil")
        with gr.Accordion("Advanced options", open=False):
            seed = gr.Slider(
                label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, randomize=True
            )
            randomize_seed = gr.Checkbox(label="Randomize seed across runs", value=True)
            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )
            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale for base",
                    minimum=2,
                    maximum=14,
                    step=0.1,
                    value=8.0,
                )
                num_inference_steps = gr.Slider(
                    label="Number of inference steps for base",
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=4,
                )
    
        gr.Examples(
            examples=examples,
            inputs=prompt,
            outputs=result,
            fn=generate,
            cache_examples=False,
        )
    
        gr.on(
            triggers=[
                prompt.submit,
                run_button.click,
            ],
            fn=generate,
            inputs=[
                prompt,
                seed,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                randomize_seed,
            ],
            outputs=[result, seed],
        )

.. code:: ipython3

    demo.queue().launch()
