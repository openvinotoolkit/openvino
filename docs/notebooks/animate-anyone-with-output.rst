Image-to-Video synthesis with AnimateAnyone and OpenVINO
========================================================

.. image:: https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/animate-anyone/animate-anyone.gif


`AnimateAnyone <https://arxiv.org/pdf/2311.17117.pdf>`__ tackles the
task of generating animation sequences from a single character image. It
builds upon diffusion models pre-trained on vast character image
datasets.

The core of AnimateAnyone is a diffusion model pre-trained on a massive
dataset of character images. This model learns the underlying character
representation and distribution, allowing for realistic and diverse
character animation. To capture the specific details and characteristics
of the input character image, AnimateAnyone incorporates a ReferenceNet
module. This module acts like an attention mechanism, focusing on the
input image and guiding the animation process to stay consistent with
the original character’s appearance. AnimateAnyone enables control over
the character’s pose during animation. This might involve using
techniques like parametric pose embedding or direct pose vector input,
allowing for the creation of various character actions and movements. To
ensure smooth transitions and temporal coherence throughout the
animation sequence, AnimateAnyone incorporates temporal modeling
techniques. This may involve recurrent architectures like LSTMs or
transformers that capture the temporal dependencies between video
frames.

Overall, AnimateAnyone combines a powerful pre-trained diffusion model
with a character-specific attention mechanism (ReferenceNet), pose
guidance, and temporal modeling to achieve controllable, high-fidelity
character animation from a single image.

Learn more in `GitHub
repo <https://github.com/MooreThreads/Moore-AnimateAnyone>`__ and
`paper <https://arxiv.org/pdf/2311.17117.pdf>`__.

.. warning::

   This tutorial requires at least **96 GB** of RAM for model conversion and **40 GB** for inference. Changing the values of ``HEIGHT`` ``WIDTH`` and ``VIDEO_LENGTH`` variables will change the memory consumption but will also affect accuracy.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Prepare base model <#prepare-base-model>`__
-  `Prepare image encoder <#prepare-image-encoder>`__
-  `Download weights <#download-weights>`__
-  `Initialize models <#initialize-models>`__
-  `Load pretrained weights <#load-pretrained-weights>`__
-  `Convert model to OpenVINO IR <#convert-model-to-openvino-ir>`__

   -  `VAE <#vae>`__
   -  `Reference UNet <#reference-unet>`__
   -  `Denoising UNet <#denoising-unet>`__
   -  `Pose Guider <#pose-guider>`__
   -  `Image Encoder <#image-encoder>`__

-  `Inference <#inference>`__
-  `Video post-processing <#video-post-processing>`__
-  `Interactive inference <#interactive-inference>`__



This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



.. code:: ipython3

    from pathlib import Path
    import requests


    REPO_PATH = Path("Moore-AnimateAnyone")
    if not REPO_PATH.exists():
        !git clone -q "https://github.com/itrushkin/Moore-AnimateAnyone.git"
    %pip install -q "torch>=2.1" torchvision einops omegaconf "diffusers<=0.24" transformers av accelerate "openvino>=2024.0" "nncf>=2.9.0" "gradio>=4.19" --extra-index-url "https://download.pytorch.org/whl/cpu"
    import sys

    sys.path.insert(0, str(REPO_PATH.resolve()))
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
    )
    open("skip_kernel_extension.py", "w").write(r.text)

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)

    %load_ext skip_kernel_extension

Note that we clone a fork of original repo with tweaked forward methods.

.. code:: ipython3

    MODEL_DIR = Path("models")
    VAE_ENCODER_PATH = MODEL_DIR / "vae_encoder.xml"
    VAE_DECODER_PATH = MODEL_DIR / "vae_decoder.xml"
    REFERENCE_UNET_PATH = MODEL_DIR / "reference_unet.xml"
    DENOISING_UNET_PATH = MODEL_DIR / "denoising_unet.xml"
    POSE_GUIDER_PATH = MODEL_DIR / "pose_guider.xml"
    IMAGE_ENCODER_PATH = MODEL_DIR / "image_encoder.xml"

    WIDTH = 448
    HEIGHT = 512
    VIDEO_LENGTH = 24

    SHOULD_CONVERT = not all(
        p.exists()
        for p in [
            VAE_ENCODER_PATH,
            VAE_DECODER_PATH,
            REFERENCE_UNET_PATH,
            DENOISING_UNET_PATH,
            POSE_GUIDER_PATH,
            IMAGE_ENCODER_PATH,
        ]
    )

.. code:: ipython3

    from datetime import datetime
    from typing import Optional, Union, List, Callable
    import math

    from PIL import Image
    import openvino as ov
    from torchvision import transforms
    from einops import repeat
    from tqdm.auto import tqdm
    from einops import rearrange
    from omegaconf import OmegaConf
    from diffusers import DDIMScheduler
    from diffusers.image_processor import VaeImageProcessor
    from transformers import CLIPImageProcessor
    import torch

    from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
    from src.utils.util import get_fps, read_frames
    from src.utils.util import save_videos_grid
    from src.pipelines.context import get_context_scheduler


.. parsed-literal::

    /home/itrushkin/.virtualenvs/test/lib/python3.10/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /home/itrushkin/.virtualenvs/test/lib/python3.10/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /home/itrushkin/.virtualenvs/test/lib/python3.10/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(


.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    from pathlib import PurePosixPath
    import gc
    import warnings

    from typing import Dict, Any
    from diffusers import AutoencoderKL
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import CLIPVisionModelWithProjection
    import nncf

    from src.models.unet_2d_condition import UNet2DConditionModel
    from src.models.unet_3d import UNet3DConditionModel
    from src.models.pose_guider import PoseGuider


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, openvino


Prepare base model
------------------



.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    local_dir = Path("./pretrained_weights/stable-diffusion-v1-5")
    local_dir.mkdir(parents=True, exist_ok=True)
    for hub_file in ["unet/config.json", "unet/diffusion_pytorch_model.bin"]:
        saved_path = local_dir / hub_file
        if saved_path.exists():
            continue
        hf_hub_download(
            repo_id="botp/stable-diffusion-v1-5",
            subfolder=PurePosixPath(saved_path.parent.name),
            filename=PurePosixPath(saved_path.name),
            local_dir=local_dir,
        )

Prepare image encoder
---------------------



.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    local_dir = Path("./pretrained_weights")
    local_dir.mkdir(parents=True, exist_ok=True)
    for hub_file in ["image_encoder/config.json", "image_encoder/pytorch_model.bin"]:
        saved_path = local_dir / hub_file
        if saved_path.exists():
            continue
        hf_hub_download(
            repo_id="lambdalabs/sd-image-variations-diffusers",
            subfolder=PurePosixPath(saved_path.parent.name),
            filename=PurePosixPath(saved_path.name),
            local_dir=local_dir,
        )

Download weights
----------------



.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    snapshot_download(
        repo_id="stabilityai/sd-vae-ft-mse", local_dir="./pretrained_weights/sd-vae-ft-mse"
    )
    snapshot_download(
        repo_id="patrolli/AnimateAnyone",
        local_dir="./pretrained_weights",
    )



.. parsed-literal::

    Fetching 5 files:   0%|          | 0/5 [00:00<?, ?it/s]



.. parsed-literal::

    Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]


.. code:: ipython3

    config = OmegaConf.load("Moore-AnimateAnyone/configs/prompts/animation.yaml")
    infer_config = OmegaConf.load("Moore-AnimateAnyone/" + config.inference_config)

Initialize models
-----------------



.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path)
    reference_unet = UNet2DConditionModel.from_pretrained(config.pretrained_base_model_path, subfolder="unet")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    )
    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256))
    image_enc = CLIPVisionModelWithProjection.from_pretrained(config.image_encoder_path)


    NUM_CHANNELS_LATENTS = denoising_unet.config.in_channels


.. parsed-literal::

    Some weights of the model checkpoint were not used when initializing UNet2DConditionModel:
     ['conv_norm_out.weight, conv_norm_out.bias, conv_out.weight, conv_out.bias']


Load pretrained weights
-----------------------



.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

Convert model to OpenVINO IR
----------------------------

The pose sequence is initially
encoded using Pose Guider and fused with multi-frame noise, followed by
the Denoising UNet conducting the denoising process for video
generation. The computational block of the Denoising UNet consists of
Spatial-Attention, Cross-Attention, and Temporal-Attention, as
illustrated in the dashed box on the right. The integration of reference
image involves two aspects. Firstly, detailed features are extracted
through ReferenceNet and utilized for Spatial-Attention. Secondly,
semantic features are extracted through the CLIP image encoder for
Cross-Attention. Temporal-Attention operates in the temporal dimension.
Finally, the VAE decoder decodes the result into a video clip.

.. image:: https://humanaigc.github.io/animate-anyone/static/images/f2_img.png

The pipeline contains 6 PyTorch modules:

- VAE encoder
- VAE decoder
- Image encoder
- Reference UNet
- Denoising UNet
- Pose Guider

For reducing memory consumption, weights compression optimization can be
applied using `NNCF <https://github.com/openvinotoolkit/nncf>`__. Weight
compression aims to reduce the memory footprint of a model. models,
which require extensive memory to store the weights during inference,
can benefit from weight compression in the following ways:

-  enabling the inference of exceptionally large models that cannot be
   accommodated in the memory of the device;

-  improving the inference performance of the models by reducing the
   latency of the memory access when computing the operations with
   weights, for example, Linear layers.

`Neural Network Compression Framework
(NNCF) <https://github.com/openvinotoolkit/nncf>`__ provides 4-bit /
8-bit mixed weight quantization as a compression method. The main
difference between weights compression and full model quantization
(post-training quantization) is that activations remain floating-point
in the case of weights compression which leads to a better accuracy. In
addition, weight compression is data-free and does not require a
calibration dataset, making it easy to use.

``nncf.compress_weights`` function can be used for performing weights
compression. The function accepts an OpenVINO model and other
compression parameters.

More details about weights compression can be found in `OpenVINO
documentation <https://docs.openvino.ai/2023.3/weight_compression.html>`__.

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    def cleanup_torchscript_cache():
        """
        Helper for removing cached model representation
        """
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        torch.jit._state._clear_class_state()

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    warnings.simplefilter("ignore", torch.jit.TracerWarning)

VAE
~~~



The VAE model has two parts, an encoder and a decoder. The encoder is
used to convert the image into a low dimensional latent representation,
which will serve as the input to the U-Net model. The decoder,
conversely, transforms the latent representation back into an image.

During latent diffusion training, the encoder is used to get the latent
representations (latents) of the images for the forward diffusion
process, which applies more and more noise at each step. During
inference, the denoised latents generated by the reverse diffusion
process are converted back into images using the VAE decoder.

As the encoder and the decoder are used independently in different parts
of the pipeline, it will be better to convert them to separate models.

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not VAE_ENCODER_PATH.exists():
        class VaeEncoder(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae

            def forward(self, x):
                return self.vae.encode(x).latent_dist.mean
        vae.eval()
        with torch.no_grad():
            vae_encoder = ov.convert_model(VaeEncoder(vae), example_input=torch.zeros(1,3,512,448))
        vae_encoder = nncf.compress_weights(vae_encoder)
        ov.save_model(vae_encoder, VAE_ENCODER_PATH)
        del vae_encoder
        cleanup_torchscript_cache()


.. parsed-literal::

    WARNING:nncf:NNCF provides best results with torch==2.1.2, while current torch version is 2.2.2+cpu. If you encounter issues, consider switching to torch==2.1.2
    INFO:nncf:Statistics of the bitwidth distribution:
    +--------------+---------------------------+-----------------------------------+
    | Num bits (N) | % all parameters (layers) |    % ratio-defining parameters    |
    |              |                           |             (layers)              |
    +==============+===========================+===================================+
    | 8            | 100% (32 / 32)            | 100% (32 / 32)                    |
    +--------------+---------------------------+-----------------------------------+



.. parsed-literal::

    Output()

















.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not VAE_DECODER_PATH.exists():
        class VaeDecoder(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae

            def forward(self, z):
                return self.vae.decode(z).sample
        vae.eval()
        with torch.no_grad():
            vae_decoder = ov.convert_model(VaeDecoder(vae), example_input=torch.zeros(1,4,HEIGHT//8,WIDTH//8))
        vae_decoder = nncf.compress_weights(vae_decoder)
        ov.save_model(vae_decoder, VAE_DECODER_PATH)
        del vae_decoder
        cleanup_torchscript_cache()
    del vae
    gc.collect()


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    +--------------+---------------------------+-----------------------------------+
    | Num bits (N) | % all parameters (layers) |    % ratio-defining parameters    |
    |              |                           |             (layers)              |
    +==============+===========================+===================================+
    | 8            | 100% (40 / 40)            | 100% (40 / 40)                    |
    +--------------+---------------------------+-----------------------------------+



.. parsed-literal::

    Output()

















Reference UNet
~~~~~~~~~~~~~~



Pipeline extracts reference attention features from all transformer
blocks inside Reference UNet model. We call the original forward pass to
obtain shapes of the outputs as they will be used in the next pipeline
step.

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not REFERENCE_UNET_PATH.exists():
        class ReferenceUNetWrapper(torch.nn.Module):
            def __init__(self, reference_unet):
                super().__init__()
                self.reference_unet = reference_unet

            def forward(self, sample, timestep, encoder_hidden_states):
                return self.reference_unet(sample, timestep, encoder_hidden_states, return_dict=False)[1]

        sample = torch.zeros(2, 4, HEIGHT // 8, WIDTH // 8)
        timestep = torch.tensor(0)
        encoder_hidden_states = torch.zeros(2, 1, 768)
        reference_unet.eval()
        with torch.no_grad():
            wrapper =  ReferenceUNetWrapper(reference_unet)
            example_input = (sample, timestep, encoder_hidden_states)
            ref_features_shapes = {k: v.shape for k, v in wrapper(*example_input).items()}
            ov_reference_unet = ov.convert_model(
                wrapper,
                example_input=example_input,
            )
        ov_reference_unet = nncf.compress_weights(ov_reference_unet)
        ov.save_model(ov_reference_unet, REFERENCE_UNET_PATH)
        del ov_reference_unet
        del wrapper
        cleanup_torchscript_cache()
    del reference_unet
    gc.collect()


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    +--------------+---------------------------+-----------------------------------+
    | Num bits (N) | % all parameters (layers) |    % ratio-defining parameters    |
    |              |                           |             (layers)              |
    +==============+===========================+===================================+
    | 8            | 100% (270 / 270)          | 100% (270 / 270)                  |
    +--------------+---------------------------+-----------------------------------+



.. parsed-literal::

    Output()

















Denoising UNet
~~~~~~~~~~~~~~



Denoising UNet is the main part of all diffusion pipelines. This model
consumes the majority of memory, so we need to reduce its size as much
as possible.

Here we make all shapes static meaning that the size of the video will
be constant.

Also, we use the ``ref_features`` input with the same tensor shapes as
output of `Reference UNet <#reference-unet>`__ model on the previous
step.

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not DENOISING_UNET_PATH.exists():
        class DenoisingUNetWrapper(torch.nn.Module):
            def __init__(self, denoising_unet):
                super().__init__()
                self.denoising_unet = denoising_unet

            def forward(
                self,
                sample,
                timestep,
                encoder_hidden_states,
                pose_cond_fea,
                ref_features
            ):
                return self.denoising_unet(
                    sample,
                    timestep,
                    encoder_hidden_states,
                    ref_features,
                    pose_cond_fea=pose_cond_fea,
                    return_dict=False)

        example_input = {
            "sample": torch.zeros(2, 4, VIDEO_LENGTH, HEIGHT // 8, WIDTH // 8),
            "timestep": torch.tensor(999),
            "encoder_hidden_states": torch.zeros(2,1,768),
            "pose_cond_fea": torch.zeros(2, 320, VIDEO_LENGTH, HEIGHT // 8, WIDTH // 8),
            "ref_features": {k: torch.zeros(shape) for k, shape in ref_features_shapes.items()}
        }

        denoising_unet.eval()
        with torch.no_grad():
            ov_denoising_unet = ov.convert_model(
                DenoisingUNetWrapper(denoising_unet),
                example_input=tuple(example_input.values())
            )
        ov_denoising_unet.inputs[0].get_node().set_partial_shape(ov.PartialShape((2, 4, VIDEO_LENGTH, HEIGHT // 8, WIDTH // 8)))
        ov_denoising_unet.inputs[2].get_node().set_partial_shape(ov.PartialShape((2, 1, 768)))
        ov_denoising_unet.inputs[3].get_node().set_partial_shape(ov.PartialShape((2, 320, VIDEO_LENGTH, HEIGHT // 8, WIDTH // 8)))
        for ov_input, shape in zip(ov_denoising_unet.inputs[4:], ref_features_shapes.values()):
            ov_input.get_node().set_partial_shape(ov.PartialShape(shape))
            ov_input.get_node().set_element_type(ov.Type.f32)
        ov_denoising_unet.validate_nodes_and_infer_types()
        ov_denoising_unet = nncf.compress_weights(ov_denoising_unet)
        ov.save_model(ov_denoising_unet, DENOISING_UNET_PATH)
        del ov_denoising_unet
        cleanup_torchscript_cache()
    del denoising_unet
    gc.collect()


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    +--------------+---------------------------+-----------------------------------+
    | Num bits (N) | % all parameters (layers) |    % ratio-defining parameters    |
    |              |                           |             (layers)              |
    +==============+===========================+===================================+
    | 8            | 100% (534 / 534)          | 100% (534 / 534)                  |
    +--------------+---------------------------+-----------------------------------+



.. parsed-literal::

    Output()

















Pose Guider
~~~~~~~~~~~



To ensure pose controllability, a lightweight pose guider is devised to
efficiently integrate pose control signals into the denoising process.

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not POSE_GUIDER_PATH.exists():
        pose_guider.eval()
        with torch.no_grad():
            ov_pose_guider = ov.convert_model(pose_guider, example_input=torch.zeros(1, 3, VIDEO_LENGTH, HEIGHT, WIDTH))
        ov_pose_guider = nncf.compress_weights(ov_pose_guider)
        ov.save_model(ov_pose_guider, POSE_GUIDER_PATH)
        del ov_pose_guider
        cleanup_torchscript_cache()
    del pose_guider
    gc.collect()


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    +--------------+---------------------------+-----------------------------------+
    | Num bits (N) | % all parameters (layers) |    % ratio-defining parameters    |
    |              |                           |             (layers)              |
    +==============+===========================+===================================+
    | 8            | 100% (8 / 8)              | 100% (8 / 8)                      |
    +--------------+---------------------------+-----------------------------------+



.. parsed-literal::

    Output()

















Image Encoder
~~~~~~~~~~~~~



Pipeline uses CLIP image encoder to generate encoder hidden states
required for both reference and denoising UNets.

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not IMAGE_ENCODER_PATH.exists():
        image_enc.eval()
        with torch.no_grad():
            ov_image_encoder = ov.convert_model(image_enc, example_input=torch.zeros(1, 3, 224, 224), input=(1, 3, 224, 224))
        ov_image_encoder = nncf.compress_weights(ov_image_encoder)
        ov.save_model(ov_image_encoder, IMAGE_ENCODER_PATH)
        del ov_image_encoder
        cleanup_torchscript_cache()
    del image_enc
    gc.collect()


.. parsed-literal::

    /home/itrushkin/.virtualenvs/test/lib/python3.10/site-packages/transformers/modeling_utils.py:4225: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    +--------------+---------------------------+-----------------------------------+
    | Num bits (N) | % all parameters (layers) |    % ratio-defining parameters    |
    |              |                           |             (layers)              |
    +==============+===========================+===================================+
    | 8            | 100% (146 / 146)          | 100% (146 / 146)                  |
    +--------------+---------------------------+-----------------------------------+



.. parsed-literal::

    Output()

















Inference
---------



We inherit from the original pipeline modifying the calls to our models
to match OpenVINO format.

.. code:: ipython3

    core = ov.Core()

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



For starting work, please select inference device from dropdown list.

.. code:: ipython3

    from notebook_utils import device_widget

    device = device_widget()




.. parsed-literal::

    Dropdown(description='Device:', index=5, options=('CPU', 'GPU.0', 'GPU.1', 'GPU.2', 'GPU.3', 'AUTO'), value='A…



.. code:: ipython3

    class OVPose2VideoPipeline(Pose2VideoPipeline):
        def __init__(
            self,
            vae_encoder_path=VAE_ENCODER_PATH,
            vae_decoder_path=VAE_DECODER_PATH,
            image_encoder_path=IMAGE_ENCODER_PATH,
            reference_unet_path=REFERENCE_UNET_PATH,
            denoising_unet_path=DENOISING_UNET_PATH,
            pose_guider_path=POSE_GUIDER_PATH,
            device=device.value,
        ):
            self.vae_encoder = core.compile_model(vae_encoder_path, device)
            self.vae_decoder = core.compile_model(vae_decoder_path, device)
            self.image_encoder = core.compile_model(image_encoder_path, device)
            self.reference_unet = core.compile_model(reference_unet_path, device)
            self.denoising_unet = core.compile_model(denoising_unet_path, device)
            self.pose_guider = core.compile_model(pose_guider_path, device)
            self.scheduler = DDIMScheduler(**OmegaConf.to_container(infer_config.noise_scheduler_kwargs))

            self.vae_scale_factor = 8
            self.clip_image_processor = CLIPImageProcessor()
            self.ref_image_processor = VaeImageProcessor(do_convert_rgb=True)
            self.cond_image_processor = VaeImageProcessor(do_convert_rgb=True, do_normalize=False)

        def decode_latents(self, latents):
            video_length = latents.shape[2]
            latents = 1 / 0.18215 * latents
            latents = rearrange(latents, "b c f h w -> (b f) c h w")
            # video = self.vae.decode(latents).sample
            video = []
            for frame_idx in tqdm(range(latents.shape[0])):
                video.append(torch.from_numpy(self.vae_decoder(latents[frame_idx : frame_idx + 1])[0]))
            video = torch.cat(video)
            video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
            video = (video / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
            video = video.cpu().float().numpy()
            return video

        def __call__(
            self,
            ref_image,
            pose_images,
            width,
            height,
            video_length,
            num_inference_steps=30,
            guidance_scale=3.5,
            num_images_per_prompt=1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "tensor",
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            context_schedule="uniform",
            context_frames=24,
            context_stride=1,
            context_overlap=4,
            context_batch_size=1,
            interpolation_factor=1,
            **kwargs,
        ):
            do_classifier_free_guidance = guidance_scale > 1.0

            # Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps

            batch_size = 1

            # Prepare clip image embeds
            clip_image = self.clip_image_processor.preprocess(ref_image.resize((224, 224)), return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image)["image_embeds"]
            clip_image_embeds = torch.from_numpy(clip_image_embeds)
            encoder_hidden_states = clip_image_embeds.unsqueeze(1)
            uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

            if do_classifier_free_guidance:
                encoder_hidden_states = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states], dim=0)

            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                4,
                width,
                height,
                video_length,
                clip_image_embeds.dtype,
                torch.device("cpu"),
                generator,
            )

            # Prepare extra step kwargs.
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # Prepare ref image latents
            ref_image_tensor = self.ref_image_processor.preprocess(ref_image, height=height, width=width)  # (bs, c, width, height)
            ref_image_latents = self.vae_encoder(ref_image_tensor)[0]
            ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)
            ref_image_latents = torch.from_numpy(ref_image_latents)

            # Prepare a list of pose condition images
            pose_cond_tensor_list = []
            for pose_image in pose_images:
                pose_cond_tensor = self.cond_image_processor.preprocess(pose_image, height=height, width=width)
                pose_cond_tensor = pose_cond_tensor.unsqueeze(2)  # (bs, c, 1, h, w)
                pose_cond_tensor_list.append(pose_cond_tensor)
            pose_cond_tensor = torch.cat(pose_cond_tensor_list, dim=2)  # (bs, c, t, h, w)
            pose_fea = self.pose_guider(pose_cond_tensor)[0]
            pose_fea = torch.from_numpy(pose_fea)

            context_scheduler = get_context_scheduler(context_schedule)

            # denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    noise_pred = torch.zeros(
                        (
                            latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                            *latents.shape[1:],
                        ),
                        device=latents.device,
                        dtype=latents.dtype,
                    )
                    counter = torch.zeros(
                        (1, 1, latents.shape[2], 1, 1),
                        device=latents.device,
                        dtype=latents.dtype,
                    )

                    # 1. Forward reference image
                    if i == 0:
                        ref_features = self.reference_unet(
                            (
                                ref_image_latents.repeat((2 if do_classifier_free_guidance else 1), 1, 1, 1),
                                torch.zeros_like(t),
                                # t,
                                encoder_hidden_states,
                            )
                        ).values()

                    context_queue = list(
                        context_scheduler(
                            0,
                            num_inference_steps,
                            latents.shape[2],
                            context_frames,
                            context_stride,
                            0,
                        )
                    )
                    num_context_batches = math.ceil(len(context_queue) / context_batch_size)

                    context_queue = list(
                        context_scheduler(
                            0,
                            num_inference_steps,
                            latents.shape[2],
                            context_frames,
                            context_stride,
                            context_overlap,
                        )
                    )

                    num_context_batches = math.ceil(len(context_queue) / context_batch_size)
                    global_context = []
                    for i in range(num_context_batches):
                        global_context.append(context_queue[i * context_batch_size : (i + 1) * context_batch_size])

                    for context in global_context:
                        # 3.1 expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents[:, :, c] for c in context]).repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        b, c, f, h, w = latent_model_input.shape
                        latent_pose_input = torch.cat([pose_fea[:, :, c] for c in context]).repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)

                        pred = self.denoising_unet(
                            (
                                latent_model_input,
                                t,
                                encoder_hidden_states[:b],
                                latent_pose_input,
                                *ref_features,
                            )
                        )[0]

                        for j, c in enumerate(context):
                            noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                            counter[:, :, c] = counter[:, :, c] + 1

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

            if interpolation_factor > 0:
                latents = self.interpolate_latents(latents, interpolation_factor, latents.device)
            # Post-processing
            images = self.decode_latents(latents)  # (b, c, f, h, w)

            # Convert to tensor
            if output_type == "tensor":
                images = torch.from_numpy(images)

            return images

.. code:: ipython3

    pipe = OVPose2VideoPipeline()

.. code:: ipython3

    pose_images = read_frames("Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-2_kps.mp4")
    src_fps = get_fps("Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-2_kps.mp4")
    ref_image = Image.open("Moore-AnimateAnyone/configs/inference/ref_images/anyone-5.png").convert("RGB")
    pose_list = []
    for pose_image_pil in pose_images[:VIDEO_LENGTH]:
        pose_list.append(pose_image_pil)

.. code:: ipython3

    video = pipe(
        ref_image,
        pose_list,
        width=WIDTH,
        height=HEIGHT,
        video_length=VIDEO_LENGTH,
    )



.. parsed-literal::

      0%|          | 0/30 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/24 [00:00<?, ?it/s]


Video post-processing
---------------------



.. code:: ipython3

    new_h, new_w = video.shape[-2:]
    pose_transform = transforms.Compose([transforms.Resize((new_h, new_w)), transforms.ToTensor()])
    pose_tensor_list = []
    for pose_image_pil in pose_images[:VIDEO_LENGTH]:
        pose_tensor_list.append(pose_transform(pose_image_pil))

    ref_image_tensor = pose_transform(ref_image)  # (c, h, w)
    ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
    ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=VIDEO_LENGTH)
    pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
    pose_tensor = pose_tensor.transpose(0, 1)
    pose_tensor = pose_tensor.unsqueeze(0)
    video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)

    save_dir = Path("./output")
    save_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    out_path = save_dir / f"{date_str}T{time_str}.mp4"
    save_videos_grid(
        video,
        str(out_path),
        n_rows=3,
        fps=src_fps,
    )

.. code:: ipython3

    from IPython.display import Video

    Video(out_path, embed=True)




.. raw:: html

    <video controls  >
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABHeBtZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjMgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAb9ZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0VVp21seljiRkJ1VoCQyIRmzl696tv/0E0dR5FQ/wwOIzrhedZbedKmoLs/V1ojl6WMQy+8EBnX6Zs6o9K5GGPOOo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" type="video/mp4">
     Your browser does not support the video tag.
     </video>



Interactive inference
---------------------



.. code:: ipython3

    import gradio as gr


    def generate(
        img,
        pose_vid,
        seed,
        guidance_scale,
        num_inference_steps,
        _=gr.Progress(track_tqdm=True),
    ):
        generator = torch.Generator().manual_seed(seed)
        pose_list = read_frames(pose_vid)[:VIDEO_LENGTH]
        video = pipe(
            img,
            pose_list,
            width=WIDTH,
            height=HEIGHT,
            video_length=VIDEO_LENGTH,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        new_h, new_w = video.shape[-2:]
        pose_transform = transforms.Compose([transforms.Resize((new_h, new_w)), transforms.ToTensor()])
        pose_tensor_list = []
        for pose_image_pil in pose_list:
            pose_tensor_list.append(pose_transform(pose_image_pil))

        ref_image_tensor = pose_transform(img)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=VIDEO_LENGTH)
        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0)
        video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)

        save_dir = Path("./output/gradio")
        save_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M")
        out_path = save_dir / f"{date_str}T{time_str}.mp4"
        save_videos_grid(
            video,
            str(out_path),
            n_rows=3,
            fps=12,
        )
        return out_path

.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/animate-anyone/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)

    from gradio_helper import make_demo

    demo = make_demo(fn=generate)

    try:
        demo.queue().launch(debug=True)
    except Exception:
        demo.queue().launch(debug=True, share=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/"
