Image-to-Video synthesis with AnimateAnyone and OpenVINO
========================================================

|image0|

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

   This tutorial requires at least **96 GB** of RAM for model conversion and **40 GB** for inference. Changing the values of ``HEIGHT``, ``WIDTH`` and ``VIDEO_LENGTH`` variables will change the memory consumption but will also affect accuracy.


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

.. |image0| image:: https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/animate-anyone/animate-anyone.gif



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
    %load_ext skip_kernel_extension


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


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
    import gradio as gr
    import ipywidgets as widgets
    import numpy as np

    from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
    from src.utils.util import get_fps, read_frames
    from src.utils.util import save_videos_grid
    from src.pipelines.context import get_context_scheduler


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-744/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-744/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-744/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
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

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino


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
            repo_id="runwayml/stable-diffusion-v1-5",
            subfolder=PurePosixPath(saved_path.parent.name),
            filename=PurePosixPath(saved_path.name),
            local_dir=local_dir,
        )



.. parsed-literal::

    diffusion_pytorch_model.bin:   0%|          | 0.00/3.44G [00:00<?, ?B/s]


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



.. parsed-literal::

    image_encoder/config.json:   0%|          | 0.00/703 [00:00<?, ?B/s]



.. parsed-literal::

    pytorch_model.bin:   0%|          | 0.00/1.22G [00:00<?, ?B/s]


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

    diffusion_pytorch_model.safetensors:   0%|          | 0.00/335M [00:00<?, ?B/s]



.. parsed-literal::

    config.json:   0%|          | 0.00/547 [00:00<?, ?B/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.46k [00:00<?, ?B/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/6.84k [00:00<?, ?B/s]



.. parsed-literal::

    diffusion_pytorch_model.bin:   0%|          | 0.00/335M [00:00<?, ?B/s]



.. parsed-literal::

    Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/154 [00:00<?, ?B/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.52k [00:00<?, ?B/s]



.. parsed-literal::

    motion_module.pth:   0%|          | 0.00/1.82G [00:00<?, ?B/s]



.. parsed-literal::

    denoising_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



.. parsed-literal::

    reference_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



.. parsed-literal::

    pose_guider.pth:   0%|          | 0.00/4.35M [00:00<?, ?B/s]


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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-744/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/modeling_utils.py:109: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      return torch.load(checkpoint_file, map_location="cpu")
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


.. parsed-literal::

    <string>:2: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
    <string>:6: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
    <string>:9: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.


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

|image01|

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

.. |image01| image:: https://humanaigc.github.io/animate-anyone/static/images/f2_img.png

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

    WARNING:nncf:NNCF provides best results with torch==2.3.*, while current torch version is 2.4.0+cpu. If you encounter issues, consider switching to torch==2.3.*
    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (32 / 32)              │ 100% (32 / 32)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



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
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (40 / 40)              │ 100% (40 / 40)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



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
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (270 / 270)            │ 100% (270 / 270)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



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
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (534 / 534)            │ 100% (534 / 534)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



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
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (8 / 8)                │ 100% (8 / 8)                           │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-744/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4689: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (146 / 146)            │ 100% (146 / 146)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



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

    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="AUTO",
        description="Device:",
        disabled=False,
    )

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



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
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABGvhtZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAayZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0VVp21ej/NR28OKH6BV7NyBEFaKR1C/e7A7luhCw782Sb/jVUNhYbW5AGR2lzufoqnUmCyMq3xdzwfof3Dqj2wEP6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" type="video/mp4">
     Your browser does not support the video tag.
     </video>



Interactive inference
---------------------



.. code:: ipython3

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


    demo = gr.Interface(
        generate,
        [
            gr.Image(label="Reference Image", type="pil"),
            gr.Video(label="Pose video"),
            gr.Slider(
                label="Seed",
                value=42,
                minimum=np.iinfo(np.int32).min,
                maximum=np.iinfo(np.int32).max,
            ),
            gr.Slider(label="Guidance scale", value=3.5, minimum=1.1, maximum=10),
            gr.Slider(label="Number of inference steps", value=30, minimum=15, maximum=100),
        ],
        "video",
        examples=[
            [
                "Moore-AnimateAnyone/configs/inference/ref_images/anyone-2.png",
                "Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-2_kps.mp4",
            ],
            [
                "Moore-AnimateAnyone/configs/inference/ref_images/anyone-10.png",
                "Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-1_kps.mp4",
            ],
            [
                "Moore-AnimateAnyone/configs/inference/ref_images/anyone-11.png",
                "Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-1_kps.mp4",
            ],
            [
                "Moore-AnimateAnyone/configs/inference/ref_images/anyone-3.png",
                "Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-2_kps.mp4",
            ],
            [
                "Moore-AnimateAnyone/configs/inference/ref_images/anyone-5.png",
                "Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-2_kps.mp4",
            ],
        ],
        allow_flagging="never",
    )
    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(debug=False, share=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/"


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860

    To create a public link, set `share=True` in `launch()`.







