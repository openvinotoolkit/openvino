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
    import ipywidgets as widgets

    from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
    from src.utils.util import get_fps, read_frames
    from src.utils.util import save_videos_grid
    from src.pipelines.context import get_context_scheduler


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-761/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-761/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-761/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
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

    diffusion_pytorch_model.bin:   0%|          | 0.00/335M [00:00<?, ?B/s]



.. parsed-literal::

    diffusion_pytorch_model.safetensors:   0%|          | 0.00/335M [00:00<?, ?B/s]



.. parsed-literal::

    config.json:   0%|          | 0.00/547 [00:00<?, ?B/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/6.84k [00:00<?, ?B/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.46k [00:00<?, ?B/s]



.. parsed-literal::

    Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.52k [00:00<?, ?B/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/154 [00:00<?, ?B/s]



.. parsed-literal::

    denoising_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



.. parsed-literal::

    reference_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



.. parsed-literal::

    motion_module.pth:   0%|          | 0.00/1.82G [00:00<?, ?B/s]



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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-761/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/modeling_utils.py:109: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-761/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4713: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
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
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABEm5tZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAbFZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0VBB0NXdLHE42Wr1CRNO3FU7UrNJPyWOmpYnpLAMHTVcI1cgtNzbK1AZUA0QlCE2/57PqRz24S3qgW0NjGXVjYtUF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" type="video/mp4">
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
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(debug=False, share=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/"


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860

    To create a public link, set `share=True` in `launch()`.







