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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
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

    README.md:   0%|          | 0.00/6.84k [00:00<?, ?B/s]



.. parsed-literal::

    diffusion_pytorch_model.bin:   0%|          | 0.00/335M [00:00<?, ?B/s]



.. parsed-literal::

    config.json:   0%|          | 0.00/547 [00:00<?, ?B/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.46k [00:00<?, ?B/s]



.. parsed-literal::

    Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/154 [00:00<?, ?B/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.52k [00:00<?, ?B/s]



.. parsed-literal::

    denoising_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



.. parsed-literal::

    reference_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



.. parsed-literal::

    pose_guider.pth:   0%|          | 0.00/4.35M [00:00<?, ?B/s]



.. parsed-literal::

    motion_module.pth:   0%|          | 0.00/1.82G [00:00<?, ?B/s]


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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4565: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
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
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABH35tZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAZ8ZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0VCiPSMWb9ykih9Gv8pzWmqZvhv+T+BX5QzzOhsydxi/zcnE4U31C8X/1MlNWEyWbGJiEisSns6DuoZYKGnbgfq5xjL9rZ/4flkzN0unYubVSm5itS3wPFpQ5sZzts7Xew9mDuqovtLxqlnZuujQOkBoasoxSuwJtPNyMEWzBrveRJSoRGZhmNk43KrgI53U9Rg6bLqOOFvGlun+jjZp0yeVB37Dv0dbqakTiAAA8HHHBQfMBjMhgaOz/vIhm8wT9wYUvSTdBTmQx6e+UUHBbcrzvYXza7wzfz8iZNeUlsfbWH6XD38eMFQ1VvK1xXXqxK54iL9F73UznYU6a6YdrIpBlKNcSSjKIwR07vCUAhKFnB7yQO9enlIEBn/NfdMioEZ3on0fSnlL/73j7iN/4oreP8t/M9PM+HCv5UU+pfUvSMLb0GnIuvwlglCy9BxmtxUEzJu4FDk7qAn82Dw8ymwNxvLN63AUH+YPn+N/1EAfs3kMtvAVdbXpV2hmJnJwSQ8wPfQ3Bicj9DT7k0KYeNvYtxyKNZ4cDIMvBMLQlnWBpTta/aF5rFEtpeoWLiAk8UHtBVk4KdT7ryzSRxVC01GKNGm0VmcfmPnotM0s4GpMBT/9JlxNa+qtpSGJF08eypLeidiSh/psznT2ir1az5uEazCfXHJf7bjCKY33UtT99faH1f6YaAoEqZcd9HBpZfAND9Npusji++itt6Zza3lQC9uf5XnHnLz7bfNFTEuKE/hxfHPHZ7TRHQkuW90pS64NU2Oh3qa1XfvGp88ttlcPQoUmkFkNKtyNUwuylCRhQEtgeCfB4j3f1NoSjOawELMlg7ToIjaErrJN+D4et+w+7ZA5ue0NHPSyS1cVypJDz5KL3hyGT7hXjq4N/H633azP2f/jTP3ozZj4wF8fIJm+qwycNqAADsSI/Q9MN66FQ3hLRAYJs5uYKDsldy5xi7zjIqCL02Fw6Tsjv9Hhar8ORxlR8Fa+UUhBlGhiTHlRv1ZtK6jf9nrCBVkshzavJHmkBOiSggshMnMPBO7x5wOAjBIqGVigstAfgGcaMYoY8OD2PY/1yelOZs55cJMqBMd5cgpnCehCK5JRp0Jqn76eXa4mvtXNWKUnVDtNC0BZS65Je7RYgzgGTKgY6lyhgL71Z5fpSOC/pebcTCklnA9xjUsONVTMS0bB9wCx2pXZwEdxSLNV1QW9c8GSs+E2CB7sjkVwKyFBI5ekV4Q4hYLFH1m6edtjWEAFtwkuKnFaAGw38IgcCs3+O9kunt0HvhagCG/cC+If8vAMT4Y6edaYOkkswvDWbP6Ojjn8fIEnpj57ulUbAytEyJxtXtCXUvgx3e+knzT3IpDe/g4pYxWHHwiz+Ngq75BNLGmjPT4ANtWT/29IjJ1rgVUwe3QywCBPi/C9YukB+hTKN9trCdgczFvSXEF4eWLCJqmvVRplD9EmecdEeyZLtIhNedufg75S2vBOwOfNSCHytnbAKc0/h8LbdJVpPa+IZbOpy5Y3sck5De/eXN+4bMS94hzE1/ghqsoYTdLiQKR3s3HSSyq7LYjDGBXlvzsz9DSdGQnK6KucytsN01yDLsIHZ61oqNUxpLUvmhR11MAGEJle45CWRD/yIRts58CgsG3s8Nrg/HfIpU5pDNdPM9wcOcoolwL2FcoAoWmwPPwFhlP5zvikA6F7TWMzsTF6FGX5HprWhioL92wr2TEUa9MvCD4Q3sclCVNwKcYY1r7XDK/y+EKraG31te2DW3F96zBIkNDGVjOvxEWKTzTafobDb59GLaDQGsPsfdoZfvcIROXGtTIcelW8DKtPNdW9RS/aM7CVFFtuUrAoRF7aaQnYAPlAPCwTVugHu2r8xydlMo/CK+iNbJBs4yaTHK8JnyKRJSf2QuGN3mtoAAAqYE2ggBeFBO600M93e1vOzAC4eyOffhcbFFQYN2VGsn2DclasC1XhgjW5KJf7Egk85eZbTFCOy0OiDZm8IyWohylG/wxIwHfKYp5YNGp3VwfoRLUf+kpINeOja20gM/Qh9iqXNBBQY4eFB0tp64xiXgYHZzLx3VtJgqtqDimYj5q8YjKK3ibCp+zVb1MIBp6uZqQAACcdlAKqIhABz/9Fp4k9H3/6aimTa6gCYgRNgQOu1jtjCBlJzGOuQFvaKYkAE9POxpUCELBQRIzOwvTqnj7NbIM1VAHLLSGn6PMJbn9bErXANsIBocKXLdqrrthI8Cc7U90kxFZWELggOrECx8tFUfSM1206CRyJ7a9vYLkQP7qi2RmjejDtMCtIGo7U8wl3zZguHwKWcZzXsArcyP8JxcTf8OstQ971DTH/Vlt17/iV/suAYn4ksERTcCEsEYc5fjgF5fw0yLcvOgPHvG25TP4ID8vrM8PqZepjJmMJi70UlC4AIOWmvp9lDXnr1X/YI09/gh5tQWQIlcpZrh0GRI9eZ4xa3zHnIcZODqVQpOGOad3svgm2WLX5EyuENXavhgokvfTXQ7KDAN8efPw9GDcUQKAnMQFe1wVx/mfzLftwT8Xva1RD6WtqQSkjTE0WPEYY13Mwcor4yvKfz50z2nsgi0f1GzIDKJEBdTeHNmKGB1/TRJwCB18GpuBRnLPhYTFRGa4W3r6v/xcCn1AIdkTDgCatwbdAgxay/xh6l7iSN3jP9gUdPKFkVcavrCyGFovUWvhD2VT4qyw0kwtp8ctsx7vymrzjo4zCX2FLYuM5mKTxhjsLfN16NBKMFQ/3PctAFQTs8g/dYAB/si4yIU/zy0eF4Rzs38nsgbTIuLKc47zMR7Gk2ZaIx3PKFqz9l5MzFeq42BAnaMJA/TXgfGIsy71PNAd0TLxanK6S7lc5umrmlV03GM+gpN6IWAAC0nwOJMP9hUL2Mm8rzA4U0r7Z2o0CEkwKqZOl/AKzgTvHA6L3F+xAZ3OppOLjkKxXbzJ4FpK8NFDwrq2q+0GJnS8HSjznajYPf+/rzlz4EQ8CuGqenD5XqRw2gctZuTGCc6UCudIqNZHIzU02ZGMN183GsZ44gQhzq1diCmf+kGcAr0nGPG1tGrXjvQvuPX02vudfuEAEMmmqvv28kBFxA3tzU97ZMFk0pn+0CzVxYwFFYnhRZIFlHUACGtjbpHYMo9C/WJaRd/e6+/CNUcMg+2URIxO0ZJlzGfQGa/HB9vNW0MWVdki0I2qOf17hEicN82kFEQ8iCdr3cDIoPQpNv9wYlTqUQtDmvPnHAQvY5783IfDcORyiDtN03vDqp+7ApVVhgsNHuyoalsrQ1XJ5CTU7U1BXhrL9JNWu7mvMZ1U6xw88mL7LFWLcMnRxJtjeh7mEwvk4T9JWU+p436kwXAIeTyNuHZiglpF1gD3UnRQwdkU8a5zeVzdELriyI304q8WwQnZ4iXV97+uhIE5ffkXROcy1T+MzswaYZJBLIcX2kDCB7ctagxGCxJn9FUPA/kUEZfWjtwzrT1IPjXQoJpUXb1QGiQ3O/EviWXliAnaWWrfpdHtbpcM++McUnT8LTgkgTSCHCMb7wQDQm3Um7cr2Yco6cFCpgiSRWmjc658MhKhT4LuhESn/QIGun49eDkeRRViW9PTPvlhIFyNBJJCRCwSj8ESC52O72GPQ2Qk1RmWR1zaob83ltogu/yT5qIAgGpuYjRoboqkxAshvd9f2Cvb5ZlV2hg16hhBNVMEnrZsfScss56O40QSkGrRlkI5Zw1KnT6HZGatXDRZet+l1XdZrLVJLJddjf0aPfI6oLD/f/CKHgbaOFiepkLa5gThc+GHSIQWAZhfVqe9Q7DFZixHSrl4LjuxW3hy+1ARUhcHfjbvRSodAtKuya+Gbb/OQSQZIb/F01WLEQV1QcqXvi8P48tEjXAg4xDUlk/Vlb1+YVH5Rq0TSuWS2VrEXmlOtXpm5mIMe2AVFI/b/URUycjuR0STjOQorwIBgxNZrccTQElLIuervLdhpi7ETmlp7lakl3QKEeBSqbTpafwWBU2hUtS7VTJF3Hh5ZppFe7Ac2CYdd/GAE+9c7BNfpNGmCNnDx4pd5xioMJB3SfRmu4d8KORVr5MdDfLetaU9fMjJ+E9sECogNLfXpw+Yx36jv+4dkwcnFkzfU5gyOsgeSxMnBd5SX8l0mzDLjBu7IUhBHg8L59fgbdAKhd0AGDee/R3PfO6kKTNbaRjPJ7CKBoMIwk3ECP6Rp633i3P/GAiucOFg4YNcEVxKc6LkVUqYkn2d7ONzhmR2/oGQE39kYar9NUZomDFDx07MRWigU+DeZXBBuD6J9iIoOf1RJ0y2W6EQvBR+RNkXWLjmzmWNTevwDk50s/ULcwEY6E2gvsZFGaT2EEncCiC8IgdmvnsgDOSrBbqpbqVA2RQBYP6XI8fQhFJMGJLz4bMdvLUEtx+wkjN70/XzEYGcOFvKjCOIppokNka7gKWdkJRQinukR0TAO5YTPh+iN1OmCQKwQtAk2nPZlkuhbbPtLahuUTuOWzXdWjMOKZvGXVePJcoZ8id1uNkbcsyQvRTa189QhYLE0L1iKpXvqnkUF7GWl37Uu6Cp0xL+9zoasiqQ9qfCg3u+3j/btESZFAMgLhI05e1d0zCFI/GSwz+gxpy4KyDJXTf7BfiPbb08U9D0bUF0bk7bXtYeiUdcpgMvN00JHm2Xr0O3JOVKqdb2LjM9QGvumkemdLoppes3TdHHIQ0Q2zBVigwjqQZwBVqVhsiRukDmYo21Rjrn2j2M7H1HaxCc3MY5WmqiPnnUiENC9ra2duPlITPOQBeUeeKs5QqiZdcX2Sh1OnY/GPt+U/m+Gbo7KO/Kqr3mfs1zTG6MxG5C67M7YwK/ogt5rfW8Y52/227v+POZB6e1kTVackCdSf95W1W/2A3Bll3xIQxkdvVFW39ZN0ZmBdDfJgGBbHQODDArlxDyb8XF2R84VIpzpbzOMP3V+aLU22i4O/yAAcxO77mbumRtmNfRyJTI3mk56xKQxy3FrhxXv6GQMoQkMseDucq4CLKhquBjGrunfl1VqoL4x9LX4dPBI8svKRXWKQgAPoj8eDcxGBRSjboRl//mdNemyHGERT1wMPnunV2k6KQhdXeiAB3aBATCHCewZ512v9lmytl7Wg+7pfaO15jR9/SAmOdGOAh+nXksij/Qbd6XO6X7P9ZBdEaSj/q8ue7VqM6QdEOriwoQcdYNaD+28zbxAG1Fzp3OhrAsfjhOXRMBJeGAJrvVnXuGG0FSHz35imeTaj4D5a7/euh+La5KGyJX859S5QcYS4/zUARjy/lYcZ7MTp20MI16KkdfKXsSs/B4vnk/4JdMwc0WtpddKB0j1yAWt997vUaeKKH2kvqQ8+KtL97JPs+vBOkDsNInH40NzXp9J8ItZYKhY6zIsqFxngEqr2OkDHd7RBPkHlwhT6lk1BeykzKL2bCivbC6B4zFsSA69sfm+o0MchLGEErivXW+yZAAAK4mUAVSIhAAo/1popaxQvxlenRCgAnKuD5slVY+5FozMZKuLmQ/HScn5WS0cp3pt7p2R5XC5eWTjBDhhj1KxMFJezAhdjh0kqxOuVdN4jLYK075znd9XBsGphblnxWHf0XgZniIlrEdiUg1JcQCSiXDdGSVG9QMq4MeixRutenTwQs0BxcMc7UptRef97ZLQeToIutvD+iGteKbymu2i6UvqoUDg2rnkwdfvYT3vSpBwegf7uHYfTEfBLm0M1inwES9YiFwxfXUEdj3/1OZ2kQkm1e2IxJKctnSNbNAfiHnb8fSoi/VhLeBI/LO0Jp7jUcV0pFjegMfRkJyaJE4hDTVgZ3iHMbiISf5atvycw6xp8EDmAKmKxQWfp7w12qTvhZuL0CDf9CHv+6qr5zpLsK262u89wJrpM/qWqJbzyh5KmfOf6iasPThf95dpJxz+q9gu9+4KWXQXOsbzCO53AO834LGkrPvWzDGSgAXpwdXH/LJ4wvuj3e2Y4vh8441ylm0gCUIyaQHpM4djM9CUDP+FjUOmwSka7DONUyNc/taOv2iLnM2ajLpR/wm+p2xNju16rY8DUo5wYw8m5lJxQOaDsBl1vPH7uv8WacWwmyk8ttIp0PUmmFJ6MFpsWY3YOcVoMBWvzxDLZTM4JWcsm6QoSK4XJAhMv9rtIJq9TGDuIr3d3icYox08Onx2Px9e7JEWCjTCtHNfkRqo9lmLtLrR8GiEH1iCCTfAArB7ZOBT3Wi5TmFLq5khDALBmW/DEpoP9Wpe1bv/iCFlnGmttciB5idon0CrjW9r75XDIdXAQoj22Lo6VqmkG22+gxdYyT+OFzXvwLdEGGra23myCOHHd5d3AF918YgPTe9N5YIvVHVN3LiX1ToADQ8pm96S86Sr+Z7vhMQubalywut9MzXdLgyZCMgy1Sq8Ma4qfQNYeoQA5FyvEoUUXj99EpRAeCLj3D1zhi/AXaKWnGkeATAUKFx9GOaK6c+0nYnn/96fYeiszdq06L8ODVkqWhYIPgdf2YXDEBHh6kZtUfvrdm/70F0xJ4XBr/anmyQqivy6/Wkq4AJluGyQE97g4pXcT1MNafbXMYkdo5vuAzdo5Lp3hg8q8yla8EQElFPoY/La58fpDQHlGIeqovZXVugSt9BsZFSiNQ42qAzcy6M629QoYUhPRlKg0DljGSYNw/WIUFcYUCMVMaKmm4wVSWD4k9Hn57gZNgbXZXQBZHvCmkLEfbYdns9UdTTP7cohoiRiWATnQCkPL+inxZ2b8aXH8gnVRI09TJAxi9E0RYlhsoRHuWlT6AsBza0EDXixA9gXmFfzeEDw5Rvg0nImK7TOzkxyUGZPrkL3Ye2BixRwDDMxejMJHKt2TaMxUEi97FDvnkwRfdeMSq+x+KgwVSNEE+XeRR9QU1o4Ndw3ROLYARUQuOqJRa1K33ex6ZEqAA9S7YxUEcqdXO653IZTGEMm9e9Fd8lDLWwFjuAR6Vrg6Br1tFA/xD7HLL79LO/Ol7J3l9M5rSj+e2n6bQIHob7a+wcEwU0ERaVujJeyju/LVp2RqXzPsHmt5TXUomBGXRsSdGNzVtBuqaXjBkB1M8QDVdz5/qtT4fV7CBU0ZmnFPiNPo3cWp0y1Iz+cUs1UDjaYq73vEhzgflal2mSNnVyQltNLjWHWJIkVpOA0xDneDF2KqMRaBpYJzgocZPd4qvfePMI8HU0dQUny4jwHzQ4BXkRxuEGNJHWPsJgmc19EHRqSuK4J08nO1/JoV7KgFAr+5dOK+MYKBoY6+zvCcBlCRcZkVMY/GRMtbpmmPYwfRnKhClpAUU4nu4Fb0YTxIC4bpRhGiiUz04AXO96pxAH+lqcyFh9uR8qpRgOO4aVhoYpRyouZmFtM3MRntU8q01VFnODjhxQ9/bO9QxQuI6EkxfAp6e7C+zArtjAzzgVC3gJ5DUPxCpKW0xScWuP2grLpCX2VN8On+2EEOULnqTlSEvfYuRCbrpzJxjqW7PxbspZFM/YGd6DC03aCfWYJDEfkpmcnwvXP2a6hABujKSj0Az6EVXgjdGTvVMSq+visr3JTbQDtrepHZ2BR7mf1cUP3cm+hDJYd2lOQeufQA/OWhdo6LFtwkadsKBsL5Wazvdo+gsj4rhpPU6SYZCeV7JFOYIQvTbpgaz0i8DyR+2C+MkUZrbkJs1zRHbNRbInDt0/426x9g4m21WL+S5d9Gl3PeaVSbkpFjn9qWq/P1Elnz01rB0TktRUUvbJSIiKjPrMbXDpYDnRs0QKfchMFRAEYnJRjlznrvP7m5+rNtmgbWLNeIhn2I8wswee2pqipBPMtML7Z2v3nJ6BsO0NTZvDyuBqJMtGn67VU41vi4bHhen/B7hjoplYezJpqIorJhy2evkEzvyQKCkgXB31YBfuLGEOR674YIV5wBr4PsUDuCiJxx03q+AC9xvTOl3JbE8npEe7KlizdMTbgqxtdq57ufVAX7ZQEQm8AXWHMjz1uTzyTCEM90m8i76+z7SIQxgnjTk/6z3RMq0zCGqnHqiccnXxbMt30RJmmxJtlq08M17NmmjLtD58/L7xP/0l7UI6q7PI4aPjW3h0ZKAYFN2S5qbuFlbxtYaJQK93Xbp0aEo2TeBUMmrnTLOwjJjYGj+SbBWMHBDLF1qTK3UiwZaRpSomzeE2I8uvjSOnhJNYFSeBMpSxah2x/A0JlSQBfMrw+NCvtYx2Zy7cS0upLShjckLdD+FItQD3AENEljZhw9DGxUyo5fdElhFj0lkTO2Co9Oqd2ldQoCLy+dL6gIGMKojgnHtg9RW3NS63YwyYnAwVvkyfiqxtIanyouEi2u1QzohfNKvJmXeJnjVdppTsZdgjol9njlsiDgjwz06+fCIpTxvWCgerBT+0ST0my+65Ut8S/CF/SV1KG/C1so067GU6fHRqcJKtY/cNI7t3SMvGOYwdPhbdXnZh8+fhkyKi3NwZQ64sjSW8mArL8KqXGVJwFl886eJhtzMTTZxAUqoTmod3PzbyIQLQWHbzqfth7bsNKm/C4HdWzTT1z7aY312s+gu/n6QFwAfZFzb4aVQdbmVdhOG5oW9+8X14COh650VHP29JQBwbC9SPX0C8D7Pken8fAb7POsnO1lft+usgDOamhgB2H8PRQfWax3Hg4FytXaCVlnXZLie0/wlVTlJhWYpNzX8IAvo9+s451WFPJS7Z0wx0SKPWxYSQqgDHQ87H3ALIxw3a1WXoI1uBUEvxBLnN+75KVqRBM3e60Lbbo6AxlymYUR6TfSkdIF2O9eqoWJ/DZaDbdG/QseAMVmZVU0rnIs/WIxzn7VZK+IDSOTf2mwdX5fy1PyCBMWP3Nz6u4kVGJKCrpun0+fS1JoNg5QFol0bNq0I2OzYITYTKIOXo/wrj04itED7Y3sz8eozQ9EFhWr3s/URorSQjjLwsyKczp8Ayyd1SbF8G+m/+Y0tKI4SHNR8Ux3nIagfWxjblB8agvuNqIQWeeph3zZEmRzamrCu9N0QCXMTVeJvLerJBBHVkM/I8xIihVrByrJHA4Shn5BOMNoaPCdazU/ZiQVlIkgNn7aQY7t5M1GzAqf/ulcHhVkHv0WZ1QFi90ybqBf70oEacsalfvfiOGi5vca1zD/LDewkZGwQm9n60dtbnIa88GSnbTpurk/JSBQyR8z8Yn9TJ57cO0tXECLnLyiqaVwu2dOK68lM+PF1q7xAAAJn2UAf6IhAAo/1popaxQvxleamBwAIY6rCXuvmiRB9D4PSSaKGGffl25/TxgLKjF8YebO4tzxtQOs+0x1PXTnfLnfPXgG86qUCmZOq+H/ujA9++reGjQF60g+0Mnt8Ysv6CX4ppuSEecG3rj4O7BuDVVjQpoxMOjWYL3X8Tb3Kgchl/4jAxs3fLhb4iR3qgv5SxBmdU4rbfHIC0tVrf60yIYlPSoec1FLOmSEXdyWNNQwyApj8K/AXfaCMmbLIm3YH8n1fCcS6FExd8m934A6HPXJBOR8X1MHt2CHC7KHmnF9pn4nqbWO8BjIJSjN74K4WwYU+Kcae0mq3IPShhNQs9R2wNkUMG1isBFDCi6nuAAkgI7YeTzeaVtMJg9OLFr2RDQ7zYUqKxa7+cV0cjCxX9kKIAR8QM2EQ+4xNqEH80mZ4mWitFeLrtm0ant2yI/Xzw8Y14XhBqsv81XCbml+Nv8A/kq7azhznnc7X66BuggZG7wVDlUkeF8ZVZSCEs3wSDqhcOvcWRi+MJGyvn4WQVkgUR+u+q4tdMGfEPkMqks/P8rV8rz/QEwBmOoMa0b6tkPyRQQ96YwPWKGUfwqGVZ5tBl5pkD64Xhmt0tnx5LZ2ZJjUnZp5oXe0Y0AgCx/y0ho5drrLxhA1EfVem7i7LSMf9gIAV9PT5BGhgAEYwRq6t9DABorBtxncpFGZ+OCjAIU757WKApLbgWTS6jnGczGDG0J4c+DNqr2sgtyYg/5+a3Eli+qfizB2jUQekIpX1Y4FMVx3aDSEkYVYrVGQG8UeG6OBB+/sjCsGD02qrrkTW51j2YaQ6LwLgVostvaameu88hGaefpl3aMFMyWpNDEE/bvhmwteCj3AwNizkOYXCw/JLBdNMn9XPMV6s1W+IUiSw47oXsb1/ejAqUhpVIgyGrI45roWLPAy2Nh5+8K4zzUsQyaU7dbqyCiTz1eel/GzjDijT/2aTMgIU3vsNzhQINeSXaUHK3kU2DvrHPphRFudqAWaMGeU7tbM4ZdJBiVUQideGG+zHw3nyjOrHWnK06j8rXUJDKdyHQJVyvfokBnEUHDRH99M6Wwiaa1mCW+tzdZFIIKy+GekgqS1GK47BrNM2f+zeRK0bmBYbbwDjAO6DEd7/orP1Irkcv1AZZx3HuDsVeIFIrtgR1uQwEyNNmSIeA7gx9/dnOvbcYzJA7X9TPTAlbzb3ngqILaYmHT1iFgVqgqL/NyaJO+QNyyD7DqFBBTUjtdXDAudifTSpetZJJdCO9Wotmk13VyEapLmtPit18gBvJP2QVD1pxivIaQNCxGYEjijbZcfjfU7jHtISMj3nxuqevX73eIv8gsp4MsnzMjN6X0Pxw2B/J2jlo+L2Tjb93GYHrR//ZHoiLAf0YIAcaw0JYIQNmyXrkApyr7Yw3lB1i2xVJqTVVUHfJ8AjS1xeVWpL87UeJzaE7XxM4zuuTmcU1ekUCdiOvkNARTXdOefqA0b0TqDVPTVp/QyeuzdZnMuj02T21hINA/24m6xe1LXCzeD33JmwDclXKtBU8nHbzoWoIqT4LTakXbRKKy+wo8cP93mSza+uHRZJegj8HszAbIA/bF30xFLyVk6wC635p7vIEfOWqfT5mKVVXV4eZZHexutVwuor5owRWlwrx1GUeiVQHhEaQco8LU2Sd/2MGiPwTtpuUP5wUMAIP+Q/kA8gKoN2mf4Y4qDzYIp0sN/rZXjmRuKuGHzRePzRZSYZJrBOglwChSNzyX26/XrTrUwPo/4YvJj/cHYMkAQ7sI/2L4sGKcvFNvvJzMjxl+ehTQ0G8eYNJoOQgnMxQx5ReehY4kHofMMqLWIiWCdGiljTIaEMVZUyAhNTT68jrjJJnoBKJOX16lGVVHvNtkQDGYXa2BJj+nh0OS5mIry+d8iKqpjAk1+B5WkXuWTRqlsgH8JdWRG1RY2+93gC2v1RcWUkenIjOy0KDy+JpeH6XC3bqS7ywsMmON1tds96mNaW9F3jZUdUozGyLVEBGsti3kMFyoZCnOXQ/ZtQwcjYq6tXOfTaok+g3avok4KMpdH6DSWwzUNtB/avYuF4ltjqKSZ/JMv805IR4YsVNa9UtX6a+4HQ830SxBQP4u1VNVI6xnoTLvGaoXqwwCw9M9wtbyaODvvcfrg+rK1fKLuzHbiTIBvvVtcbzWJds34hNGJtMJa0dxzT7jjI5ugpzkq3XjWHxw0oDRsTJoBwmCwmzUWijYlwwkHfb8YDYLBt6+nCcKBoYIc565AQ9ylSyxPWI1yWOBIAMf3ag9TW28NVWDQdl1qemAptFvRjypA/Q/Z2l6U28bTyK7r5zOuGPlblw+UuLWQ7cmL2S7NThMDm5OQPCyQbEF07OQW/bad9nR16hdNs5qgEmKA3vPAE9pU5oYxB6dLnqNejo//GvvR01iAMXDM4+VndEAuSTE0XrngTsNvKPB4jzX2gFwkxv2g+whnTOSFrY9JWo9NGgaNa5hHQ4jOiIjY79De6gcDygPBYmsQDJ/ow9wwaj/PU34BASvf/nGFFq5VsHv9MBwwvnsE4YETO+KXEBltG0nFgULdFuRxCE73mrvpog7WQ2VG1BipgIqRTiPH/hqZeaBds/Og8lbQSGNlVZpEO5zT2/xUKLILMJ5lM9O4ycpyywBfbID+R4T36C8+0guiSmGwl+2AXYnbgshT3++6Mn3TNpE4mxrkNPJxeq1rTggKpTVP9aCoxXhGld4pkSCaLI+4aDPbEn4yPQRSLtL0GMEKXUDB8mkdYpBVZRcpi+l/9Gl/0jBXlAkHGt1oZr6iYKvxlOrvuSqasOWYGvcZXAg+fNBA+BjsRb2nmVfYdlLaUimju4qygcT5PrQ0+HvG8dIk9WLReeaLA14Mw9hAWJw9HZIyn6QENtrXmtUy6eOtwdJ6ihRogeaE81DzeOfbXSephXWZWaV433lG8GGBEAtyTaoqlywRgoyMaqrGIKZSyK5zknDf+dkHzH0LzoCsR9zPAB8Ik2pDJ4JogWwxlQFfup3Um3aKm5EBvhgCck9Jp6nurNXK6wSQqZa6wYMXJOW+JW36TNKB6S/PxUwQ2CvqVB/BUZEpPazeUtrIkHKqi/xlVmK7z9TNQz/PcrVz6syeQsKA/zJenwZVIItoejy+iuMEH7RTWtRq5n++eVjtus26CsKBDHnrW5rWAgGYTZTvzObXeOZWYduCtFacepF+AmYMs3/uzK1Pf2PVG7ZqMuU+kI8Jr1xSwgObIIV4DLiYTtBIGUjcTnpUTtnWag3Yaa5JW9QsulLA8QAABzhlAC0wiEACj9aaKWsUL8ZXlkOs4Av/kcA5e8WCjH7t/cficetssv2ixu5xCm6RYRknMhHGH4tNOo27GBUwmNpo026sNScN3GGsSoeapmeEWZ67IKz7Y0g50glMgx1BkSli5rX5oExX46GFxrUwoPeF2UEugXJIfCYd8er1gOH4YqMZE21iz8tI959BsgPZRbhpUzDJC/BDssbdygSFNEFTfhbg2OkWJ0e+jS/89nO8skcLYBA/asdy1YKvM4aFgNZu8woqzf2XqtsUgnWNbD603qG2Az9ZXH6QhRsLpj9XpeF4psAEBG4BIquVbMebD6Egu/bwHOkQba2vWjTq4yVNE2UzGvyW1XILGh6mKB67eLmEkT1dUWKhzVWkvEa+0yAPNiRD2Mz248esxdph53NnIZhLDG1qgAlYk67wtaPyraXCiHyR3eHJnSqWgWYv7Hc8npsRU647yiZa0Kuo3Sr4+g+o6IUkdUoqAHc6HpuJSxfvhULatyx7MchqY4tEowbQPJlRo0T453KEoi+wG7DcGFdZ1tximzN7FxaiU2J0MnhhVWmf5QlTJEAFkp4dPUfSQ9HuxMrvQ+jYc79WnEwmOLDaGksMX6/luIHMpOr2EFa5Dgg9xqMNmfIIyWLbAbJPtfxtHUYq/00k6TQfWgOIYXLE2m6gkuNz04c7dmyTsFukdzT5QOJr/FPRUMOszfRO9nWNMbxT6LNX8W/6lZmd8oIwvTtSp+PeuWl0awbM58tz/IidAjVsc9aOg094/akLJK824WW3xmFdj47ThUoXHn5v/uYrctjSOsgvzoA6/CBRrbclHWbTpOOOAU86RIF92QG0NvOYogLrzofSfG08EtQPtFZWb7TZUruniOZuZLAYzl5zHQQN26i9lmgbRJFROodLzoKNhz/RrAf2rVoxtItI8/7ubaP0RgoWua8gp5TWUqllu7zanDofj6mVDE016L/BkXqgQxTBKt9xNy8runLM/kmh2PLwd+4478gjZV3VDptYVMMj/6TXgCuBng8eqDAzGn/mSu+OH6B28AFajScMfoisMgXCDV1MvLLFZJxN+qtj6z0VNUl3iYdlJYuF4hFZwtx2pkhBJgj5ng6cGvOCW1zKhh0dlLsjAsZeu+yBnOFy3GJqiYkH+OV7U6EUk58re1ZWiVTAWDSWV7Pds9IUopUS7Uo5GyCVRpwnl5UOFbCWNW3Lpjs9ijNoP4+ZlF3yMnLZ7+jU5G2OT1O7/6YVuJuV6HfwuKwF72WMVikn/MEpeTztvQVvit5vwgFDb2OSH7KCBsKNIARJFdx4zeQrqKr97l2/dGW1PiRRCYM1Zzkq1kP6sPaGu+H1f9y9rmp8L7O1VrUF5cFxIP19/mX6luRkcar/HwQkd+VXR3GBzDaj9exgAmTN/Sh4tDaqqcHat3smK93hkWs6iwqtmRG0v/icSSOvNhBW9cKW4sxN3qPAdTi5r5WCf+IU1wwR/8JRGJTwFF8KPyJxb2CVD8xzLx7wswvMw+CQFsuwmFFfUBrr2GXCY6j/GhYLlkdVK6Q4QTJLFtXwuIN/5LSSuYrRz1Hug2uwxNBsiHODLy951bzKH8QNm5MRdma8MuuxUQjuetoyFqiZ/qltFXnXRV2CGzCurX4nZ8QcElr3SMK6nlaNsZtjn3OXX0i9iXWHO3HZrTDi8zTIFvZgfMzGwELF5llMQ99OtXJtC4IAryzdKG/XpOavMAZKmdlC5lTkHKTwyUw6zeDsw3iIo2FZ8s18/8U8dNM1TYgp5mmQTiapDCGQIuON5icup3X98dPkT6SDqod+8SXILXhLEvur9kcyxjy0dLADNdWExkZrc6GZxxl4d38gw1/Gk+gyAHLy/ctcKqa6ng86mYq5S03DM+XlFPJnjglnzfqDsbelpqiuHoIAVxDGlnI1FrIZ9cDTeD0+WYBpNh3pMdHQlKA/qK3oYSXvhT8BGXEPFuKYCJkvZb6+z/MHzbLCVX/Wl7B0cdiPRSxzQ1nmQE0IxZMTX/38ep0jshKxUjmmSffKVLP7mz1w8U+O2QI+uJZWQb/QUxYCgrqCUazXLmkTNU7/eO2ukazgf+59407YefL+7NfjV5rdYw+DikTyXqITnyeoCKB7/txpW7p2Pw23jcpGoA9F0Ivkl0Ssx/uf7xrMhBbjkOHZmo5/rNF+3kVuJ/qVpatcUoEvK8w6h4xYkeuMY71JXqh/O5yaosPPEH3Zh/RA677MzldSoE4OAetID63j+vAHepuwx3g6fdKH4BVJFLpvHx8E9JPrXUg7uxSrFl8ZugHCTaRkNNtPrUJIkBPJ7BGFow/dLm31gbMLeizPDtING9Lkq1xV7+0Z6raU9f/UVRFUudqX8t4ZPlnaN7eKSTpln7hR/Bb3FMgeNm5nakAqpkAoS+4J7OoiGlg+f1COypMsA/ZaRjb2ww/hy2quz8juORbZ7tIM3aix2mY5FwbqSV5kmpMAAAT1ZQA30IhAAo/WmilrFC/GV6dEKADHpzEkf7dzeZuFVJaY5HNelkPWl0O3lLbmwtMguQMh0MBQgP1vn1gXOkUr3JpiHMlXgKTXx5fQ/hLCQufQz2wJrOra8aDmTkFMDKz0VHUGIVGXPC0CmC2aDbvibyXNLY9Oa2sh8Bc0zKuEYaJzMfpx/8ns9+OkMzrz7Pn5ax9tKb3iiZP6ozfLTTHK+KWm/nlX6/4ObHbcGIg3EMw/ezNKU/8VVg1TQaVuq8Hpn66tY+bI5UWHlCwnQBgwYO5wHMitQ/GTr85dlPI+Ci4KbuEyPPytnBnqcwAD+/uVXeo6YDpFpLcV7hxoHN2mXR3nadZwSKUQdnuB+LnXJry2Yna5n0Pqag80P4DIdiopUlgZ3LDk4EekNyrWSxrZV/R62GqMLDYEi01Uh7a9NEx0U9qs7/vvQZxyRJ8YtCNwUjYeABvBWPKcOGc8C0tF169KuOJy8wemdYAveMxvCeVZFRVHF6zGu8oaDbqzn9KL29yOaVwk+ZRFSG/e3Fgaubb3Z80H2KdbVO9zSR0YzwRhS5jl8Z5UmGpN5y0ywCoc4vsyPswmZ64HoKH+51Kz6vs1URhqEcUi5L32aT52t58UjRot+JXN5sH2+dy5p/VRP0i8U0dAwnL/4Ufd2uh8VuCZdNrRhFGlaIChsfwIqL+Cx5ZEd2d8APbNxEjDb80zcTdyq3P3SDwB5bUk7UV6vGkMSyt1m1aZD0wvxioTLO1zOBAvSrh0jMWtFSCgsjOwJu7BCUVBzhwL+1mXGLbrm7+FHrLTm+cUNmxsisRQsz5YiKxpTq0ZLWN0tpxiIAe7uiXQg//9KYO7O1/rDYHv6FtRiiilBV22hkrfBuDQ3dJR4csyGkI7K3qk7p+ceYOCym6+8WGnTcFklFYjEYFekbXhG5ZGwiMlx6oJB/IHdD4NLfYxDpjadAb7h85uORA4Xj94t7MXRtmoGitPa5pNo37wQC5hQmM5sbPntSaaw4aqPrvwz060mJhvPGpUy75BJpq8ktHj1Po/zmTXu70SQhX8Woa0qvHd4T52jtiejj/OiOzn8j6sGZIFKLNYk7eLeBqC6OvRmt0SIX1RkqDRJSAkmc+04PlwZU1kOudbzZTgaw66i1mbjmOoc3ulWsTQRrj4VnbS1EEAh9mwZUWacVthvfGAe49wENBbKjETUtUfFrtwozkFgwJlCUBSiTJwX+QK8QDQlwq5tdrIRdK2RjYOh51JujBePsHDvXjJSSuaKaQ8NPyerVJjj7cDqMRS8dfT89voU61HAef/BeJLUEVQ79ovcGubwyQoVKunrBWRYqOoJWyru6QQ1Xbhfnp+Yg5oFETOimlN1Vl8EkIiadjMeBoYkrkIsxc0Vc6gcDe0GfBmlgt8dWL30Jv9jNrydIAftD47wYYy4teJadveNofmm9ygMDsm0dB9snW2lxRL1FSJov12i4o9KTyCVmovWpfotC5EBpN+fMMKkJMaOIe7niHym3qkgU8l9L+wysb0oqdqrpVHqL8+r+HxI7DON/1OyZVRVtwfbKjDFIIiV5A8j+6XghdX87xteFKmOx4B/unvIpFhDjCbs861pTkRXfoAPtVu6L0giUL1EXo3FR6ecIVzNhkHTNmRlsXg4S/VtB+aGHzLKfdQr8aaiVa74MFb3L5b8xD2mIE/OA/xdsLRUEaBAAAGSmUAEJwiEAGP0LniUGYv/k0DYsdngAQs0ZhbUhMr1zEc7mJruVu+Gw/g4F47HwJcZSnQ/2kKLzIpPIHgaJVzE5OEBhpTpYhkbl2PijJJaBu6VXe9zsaN6maLevS0H3TlJgaBz5Qkl9kZUGjpr0VDPywAxHdGfY2wGlJrBPylqQ054F9lOFx22eo9kE+sk75iayWa+U1KOVF+xW32Zwj/NfV3eDeVYyRUxeUhmWLx5EZ05GeTLv+B/mk92A2erKUqsVPp7n797EoRkEKhJkJy3ykb15ebF6jRjEDd3vr0ypYkXvCTCumA0IFw/cKnBgyQHN2mfiQSsrCmX7MougJOy59Uz7q6yDUG6tkreGYAAA0EbWzWU/6uAqBwsZr4/QuGI/FahjsO80VTiuJx5UIvtCQJ4zwQfrGJ4umw2WSng5XSUy2dNNAFkzF5CcU2FXxRj2F6gDb6J+yqUPqIjMZHX2hUUCJaLLWMN8xtCCstvWRSoZ2YeoycGU0LQyQn9Yr/sQYUYn1vV301kzKgX7imrxNrAkqViZO+iqnDIiVwfAZaNnu4GIwa5Mu2At2TnhFWN26z3A3PteLpSsUu6b6exmwA1LPWo/9tHDuMzO32kt7lcQUuGuC4qbOMif+/gvkmIoUiaWMC8C0JM8O8P2T5xs8xNjyjwrxpgS3utbqflvZyKokdgodksFwBzq1uK7vrrZQCnKxe4fNluoctaPbqwgs9qGAQpqR6EFp2FmNDJZUOKiMDq2a4GRXYyPLibhKQtSDn+xcQ7/32/E/TCWOUADD7kBXR7IRcOfwAHkMRAHZragkc4y4V5nUZ+9qQSIQ0KRycUc22FsbbhC+f0+CweUd8rAG5JCUDDcmaQdvE+S6SVl0L5L06+gcAeqQf8qIDpYD033JAGkjxNyNIGaTK55bvq3fP3N+MwyinhSYgAAAVo5n6OfZDW6mmY7WDthbDa5YBoMtyGFVJrgbyFOxA3ewJ8WS/sFg8AFaGDq7SHlcp52MBi33ceDrWeI5tNbdzLYls6PKAOpaLbjKAU6czu0ccQpT1v6/CaEhFpoJO3lVu1zS5EgBasGAWMHfqhW+Jov1ycp2EqvYC4HeWgk0EgKqsQ4/31eMwfSO+HSDq0wWRXzkteE0FiSfMQga3v25aEMLT0kzpwfpLt9y809DG/mM2B9twUVcBjDTRZbrIjrqrP92e2zWmLhF3Td+PLuG9WM1jCip5ISNxbfJv/u8x7MKiN7gRAf6PpKlnmcsaYg6mouBPNUk/2tVMXQZmRBfhCEjzLIkt3FPeEgaZievYi1zPAfVLDi5BRZmXRlqLrfIzqVg0ps9MWAotyqCSujj8Viy7sMUAcaWZwqtTpaQFkOYva6RPoSI/GegzDgJpH5Mg7pXMBkGgnBPZgt9fLH91wkNhDvJbPtvNrt5dnkESXXhgkB01XEfAq9VQKZ76ycAMmgNLtugH6i9o8D27P4ZWCYlQRXlgYAAAI9bSjLhiv3lDBEXr/ypONCA8BqMpYw6FoeUtZ0GVajsKa1LUexMlgfYt3I6W8Ojg8coqlL7ySOtzDxQuFo/+CA9slvZ7ImbmdotK0bhcUzCnvW6SUs9zD+lWjJMA9SBYYTnqD/aqMlCeMmNQ50ESLtsFeYwSXek1P+2qMnVF/o+QWlhjVEEjrK2pjDHR5gb7gHudi/LCzl/18uwRUDBcx4c8HXxlmmOOd2XFjQhEgW0OCGhcXwiDjo00w8YOZf/YPwaIbuAyd+dFYl1HtMryFRmtFyViArM2P/QkEKIno/6wEXp4OBQeKC6Dg5m9TU/CI4PjfKcoz06BbY6PId7NVezvv9inQqFtAjXwttqPRnGFVDYF1R3pDNnEm6X+BsHqp1e1AioxAmUBlJlI7Ns6GHrwHdvlq7shPVHQ3e0FwjPXAAf5btRlkAsVcEpmVOfc7oClWL8oDe8bpCk9VbzXChtlMe4E7EZ+0UmtscPEwrAONy4Pijf6T8A/kaN9cAa1cowQrbx3sgHJiVyEpkKG8deNzZDeKTyT1GQ/pV6F8RyjeZhdh2cCPWwiIuv195vNQEQj47D5FG2vcxigRMYJgPix9xjvPOjxIw9ni7CUPhPnjoDJ8LOC3H38gyXxBuYw6jNlADlStrdzzFqK1BxxtB8XAAADoGUAE0QiEAHP0WniT0ff+zEr2S67gCZxVmStIth5uJ3xsavW1t94UPKHvMAtsmg6lQ4OncIO/V+VbJN+K0A5o+0YSgUKfsrp2kPMr/VBP/1R4T0WU/jBA6b8inewrz9RzjjtCF4EkHpQ7KFVAIJMhfMc7HqLlAhgFMYzfV3TQjKEPNaFyTvMTFIJwSWhvzuo7dSv1kiAmr/QqYnP3Ja48eZM49yvdtMic3+8AQ0s9eT1H1wIhH2S5A15UVoRi1z9PatF4QKnVza6yNBou5l+3ZbMHY0tTY6PSSq7/2tgI0FSz6zPn6zzQvs2sgaVNUmfVA3dWkT8Jfgm5jJ5IShcVZfzMiwq727we3XmyEy5bwpPy+MktC7E1rYPkwfw2ssT69iC/ayxKQ1y4IiELX2OiEOWgeKeoOo1Z4lEV03oCT5iK3JDVRhO2q6LHSkZTxUuHYn1gqAAG0HPJpL/5ZF2QdkYX//64+CAA+CPvFe86jIPSBzP2uSehQDhailiu9nl0AUUQ3FFgHa9Oim4eatFoVOhfEAr4mwcA/0JS7D4IYnL2rEz9BE3iffzADS4lOobGU70PYADXcek0HBFzXt3JEfwOdlMkC2fpldFX84HHCIySHD8HmNzpRPzJ0+X3F9kJmrqe4vqIxMN3kAdnslcEq0Yl32X9XSFuiFRgMqS8ljT3B+hou+QPHhwCOsJtE7FH1R1SNX4oB13wpHqp9LCdZqu0pxSsfpSnFnvuergU2mgHmXM+vTGpNuTqENgU8SJ60UB+lKzN7ow56mzFw/m332wgUGp973pIurBeF7wz/oCd+e1W25yWfy6ztmY3xP84Co9iXu/Fi2pV5U/i52H83EePkB3qn3Kh5C4fmfm39rtUWe6hbcvP6O/E+JlrJsL1FdIkRDq4huxtU2vG7j22qEOq5Ic6C30rUdt/6K/OsAP3YefATLFwA8kok3tAPOlr4Ep3SVm/ldwH1zuMfoYD4Sd24WFejCNrNrOxYV12M8hz7tqxfgjxXdEfaVDgvRneUmBRXTh0O2u8xUAvxDO3Ukgj9XwIHgpi6LVfBRhU9Udd9lj4PXnJDURfBwATJZUF/+gbr+FGTP1uA5C6H0RCyjv5S6sztv9TUSEs0j16RaG0G7nrLKy4Ru/OgrLeUMGk9+eMMkpDis5fFfTAAADAAI819pDdoE2IpDtw9T5NM7/bDAAVZADWHihR5VHOk8MEekAAMnug/lCRvctf9lzMYEAAAErQZokbEOPGvSKMAAAAwACWpzEWLAUcZR1mUcD2uD+xK7Q9A3WNg8eGD1ip0WBfCdE0/tOwQNXXE+GtM1a6dJ/aoE/NgrCcGBbIFKQm1TLLisc2W8nhDkAI3VfVJZhYZaQ+Y5nBmpVwh+LCF9RykHopguQlBKf55vl4gaM4ZzfHUIUxnP3EhbD5kq8Ytok1olciPtmWzBIwXzihI3ylj3gh4qISJSRThG04DDTVbl3eQ/3rP3RA2Se6r73Ii7kFH8ASPJwga0B8KQkRismebui5ao0NWFfqnoc9m+LoKfyI7/LlSJkCLqlrry6Cs40Gihwb2YqgDjrSJoZSleQrmpkP0zGURDvtZ4k/JKzmngITcYEPo/U4IcUmfDWW4fq/QNenxfY0EQTu+w/wZIAAAERQQCqmiRsQ48TviflRQk+7x9xdWy+ofMgPYlW33KREndB9RODQon5hWImgKjfF3FG6ZTO21LhyPCVO6dNxPwmA4AaRS3J2qtuzcDhQcEq93r0L3oBxgfcybA+lE66EVC4PWgh1obcJutIQAAECc5EFLjpDHrP0VDVQLHsK8dCyfO8CAsbyNLl54efK10WwkoetrFjB4xTG65Y7zjLYj/FiCXpA0/1pwfgpu2l8x43bPDqAbm0FurJX1DLi+27KS6uxnWOxoe7MexFgDfD0vatBddxRq4djGeDBC4csU67m4u6H1EAc8NU79/baOMogioxfhm2FoTUiW2tgExmYCCr6/UqESDneRbYudPCG6dAHObcAAAAykEAVSaJGxBE/9fb+ADi7ugBoOE5XiH7BAHQQ0ISVVlAumxQAJbZpJuZXCMoo6M/O3+XoSziPaRC54gwFX9VMFRHe3Qi8vpOuB7qDIuCAiOFxWLf9pKFmCNFaysNHmqUF2HzNjqtBls/E4VP5GcEIUE7YYis7c+Zlm7wsP4p6eDoB7VsKJXigQAAM4fS6jwxiKK+qKVjWLYm2ZnrPxqIjepq63wLA2ZVIuHfodp0S5ociEZb8EaKJO+mNXUMJY7aMmVx8qVEeGiJx2YAAADgQQB/pokbEET/IQ6p2A/hoLqpHioS5upVASLUHvANJEghPFHLO5Jq8nQU4p3+pF5RbozstgfbiMJeu8DpAHujrOyQPwG82yOIQoxx1cttU43jT9HfrDYo1Juu5g6Il0spxLxga7THulS7VyZSoZyd+erYA23sVQk6S1oJE7w5p5oVlOXtwyW9K1y14vrdS2pcTgQiTH6j2JdshE1xjvGZKS5jAWOiNsdWcJwnUpgTCmz/QgGgB6XjyE/MOuS00rGLk21E4Ro29Hv2545JnXV3LStWFMQNIai5HcRXlz0I9UAAAACgQQAtMaJGxBE/I31L+wW9Bw4I2ue1Ji38cuKfAbc63yJH9JSaZji/AFqCw0DLnr5sEgONZ7SQj5IPuSijJGWK4GN7rmphI1FzGaxjeA1XzmKgnMJGphPbVzvN4Ba51e+85UbjRNWBGCQfLzRJXHu/ktykLrIO0vdMa8zakZfcGD4PoTnebyPZKxt88QtxLrCXURgyo7x1KMrOfMhAbtu5gAAAAIlBADfRokbEET/X2/gANWeNOZuAA7OSsgnjtQBCTGOAgHxAmTDudZeW81xMArduvfqP4XsPM6wxenH2ggab3aKm3ULPYW+gJNBK5rsKfKmXWwrxH2TAT+YOaZQSftK4G+50hQL620dmEOXGEUBwnjqY5DGI3sHWn0F3WDddkBXb8l2yrfq4q/DdWAAAAFVBABCcaJGxDj8T7k18R82KFqTjBsp5wjkQ+cAAIpIxFHxN7uz91xwcLst8S/N9SL7K5uZ9yqV1O8UWXiMsTuc84E42h+NCpFN5jd8UxzzrrZ339seAAAAAMEEAE0RokbEOPxsINAAAAwChQDyhhiAF1M0U8hyfdD+ttEWmy2Uh71PCU9j2CHr/YAAAAFZBnkJ4gof/KNe/KVpTSD17kQNW8sQ47NxqHHxRPCsZDYgRJ/rF1MK3596GRgrhuXt+AA+4fmzKf1JrYQSbbEhplq0D13AwSbLrGppH+WtXA4lBGhMXWQAAAGNBAKqeQniCh/9UOcSALDowX4CrdKItW0a+bx1Pq0xWZ9XRZuA9i+BqWOOu8jdRu+V1JmzBO1gtdVgmGsTLGkWseWBeQ4XXN7SgvoyZyxOhBaAU0Yanu7PQ163MDEv8v2l98MEAAABiQQBVJ5CeIKH/bOHz/WMiunLSmwFk3hNw3CmowVkBxLrQZcsTD/eUX0UHbvf+GAJDsMxaz/krrQu2uQEbKgqMk2NT2AqAvq8wZxdeKeqXXRDLiWSkks1RBmVFR2m2qrP9RIEAAACIQQB/p5CeIKH/WvemFZJUOAPrHPYj9whcCBzKt8jLpncTWFPiqKomYhx4kD9hT+Ze9yhkqPsQlN3XSuLa4pVT0vvhX4R8Px6ueMLKuyc10Zam/kU9284v+UHH3ncxB5pmogj8yX+mjh/ZSqauPvZVyTfUziqMZPI3SnDj/Edg1WOIPpGh1ToYZQAAAGNBAC0x5CeIKH9eSHKwfZg5AQRjjq9DwJvmj3sXYyYP7wKYLgmMOz67WYgg5NKYKrFaRgWV7GiA8phURKUrReoMInA3zSH3lxCKh8INwZ+zkiVllbNtBpBxiSYWU5RxzIpfhMEAAAA/QQA30eQniCh/7bszT7PrCKP4C1di3XS6MKgx+Ppetd/n5Mj8t865uz+mMsEk3iAbTiVFWi9u3fbRFODNx0dRAAAANEEAEJx5CeIKH23OFEztc2/b9mCVzkkJ6N6pQzo0daq6AcZm59oiQ1jTfpx6eRug6PGC2xEAAAAVQQATRHkJ4gofbnkMAazPC4yDSpHzAAAASgGeYXRBY/8rWmTYEfRZtxXMNpcXslVJWDjqiyr3ugjKZ8rc7rLYfvqv/oa+ujKA6U9ers3KhmudlVprhPVqM6qFlpCN6oCJRtqcAAAAUAEAqp5hdEFj/1i0EKvCnMQNdN/EELFMdrzxmMjRohdl9Cmeob672+1XegA+3+xVMcCDQEiQ0pF42NnTfJ5bnwwRjz1lTNTQ3wNP13O78hAgAAAAOAEAVSeYXRBY/18hIKzESQYFnhF0KW5LHik5TOgNfEhv7rtTatsDOFSuUg5Pf14ZiV17B1KYP35PAAAASwEAf6eYXRBY/1z0ah6q2EWL+9IxgtZm8VhSXFNUzXDGoRVfH3WG8wgEwgqk/SN+tBBabTdVgSBv6hj48/TmiKeSnUFj2f8ZPZV3gAAAADoBAC0x5hdEFj9hi7B9g5ZhiVPokXIGPm0hsJ0C+yrDoRX/pJ/MxMAtEpTYIvlNhdME/C8uJrvYtT7YAAAAMgEAN9HmF0QWP1ivJHmQTOhZmrC5ri76ryP28vszKXIrlvptdPX7mREvzzFdlOLEcC+3AAAAJgEAEJx5hdEFj1O1zooU5mx9L2W7qBSw/Avby1Yt9S0W1yGze1xgAAAADwEAE0R5hdEFj3HZRACXgAAAAEYBnmNqQWP/KqKPqvn0X5yysBbMeF54KGN4HIwjmkF0GLnNuEWROageBxl0keVJU5h8Bgz9BsOoVm+pqsrSJ/lYSLqNSTalAAAAPwEAqp5jakFj/1gebYtxOYumwJe663BlUCZT2hj04wMuGHJTQoEvjeHJ6A08e92c3HwQc1/E8psrDPMIRC2BIQAAAFkBAFUnmNqQWP9czLuBT1DqzrexThx/d2f7rZRXe3kcvYY2kbc282brAvA8UWSs30NYhdTmBL3RTsp0pCjx8LDfaE+ryJ77IGZ+KOC9A1fc3plLDTIYF1Zw4QAAAFkBAH+nmNqQWP9fE1P7Pug1ZHIxcOIlKDspTrAEcrolyBR5W1KY+m4XP0hueYpagvwXGEjBxF0659ErSb/QF6LV0OHxLPELrHhAyRWt+xZbdTMzVpgi275l8QAAAD8BAC0x5jakFj9hVaGOtGb6/huDygW9DbILXUzy1CmRCjwqCrGtECvvEN1RSn0h9/XFL/lFQ0w3CmuCfnU7irEAAAAvAQA30eY2pBY/WFN5xomfPLkrFZ5WVydF65RKF9RXbNy2fQQb8oyHKF3+80DbOyUAAAAvAQAQnHmNqQWPWC8SFRNtulEgdPMlEvmr5CR4NGX3bd90nEhznyOPRJFSnuNap/EAAAAbAQATRHmNqQWPcx02cfnjysen3mskVF78AJeBAAACC0GaaEmoQWiZTAhR/wxnZWgAAAMAAAjw6onm51ivusZigUFLcxXa224Jq1FOQ87aly8iDn3HvD+2vS28KScLtmIUUrVZdksuEeJ23Q7LEuNjaROhAsy9IjPGSBtsvS0rbqU5RnX7PAk9xfOr737iAtYjSW1TLLaTKYnCo7Yn/Bo83WqM/BNpXzOMNpN92TUeEV0wV/RahlQvoSsSy1N+U3uu7kSAvYgRd9y338/7ITQJ50Pk+qweBbgdcmjGsnXujMyWAzhPSimwazzh69K7BIJ5/0qg9Qc25BiKW9VI3O9Iq7orO/5XyTyh00R2+9j76LPrC/CVPdUsJAGtHisWv93ZOymBoARke793DXsTeq00tDxaAiwD65h5a74RZlElPGKcokW4rOOvqzIUSMTsCv8oMtzJQZwQFR3b3hS6bi3iX/z+Gb+xUcBzgFNzEO4TRbLgnm0JnMpnLjaFkghFz23PcGVh1T7NcomTo+CD/iPbNqpoK3LwJxiCdWhffRJ6RXeSliu023AKyG8gfGc1hRKCnm7196P6O7bO0sq5wyVjxc/4RyCtL5hmoSorh3IDHNwqgcNDrR5EJ3a2aiLyvs66Nw0Y1WfMdHAf52qc9hamnXgq2ikYmSX/iqBL5I7h9PHumME243yulQzG8Br8DqB7SShY4Xb7/Sgfb5ViqNlRltefeUnFtbHQtccAAAGhQQCqmmhJqEFomUwIWf8L/uibmmC9xwhnb62lc+DVmyC1HW4DypZ5WssSuUoTvf1cD07NgpSmStRwjLtYXSo/OsFhAAADAASxrtcdYNyfItp+BGpE/+Y0eySVu+zTV5YCqnN32K5xLb9esX/cDQSbtQJG2Xv+uE93t9GvyoMbXAg2z2f7QibAl4/xT6G8LEPZVHr2OP58mLMAA8m5912grUY4RjQEFQSmTQfAOZzR8NU2w3t7enTuBJK9CTepNdD2FuJZvb/ZPu1P4OiCeFZ+DFuZXq58lbtChF/YsxpUT23bgwNMd6+KBjVlTJN+ekf1rI5m0YImHZetTWIArJuoJFsW/Q8sYPyWLLl6+Hqhdn5b9ECYkOPSNIuJFO5kuJWBR3ZgNAWe3WwL/znvruQfg4Ck3CPYuLA6ExLxzUiLD5Gxv8UlBpXH/msf1NPzYDAh0t5yoT/0K0AFOESOKd19OfOb/CEhrT3V/vhdGMAAChEN3kOD1do4fHCuoaXj/pacbvKVneEK3EC+WguyDg9S+equtARtOQrS9PBx7ZXkFbkhAAABfUEAVSaaEmoQWiZTAhp/El6XQywZf0mINUdASMDYwMzHOuucvnJkFYEDvZ5ToAifNgAAHjjGwQBps+9MKQoZ1EMnhkf1jGxv5uGDXb4QsOQPW3xN8kqqIPra/8Fb4k3W1L2du/ISZwqb4QSGWV0/Lo0UpU1bUUiIfKen2NqWqylr8y2oES1mgOSPDMVQ26wwBbbhpSZWnAwyjasAfEzDKwc2yW3V3JLep0VWDSdsn4pc1QyBN5T2upTqkMzXM0/xDTDbaRSRWiOlqh/xMtCqTltKVm+7CKFJ7umx6D4MWuNG5n6HoullpaA6vU95AhQmwLaGgkIgqlroS4bMuSr1BQV4AsvopfTRw+tLOZG0RIw9Ul8096jNZJa0rO89yZNsMqYu9MB1fPd0YZs2rB3TD42390uRPHesn/LsB4q7c5xUgen5OTIWH45tvSLs2XmbX5PNrjbnvdtQBHg8/EVdWS/BZAf7r66ORtRBqMU4PopbSklMWpGdUVu+D3DgYQAAAktBAH+mmhJqEFomUwIafxJelzfPXFVKdv/k/rldGeeZGa0PVSkT46pFjO7kxAGmmoq2GBsGxOQoMiABGqFkCVJebokinybYEHEgDXExoSSQ9eSMas257+GmIZkxela+95N+IZZyPRrInr+8REZLgYWqt0vQ0lvuWbmGDM8elqippg4LenymGioq9MzX2bbL5eDQ4hzBLlWfR/KUnFtukdPPuc9TzVUjl+s/c3uPkdEb1NGc8n69px/G0o2HJN9f6M7yQixMkFcGBof4/A2Qm6mvvuSTfI3Wyd7cVydEDPZJ91YgwlGs2u1ldvkRHWiB6nABLCoXRghLBRyi2LsEvLFYjnTq6W1yMenFOWgNzlCi5t5MdRTLGB4ybsmmTfGHwj5fJiSl553Xzn+e2Cia6+XG7+vY3EzdkN/f+EF70rbyLbnkHLlXZeZ2/2ybSPu0CiLMHhhswufwaEMzpovXXq9rCm8ytN4TLdFTX9o3W9O3vSH3hl8zLtWUkSUX45Qlm5/DOpytn6urhkl8hdFTXpmUjh1YgBtQRM33NtYLcRb4l+RXBywCdHx9ggd0qTa1166ewuMIecSn9ANSUh6M5UrXeRtb0t34jYpQ/wu6c/zFVNgnvL/XJfR76ngaoBTV3BCLYtHWUFu4l6t9KMpjLXGBfYd8wWbTX/H7w4Fx14LXZRKLqFQ4mjIieUEmeWKKIJu+vjABS8qY0m70I4T36J0SuBQ1eCAL8N52eMvVBWyxtX+F/whcU3jet9TBKFlgjQjLhgDK0cP5Tk3JQQAAARBBAC0xpoSahBaJlMCGn5g4rJKqHzF1BmkawxwcPs6ASLcNEkl6Mw4u9jXLdYV0VOneeENAOO5OSnmGNPKOiVN/BjzZ1oA5/x9cNQsdT12kIY8w4dr2QvWrie0APaThQUU82IWVi1hM7LYMwS/rcWt6qqAyybmfRcI4J4n3RQ9mLS9Q1ItdqASEsHz67ZnPqtZxgmF4hE7IBgOcm96ka/fpb+theJtdqiovQOjcYssbuu3BuGSghyujIKxXbOr1UrZi52bD4eE3oEkcxWEQKOQwWBaqln5AQ7Nhn3rogCZ+fRemU82tj6hCi2iJTb1SZEUlgvI42i0V9GCnxiFRZG8MW5n//yW9QiyW1u/w4qHZowAAANtBADfRpoSahBaJlMCGnxl8u8VwQAAD4YDiLAPJTV1wY6wAYi4EeGR6AANJvIWNax0GhTCjT/ohxx1Lq1ojEnuNyh722sSg7jBumg3WvLIotoxzVh65c0gY0PNUIPsenL68f/A07zbhuZuz9CJlPfCOE9xWrsFAP5ZKBs+wc/ZHk2eIsAMYHU9AomQbvl8IUUFFvWaikaPj1f5nyrfkze/hoEBJhORlgOkW6mqgGRnyptXcLqQcX0jKaJTxeaEHCfYKnMpR2Yi1AwvySoH4U4xVdIKtTYQQ+g3jGIEAAAD0QQAQnGmhJqEFomUwIUf/DuuYE3AAAEr88vum5KxBi1tCAO3o+R4azDCa15q5Kt0rV2eQ9bFgikKoN7MOV5hdLVnbD+jM0lERUDThgWf3pSteD14rSvqD4qj7H6jV9ZSJRXULHMAfJ0AFBOvjmRrNkTYZnPa4MZqTIBMjAjsLQl15Ew78EE2AXELWfdBIV3V8VKn9jf0toNDmEr9vbAaV4Y9sCyDdlqIUZsApoiS6bCx35BA30d0BCxJ/g1Q7a1zMBVjTJPs1w50tfoaNKA0B3f5Cs7h+s8F72GSCJmCveOnCFHMF6BB6mm81Y/ADQeqtprpMQQAAAKZBABNEaaEmoQWiZTAhR/8scdw25GlAvjLcAAADAAAEjVVwD5easbiO4veDOBNwSl1qGgjRbXZu5+2AlZpxK1ouXP3qvz8P1On2vVXPbFGaFrFD4OXPGRdu+bsKHcGOG/PVPVF5qOQlLQ/ySpzgaLkVo9KoDNc20g3D+efHrJUyyw1LqdgoT7NU87MCgAfu/fQH+Z9xHcBNqBv1EU2EBoYqVGKcHdjfAAAAkUGehkURLBI/J208CmsGuDy3iXeqQa9NdAhWtoTND0Wd1gzBl28FJNolD1Gq94Njcku0MWxT90Q8x5ZRMJjS/ekYSqOilbrcN7tlHpCN0D1kPzByAoV4AF3T0nzjh+IQTlOuXmdq4xowDwGc2pgXaDTgr3bxxiNuBTF/HFHOKINNmsN05byKvXM8Bz1VTCzHS0EAAACWQQCqnoZFESwSP1I2eYEQLKDZre/8OnYwmY4kOZA0ZVN6IzGJ10LcFMZh5hzjAKdnLyNaWTosNIKOOgVWdyyy32D/jvzxFA6y7AlRmv/l8va9ROBEtysXqYDtHki5ZNPZVTWXDNO9xfhuwZBhc5XlPW9Vcf8kISp3G5g7nOlslvgsu/XGuSmjmuD0OYesD0oLVsGuvJuxAAAAo0EAVSehkURLBI9V9EeoO9SEiOV84QYUcDegI8L1E/wGmeCBrDOyheeHFjeaNr1O3nqlX7Vze5Jh5gtcVim0Lh2TqUGjUX3dno3rS/mqp+TFPBrDfBHdhjKJZpZE9gQ//C2oLUKe/QAvt5MKS+a3xybNRbWkb+8wjdwu7qcxKjM2es4pxNkB9ubW2yjObebpF+3l42X2EpQ8H085l7J34JDGq2EAAAD1QQB/p6GRREsEj1kKbppNcLeq67ZT0fqVccQsae+2SVd3iyS1chRta2A7UBWFAsoiSaORHXl+qhy4Ae9VGmEV3jW02E9KonA0B/Ag+NWrfeQKgikuhX4z6vvQBj6V8GzNxuiplK0fnfb1GoG8Fr+XsLtxvRCpRSpkUr2JIFwshMGRAxzqCwRv3AdpVRNh2d3srtKG8Zu1FPZegeVPahB2SR+mYhd/6j2A7S6PkdbEB2ez7FRCEtjm5WgVH2wirUF1hwDqwqJYwC6HbCDLa7kYEtVvGIfuSyb/xMfzOPoi/yyGr4dsoI/ma8bjka8gL7wc3Xon/EsAAABrQQAtMehkURLBI/9Y6lWnILVPyPdblwXuHVFOu0fCfcO+xdOtp/7ciUKzTNPu5JSDViOPPxGJPW/DJUmLOA3zFitcGJtLPdixgjN0681V9DH3etzAvubLpsfoSJvR/HS5FNfQ9RTkUjX4ClEAAABNQQA30ehkURLBI/+pU0o1GrDaPuhou9NYg8G+miHMmhNEZ9bSP73TmN0HQdLiKE5IkeshI1tp2VlsRIQuIClBwQe5ZuC4ZcnSMMMlO8MAAABYQQAQnHoZFESwSP9RXgS4VgTfdP4iXnGiBmPJFmxA80xgcNCKxnzg2Kmterq/YHHyonGka8/Ndgt9X4A+NhPNCYkLWbmPBMaKPLfl5ri8FJlJA3gtBuWVPwAAADxBABNEehkURLBI/2pRHukNEbHJkeWA8l1PzdY+cR2aWCj93f7NvugZUXozWPw8DTu94C4O6L04GEjOm+EAAAA/AZ6ldEFT/yndH0XtSD8tXoSqGMZ8Mwj+YzMFVDpAqfdf0gXP4tOwBX9rZvBVax4GQHERoJ4SjZzQGZxkIdmxAAAAUAEAqp6ldEFT/1W/xYmipHd9KSNUfuFcs78t19tdlCV3AH9L/dfhpZUhH0RHoqfKxlZA8vLMrZomyx+9/EPMBmk/JjKQrZKCCdpcBU7YGoBRAAAAXwEAVSepXRBU/1q4H2yIWxlpafpJB2gRdiWzcLpe3IpoULaN2lJ2Ph/6nNkT8Kd1yHOFy7ylHkO14oODkJKlfOVCseKZiEK+QtPLrviNm7B4ml143GzaCXZinsrmjlwXAAAAXgEAf6epXRBU/1z4H2x3UnGm4MZcP/CiEbV92F4SIoWlsIte+39KvQJyZeoc9XOIsyuD1L2UCaG0tQuc5qL/syxUP8QEpFmMY3ezv71iHczSBZEauirON8QRUeEi9iEAAAA5AQAtMepXRBU/X1QSkFYjUW8QeY1Rt6tqxMXzRsWdnWZL1v/j484OnT6Li0kR6igtovCmSqmq2BojAAAAJgEAN9HqV0QVP1YoRZEabV9F1R51ci0cYdqV5rEoc5W9boJFUlkZAAAALwEAEJx6ldEFT1YpVi85Dd5AI2yJKJrYSgLzIP073ExGZrbQ8u8UBF9E967kIsuHAAAAGwEAE0R6ldEFT2+JRqpdF5NCO/RzzWD8yWxWgQAAAEwBnqdqQTP/KBdvY9ypuwr17OsyVGn6DbBhSya3cVowuUdgFaEHKrfzDQHUYCwEgZYJ8pZMNINfcfs5VoIKYBWOMlaAl5AJAz73QUWAAAAAXQEAqp6nakEz/1Kh5rtwGyO74j7YrdjbzrNsTdd0I090mQA8gvMi4+dTcscrSx1dmgs5IO0v7wPuXfDgbbeFA26ZR9NFohE1YTvCa7aRZ0edBG46rGUJTacugJNWwAAAAFMBAFUnqdqQTP9XF0yeZXawruXP06S3S536HP3VbmEaVv6hRYHfd52hs0pPc8n1M8xsq1lHUW92JCUjpIM1c7z3AjqZOF2tpVHRaPHlem5LFTixYAAAAIABAH+nqdqQTP9ZWwMzF/jiK2aKmFei21Gvpk2rPQTgfa3vwFzxkDEuUWsm7ZpMbTT+Bo8FWf8qr6fqz8isaYriP50nXlXZh12REWcDbUJb85zWCABs6Plym0hI2e+fyx2Z+0QlM/Zqfe7+8b6xNBWA2DL7BtiOjb8ha4e0viBrIwAAAF8BAC0x6nakEz9VXHpfJTObor4PLiXc82bk6htAO+Io2UYL9mCeHPcMVoOpPQQpo6fig7MDYgVY5tHlTmpqxeI4BYHXpLniGhSe3qktcyrX2D6I0uAT6tKmOSW8+acL0AAAAEsBADfR6nakEz9SpFktXSkrUPL/By9IwTl9elmFOqYROEJaaaLljct0WBKH8E2D+Ug6MMHmrBDXiZKLbb5z7D2W0Pa7Zf4JMUmK+RAAAABHAQAQnHqdqQTPUqM3Dpe5tyUAhf1rz9R3MdQu79q765g8c/Tvqw7qDezozwL/o0raaMN8bgi+j6XUW2CmelNSlwqhLTtFCh0AAAAcAQATRHqdqQTPJ7/m43I2s9tnezz5zKsbyWCNQAAAAxtBmqxJqEFsmUwI5/8Bp1aiis+dO5DwY+BjtUCxJagOHaRwW0IK7FMqTmxSMduJMofVHr6EIdETGIzB9eRgkmjIuyNPaEWg7P+ZdiYr5UwFkS+J+CpPM0x9hLTfXfCsBNV3YRb+Z0+WYeyF6qLYlQmc8K+GsCvYoCBD3qlmbcFdU2IFWk7pAAADAFrHE/9vPrAr1pjPWpgVx/MkbN7T1/jVeFVUkOxpF4eobFLNkW+dvZrnZ0tRDN8lESqp58Myny5AwewvF9Ce3YjgyAGD3/ZEbuxSdvborkC3X2b/QT3l81RMRbKftFJZ2JAAntB5YtLINa+zO6jE4x/A3oOGH4HDHIgFcZ0dERrHILHvtDEusS/+xt03szCj1S3p/lJLl8qBMkdjLpcOWn9mNuVEcv6U0z7aCLRHDE8VXx8JIhinEZeL1fx0mTAGvlBIbSMHfe/fFsJrVV6N+bUw5WYt2opo7FlM52EJx9mXtFE1eUTErndhp+14758V02l4rba0RSne0mhML6tjV2kQXq7tdgmIrrSXKrLYy4E6XdNYlbjZA+hNTmS62pId3DGJOeO1bE2Wcx6vPW04rSysWHPhEjMqEST6fPYhRP6YUOa0mMpDNwpSrvszZwPE1ThzUpEBQ3O12Zyfc/HsLoG4PS+YRuV1Ch+vIIU2AxcXRN/8KgTdxDWGyXmR4S3APMjW7yZPS6EJyIIj7nGFQ2mQVz0KWLdMZVqsaTcI/UpRUyCO3WFUEEVzBlB9PU2BC9s/9pXnozlHTuwyLngyrQgaBtubw1T5517MKAHsGUtgpMi/LQFXe5i+Iw1t5X9eFMN6CWBvV54aKLPeOh7fL96/FAtCzRRb7hw0v9qeSEwgDZeBMsQ04ZTYlWUMtropSnTE71dKf2HVmFsETpJIwmOYLFqALUjXqgNeojlwnkeR2sR2gvZPoxcw25UJ9mcCNxB8tuDdVs8pQ3bvjVFzXlHOx7xWZXDkOrk8MBDiral4y8gzzuhBlHfQHh+q3lvpjtrhAWAcIkY7+fQQDUbwG8eui8bno0vEYWFzPzzBsXz4/oAAAALpQQCqmqxJqEFsmUwI5/+mNEGoQAAbl4szeejWfhUUIpo7ps/1mYHGsMMJvbNUt3uId1xWb1woyzsoZQGOp2b8cL8sRfRvWpCeWC15mje/864T/1xX+FOb0P9KfVYT5Reqvw5h1OfGdKvH8rhgC3Jcc/eyf4uHCaZNKQyD3TXkrGaTkqCdOHlPQs+FvO9Ad3Xs6QeJYmjnmB2PAsYiSXtkTh0cOVpfpRCDATkDcg7F2CvttFeywWwQjezZMHDncWtdn+9o5uqri47Y0Li2NpazwQmyJdO3pebJvL/qh0Qt4rvSFgk5fMcUc3fMBIcLfKY2e5zJ80BM9SxDbv1qNci0hZ2gypTWtNIIihlMggxpRdm+OKlDptVVHHWDflATD3SkRYVYG14NiBkSFcLHOYKJbj1kmmgYDa/MO6JfHZQNN4oTGWcYwq2ZxEVeY0T/W4nghXtjcPNcv5/Lc+Br4Z18+qOsiI4vLwD/fT+5RLlnvdp9Ee3ID5Y8ca428wsrBVJ1aw7dtH8mebrbdfqKH2wUZJvQGAy6bcfuEPII37XVs0stxIDlEZ0TnyxBegrHxQzTftXDsxXUBujDKnDa10Z6mOLJisN4Ao2zg750SgJjssGKgAsYPJcqJb+Ovzk2IPquhFklWt/9nHr3LVFF/u2QtUo4DIvrTg2u2aueJnlUMeie5YVf3Ks8RCjmU+QNHQ4zaZMcV+vKOfRm+fgdkq/fr4xK6Ud9jupwAM6Cd31JJSPuqpo0yeWtR+nes2e1WgmPM9Y13dWMyvQGdJvd1KXeOFUZXgczjnR1MVo0QHbwIrMLTS7yDoI2Inn8/EU92WQqTQSHjcVO4pBwSpXs8CoigQuz5WSFaz4jhBTj8xcQRFv01m0UMJaZ6F5eYTlOCuND8u/JqdqvgUSFBSpTDlemR3YxSgQxqqbTjIA42JfNrb7Hw72Lcz4vBgqGzQpf64e8w0FpWtL5Q7x49PUxDlg1FEN4knjW0NufwAAAAjhBAFUmqxJqEFsmUwISf5cokR32jm+zUPK++eM13mGdjtXCls+VXkY3dCmtyYE0prl3tOVD9bqUu16JKZZguEA2ZkHnMCIYAf1rLg+pZL1JSxOW4vx5xovISjyXkZyZwXmUtyXexVto3t46JpKqdbSo97aiJyMgM39r7l1JnSyqnPHtdi/l467PGtYORyOjHfItpiGfs2FaIwqyUsBrhmL/zzuxP84dRcq9Iy6sTfAfZVShs2DVaLTZjqkPl7JwfuMreIEAa5xPJrXa0CuiCH308pfmo5f2amzmmWnJutIXBotVcuWsmQqflyUKLy17kvAkVxxoTwsM2po5X3JYk77G2ZgTVYcSgE2MpbcnzMHItK+5j3CaL08cV/5GrjyUJILIUyXi2rC+AVOQ0TsEmGp+mDdLTwnvucuRCDVb/GKEmH2GAubNhiL0ZF32G10dQgODnJUPq4wakI2ZuVVjcNpDXGpOqNb7qDDDI20a/90BozIPsGuJ/praBL5ORboFMbE7mEeb0abTJnV4Zidmz5MpdT1qgOexTgILwd2WyFpATldwKVLnZkn0VgJ8sc1kH8bBW6wmp5xbj0EjEBmUsGxFgf+Wbuld6ixkaMOCMPtsbS8syCkNa3OuzyrAUzMD+3YWlnWu3K+9HEDE3AFxRQ06ApKUvFTakfJzUyXmvqWKChVMbL0Dilm/uPTD9idi9mzuH6sjNgtRjtA4Ko+X3FIwdEos5JpdYiZab+LClzDcKsKFUe+vacSgAAADU0EAf6arEmoQWyZTAhJ/l+8JFdhaToSgAGNcKfo8jOh+z7eTewIBuYw6pCyGoUjSILVOsg0gCUtGU7dAEGYhX6Xs00Vg1jS27Nwu6SI1zz4mnf6qz3kp0ofuiOHbX1QdPvOjV5lPXJ7yo56UzmJmzQf0rL4kw5F0CfRJlRm9sj0bbPkgcMzw7I8ft6FGpKFgojNne+ydn+fFWeqZ7N9exfwonev9ux0p0+AdAiPsx0NJ2L2+frnIjB5ApevqJF3pfwaSnWJspLc/mroodJQyuNXd8jDkySgLkKNgRV7dQTF1+R3oaHVM+30Ebok3Bm3rLW9vDdjIWj8SsJYtEHDHxojrtiWtk/QnQkXWg/aK6+/93CXfXUjwNpqtLvmfKEAFBV+Gf6Z6KvSpP5KJmSgdIc7PC6R5eaHThA3w43GJziprpAM5YfFb0dH3Q5KBxPFK/kdYUZWNjYHJrgetlVZKgSmwG55wH57f/FQo0WimEYJg/KeZjl1N5Fd9/A+GDO250WveI5qeLcAOGR1sevsul0q64NZewaJkrObtPNJqBsy0Edpokl8oyudLb9wQI5YmzimSuds+3QnWg2rhaf3qFImCsm4xrXXu7VyryeKIYm8a/M+oiLIT8q95X8nFzTB5Zv+htxyJLxu/elT8BsO4SwDGM9EXBVFr7OqQSMU5p2OUbrBPTp4LYP7WQ9zgyTW/ZtiIrclGadWuRbZ8fka86qvrcccTu7kUxYdramT1ZeIVWXx0uf/CLVFlohHjFJBxaagS21U1mntswh4y5lxLbsULwZhpL22L7nnVL4Y3dCjLQo3tAxF1Avi2KzjBfngIRjEfpxRH46y5Cu0npogzmGaK6mP6yAZBnl9KIBcdpZLgpMgcHd3l+1S9uAGSKf7pF/Z65oUa2yKjrIDFIhOL+jRorl3oeRRNBmytcfdmSand2wJFQzDEONi0KiZHC1MAd0fm/NB9kmzuDtkOg4nL9ejixIwJDbthzrH6SY108UbAm/QvakBjfBQUVPDhHQQrJgacFcvZ7G0k5LaUa9aEwWRkAhAULC4iMGZBuyaRSLlEAs8cC57EN5qtLrQ6urI5gkbvMB2BWLeVlHKj/FoHRKUdtcXvF+vxzmpGRCpGk/k76c+AAAABiEEALTGqxJqEFsmUwISfButVhf+T1K1qAuodM7s/Hs9my13flW5l32FzFUPGcUa2dm3bCXTkjlemGO5gntuC7OsXoDEZP3Eesu9Ju9CGNwJDycKUxNi3+N37XmhT3CodDG+x8sYQtnTuJrque71mHSxoyX9xWTKcP7ub7TkGVHEMHvLGarwCIs9+Nd1CSX82D1YOSC87mX92H5/Fh08+Ja49wuNwAI+SWEclUHKeyGwaUxyL/4sNCmexHVQ4s5I5Xe44wf5pvo7JNN8R80wUXSQdfRtj88Y4z790doFptdbfPH7a7M6OlNRDo8BdSnX8hqvjXyR2wd1311CueIeKhvMfpBOERHzdV96zV57wQQFmRly1DLDnho7Fs4BOFDqKkzKTs/w+puGw5YJbKKqMMCooEoymVWpKZ/v3nLRIEUEeblnq/9pIpFkXisJJ0KRtlHp/i29UEkLWDdCUWwqAkkwqy8yQSLlFW/eGg9hDggUAZRPG7Eib/BaoniR5rITStPI2PdBwLWqAAAABb0EAN9GqxJqEFsmUwISfMbShMlABIjuoMnbXjY2JEwYDBNdISHngLsAAc+fr7GjkuwGSE+WYVDqrgx1GWcFT3BPz0b7jImYSF39LdR7ztgguwABsXWmYII1D2dA56+O/8Y2GuRoPcxSSYfjy24bUD4SQtK5wWjvg0F5WVL6IuIqdeNVWhgEcp7Kv8YWrtu6kvh9WpH1JX7gQe9dxL3Q5vEo2SFTfMoHx+zuaoLFVrzbZDgyZnIpr0Prach7ytk3D64y1zqtUFATrN4rM0/n3O0hSIH8JIaqi7UCnKixq1EHSDpWTNUm7DuG/p5RxujCFI5/6gp1Qfkx5zvD7+rO8sKaYmUqqR4eUSm8NTCw6VE7RZj+FCpVGZmgY122HMuWBdWJ6jh087qS+WhhtEw2ze2KMicKzd6oDBflm/FbGd8mNel63YP/7Uvt9z4UcqD1cpWgNDrGuGbjCEJaX1q619+Muc/wrlb/4MHocbB07TiAAAAG8QQAQnGqxJqEFsmUwI59Gsg1FqI9duzv9WCN6jBQmRgAA0qBdLfvF8Gy2YwttGWzwD/o181vAOEdjSZZ24XGas99mqgGlJz/LL45xg17r21XJraxshwbOB3YcIQV9McoRcdDkja7utAfEwtuFT7MmOpy2Ub/yf9ow+fu0Rvs01jiy6l19SCfE+2kI6XA1/SKGlHr4y//v5cb57HviLJDw9TqmDpa3SQNdnNdoyXBbJpv5cEHmcqFJbinevM6qMWFzaVXeBFuJUCZLw37lAnuS1GZY3jk1LYkizJhyxbc5JHEzoDezJ0Ldq4LVDW1Gi2qu+F2sDGdq7MCvXIQaSIOvxCKMQyIs/Hx3qsotdxzx/PHWmTL8zgY9GAv/WPUE5SlNREmd7on1SZJJzYPha0qNwBKVhODV37pX6WTmRciB8SKEm5kU++owPeV6sDBb9ES15JRLvaLjddHCF5xLdGLvhd/xjyb8LVRstZOv2oKsuzef7dE9K3WHc1YkzrygGqa9Y0fxcs4i5U33ca3HEoB0GSkpiEz6ADe4b/tOe9Wt2YjEQ/TXFaTWElFIbhijGAVtP/Ct4zERIAOOZeF4AAAA3UEAE0RqsSahBbJlMCOfMK+yuIpIAAAGVZu9rH9oXcsbdBtElyKPR9XxgjF/E/NJBxAy6ZM0x0Ltk+MfOCrHz0JRMI2g15cFNV+KrIXSehTmb9El05ODilwg3nswKEtZ2yVerRSsWvNiugYd8Sei1MdsoZNdLkcVah7UOQNX3j3PMHVsaoXNfgeGss1zI8Zbl3PPt3zhEVuWTjb8hlumQZ9U155pnGpN2LDE/7iqR1uCo88Jkob6FDKaFt5owh0QwY8WZG55DT8BITh1Aty95KzetUPeFJzA9wnuYU3AAAAA90GeykUVLDj/I3JPfqjWYP/bWLPgRSSbmkelTI0ZTn+VefGRtxtkfQh4ytxGyqr5Wuq2swg+dEMzvT1eydH3xM8phDEWH1BbZM+wtrPbR5t70MeKsU3zvs3VSqkEK+XexZDUI9qkjeLaVfGscx/sI36p0ORikO1oshlbpBfv72VFKO6aaSl5E74EniOmchXdacS69J9Jjl31HZYjQPNmWssrdu7JzdCiojBWVhp60yHSp6qMBSlcX6IvqBNqKETQ/Z31xCRBajY7bcgvzRwoBsp6iYm6HY4kn9vyIhmc1fTX9ntfcgeWXkMfgAf/+yGZP24ptovaKIEAAAEeQQCqnspFFSw4/+eduZXiwkUvZQ1jc9od2xjOkt3qPlPhtryBDzhdZnHrZtZu5r1Z5dDMG9zbpEFRIxvXC1WPBdo69L9FAZI79FMkLT6PNfngZ89/q4OdboIT7bxbKe6Ddt50szJOIUBKOk684V+YgMsdkup0HF3CKGfAWLs5SYb7I0cHOPCtzqZsXj6D/68QWt5yR2RwS0HkATw6eBW+SFpEqKo+gx2dGtENN6mVr6T4Xi3p7oikb3KS/jZ6wJ58+fkjzBRlV5R+ZDybvQJrH0/2DmkSVFfpn6e2UjaSCozjU++HLLvh8HY5+jOtwKxguxIXqpz3IZUN7SmJK9kcLMaH+4fRg0bHVEBlJsJ4i9vy5VE+mnqcBmMQJvdmpQAAAO5BAFUnspFFSw4/THifOEZlWg25OiLl0WDIsflQybO9cJ+iAUj3vu2u0NQRLPJNi1svDTLVi/Hz4usf2Zdn12DP0eaZa4dC2Ov86QsfAjeMvHLrG3N4A/2UAHbeZboHREVmNZTVqMQxvGq6pv5eoQ9HQFX8jLzxdE+gGKSdmTzrS04XRpaF6t6Av2feVBj0aR4j0ZRWt5pgter/sA0qyb4i7QplD0bshoLZ5bri28uznadGdjX40hAkn9/k8a3d+Pm+x2IPHwHFP6+/KINEorWtInzqeHzESWZoAS8hhTGht3HvaTxudd20p7vi+5dRAAAB7EEAf6eykUVLDj9slgt3OCOo7mRgOUhOy/pxB/XhriWueLCSA57fkczORszASIJPlQy+LPwYsJrybO4tGsMQTJAt5hk5qbeJCOa5w7WaCobTtwELPqcbdbVl3/svsRxzWVe2EqiVR0KnqEWsvaFHXNEtlAxzo9Cw/yVm3gmgxvWyxl+EnkXDb/niZV6ga7aOcRX57fUO9xenWKLFAzCXGuRLDRVKNAeU5E+Lwqb6DWFbrybbbEUoV5//fgfamYH7XOU81BZuVkP+ufxFE9WBKFP8dUb7ROoH8w/Y0ffbXf8KSCy2YMVvcffnPqW8PsD03CQRV9AqUxDNzPddfyZEBp/m7Em5wjZK/c3mpMiUrIXXyBJ7CcfmiCcRpClbvygKXi+NXld+kch/0EFS0q70dw3Zp49jBl36GEwzyyTYa91+kwmM70bFHfS4vEXGiJgZHXHPcyLQ7bkGezxGxt+d2OAA6QU0kaQeRZVUhjtro5e7Ng/wyTvYi6Tf5WO2jyev+Z4U6+aq3mPyG9auSQolS90D/jWoZ7Ge901cIBTcdXBo8KMdO+hF4QAHx6CJMvb0C0A8UTHP48cGR4rl8zzWkDJ4XWFmQ2febvIgzmkumEET9g/QBai2Iydd38P4RPMGjXFJErpm06C2MQ3iywAAAJZBAC0x7KRRUsOP4XWG/1mbmUeK4i6sPwKnaQB4yY8ra8QOpRblqII6nOIW14+2IvDDaC6HcUiMUCkkqoUOT83ki+ZNTnUTuAAeodg9iF8hqifE+0cC7HoMLzCX/iO/b7iW8fEBHIK2rycvppr9O7FAZ50/THTYnRpsS02VteNz/U0Kuak8erJ9eUqSTTOEAbHd/vmVwtEAAACYQQA30eykUVLDj6YJxoIR5ujlLhYjKlvR/9Mh4HQh7ZiGG1zS8z90AHdmCyhlqfL9OZpCkJjYFm1mHm4pKD5pJ2LAX38RY5beetkBPx+x+BiPPXens2Opp+xm9migFXzsN0+6LEYsTYfdNm9q+cuePc8aulB2DWVndPn1Xxuu/hpt/QZNZ+d1J8qt2lmJohCbxZsZ0LbwIMsAAACVQQAQnHspFFSw4/9qFkQKaLgYenBKEK205zzoWQZ/qixVvZ0zyy+Jhp96f+T1hoMmDpx0DRRU4wDwKCzP1f0J/DE4rgteyMtcYzP1m6ZCwCmmn9n+UQhDSYcV0WGrhtMdOLQffepRjSanh8o2RWV+50q0YgyDSTK+b9UpywetKXHkkOY9z+p6GTl/3jA0cxpGbGA5lqEAAAA2QQATRHspFFSw4/+xjLYQWLkWl74G8qgVYd+i0NYsgUtx1E/vztUYv9Ah8FdBHEkiI6KwP7bTAAAARwGe6XRBE/8lzEu89xj7UW8Dt0rnMP7UWy98qdXVwlgW69/p/omeHPyia3O3Tl+h04gULwSHOlE1Sh4wI9vczBNXAxnIrYsbAAAAXQEAqp7pdEET/09ONAqh+xYj0egFygejB4W4OnyzysLQztauG878SM+kGUYUF8rzIM6QI794/YPLi1lQNwBYkOje3Bgnm5Pu/cbeW/jyrjmRdgas8Lop/g1a4+wBCQAAAG8BAFUnul0QRP9TlQymh3OpFp/vcIdWqf/snWKqsn/s7xQRk9HsC/2YzW5bvVID/a47H+JclzIGj3ufYAO65D+/0mg6Eqf6yIG9Z+LBwI1k63luvAmnqc2moFrUmxVZRU7rS6EeJyQQqNLDRTMRNcAAAACpAQB/p7pdEET/U+KtL4SCo15ZQi7fcUpR3ejFAuKk+RDGRJJYmtosXFdUMJeumgTwyxtuq53N7G3mhGvxNdsSgM9CcKFhhrNBKGjneoaNkMoVjXwvNppmHcodZI/yV00Jsl0eDS+ySfZxotJ7330nUZoAbpPa8/1U/2BgHkoTGLhHShOCv3We3LTVbrfFJCZtygEEXMRd6e8C0nQqPooP1PZG9OBHWZ17wAAAAFgBAC0x7pdEET9R9c6NBpMuI3h76getGD9WcvDvuPMlAQ4vZNiTbJBScAqIwSIY+WYU1BQQb4mBPNlZPkmhD37wzhsz992JXmrvJzfq3vo1fpNPOQOmV5KAAAAARAEAN9Hul0QRP6UToWDjtS3Re/1okTwJfrPcVupbxBgspEEbtzDKBm3zvT9NDlz3avv5Utnv7GVVCkgL/JgKflkTLNXcAAAARQEAEJx7pdEET08+NtYcl05j4qUJZfDwKRht7UIwLPG/HIzQydyrP8jM8io82zZruf5SbYuctOhatULSrVAtJsk0utupyAAAACQBABNEe6XRBE+6jRwXVlJxQLBXbJ5rlb2Ry+p16Z9lJWw4HWAAAADPAZ7rakPPI9n23ATdAMurrbwrlb4Qu5JyRythh3wJn2W4ewKOOoxTG7j73v0XquYt2vDc5x0njcrP3YTlPm52kiOh1E9eVh+QHlghAOrWwpRPMWQ9BuTOe+3Sfkxe70l7czvIbWvKnza+lH+9XIzKNpifyNscf5JAEXoM3TOdXGkg4Dbs+lIMHdNaSs58AeaJTKg4gUTCtBMeDA/1faQPqBRST9hbGABqL3ryHy7ZS8t9aabLaue52+5iHSnWNnwDDvkp5Stt3JtorK95IdwOAAAAtAEAqp7rakPPS1GUfHnAK7+otVn/UI/VpJF1M3dW+UuZxyAxr2ZKYrfvO+N5dFVhXsKeCyJ9HtgfGOp46O1xSjOa7rKFad8fWJC0O4ZDZN/9m1fdl8/4LEeN1/SsHc8tskzT30LL1usnCJw1nxh2PqhsSgQv2r2JWuKTFUbizlVZhP7sQbNZkYEjrSzUiibvzsPFvqluUm39xuVZSoZggD0YMYqFBYoIjuHRxFgxxUzIw+OMmAAAAJABAFUnutqQ8/9NtIZM9EF7sm/PNooiV2ADooldPTGzUm5/M3TG2SulLTUH82jMcFv/dynopDZPJyX4aR706kTZ8v6n88Kde3APLUVspb96CHDkzuz3EmjowZzzOACJKTVGZ55nYLp+QFCnrvrAVm4tqQhjOpvjqkCdfdFxm5At+nEK7hI4ULKaU6A7ARoGwsAAAAD3AQB/p7rakPP/T52EGtqDU0c3Qsu+1DcjRGVDg/o/IHs5H0tPzEbSmkU7oUoCQrQ1QE8C+9sMCxbXp2yXih236rJ4i9QiB+G5ngtx3QaH9IG8DY5NhI4jZEUNa/r9F/I+aFBn+sT60JLZqHZSTS81EowfC20G7BUYW8UDplWUZw5xtk2ueGCkP/o/9bQdyxhlv5E2xE7h2sZ8110CExSveL5zOtO2NVkfXWxUQxrppoKFtbAzZGSOY8tZxACRepYDjBwINaMcDt89Y04r1ulPe3GT+1yiBY2I8lTolf9ImjJg8PJZMCaul2FoqRi2kwQxKRoDOPf0iAAAAFgBAC0x7rakPP9NskessIkKyCacYi3DRNxegwjHmSxv1s5E+I2qzBSkxqqa3FJIz7wUkQEs9jhQ7AL7ahyHwYOeaOF+5dD7Yz96894weqXfPcxBicTQpKqAAAAAZAEAN9HutqQ8/6AWFaxBzgWNEU8GL2cRtdBC8xijNpegW/FAwfc3T1YGjlmoU+LMYRmWbHBgFe7IdN321xt7SnIfbNTAfk+1nNz4KCb7lNprGKiE7MI0wqHYOBBVCKMEt77vvGoAAACuAQAQnHutqQ8/TT35rBqdagV+F8H4fBVpVqc3GEpHOXetAAeYIo3+MM6hh9WjdOZ/SDbY2Wos7YOJEFV5LlxC8uG9mxh/0OjBXzQDlEKJiL6yF8UGLigRqCQnyNSpmaxKUHwmH7yXBE2XbLBTi2HTT8a3JT/zA09etcBKQXA94dXF/4uqFJ44Y9ArCCH0gymYxSPSYn4QvMIR90wnPr9J+qdZEPyteNViLMh8QomAAAAAUgEAE0R7rakPPy65c8aeu4AdZgdgKlN/nBVfsPJdkWxNT9RDLEu2sQHpuiJIO+oqVJ0wHgUyVX6VCh6qCloBOQT8iXWWntXkmeZ2RSHie/PGC2gAAAUTQZrwSahBbJlMCMf/AWBiQJFaAjiKdjaFWcwCxLh00f1HwAuBjOji8B+607s98D3xB7bMN21t8YC+lQA27XwX8oq/yUZbF+gtuHiaR9CFE90xl0w4nm3hsZd/NWhsjiA6XboD5yYcP3uMtab/w/hA2M1ZAPvwTDFJG7gOkPUJz2owypzaCAjBgtHfHwU6lpTvCrz6TfhiDEtL/w3pzgxoT3ArrNoHPyTHtEEeG3oSLdPKVX2Nb324+5HBMVpRT193cRonvxUPADE7c/7FwmheyG17+fhzzbEg0AtqoDwZziY5864qJfkmr5BK8kpvefq37pZPBBNW/MAriKqiLVv7USvvo5EAfR/xRvf5wE3gDZH0HyTTb/glqWyo1+jiBhlRD4wGoyeQo7hUmUiCndZAoDi+mHG+OjM1zC4FrWD4uSlkVMvqZkkysdHiQo/fOE5it1bj8QYuOreKeBiaDS+on7SJghZvBDjp66LiZb7/1RdpKMfBZdkBXODf0Pl3qLrhVm1okF+x85pmOa3IAw4+GzjeQvHXCSeQc6e872beisEYNljG9pVzlIKc2lwzGjx2yYEHTo0/C58eSRrvN0LI6p2v2GjOBf7cJni39pGsXrw1bM1ymDC9sw8VVlgI/45M+SeEj8I9PQkzmhozJWmNBZBtVGRMokjo8qAvklOTOqzHnf554FAwMbFb/tljJYD4adfeC/Tti5A8XPBOM2sRlj6NwYBVepvHOe7yEp55egq6a62sTo+NRUXYmqWBOe6Op2moWTQ/R8tpw053JfC+xYTXuSDiGTawd3W+6kDSykDP3jqPgg8wnQ5M2EObGZjx1s3xbjOgwEd5ppWL35bn3GrndXfZnUdFZJt1Qt/6TpL10JV6yJ0oXIOu9CBx1ZDC2dpnFMrJd5cAVciUgOqHxmy+Ov+N2gv5adGRX6OmwbDhaEIv3bco5hvHRnM+mgJ+DQQEdTuX4jRTCyuBFJUgdCwwFHaejurkHrEjNA5AFdnE8uI/WOVDHfRsJDMaqRGmW15NvCr1LSLiqeARvPMs31qZoC1Jg7gnFWo6PmHEoAO+bkm8gU1Po+lpdGpp3ElzUdXdLsK4MLC75LO8NzJ5zK6KNy61Xdh160h31uzedvvGKlbJELdk5Tfn8g+/3y1mwuvdvjvKjCxXXahhur0VlaQg5ZUqcj7N7x+Ev6IEXBjJ6ALOmUvgOR+mKWfC/0mZtbCVu6rpTevzVGJIJsCUwPUC1+gG46tWhQeobFsUIic0usv3d5AmcYj3MV/0TBAcgHJeUoHgV3AD+rtMPvHLmXd65fQcyr97OP+W236r9TZxOL8J9/4tUTTstVwQqqBkUtiWmbzQYZlmgoSFnr8a3ybCVw3+2u/AQJ5p9MQRDY/OL2Rgc5UWdAfdJ5YY89ii9WbkDkG9IpwNQPPO0kwcGx0lHSOeLGRLVV0nWrwPfwuqtz2lRD4k0qSRoA6+ikSND1p9XptMAnywleCQXoBHJTXKZk2ASlAWS8OF9VplQFESF63XH6u1Z+7w4bbfnBxU6k/R3JE8lfLikLr4iPA41Y4iKuIVjK1pmMv7UwUI1mq25TynPKe+RbhrvADC5lSh6VClIkfjmU+VL4Bg0WaSBS1GDfPQf0UWNKgY/3JntfWqVqze08Zb3i1TAhNRmc5vaWC3bcBRAhFVi0szz8pM1XndqjWvXF4Tb+3KWJnzKbYh/hkZpZEfyeOMVRAvAPyWewVjAAAD1EEAqprwSahBbJlMCMf/AumrVjp7IJ3IXNc2Iw4d9P19fzBih0uj5mrw4IgErRcAEW2pwnIVEWMC1hCg1LVSodLXF87knPg9LRDvpueAAAY/ERjcO1AQypOoLYTYMpv0iDMdxuKiEKX7ZikiDBeaCbdep+ty2Lel7AHYJcXgBwZ4vIKMuqY0ytydiSyfQoTYLPMt8VIFl16bq6DXtIMCO/7n8m6wxwQ3Ofu1hXquc/osLBIxMNbJ1iDU+JMtSr3HFpCrFrpsJQV6X85/zPECvM0lN10e4I2x10B9EdvPkrB3KyU3jAkeE9IAkkq4zRxZbwXR8l1P0S2YxcRoWNdUO5yZU8HqsfW7uEbr8nMfquw76rPYpmY8ZClh/l3jr+cFQ3LZeIyjudFYnU7HqTkizTubcU41wlFC4Hmfmg54EyqNfOyr6bsjhetM+5UkhTf1N5Dr4ByVkvHsPkAH+gpDuKkdDx/TGP2ZtjZWUyDYJWO+eYG8d5gfyDTb1vrpkIHAR8J1BSMfiTS74jO1TL1q9j+1J4RzJe5xZj7sjNb0pTUHiSJdwxN+BVSZwPo0arykf0rmIqU7TWdnz9Z9JKvbxKWedJQ/AmXnS/bPiwwv0uxylGeSRTGLvsTE6/VpUZXy+7VxArjg3P9Pg/HtL2ZismKeDWQfBoXcyiQgo8n+o3PrkxhOy5uu0QP0uLFdjAFRoJrhjlDrMVGjoqDGAtqWaGn7FqdlZITLLVk/q/uLh+6ixcnG8mUjZjwY75SMCt5oSUqVZcYQKs6b4zik55kN4vx28BLAOtAHQl8KR895wePvCn/FFrbejaRrEK1wDqfSip4PSmPMwfHUtcYHeZrQkrkU+WiWVjTsbtYioRvBs2i8N/r7Zc9ZtfTTp2tweTGVCDojDftpriHu77lxXUYtGkxmUtQFiPe3THsMgm84do/KeovD6LpvBIkEnbl3RbNVtjHqseF+avkjiti2TUTD35WUDnIDeOeHfe15WgHhSyl8+PL4MFiaiwMXxLR+TggrNuB8Z9NbTMcpbmUPZc1rdWbm2EhXef/ZOogSJ7Y4tXIOXHuG2A1EuurQebYcQji2JMldsO+wvCBs5l9/dYRPbJH65G9NQix+Ht+X11ZYv14qH2xoWX7ldWCv/nCg2wzWMn0YITAbHV1gWTOz4ytykQvqWv7XcQU5izs/xCx/osRNfuC3UtSy3zWSTRkdMi/ofADQsTDz6+W5EPvy9LBF/NVyecJdmafBIB/OnTjUCg3g4tnw4ONZz3FTgmBZ1knyO5FcJihrMBNF+hynbxe4kDp+AKvBAAADJkEAVSa8EmoQWyZTAjn/A7mK2KewbtnMktH+fAMrBjwq2a5QHvagQzoL7dsQjXNseVvycRcw9+EMLnxXkeB0rjYuv//0ba8KW6b08SUPYfebnBiNVPKQW9HsfhX/gcUAlf3T/I7c744EJPr0UDrbCKs7SoN695+fuicvyUn4dSQKzNzew0HHifr9BDeAlHNm7NNCAbvsuJ6YrVdqEHZjr/Z4A/w3i+PoVKGnSZVkgzSJn78WixyWk/PGF4wKPjuibMEs+4hO8Rlf+y7lMxkJ1pvlCXPE3PWbSRdlg138kOvhyOUFyZnwXHwd+oNF+B/MuSytiW/2IiYRm4A4l1o/KTsyCCPvW3ZeKt5JVzP/kTu9HqWkadjmP9g6SemogXEeN3gCO8gYxwVpxO9mFuAJTNE+cYLnMD+yWEKlt8Ov8NAvtqAnvM6oeKQL5izifbLvoBLcxKsoxoFNEcVSMTEf0993P5rK8id67IFBsfSwvtS4L1LoSe71O65fx3J2oXinlkl7+Bv33jCKw3X3qeCF6xxdBlpjpeV6GdjY1neFUTHxIuki2QxHbk9tb77TNohKgzsABWZDbi9jhqXEvqCfti3hCgzV8JGLOf07U6Mh+NFnMYuy7FGbeRHbNBrgitSeUQ6dEAVz4Uu8OOXuPJ6lAyUhN5t1FddnRebkWjHcPrJBb+6T6kpFPdlrXkFUY6DP3aPYJU9aJdMGGotsaGT+wCPzJOEk0iUWGkiBDQ57huniR3V4yPbagRFEM0XoZShcEbVIDza9E1yX3h2BNHemwGGhIfUETWf4Mrs8s/vuYXrTqlh9rTXx1FrcRWeCEmggYqi1AI65nd26qNLY3WDS+xY40IQrO6AVuDpBgniMez+JtBPC6yAShC8bVJrcY01d3mcqH60XRlyNo2TfrUuGcRsyz7UImJd7aRRdLPD8ux5TR3EpnGLtR/a3ejkr1hGNm27wC1S281+5pjvxlLQGaPHjMHlnrltJVrVUd45/X8OP4Emz3rkSb7LMWMwb1cpLMz7fJkFg5Pxub72FYBU/6S2GQj7Ex+j0Y6qBDa92fQ2w0dlLrIwtAAAGOkEAf6a8EmoQWyZTAjn/HBi9Y7RCgVsnuqB+6PKQGsTf2KhragBiGpwPFtY5fuWRhb5MtjYEiDHOuOVqiHU3WoZ9WdzKD/0EgNmSji51BU+Oxn+RxQR8/F8aCGC7fphVP+SS47qDF9sEqPG8QqJ/A5Fqov3EkGgt7xl+lk0YXLPt+qKYYDm+FZaCZWkpX/5yVn2IkLVCHanxMb2jCKibMHOZGFWLLDYNRO2BZBOk19Y0raSxM7220hNWusMJ6EqXQMPJdLH1Yx9CSJuG5mA1D2MaV+i2dgksITZfNwH9gPkVJ3nVwiezB0pnW8+tle8D6aetVfBj7PVAK1LiJ+ppjCBTnDh+1raD3N3E3AXM3hpM3Rv2HfRMVTv0U7ET0ouYer5R0DDYUYLoI7MS7YfwH3jXXFkJFhrRWRR423pN1t7PSo2BnNZoEmO1HTk8E7tiQQLUKzc4Cf1w5HLtXvRr8Lyglav3K0G/aPYh6ZGjn2BznoxwlRVfYzrzSm5nu8lzoRiRSLAsvi/zSS8YBwYRPf1W4+vZeP6lcP54fswlzEKFSZeYbA3pqRK2UkR5Wypp71fZ66iy+TOpIvKudIkbVHSDoQ5ALbQlPw8uQnKOR5Qg0xrLfdFEpKFEGBBC0jA//wgGAs2RNCKUpx7YYAg/qesiQGmZ/KPwlugk3ipF8uQHRDe5JYkG6xxkLCMXf3c8ANBsDH+HM3RfFfB3mStSup9b7SnwNsSXo7+9rs9qVcTETlhmi4QEZgYEpQ1e00yKgHEvkuuNAu9GHE6UySJ8B4zvoKE1RQlHZMsB/Rk1kQAR0t5Cxbva/pBV3HKhzFmGhqOteecNYRGuyagCCyE0GDt9qXJMxgOdB2nSV7Z91DG622UEyAawj66bWW6grbFlM8kz3LeKUdl+JFTL/yM+agUYRhszEZYc75V2Hz2aRdI4hJCfD8bh3sOrqHeaqN1MonL67pVwWheDiinzBN8sgQ6o9ghmWmcVZlL/NdffvHX6zX8fHH6Pqw61/VBlMAhnLHpd69ywpR2V+qO/nJSzesxGLpj4mKYjKH39WALGzUR1nAsh/xZ8TGij6+XT0wSZTncNgUtoFjeKQytZmpyAVpIusPMvRzAlzN4dpSG9ZqFRDG6gxDn8M48K7GlTaEOAO4KcHxkOksRUZGTd9N8+jSar1Iws6uXw+dyo96wkrsCP75ZdyrgOSsPJHqbk38da9iAXf+1VLkGddhXayFRJ2xMCQp62/kGrtmTGRpA0hyJY9KUf0gpcEsxvg1183ejsJ2KduCjmsa5Lt+cS17zqs/2km3erTQ5s45uHW5fbAq3TtDfah1/OHMm3qPgwNBUAgtEjTtZIFUeVX4DyGOu2r6CbsLe/ksDCGtj88dMsxBIJKvpv6ytKxkMXBVKvS+5aQDJczHr8OPi/jW45WYei1A2s1HG7XxYC9ajZnnD7F9qHpXKECt7pL90hLb+cBUsjO2FlLaONu1GXYSaZbBhchLXSF/jc3Ps2YL22MN0xdlywnstTCh/5YPOSBHliH/QkF5byTJquEstbaHA/HjYa0a2c4RBeev03oirAHpP+La5VHZi/cgr7lXcFnxittUCQn65XBiah4hculXLMBZJPYVTRm3CTpAd7UUBrIjmnhYLFhkqqioD8XepF/OxMenr46VuetgpdoHl+g62vt3nPAY7f68i/Wqef1GGl3TqqF3EgghDWi20gLRA72m7T1SPgPW1vHFoj/Jb5wZJsBqhSVdbt7NWHfH2Y4W/5VUfFsCEf1WJ+NhchYwZ2PLcmHPzWF3lX7Cus+pIgM9hFG+r2fxlaAQ1uFSJCj5quHgzcVzTb8c2KYzdmJvbFlhzm5D8p4u/HvkRMJzLfx21Kqh25rJBhC08jwZy+JO23OAgxJs/ufGkYNbaQ2jgDTJkv9WmrDtzpG5bAdNCx3jUqOXbU2B2gHRWaTASQrem0NftND0dIeCfcwA5wuMmq4mbK5n7zfwbx72YUpXJl7MV3MrpLoF3LYV+UPkaABNB/nX6hADnXEUJJ+Wh5Wh7sryIcedJR4hWbXmTHbJledk7ih5QQuC7/NpxHQ3KkQMOv6OfjKDn9W0l+yszTgXorHtaYb3/Q2vtA2XsNbp2VxPEAAAKbQQAtMa8EmoQWyZTAjn8DoGCQAR0cBBlAdZB8Pr/6g6rvjMzr7lKGQhuwXTkMVvicvTizaLUKnFAR9kCYY/qWDuKdIrfojXfdq5NUZ6/IzD0Hi0FZL3ixHTI3Lh+uBcdHkH5qPTxE0ISu6oDOROdojxMU5aHqY7bFgAndOARJ9udtTi+C7RE2clS13px1N5txsAIWuh+JpyJttTNwkdpbIuibBz/ob85zM553vDda2DdyYfyTrgYXSzmEKW9jJKcS3bnoEadCd4H6BKCj4eszxxYviYp4W2LMjh9YucFDIADYo2CDdkGR8hTrJUB6qOl7JC8RF/yscmti7rTIBWmmQ6kOJM60s/Wpgzj7o5HyFWLMnKsBO76Vyom9i5AM79qY1ngTflPj+aCXyvIjkBX26RQtrW485HrdCGJOaalwDJ6LlwhrkhpKxBKQx3e/GOIr+l4LdP66yZL9mvxllhhnhyqZvbR7lXSUOFChH4anCrjuaGVDMOogXziYUdudYYb7KxGMM8F3VmdhIMKcoZRoC/+nhKHDZH4pyQErOi3yse1PxIeE4rFotSDcf0WblLrRjuQK5AmTRLmOTyvfe5LsD14NolksTKTw/4ZuastQUgCnleM505gFsZt2D1kF8eySj2L9rAtMX/rTyezOzteNp5SqVkRaDm3PjQtyqCqUFLiBpZAmcIor3bHkzKfCHef1U8yPyreKwSsfBbV9hFCKqBpMl9bIEEXEcAf2zqAjAyb8Gw3pJ5I1qkrJM2FuHIGmGla1q2C/MRCYswiKmgsbZHGxA8echqAVe7Q8aTIhWdYKFxT8sJgnmXUkpndrZ+6YCi+HP8nFQGOH9hFKhiD42knSw+4Ox5Wr96a3xUQegf30vEIOMdzaW58soQAAAq5BADfRrwSahBbJlMCOfwlEs1zTWExcgBf9GbCnAmACxYpukdsfY64zPISCMl/rERVxTGfTOhQmgoIxnSmCLy4yg1NhkyaTNVF9MMuS06AABM8HkVD23TOO2gszKgpnNAFF2f0iAWGJSL8ZutF7K5hWE3RiEMhVoO8PZYfguAAXeJQXVcN38WFQ17s1nPYKpuSOyaXR5Gyh+HDU+akGFLYMs64x5Y6b435XXgaTIhEIAsFcQA1lx+UaOWQSR5FA/kJ5BM7s9GdbCwqpJZZpilpwyBICtzVRGlRuDCK5oCF92Fs2BGKGXlsIsV8PadfLq5LDjhiF2Yr4bCu7rg0NPiY4db/XGdrbHlH4sJ4zo5NlkthAxVDmNOfkfKxV5xXEgdtHTX1uBrKyscUAHxj+TYhVxvl9MkCANIK/80YgfywI0k1Svwaius9EwTSPeHm4XYNcyCc8yHn3BogSDUvAZQOoXd5MdbF7vwewt/qWNRj3++aP4tJ3Idz3OxpdgLIgP8bARV7SSLfXqNrxqVQDqBWaNd7EUajP9ck8Hvbh0H6QeM/aLOAIy0UNcv46hRIz9qzMIp2wxksKpN8/TS8YoFUlKsTxCXtR29u+h2SVXPyEexHRdPxT8oS17RRVGB9jW6CUuVyoGSi9/33hpPceqqCuIA0SrFtwk9yDhvGnkdwcuA1VV4+zHSFNjFdv4FVaiEKHiEOg0vYfAU/HsYllyznK5dWthLYTf13p2p5b/y+tBfk7mvIO2rOpXvoELs3KAL0r3hA1MVHXKETKP4j+ye4xIXxAFyrGJDUPqIvUQztyxsJO7kX+EGF1QWn07Cw7gyABWEHGtrK8FwvBqoSyYHrioQ5anMk9kFfpOEluipdNh++5A1xBYTY5uS2cBChoDTNadel74uPf8zZlncWt8QAAAtdBABCca8EmoQWyZTAinyzol3jy8AZjRtoeGao3Rwahc3mNAk0DW6UgckNRNIt7h7Fb0y1j8WqfveQi8O+GmlCeE+sQw5m5dGBWbkAlUGVZqy/1bfjpAAADAAkhLrOxf6hSG/Hbd5LlNYEGuCoGHIrZrUeGSsU77GSHhAa3yKn7bpX9wEm+hMwELhqaaSEs0rfijNEv9ZbioxvZ1Q8hd9Kl/86oxMvZPbBHlKRgvsYTH32COlZCf5+cB1I5w2d5DL3f2Y9/negqwEjayshq3uPUSntc5/s4cDoPcOVyA4DSQry4HR5UUx0J0wtmr6I/Ji+XQf8v9rxOe5/QU5HR58H/sU6KEn7ATs7HM9oeg2nVLBPG3Q+/+IaZ085wH1qo144Ou8zNO99OvG9IpMVyNcpy2NJo3L5lazri5hXeFSpg78TBniPSwiRlG2RDDEzVeqM1lTFUYRt9AR6eVBRuZbJ6Wt5b5t+opnNuUNmVAcrT1QAFlph3QUre6WY0rYNqg85aGHVMaqt3aqayz+qPayJO2zrX39yHTedw+3qvq+yVzqD+iSQDFOYVKqA7jNXrTYRkuEvahQBbuAsWCenQTWEMySB7t53DcLx84otLBByisnN+3vlW0yQmZnAfniV75CYvMMPh7rBvG7+DfU4rbBgC9xj2AlE5cxH2aUfhWdSpJtTRacrS3x9TnIMjKmmTokvr1leRoU53j8y8FePs+E3y/hBr5NkcyVC/vAn0NBpCuvv2woT5CCsTd0AHoURPoquuoKNqypOGYSYE1eDsYIIeq9KSpAMmDFHh0NJ+0RJnrQ+6py484f4tbCwpnqsU//rC5TtJ+hWmVIYv35ezS/XitFGHCxQ9W+Wa4yL6vE87YO/fRza/qOWgD/GdyrHvpjSmtVXjhV9eywx2xiN00NkMntcfpyxlD8D+lOKH5xNFO8emSGYzM/hpwl+5wD0PxaiMFTAXnX89AAABmUEAE0RrwSahBbJlMCKfLOm8+s1p/VvwBMjVVlCxOnLZu/pKHPwoAsiDeO/xIowrpopnTMiSzjJwIODngA8J7cZQ0/Ger1AhHRUs2jKIXEurOCxCyHVL3aLyg1XVl7wgukGz2fB/ATkeNfwuZWxcHtVE//riarySkWVRIjgVfDb3bbKStcVLeoQ8E8VFaPR48CpnWmCwEfja9QLEGjNWeqirCpjTafTjA4V3Yv5KGKLfxIInW+bvylZF1WDDAeIMUq64ZSyPo/3l4/t0+ORry9L9LlO0QpwRcMkRFCGWE3b/d1mso2W+BQEvsrXv7FTz+y86GkkWAJQ69gcOsOXge8H96jpwAHrOcECi3CiGdIDllFiRUXLneikxCELBIrFN9D6oynQR2sQtDDSA1kNKkhcDfGJw/dr5Jm31hxKsDq/3VVsC3CYOB8mORHF3NGvYyZNW0HlMJrjvdKYPZ21uABmb2UMXwma+GH1t1kGR72MnpoM04eXD2CijmQYCtUHREyzWgtlVQYsaEGnFRZBGK57pZuaiuAIprSEAAAIsQZ8ORRUsLP8ed/tqp4T5j7J5TJjJA1R6FAOplgb8EGhKVzdInkD44OA6wA40ZuwUlG7aXMpVN8Gd20BM9oCcON5dLQ9AZeAu0JJGdFgiH3jq0JIQJ+Si14RIksXAieEpN9HfJo05D6Bce5Mj0UR8EV9oYGRCIZfilZqvIRGSJa5mSrvADfR0Re6+VyaZ8j/I9AkPFEa2q90GvdgXJG0avA/QmALQ+RUWuWlcNkakdPyphrMXeViquc27grjYmxNcNZb613tVPhbXPWzy7b6L4EJ8lLOMxcVmhIAQ1OzeSW794uvVnbpwqsXDU98woC36EkoOGQ2OA8BzwQCIeJDbedpcTvQEse1vftqJn48vOW707z8uKi23aI62nXQ29MitXXWySQ2hpyoBQmn2GvYWfO+X/l/ej/+2TgHyJYJCrV2tQ15xIGHXCZLAXc9s/DNhaqSHRCYvBCvYiYrFbULkRRQqlXF1n6c4SR9SguDh0XhowP1NYRmUHfHuSGjsNLPwLCoph+A8vEqt8S3bOnnnhuP0qtEN//pB0ai9hesMZipwDRhI2evnXNIh84Y8LwJ4WtJYqQl/JSfl9uBUyaqFF6HHEKqo0BCBHO0c8p29ZFwUshjDrtiuKLjIuOBOE+PCpgbiE7ujxFKpsYYVDFoYevS+X+igzAeHgt5AQr/CbMSU+6meD3A3jZH7i94YJz/uEanp6pID4ziNILyI03iS81U2nI/U52jegN2REQAAAdBBAKqfDkUVLCz/PYqLgjO0DD7rxYxjHx6NzPuNgyf5REIOQRA1nODcB2lYNpLZZPlkUAJUSeKi6ijgducE4vmZYGhwlzGv9kXqH7DHUHNSmAp+mQpAtMCHl6FEJFVlr1zZlyihNPynT+FUQjSzh8y83r/YEEIQDXO2VINlUPWX6xGRjdnWL9A4/7WOZ9B2cO7RI9nnBiFr3YnqwY49YZBD5VqDrmBSHkE//QPo4JkTny3uvveKtXs7NpUQOhYmPjCykaVbQQ4qstzPImg9bjWN2TDanUUct+flhmJdy8yRoAZ8XFsoR+KxCXnIVxoJwFcZyQKdNa7QgDRnTg1s8yX8xfJk7NP6DNbazyOLWwu6Nr8R2Pk49EZ9N425JFLEzE96EUpRl5D3Zt6KYGRLJxFQ2WM7LvF92idG79Gh81N/8dcdYA07DLm3Or/Repo5n3cNYE2ezBAOWgw2saXL7QLwiqSxarI70W7RYTDZn5qsrCmx0vYo2E+DlcL/7fPgnMoju+CIuzZSFKFOg2GGgN6U31LfyAlqYuuZc9HhZWqMEcCxkaf/aIYhtrGk08ILRTdMZ95nw4Kf9CFXTdBprakKKTqH3FTVpv2G+9BcIVXywQAAAfVBAFUnw5FFSwo/PTQvmEIinSTCqXehyY22AK1x3WbAfrrTujf3DzS4ish7heE2xlggqCvBQE+SaFIQU2Tjeafv6YBHOCdkc1R8LpvEEbuH+sM/Yt0nce75nhw9k2AY3O2bezpnAgutmaXkWfrAidkP6X8XwTZPD0OtK+9Mnnlrr6ypotDARcTd9z9geJ9zjCdyc6q3IZlKButEEEzVRLSKgWhY13Z34ksDbZP9WL7q3sUGoZPVMZDE+lkTx4m7rc8Y1YCjwvcGUta9WibKW67Sohae1jHDDT4QJb//sc+1BnjtgTox5Rwh9HSl0Kw2aZPmPNNcegFlQZfjd7Sj62CYfp5I2yKZlwFGQDpHQjF6cF8HUjXtumF5jaJSG+IC/3ntUOBU6oT1Bd/GsC0zm4lWXkvQuS+deJjV4hZhIpc1Y3TMQjsk5S9ozTElYuswMd4DX0uFoC/BwPutloQI2YI+uSjgbohcbGlJNtCCrbS+B1iD5jc5KM9EWKSqYeJ6HGh/jgOBAXbZXvYeZQOvsN+CMSkj8+v0D2RBlsNDEVVE3RgkTUguAPRf/pTliuyRMiw+dNrFKvjpSQC8m9ru0DqiKazMUsBD0WbHaUkvYUsrlrVYubX5xc0EsHWU7kpGb/7zVJNuOwS5Jg6IUlg4k2jvREKkT8EAAAQ4QQB/p8ORRUsKP2EPFAFjjC6Q6bnBbZSM1D9/+oNV+WrjYdnkUtBDYMzLqa2vWenM6hBWxHbhnnhZV8HTncL7PsRr0RG/4XLjpskIJ7beaUxYPhSxchCZumJDPm6hUhgeyyAms/bGG0imhsFzsGuht5CMbA+/tMDQJTO8XVqWTMojHC/l/MhpY2wLSjDJWl/QOgGn4gfWXRdRV4ApTKmofj8INUm3w/71cu+YH4an+qGBwCKBcpmIs8Br7MnPxfX0t2cw3qF9a6oGNlbc02+Q07Kqu5shd+icJF9EDe5xvv7PUByDUExhhwM1x8hBsAT4T+r74fn4F8bF3hbtnkKyp8H4tlezcMPLcpE+nNDSzV4v3pqIBACn5wMRrrkLaD+khVBcLS278ncBlzagXMAemu1GlDMFYuqjZk8/yUquGwvXifsbFRolgzXmW+cfmebhm53Wz32tcxbBfkUgWKtXlAmUdiEy5HG78fNzSyYoOO6Z/ghmz21Q8261eZkCbgZRHEgopAsvqvwtyDs/jsoVHz3So9l56Be4MZGa9eHi0XOCQEJs8uXYRCfXgZmSo6zxi82lIHBYD1UOfk/h7A7rgSiLTxyN0dnEtcAbQ05Oj/M8jXE6ErRHhtZZuymCS6NvdV3LuJkFZ2iVpKmyqiYVIvQlMuMz672iqMOLdyoD9lpQKXNmOZPtnUvOmIyZe9SIDu3cZgJjpmZV/nMMN8buOgCVckVeS5PyJk/zBb0JnrkBEmZGPl+wgn2Vhw2JMaCi4Cgjl4bxtrQVcMuTzGcuny7YGgwkwsiHN6YGhv9e4aXK0IPK1/qg+MYZKK4SbWmsBo8pKo2VboeGhnRVllZjWVpM3ioTk6/+IskWTG/biFiVLvuYJ0psOwLXtfYlUpxLQ4V3ZhdaA77SQVSE7mfCIn5mQZwxqe7Vpm1ZsALyxc3UM9ZboQg7Y9aQK86OimmGbAb+gOiuZhp5E6z8gpnLlF3xPK+hNqfei0X+278xnzH1R6xxu1OxsS05uPo843FR2BlBRzsF0Dq+1W7rb06JcnEUmMLhx/1DIacgXSGN5OVjEo1pRkbHs6A0pVrMl2RHfJjbTZjgeNpYUn0gwBeQA5m8JFU/Cml7c5drKcztDN4C1IC4HYI65YqSz11z8kNvY/Hmws4q7m0TA0/TglO13IV0qix+QXaFMcLFUar1JCy0nRFb8CzYOfcabCM60ypNbHOXyQifWE8TLUEVo91yWtMrYlo4J1kPSgGn6EEAurYE/ldWDSqmVIq88ysgPsDaXAwRD0fjRtp+NIT9bn6wMiFucTP+q99z4ZUxDbxrES0PyY8dsBOTqv7lu/r9eJ/Bh0183NvCzeLgPWqHK6AYMXhcr1lE1cPYopTVrmEU4bSgDQ10MtNLHDLf0tkKmbIhrxRGd91OWmPEcWARHVvucDlzBJlKRwE7AAABREEALTHw5FFSwo/YlsuR22wlcXFhKacCXz9P3JZEuPUU4n5i6gt081/9cfNcNv3uYGoIYiwAV3aDd+VBNmMz+h5esgfdJbDYim5VuBWp2/oWKdlzhHa59kPwYIRTUKB6b2oV9T5jg9npIf8cOI+rs2XK1J3cJ2egRh7eILPIdkv/+3vWYjoJOIqouS63KdXVDq5sMbviOm/ongOmxW0dQPzzaDS92hAXQWc0DQGD8knGwxT+Z4RV/B7v5pr+su2rw9zYylK2iKZHyhAoJwk1jy+oh4TNJdqhEu3HGbmEm9ushRj4OPdYmRertrEIrGF/6fi9XBSntxypUAlAdafCNUHt8G9LnvSBUgry2yYBVh1h6PqGnB7g4o0qZ5/2HLf0qhBbA2y519Y3RJuUTqdk+KifrxuBbcfTEqSaIBms34/f8AaK4QAAAUZBADfR8ORRUsKPh4OTBucVSgPX6amIASt2jXkRAun2BPcuvQ8Y5lkYDHxrxtiDNIu2ihDEWdVmgLkDNdi02Hkl6FZdbIT4DAi5dr7pazwmE/5foN1tF54WUyQDxE8rUAQ1QaLlhNfiK6hfNEuZbApPE002uCeTQqY1HOSwXa+JZDkHEzHNGg7pXXKyh6L4HQ96/daJ0UK6Jgb00YixjA3IKof725kJ6wlvJUd6JvRp3n4b5QOY3gcuDHnrJbC27igbv9JEdHIyadL0yDGrRxhaDRQRdILiNwe5mV0wtYVk8ppY+8TCXvvQRr4CnguUJFp4CwPgRvSgtcAjpVdZm0Zk1IC9nqGQlVFt5wSILm9nzEBBBRBzSrINHawiHjixY3lXA7FdJxQHlbCbRrHH9NX7jvTEOMg0JLbMF3lBr55rrNE0BsyjUQAAARdBABCcfDkUVLCj/5l0R63fnB/2PAm9jrn+ql7ZOUZeJPh8aGrotG33hCctte5CQe4CWARFM76p3W3cN2wjPJMd/ADVoyWy46tbDT2/WOR2eOti0CDYcEoFx3OjjpTXkSUm7lkiBkg7rBC6wwCcRS8lqtREEYrjdMv/gZHjx/i07pVWAJwx5jtdOeV83H9JD5rrrLULdW/YuH+7qWzCdeFBoEIEeKy0W1pHTQ367QQmn0MjFRzgcjzEMJwsVLW3do20KUPwSHDvheyQ2x77AKQGqpHVY2EHJa5tF1gAAsL8mKPktp2EjBqUA7SqO+35ik0h5UE9LK1G5jCrVrdCENfLEstkO3IdgYtk6ppvC5igSsfSUuiMgEEAAACGQQATRHw5FFSwo/9pfyNo4135vt6Xob4fNP6TH0FiVgdxrKhfDCP/RlPJTKiTm/xeLonRZcaMdCxl80bHDAwqCIWGahzO2Ws9KUQO886OCoTQ3cH/npI7tiZ0fWZcJppKxoRDNczw8EtJ8Qm7vggJg15xdzHncdAAWSOhs2bHUvc7MfjCohcAAAE9AZ8tdEOPI8C6WTk8pf75yVJWwHeHSSh45V1LagwOvT45MWxsBsyxUBm4TKANNW9r1DR6sfgYFe7ems3OIzBBkNGeXDcNNgJ0Fd2ytLQ0wHu17fcexls1q3YEaebkBX1Nt5RUX4cpwwwg5hjZWx2FEVCSBWv2ESSkep1EL3STU0P2ppN2cbuwB8O8IsPaIXddCH3wSueWQu2OrW1UwbVg3T1OEdz/LFULjsHwwyV1fjl16FUwbHR45e57U8i5uJL2cbhizy9S0f6Xo1Tbh3NnvJTotI4iTiwr49QiAj6VMFZaVa87ATdnfiLFiMGJUiFLqt453l5tkNu3Mb2B658aXtise6JCDeDFvEjurruD3lwAX2rzGShhYWEUAottDVaTpdYEtc6E0NNh2UmBSzBhD65Fkrd3rogCZNuUsOEAAADrAQCqny10Q09EAzOk7izI0AOKSv1tump/Ykq7PYRinRGTCEuI+AP+QH2/cAe99IW+abX3BPAs6hzr694aiNWaEHwki2lh1vpIyo+Psu3YVqd2FtsCJUP2O2ygvM1bRWJwqgeZxUxC4djubgfhv2hHy8W+tIfyBBUUyekSFYKGeuKyqXdJoKRT2M73GxAJa8bIteaKtN0WjDo3mfN9Hel7YRoiF1kHUoeR75RjN7B4pAHEUT7Im8doSmtJn2It8TsKJLSSxM6fjDm21ojQi4koP22GFVfEBX7HACgXQAuCFsRTOmOiXTIsbCJRkQAAAMIBAFUny10Q0/9HJTA39vTPuyndNGMASjFKRLsX5czUtdx93CB/ErhsofOfN2BNqrAm81jMmBsjPfTpmmrOUxMygh71cjJeZIDorJIuXtTcj4A3GiBshFBQGmZb1N59A0MtkzE++3Vlgjw0CJF2Dnwz+qTEPRJCJW5QlBj9NorLKG/apEr2jzYhgUa+PeB198ODKRP9wi3IBU9U9+qyb0WF69hUyiofDwRfr6OfPYw1lZgjVcdH3GCH8ijjI1Nr1sV5CQAAAYgBAH+ny10Q0/9LIdX1RZ7yj3uLpfYVAgbpDZybPVrDWXdWZDwAtwiW9IB2qgBaEDzhtOn9jZiQQwRD6DuGxdVRvO5b1VHeKuAaTOoEh29wUi+GhShL2PsKZKcTwOznsYGmdtD3NJj3I4DBg7pONN2vJ139VAI9l6Gn3KRVusA8CdPV8iXttWP5uwQN/XA8S4oWi6C8umygL0htMQhHHFxP7nFfVz30ssdiAX0mbMMZphr7Q34O8vhnTJAIvZKyq7tjetIjfRz+LlB/8O4EWROoV3QtFJnYVTv/EVxpkhGjXjmCcZC7cgJtU/lxncj6wQMvgkDwp0K9FQi9svDQElE99O7OnYHeVNtG1w2Qs4hlYnY/zqBCZ2akLocDRM2KhIBMfY5DqKHDHE8PHF9TPSqIrMliRns47xAdgCOURWAjgxRNrHQ12wB7YHQ2Bn1lXtYmVS26RIMNShSvoHIyICjk3S38GRKKnm9Y/aaDbk9znLwshNTs2nEza9RXYX0E8OXF1r6Qgw4xTwAAAJ0BAC0x8tdENP9F8T7fDJyUrIvcvJMN5VZYO+dCOSUoh4SKDJE966mtewnmnT+poPfx4u2xCjobnDlyeCV7f6lVXeVn1W8AWbsBP5R5Ue722NEI4oKkDZugBwcdIT6FUDpq7xGc1QyeS+XLPdVlPQ2U9mQBG9dOjpMzGhtTxgqVcBC5lKrkhRHRS+f5phNlwrOq3D7mB2L3X03ynfpxAAAApQEAN9Hy10Q0/5KP4YKak6pGahdMg33sD/rX4yC7S0NXx+xZAOQ6ipfnMg/VsWfoHGEH4XwQZr/nc5e2xfn/MU9DRCQwwPulvDuAfIhjVAK+mh6GGdtT852rkv3sAt5Jzh6LMWZuXsN82rBxGHcKXr4Ut11k0OL+r73ICfwVS7CbZyX/y4y2g1vhCGtQs97N1L2TJtUg4Dfi3F1MAkz992fn1Yj4xwAAALABABCcfLXRDT+kkWF+6jZHezI2wobveS+4zJbvf5YMfozQsWZ3+oIO5UyfZeonOi1P7CK7N+DYho2U4fspYACSIkyD3efWwUttRhSEiNH7ASVqDAycKZY6sNS9QRY4+l1Gv8no/sFx7drC7QvC9MZ632AMQ67ySkEw5unkxMHG6xRs+o7UwYTN2FvwNCC8sAlnZ1uYHV02yTYuS5qmuNojj87pt89FvqNoF6DXKBBQgQAAAFMBABNEfLXRDT8thE3qCW1GKYlAdzypNeLuR/MmCxg1PLdqBkoNF6OxsLsJQuKa+oWxj2PwxET6+zXc70viqmQojY5Yi0b4pdoA/ZiV/e2rXjgHLQAAATMBny9qQw8f3J9U82Jo739K35tuQsgp5IJIun5awQkqZVBpUmwPAVrTLUL2fIy02HMUKwE+ixjAA/D8nogr6ue8eURJfmdhuTBlK+nCSeCcjibfjLoGiFEPRDTn99hKB/dnJ00QUeixAio/6ijEye4GL/0uQSeQn7sB708yJH+jf5jE0wT1A/YGrBiZKATxImVO3YnC7KZL6Trskc7428pm26NxZoWwGR64yTWyh2maOZIs8PJN/d8tyr+YZRuPW5m8W5RMEBd2dZeGU4Qn3esb3vRKn+ESlHNTCfZoaBJpd9xuEhs6euZtZGWA9++corqgB/4eukm+J9mHpFPsf+tP9+2BXYFdLRDlqpFXzVAUMr2yyFilnKnJInmP7G4GXoAY82I02BeUONZVeTgePIIb+BhoAAABMQEAqp8vakMPQK0j5vNKcRnqjETJ7VcsgzifYjO8z/4Pal8dyyHTr7KMzSv0xo0OKlzp9tcjks2PaFFV6zpZFtK8/ODUmRMt3oChIslFZMrMWyf0BRLwCJZwPI8A+7I2o7/kf6IKny+O9qKpVQPzuxPOMGGphkYJZPf0L3TgKIj3XTDtyMHBYkyUjxOpGRZxiJiPzrXJsUq9pKQ5nwSP9WPHUgvi0ufLX2cKhP5SC8jo8U5vcec5sodhXZTeCrIKoJHvtNbY9ZX87hTAFT+c/8Qfx+MrlDZXc/324ojLNWF99ODh5U97O+pLYXCBF4JIMXehLQAuS2CMCtiPnapkL2ev6wTpCQ5teCYqX/37hUpMid2wDc9wFBzFBP8bSjJFH0rCOS6ZZ93SAcK/g5PiSv0wAAAAyQEAVSfL2pDD/0LMM2s0hq4xSgC9TJ88hJawYQUe/wRUqExcBG4GZDXvpkoFMC7P3+zIPYGdLusWfAfnYhWCfu7eKBCZsc2tWr5kJqUbK/5gN0Vv079Ox9xoYYaeysei0NflMSwi53VCS5WlZ0Aq4iqqgnlj72nkO4TsZUImyALCFN7NfVpsBzIivm7AilAOkTa6qBjhzpmym/UFSj6jyzUOX+5KW9X6v7Kf4vsAuoFv2pjYGMXz1zKxXcljNxwIPW4Wt4xPzU+wbAAAAdQBAH+ny9qQw/9FMJSe1lrHLc/YTxzf1ERVtKrAdQBJrBBogT6P/9HD09S0gaWEtutMs4I4Yc7UUYinqOjCv+d4anwVUftPrqQa4Ecf//pL5wu6Ry4dGX74fhIgPU9xIjbmXyL3ELOuyuZKM3VdalCWpjsmv6BYaHDH0IzH5Fu9s0IQ5toYesbK55M7VIPgFt5Uy0m85q5DGEYo81ia/JND518k/U1PRDre8S9yj6ujrrKWhneuA0Cqj6/fPSaiF3beagGUasr6lbDoa8zjmKe0qezr432ldepCe4TxUGKHXBj6e9G84jnLJQ+i7hh+u216qksN1lDkXjCkHcbtlDVBUdMfFrsP9mGLKCFS7HHyQLxSbzzVKUCAryAXdKXR/CKxitkZOmf9GNmQwi5mSJKtwaNd7Fl9EwK5aE1BiYCDeLZe5o/N8+gHUxDML55WPlfY1xrNac10mqS7Zh+OTZb9Gi1F6lrV4Dj/BLjRsggmXPb77/OkJfzQbI1e+ZuJBqrXBSKkUdkIxHnxq8F57Kh0sMeGSaME+poSpkIaZNyOH/PZBJ9fOzcvzKM8CeqmhcUZKDb1qEtKmG2XcHuDh5IGoAiy0tIg9ubCw0zYLlsVHSZ1H4AAAAE0AQAtMfL2pDD/RLfV4mUPfq6mSTG15lH51YiFbts/BgezPpWbn54hpcMpqKGQwOda4pgNAykD+1AWSwxxNScKru8PYHjMyH670mQ4Ucad11flk5iNhlzWaAwybuXyOlcpJv9t6svV80pnDKeZFR0zdKLxCrxUOOBcahoQI6CxI8IrmMBxN5lTrCyAQ3HwucPdIVUOVaYVckdHyvpHQ1Nbz/Un4FsYkBO7rbbY3jci8t186HmN122qGwhEeeblW7k+aGNdEIi5X2SS4UDe1CebOUmOdH+opdgXhnD3WnEBfl38tJyyrvcT1zBz6auVjVJr3ueeVgNIaBIPWB8WuUkHEWu9zN5fD6bnUIH+BXdBgd6PXHR/+lrclV1u6Fryes1H4bue+iRx+3faIyf08bTSzTuIk24AAAEYAQA30fL2pDD/j+OHZjaNTJrLTOPgWhNCrK97eTNaqlfUHI5OeldJwJbuFi8h/0hL9W3HjxzONt7YjXWJGK424WY2L1KOjXl5yZ7AakZ4ckLn9uUl+aTdXqfqPQGWKgajPAcqtmtyK22yEGD1cLwIQxdRNA/l6AZBXKj8XaJ8rI3z2Mt3mhgtdSGLnZ/17uHISu53nyIWLAC0lNFVBBHEYH0u7A8ARqzbscHHa86Y9LF5zE+mP2GfuspLJSX9VahqBDeiCFvdo/8xjGk+jjRU+qGaz7YWAYXH4Bl8spRCZgUUofunOvirc+zXHPvWMJ4twpK2Af/MzUp7wN19uJHBYUIzBzbm3fPy2D9bcVwDzOWJuyzFjqco4AAAANgBABCcfL2pDD9Epa3XmHbBvgwsYftgMpqiic0wElsYe9gnClrU1sgZ6SMaGxiJ8foefvJVwQV+k/+I1gIP/HLsJJVOasO5fUi6DOdD45Fl90mzI4ylyuHQW8ZaCbixrhsKxfTMuPW/ltOh4ljW0kWBjSQCo7ZpCNIVdDbZZg6PCVCdR6A46L5cDhs/4/ia0dv/qgVTbbX/eUOXnmWz4t4tJ8tyfL1O9VEUN/r+B4rV9uCR9RLXEafWupWv3tRnphpJyeunLg0ZLw5+qv+Zm0ewwRzio6+qt9QAAAB1AQATRHy9qQs/VDj0RXVoAF1IOMb20he5auUVFrTAbaaW13PnudtUjPWFl5Z/h3bVGJucwzJTHHtUV67c3lw/2RD8psxkVf6455GqOcP6DVeDIoMK64AdgxvlaSmxanf+IXQFZg2lwOcRpHJgPzi9vtTDoBTQAAAEI0GbNEmoQWyZTAjn/wHnyWX7Fb1FxjaozVxei6qgcs0B+RPQs/n7t+2NcfyL1UimhrlvFE03Wl0wg5sVfYf80yHSuLmz64xIp/vWk25VDPZsGZpWZPPGGarKzxscVyPnSUHm1KQDiUFwrxJmZBtXpdaZ/A9qwYOEsGEAAx9iqnLilfRDLSYAOSm/3zpHIBgLs6acdqQXwxJ/Y+wdrlpuTCmzywJREE8KSgozpLYeGPZ8Iu6x/ARXC0CafGXyB1Y4DyycI7C5jz2n3mNEAb+pPFxBRfrkHMcHoL0KWVLVOIq4p6IFWwqiK1IfnkgAkzSLeAxVfKpVjzcxI8vBh4IIjSRAP+TD5K0zrDrz9jAWasVZkHm9tdCnd9A3HabSfgR9mJp88hWJ3ncVSiWyjfjNgWXfCCG+t6az52i7PXXwZA3fqflN6tGANmMGfVaxf9xKZ5yhJxYoLJD84rKUL2Ly1mqY5Do/J74Nl4I2Z5oFrgTwDOc4EtE+lng+NyxhLe0EbgLB21YcVgfPLKQN6Zb2x4Jss00CMYh51OQN4AGO/A4oLVfWP3JKi9hRjBHZC53Px4zzTdcYst5c/owkKh/QZG1vHJcMMAzo8lRBnjGVcyLQLngdpomDHua4FNqgR/NydA9eR7Yz0plEVpF7HXF10I45Wl5IMOXCRUyTVTJYJ7jsFFyXZDuxszkNvANvbdzAGn1WBGqjZrCVQ35qdwNNYquuY4LLLJyaquJWJ8MpKmwOH31Y2rwKvWK4OoNf6wanub3yMQhTh057n025p7zBHfHpyeEqSwCy5/O+p/2r97dsWFC1F8q719W4IgJWGftk2lpVJ6PquJICJYbqcJdOJeXQsYvmSETpFql58+U6VHIdACRYxf4BR68Qvs/+YiDIPFf9oVfeu6Sx1ICm4kINKo/vZ6WWOkL+YYa/wBFhPBcAcJRk1dQCWD/xP2B0yZgJU6KKpeLRbF7lSE9MoGp2cVR6pZwciOrgZQXYI0gUTVK0e7CPC7eNaxztdtN4z+E3JrI5QI1szSoS22PrMfr+IE606iSGRPIVijbbjqkuxQzl82wpvUv4zej+SLEflSUq5kbqJNXaRjNmHXBtRSdKEFzxzlgpSZf4BhXrn3x4JrH0zGyuusS5xpFv2IDKgd5G1C+w8J06qvk5G13uRMM22LRgGDQ6zaz2XG1NROdX8FWdee9p5Z9k8MhYN7s8ZSwmdGhp+Jgt7cEVG26DWAyWIbnGp7IgeAMou0cL/OuNE+G/aLeZa1H7CVmr5avqYrr6dGfYM77qpshavPzq/FiZUxGRFEGv+jhYB2EPoTxWyHvkoAQhX1K+DZTa2c54StnX+2fJo0Dt5S6VXtjW/61AfsjLJrZ7iATGO2hLpqnhBn8BMz8R5/EBqxzuV0h2xmSS1P8OMAAAA/JBAKqbNEmoQWyZTAjn/wNqPdO9SSzZNXMUDX86K5RsA8SYvoniwalSw5UeVsAWvFB2IuiozB/pw6ZgjEiqPvNhdNn0mSlLoV7LRbtAL52Ntehf7ogzLaTLK57peAAFEtZ34zo0viB3Sw1fABbiU53UB4BfV1vqhKU62iTMb1SQ+nnuz/xqmowaL7eimKuotOwY/L+SsbaJfxzZhZMLOWduY33VYX+j8IxdGOIehpb3I8tc/m41OoOyn2h8TIFLEUUe4vMLt9oOTKQglb2tizglmi5drWst/pthEurNY2npfkaBqJ7WCzdCLMQv1bBiAqSj17+Ahv/uv0HmHDJLjkDeakvTNvHgWshCC8fyjirYb+InS6JsFMyVmXFkn0rno5rmKxHZTmvlhLgKkPeG8PCmz7wUtgWU1EOWaUeL/5FqZuVh1ySBQlJ0tNAaFWpqCe57J7bedfrDewO/vFfe29jMD+1ralc39NgiTlP09LjR9V7t+xnTNwGIWObrwpojE1l2oNgDLBFVbOXzpHlF6jZTAenDxJ/69LXfPKDfemrV+9a1+fY7wTfMkhjiswrcDArCZzP/RcgiSjXVclyS7cOdiQo6Z1A7dnMoB8IpvDtRlO/b3e15CN2FTfa67cXA5JJ32fog8ysKgbqxytx/O6B+0BXsYDrKRSgRhnlsyolxRJ9/vTkJ499bPa0CrUiPrrgXj6nTG/HBdrClu0SIeUU977HzVS+TBMdy4nhLGR2I20g9ifl2zwogrCgmyWMoaaD2rNtstgz8NGlAdcIAWucWyQ2ARKTrG05GRPCQRNTnRfRX7w0oVGxUChtU8Gh2wRbd4P5E3Jie1+ixi8cWMJHLaKASMGgmI7xNh9WEVHmfzZbbd7dMXgOBo/rgSX6NsTw5LHbgIc9IE1ufJ/8kJlBIAp5bsqqwuvX9q9Ia8H/Zaou5ojGBibTm75H7p0rFypSmhZ7LiePE4p4RDw1NIQ+xbXNzDlZTvkW4NhLBQ7EuyKypLRkby66h7/jeE9salukJZG49NgR9FrIBxlcOT4pySbEaX59pY3C/AtPpPKZ2bqSbkATbCAAXo5geZumchXNBnCDjtsAj1eX1jgZQUWgrovAy/YMjmL2P8AafB9ekmqCkf8WFeMmWx01rMWbyQgnrp5Rstho9grQDWXl8lFyeQMurvwq0AREvhX4ZPGKwgnvTt/4IBnzx/PWO7KRdg9cdRy5g8KrfEO4578wrCKi+tEsk6g2HkFQsSIuYrL9KmDKHQPt9AeUN0X8TU3dPwwudsNhTeayUYmJE+rjLo71dVd2EmQOzSwtBDjeifgtKU1ofXyh6haKIBbekKdZOVZGzQAAAA/lBAFUmzRJqEFsmUwIQfwR+9bpI9XwXGEzROOuvsD5DTaQu3jOWIl7VSs7k6hzbhmchVqUf7dMAHYMe554404UC9457TCcFepy6y/bq0p6zZjWeTfkZh9pceje4I8uyOdQVhJ5cAeyAAAADAcp4+/oBFBykCQIxyvJkNJ+lTYMBb6HK0YrJnaF76MHougELvLIw6Pm7g/tATJ4qiGDrUKC6z5mPlTa8hSHeVLDLH8joj8EWTsG1lLHkg1mYXd4c9k+48bzQBrMnMsJWUae8rTOnIdF+II6Q2Gy47ikWUhIMgXUxB4pABAylpYPdj4L35r1XYrMaa1T+KLrphMfyMQLSOujvoeiDtnwcyzoilvkurZegHfb0DbNznPVL3+aCGKE14moT8n2AWqTVUxisUHACMk2N0IAzl1mM2yddabgh8iVfULkHHnMAOupUGxVXMvheIWboSbe9Mvjk/PDULKSyJNxPdeQxQygJPaDFs8vr4N2CalZuDdEcW/HCefzsN9+rwRPGB9JEiK3xsxpgHC+PBfqbl6fez5IR8bf8JUFze+VoUYBUKa3jsINijxKAh0F1oenKE2FDiznKkVtcVAZcmnFIJ2J28kl2jA9KnzgqlheWhXXttb0pWB30Uo8UzZI4CDHGJY4LbR8UR0CDh1kPgQ/lxVT6a+lh0VI8L7SfsrQsVDSlvmbZvkjGCkqvfsrexlpdBsceJVo63uKeULFnLOGIK2VYYMHM6CVcTCJstHdhi4AYPnZf4IYyjXqGPerKQ5AHAQ9t308Icg0mZY4i496nrFROpXX7HShzk8KSa0PDGeJNy0SuPxoz6acUsv2wGKMb6cDQSmFkAFjoiZokOb+EEmtOeiNcvnund9bCBDqAr625eJHrZrrAQA9ivsI+lbISnmsLqfE3MY7x4okYlyr4zc8iZfYI6n1gGyoFiUBn/M0g9Dkbt975bRL84yjNSTmDnUwwWoYoZOtGDWuJyWEoMSsFzUq3Pa3UE8mGkiASZ73nL4qe4PNCe+Jr6EIk+v9iXhHHJYHahgfrOyOZfHlkRBsYbuubFo1r3zyWyXkj/1+7GGIpkZGFG/EOwGRMDF2cPqdB4aHf8Ibnf1YKZ1/1wDbocBPaeJZ5lKOyCM9t46lpZkKGqn5+dFSh3vURLPTAvQjjRK490ILrHtQ0577KRViHIwMMcI8ZRIkgX9ip8M0oYGTJjS/1jX0GuItVNRyYxo2F6Gx7p7x/Y8AkWVAi/1Z9BXYT4lpL8ZW1LD2pUo4j2hy0S1ssu531cb4qimCMyoDHUg5xFeyrcNMlJvOdTh3EXcegalMoIdOYUH51kW6FNTsfOHnPT+9oX4QECn6rZyAAXeAAAAXrQQB/ps0SahBbJlMCEH8Nu2HTFOYHJM7NCvst1TtEwySV1RZZdFRcllmLZgBZHIsviWummC5XrqxD37+ZXqCjjwcQbd1ZF0RwB6zQhKNj3M2uZJB2CfFbUjz/Z8JsjfUmhgYGz7J95RFKw9eB8L3thhkUATGvk8a1SflkOwseVa3fX82EBiObYa4NXYJgZhhmNvFwkmIaR42BlwBzAotWyzcg/gDpHJmARVM5TFNFx8uzKTKHH+Qzxnu7n/nxAg/ZcqW0XS1zwwPKCSkAxKhiDVNq/44ksq9ZnUD+w/36r1MA6ecHzDFNqvgMACZszcuE6OSytK5RXO2DvjH1Oh7zG5mcGC9sQlpA5oxGJZPmJBDxC9X/1xhqg1ZTyIG4D/OoqNMWXFT7JdASGlYQdsZTX2SJ4Tx436PeGqH2JVoP5gRLurQ+laZ3nQJXKPN8imlOKE5AZjSeUx7050x7MbdRY0wQtII9HNd49mlJs6zrcCBKFBxPcD1/+VDSkUMMXt7ntuVdKbnDZIly3UUIIUMM1wRiMPWOBGVasOZAKFg0iyPQibCPwsUm/oMLiFg4GLXHAK5xy+aFik9mh9QegG/rV1ou295yniswuAsM2bzTLSelpj7pZIj3Dqwyyd9pyOs0WnwQDaEllFwedOERZ9YLxo1fKdZjJKGF4w3GFUU+nle4h2IuIEGdCwQU0h3+yDJYp7fg0WPcTh9pGQqku9lDflmr9SDPSdpQTZl4oBKQ/UqI5/cKuFkJ23o59THytfAnSUBAYMHlS6IdNONlFmR5onxNMDmiXepjjRSwxJ+s8qnawNMxwdVwOfcUrRFtWmcSKe3A8KWmiLA7cxiseyJDffddGR3szLRb+T1XK01lcaJ049dSG7bkFEly/ELkYD6G6efzKi3hluajNbPKiSOB6GTXgk3bLUTKFuLZbe0d75UXShF50JQQyaQLfPjjl+Y30UaI8yC50mhBXNlwXvo1fs1k+9VivxA4A1EuOCzWAU11U6paoewbCq9586NdemACmX4QHRaZzoJEI/QYqf/mnr9KFCpYkIhE3te5gZqpn8UNyy+bQZ+gFppwpVv1Wu3M4IbGgjBsYN4L05whh3MDcAY5XT6KV+6fO5tVuwGftbFPU48G6/3OPJSFR1m2b/zCpc+QaWtFiQ6m+HSV+3CdUy6UrGa4APdtyknXoFwSWJbvrQFCMvYpM41qoGd3BnikS6iL4stQwI5WFnz7DxF6xMC3+7UgyzAn2bVMnzey1+LrUM+nJOJNic2FKyK0cwbSRnpoU/PjbMrAH73jlimGfprZoB+H/XvrOa3+eQ0U0E661x/VjJdgDGfE1G1/4yBJWrTHJPctQs8CixULwSHmB0/mUETBgMPtQKisESN3Aq3h4zgJNLeJZDCM+n3yjlKslXlWcL2hFYOBQrbOXOObwYVoJ8wkQ8gJp+FH6eDNfN1RWW9u0HJHbIUQUViQpqa9wky0Rcwinm46BHFcx1RNB53Ewy7pgwzqnoDsTGd8riMuJ9E429abC9UftVJV6MUtzgjrRTORUKIC8Yv1uItGn5RWAVO6rUGMHJx/oglpihfVPv0JEA/bAZGY37UqDPWRArmYz5/qZE3MfTRaBNfu9rqIIk0FYi1vIstXlDs3yccTw2M9XvqEI1IctsU95PcZbQzsfdwiXRHtq6VXcIH8Be5k+btVlEtgvc+4Zo/qqhfR9kgxUvYaP7y3Gl97NizOh5ojVR0O5R0mJ0vDuWY+mUIcTgFhN3zMEdTJC5K+7rpxDRR2QnrUdibH8v0muWsACI9W7+Yr/11j4/S2OcqQ3sun1UhDLRzYhivefK4ZOE1N1l1thZzNYkteIEqD4HYt4j+0i3FwyEwteOjxaiOfGEoz6vcaCLzEriQd2PMF9cfjZKJ3J/2+vxaLgG2SdrKjn9wQgMlC+WY1GkblSMZN29Na3u9TmyuD8y2nO0daumaJdcnkfEIpznArVy18EkyPB1TWEzWWvhR6yBvUZsO62pSq6/pJlfMFOUTAAAACCEEALTGzRJqEFsmUwIQfBKylb9UbKc5feqiW9jlVCZYzxuaAViGKC9t4v2bFx6SJzF5HyfOSMCc33nMQKgBiQtH1ze2D5+fubHtZ1cxWDXiyO9D93CT3bGc3cjpA5MuZ6VPBtqy2gsgyD58WGjX28x3XcL/9X6wHCHWisN4tdTsSv2eaDtrsrh5uo1CIg+V/F7ZGlUhXdy2y9JQZH4C9F06AC+Z9cgmGvugAb2+OtYvBf2KaWlk5g2zvkZOc3ikPKhYixP2IVcjcNrP0AKEhKwP//FveNhF31a2C3X0M+u4vw1CD6E/aks3BHX/0z5Otii3HaMt9bF9gGHaT1kgGfFtaCFa/1stVf94dudiDEkohmoatubwwd6t7EGSWtUFf7KzQZhyyYpDtOf+UE2eXfCJrE3In5YAqnAMG077cD7JW8NjgwoQc2Kgo1HmjjosBBc80eFVfmU2aYJQDteAazJLVQ6GKB5g8eHA9Hv8ErywWfYDhkgpZa1W/7QS8M7aoFoe/uo8k0rsuEAIArq9qJIEwC4r4/99Fru/F9LvYNXSraI8vcALB6ctbKNgitK9gw3sRgBLxjwxwV9JZkKxjAImoRi4SseLKPlCF/F846j2o4mNqTxgCfuPPTdofIwXVKUBR8aRZsIbBdPD8/atBy4mccX7fOFd+/VmY/lMyX6FTLI9P2UWjo9gAAAJwQQA30bNEmoQWyZTAhB8GxZ9YABb4AAYxFZk0M4HnF+pkTcRQMRxXECArvE5sfKID8r4YhlblMdoAAkdzN1aKasHiqx21I19shOUitHsBcvEltp0EnBJorhf8ArT8S1dcQG44pSuJ632AQxhq5bkMSQl8Z4J1m+/Q+ieK4nUP7Y7LfMDrRRw9oYd/T/Y9IiRLchSNxujkQKS6XFALzciZ1TicAo+E8KE5iZ73Z3vxeVAsbj8C1Vy3CaOEJT5tJQ9ytwPOHLiGD8kTyn9DlzJh51iT2P+zVyk8r1sk8fFEfInJjsG/SZ55JdLa/OksK6c8u41GjjZNpvQ4c5XUfBfmswDt9s33Hv4M0RYLCgNf96RGJmKevafJSF76UKHm/bOJMK8mo7G21msbYT0mjt214rp4GHJR9CIFyvnJkzZfw9TmCVfN2MxtZlzMSfyuY3I4fZcaWFskqobaYUo+3OLgPoAFmu2gtBseh5+mx5m5EORONdXkbYuDjaN+oFOdjqKQ+kS2ADfXoeA4J3SRTm3uDaD0DTbu7XLDiSBla9TYYZTaIwGoEuRlR7qPi8tYSPLNehWb0EqEcLCr++Qa6nRTWybQBAZrjGnMpFYXkRgMq1Txe1xqtBHzY+AWhQDVGAYKPziMzAMRXbVrdTPf4JbLfq4+CSFQuYky4fHAITlekxJWhNFAXkb8Ssxc3moX3J6ABjaiV5A6kKyjFe76YpxF61UiDOx8KWaca/y9W2swfUXBRZmbhwF+UpnYakKMT7xglv93q25wFLozsJaLH6vTrD79bJs77liHL86X0cYafjSf5vhygP6MNFP89zebOuC4AAACZkEAEJxs0SahBbJlMCOfA8YBRjJabmPgyHZQjSLUkWr63x8D2/ooqgs6tcyccmynCCZAxV/fkAAAAwKcaRzShEirgLqHVvoE6o0HmJq6WgwOp+1tN9Pu8paX5AZHg5Hkj6rqU8CZ3+JcHSqlxqDVN2pe0LQiwM/3v5gFFF+3ifyTZJwsnumEgbsLcApEUyxLDJxuH8U0I2SThvNCtllHeSCFDXkkz8EWsv+UIHPDOl82EuJ5tQtsBkGKdtDrNlJhJylc+2IOm8+uwmLEn9mIL7jjaRFyTcFAZyYLy4oBdiU+/WYHmuWxSMRPAC/KH2Ub38SQwfLWNZNwSZN9CIA251XQkjJ/pMjdEGfwrIzPTuI2l9rPt2r5AYZmE3tq8J/xbLc6udsSVdQn3KK2nI7rK3W8gNCPUKRuC7MCOOTqUqX2K3Ud8qnNxcQo7E8LIp57uMIObj1vil3GtpKAl3Q3AwdqCRqTjfjjYeNvyzlOiIAJFITXFpve0OXDgWBJ7Ipq2lJFsLQ2pwIFZLC7JylKxpPjujg55B3W1vS1Ga2ge1Udx62hgWRdXC82IG3sP0XHj32L8ab9kWCFYbflY+n6ck0JNoVONAr97oTYf5J+L5Pp15LzD9keoIX8GNbG6oFU/wtymFa7TyVBwC971P3TJkGlA4A67VKv3lvSV2LSbDg9o4UPBnvcmGdypyvNr2qn/TJ2kZEWlXmg4QzsvTtAHmAYOiOCv50O+0jYUYZzup5NNIc+1B1cnmo/qHCmBk+EkJhviYTge5M/d012t+TLwtzWVExLIntq5twQrPnQyvxkyn1zKb1gAAABY0EAE0Rs0SahBbJlMCOfBX4NSAAAAwAAAwN6KS7VLm59qPPZAkgFAQA5rs0mkcf7/8zkiREsJkkItr8xzZLqCL8PxI24m2ezaDw2bkHe4QrkGjzo+Ur5CIAyGMpSRz6oczyoBLWAQUfLFnHJiuhd/rgovxBE3c9MVWEGIaiYWQfQIuZ2ERltm47FbZhcPRbeFR1cLlhi3x0OPm4DcKbZ36DGH1PYn2MgKjTMe+KEdcCNTigYD5McpdWZc8FFCtO89QmF05DDwvyrnwqw+LCDnlVO8nBCwxmvdOtfQM0d21PCzd8UxVo9B0fHKGVEiaRgYjotWT3z8Ft/MuO+Bo+aiS8d5js8mE/a4LvTbUgkX5ysxcQaK7vnUmPSGBqyiifPcKNnxG/nSr9OpUrEX6PRX1KDVWktOHk9mX8NXlaiG8JqZIdwqcpqw/JDQSdyK2ednSPJclS/1DfNObGMi5pUVYgAWUAAAAKEQZ9SRRUsJP8cB5XgBvebPTyHqySahDbHK//PG7TqBlyRTAu5H1jarSSTicIbf/s/C/hah7LoZpM5hem+rsqSq0nr7EyZORqfY3pVOq3n5j3plJ/wFVIGs/1llFL7onnmNg1ePmwe89Se36NWmnNrP1JnRG6jJMsquTKYtFziqAYE/A2h04UNHVYOIajxi04XmSA6zNX7cUW6yaC0V0C3DWoFwkUBNscjR8mG3VjzvQsOPE+RB3HsUBsRy82TilxC1UtVAEw2GYEnM6vZ3OvU5xpfoddSpJkB11+x8sHGznevwu/AY/bkFwqalH9JU3mfemabILO+0qfhaaG7Iq89qz5SBl1rKEoDnIOtk184ibN8Jk4g1fl6t1CP8A7OlgL0MIFoyQ8h+6YfNTuvQOSPyZdTQgQ250drNQLEDiiQ5erzz5/w4o29I58QyGeEaPreFUqjcqbeliCFVXJxdqjA/jtlJfdnRgkPXn72MgZ6jh0DZYk9463+bMMvv1bA0lvNuj1P6j39obRTa3Qu2pmleSrQ0oPy/v2NlNIKNlEzB+U3kKDNp+IeilMixJpZg/KFzV6mTGOB7D7iaRExU3Z1fbIRxnTcAVTXzLGJRHLIN+7cip9BthGRpsxIGbcLVc2YyVJtVbt3Xhu89DBhLdkiH+We71hiW0QC1Kcgt+wFDPoqeYWjxktXgJtVaOKWoVFVRbDrj+cM/KRf4RnFYPZH3NFTBQCf1MgRJHvo8gapTHI9iszgLwCt98vJ89WQEeQom4j/zRfYH510pFSd3x7cEjGN6QvCqcz6z5lz9UWKiTwgYTB5e1cNrou+Ky6IyVlu5yhSp+tyO3O1QhLNSjndlLwsDNEAAAI8QQCqn1JFFSwk/zpbvhmgxOuDL7DI4AVW0mS/5YzErvEOvjETFQBpp8p42nxd7X+Febn7gTbNAbMtFWoSEKIk5gWFJc1VYNBnsq5PiZllY7knPECTlQY0FLKFbZvzo1jl3XjsRGoQi1+er0pXCCqE9TPY/L6HxzMOfS0AUz60OquHWfpzYP3Xaqg2zIMtrV0rU2APl1dTOM2Bw75Sfb635wWv3/6DkN2hzZE8xYWXPX5NhD3wV/DwyQBk6szQd8mk7iUNPupGvhFL5NlXM20MxzpuvhDPcDvkiCxxMbTCHZN+S6qXRL+Lo2refFn8d8L6SrG0xAJ4wREOZX4P/1eIX18HOcwtySQ0xw2p8P7MlfxOkVbLyu0armvPEu9G4lf5BNiD2npQijEExXgNI4neA1PW/zD67FHs1SrXUI8NIbkugyifITG+F3324cWRWUlC1q6d20E6tPnjsR2ykJ8x4VRMoT2E6GfzUkysMU73L3xBJxD7rlN7g25cfjMdt58nbku6rzFzOFd93peDzZb/eX2Eg9Tdbm6KREill3Z4Ozr527zp/SZgYGpz6iOuU/R5Aycr2P5rMoAdNFdiFZMdR1wc3nG8KdkSwCJja6VmjRfg4sSbaPlCQ/olT09TeLxx50FYdHIXqUGnLE7SQrztPsInLgeQx7vj6cj9w5zfClWHXippHhy4ZrDAmEwbUVAU8y2d5+8XeOUTPFNjUmSkpJ7BypHDMNRV6xW3aLW4vFaqw8lpvuMs/PqArT0AAAHwQQBVJ9SRRUsJPzu3oveEZRBpXICvQF/zxuRLHQ+Q9ziAPUZ9QmjLVwagrwYq8edriGWaxpNBkqTg/opMg63BU1s5hAZskTIP1DpnJV7uMscDxjSuBdYSVwi7bK07nDPvH3Xkdf3KwT1YmDulGEDJtrFc9W1+kuhbTaNCG+Q7cEOLS1WF5SqhlTIKrhxgxcfVNOEtxQW6ybrvLhfarBuepJaPq3PiZJ0QgrDdmd3rkIGANJ2a6wEzFgNC2+JiV2yBab5BVAl620RcWNStQXeLlQoRw/Ek/hWdcbqaQRTuGbqGDTxr0ZCVRqd8yr5q2/34MZB9N22OycuUVYQdmkgXXA/KhHAwoHum/iMqqwiJMJhFWANUnbIx1NUFjbnhyDCNUflHTCRYf6smih1Gl47sSQT/+t+g8KZzOh2P3U4Ew5lPwuPoehhw5Ixcc6bO3XiOLEPjygIJRoB1G3eQAg9XwtrZCwKMWNh1TrFMimOmnSWzNjMGgjMx3DlczWNtlRHhXVW39EkVNjQXmYWqar6HmWP6jOWquZGXd9Bwp1GVCV6yqSGG6rils2jkVwzlPcT3jrcm3hUsRlWwIUFcgb5ugCxrW82wzvABtDsxp9rbh67NV+6LSGvAFNt22O3U4s4Cqq7edDvpg49b11sAuiAuQwAABOBBAH+n1JFFSwk/XPY1CdCLyZOxcdw5hLRZPvwBwEBaOa9UXOYpUzN0AfWi087KPt2YH//4kO5hWgvZkynxaQI1fazYqOPGLUiL0/lkau7f0PMsJIhc+AjYBxOAmMHbq4ZUG8yza6BruoSNPwN6btcUcKwAlrkF0TDEGKOshpRj/U8hY2s1sjpPI1HHg8PXB6u62IaYS9S0QirbhJZeujw2CQJwJWwnwGujDTaw7Fjzz/0Yrv5Ze1WxgcpJNn2gepNbmJ6uh7ZN5AyoKEXbPgPkaxH1ha2DLehhgK4ZhxSUI9wcVK4dFhY7V3VzwhKCUvx109MBJY/a2IPWdOa9EkwrKTm7pm253mdwJWV0mSjd8Ff/zrsV83WNWfhU/VYx/Ew6m43jEYK47qKdhQDbaOQZrfiC2Ov3addx9ryvRriEyJFtkJD4N/e36m86eaFiDCfS8auRti7S/91n5RdTyCjgwlgLYcXyoGU/xEi5JaoE8ETvM5sRGJHgiVQ0e0/4YEwMP5a5/uVqC7exiF/YKt4Srrq5plRBwE64IUuhmVB9stJNcsCTJ2H3KeVUKgv37sEmvi4XrVauC5arzhEvj2r3Xvn4egL2avZ2x3C4QhDvpvbcO8QkfosxB4Th2KlKfh3eoufUkP5EGXD0iJ9bYc1UTMXfT5pmovxLTtEM9c8dgOJoDYAjiLLbi2H1sBXg/Lmk3wlqeNgp8lYED6b/ShARek1y4ImpDQuAeYQGRVOTknLZKnwHhwfMaJZI/ysYyncB8cwgtKYy0zFfwo9/NAeX6EF+4zTJic/43b/D7Md2sHK2DEy4C/B1BunrN3IYLm2I3o7v8ch3t+kd929wrpO24YET36I/ZBCsxidQIuYK2iVRxTmr/8AHU4CzdhpRP5QPEKX5vF3ChrK25trVPNqPnHhTN+3TdxVY0zB6Fd7YrzdKanuHtlxBZUv2z/ARTysFqk+Ugc4srpYiTGLwllej0qQAolNgUDaAEnzzFb/9WFW4GMwXK2sOlbhq38jZwL/dFpSo3u51W3ov4DQYfcgER9ZimB+OVv/2e4KLSc/ifJY2LEytRklWRoyK1XDwnKGqFEU9k3T4HJ6NR1nYjKfEZY2xW8dSecF2TsGUGqmlfgt0zK8D9Y3TGyTM1wlL3XobRwCwY7OE8yIIAAnClyr/8LaSKJxKDmlsa9y3GGfpFOS8Tb/GxSMRkIdoX/UcRJpq9jiCUpReywH28pSa44ba/lXSWeAyGGc6gsIhm/2u6+UNZ3htAPq+rIt0KT7NtWAVVDTJ3BFEFIhyri/hQ3A4ScR3maWP0R/LbUPEaFbByCwJrHUFBFEpTgIXDPviF+Z4ABXlJSY0R72tHHBml+4DRWWZ1SpnB1Q8XQZyKTDdUJw/j8Rx9PKk/p3BxzxD2AaKqGyt0D0U5tNMlrQQT11P541B2uQxsaNXzMuIg0wsAyyJ2dmXOCRbYMoX8dh7H1TVkDlo4c22a1Ee9bNpPa6h80KsE3rY7ut+7lfBkr671/tbof0tAii0x6iNK2pJOSlv0hLRJJUuiwIvd5f27Sc476Q/b6Gcqy2NtDqyqWIidaEZDv1Xrmt2uW3XwhawbTcr2iPo55mRsur/GElodqilIgpWNJmnm//kUvxKcFXBVbbCNpCPI8Is1V9yww9QMqEAAAFmQQAtMfUkUVLCT9e1C/436NfBW17buSN3tW0wAjmll/FyxobEnzNukydQf3WR7a4L7RxfXgGGDvBQt7ZWjSGNRQNe7/36GoEOAJqzbIKiO4LoLvkwUWeR1sC6vhgFrjeEvbmd9nI5ZwwBpuYpt8azsFh3HwOP7OmCdeqXM39FMtum4bX0290lGvvKQcELZw4wQI6U7qnk6IIl0tdU4pY9z2Z3Zj9R9M2/Ko1UJCWCimRbe4swGL4OYQpfytXjDLCjBfyPqbu2fihblvhZ3LdBWh2pLJGgoP7BDgGhDRTGpDZNsWTQGDybiNdzT18uJqO/ij6TIsD4oWVEaEmcpcvBaZTiwbZuFOCLYEqFPHr8yan5RckOri0UM5cfMUs+hYRFFl+QrO34gb+hJv5P3KYEwgN3wgIAwIGpE+yUwptmiKr8/KftPllNMh8SMFRPt7U+4ObhI/z+7M1qKtQrINZqrofRHZu9QQAAAdhBADfR9SRRUsJPg6LHQOqOi1PjaYf3DN/kffYZU9fXVe/WKWqpt/tz7H8bNBxuYWT2DYbgrS8qs6cgNBY65HNwtgLlzKHhZPBFRJYkOxn+H8GWpQ9EKhq0sY5v0wGpyQNwalFclHyIuBexKEmPcLUbe67PyPafSbbeg8Wq1FtnUZVPgVjy0tXGxS12y1SMXv8C2HWJ2LHY68QMDFgR4HmSReRNXhFU01MKWR2z7CTI4wNir2Zw0OmJNlzvStmzwiXPCBBUDW3j3a43V5HGjXU3B05g8pW1NatxL8d8tMtYXvuVk6EeVgC+gQjzDoXSoT7IScPZTDzA52NhYwOtSaKyqo5z1vgvyPp2TjHeZ9Tw2V1ryKX2r2S4cCoL+INdFoiKSOZBvUvp+OmpAmC52j4IL3VS32Kbgrq6T6qXHMnu2CU9KlZM4ZT9Ucze+i5OhPN3VbDzhnYc3Be+woloa95D7mLGNFdThPy/QuS9JYGVST8l1nd2d2kJS8sAiP8zWXLgG4fQDMnVU+BAZJ9Fv7NMhXIpFpyu5MqTa/Q+nVopg1hePo8CxQY90rdEx84XFQaBJ3QtaLx6+7sLdzyup/+Efw2s7QHJxnFoFv1QH6lENeO55LDbAJwhAAABV0EAEJx9SRRUsJP/iiC3NnkdS4SLeeWCrLoo+yZac6sj6r7GgN8WjfDHKNENltoxkzOewQ3ZmN5fjXtuTAJzm8JOE4Q/mKhewNY74Dfc5E/R+2/k6JXc+/vm/LW/waLovjJa4sqdApobKDKBLg0ytt3JBNjr0F7bk1gw0CYOkFOBtW2cXvQWK84tImoh+vXe7etY3CKRkH17VhjboZK+zbzzWsFZwuB6N2CkVPoT0ytLKnjkaZJT4cI4JNYlAmJIgnB8Y0e8RSwXWDyJHRc7KWF3hEwZl/88v5QQY2gDGcQT0J8KDlwev3GgNQs4YmAwSgQKwN/UBDwkZOIRSa+qtoIW4zh4Q3vmoNnlsAiqBX9NZLq8SnYxORQz900YInV1dA1quUN7RDLAfFQeSv1DSJTrprUi6bH1QCpaKyf9jaVIJj/M2X+al4qTFQwldZ7eVXxVclM+x4EAAADNQQATRH1JFFSwk/9LzwinkTAHWMZfrRYNYT1Mssy5t5qQ8qQjLd0HgONo0bPbDSWzI4Ss10rKLiIeujnGkf+gsMP9JbMPa30caAXu6WieLolELy+8dT+O1zoxrPBKt3jH1TY3ESZAaAMyR5YfgDsVUu6qH7cX/RqDOPyx97Tno3GDa2kjOQq0P8Wv2uFyO2jXcmcuWUre7kl/3zdxu9qa1nvE9vn8dcOBFk/E3eNMEKgTXthnkFuxHKyNBv1BleE5J3pzEkA/8oU+OwCPgQAAAY8Bn3F0Qs8eeYunwsSUSHHK9OK53GRbrbLgj+mxIFdaeOPHyX6FZTiO4dvpErZUNR0zD1bKmasrs0QBFscBhBuiMFFXQmDPf6RE67JitU1mvG47yro52qrW47NIkCk3eI0fAVeqKspgz+1T2f8VsWYqAgz9pzfgg9EWUV9VcFEH4TS2b/bXJWb2ndZw7YB1SwqLuwt8nP8TA03SkGFmJngBFmkJxFnYFbPQ7mma6VZDJe/3E4MXHD8xfp5hP8gBqt1LkN/zvqDITQ7zRZ5133DkdHUqoV/eTaVBHLjWSCDySf4agASMtlCvs5j1J0pBCkIh3zCkzISYdwok/G+FGkR+W7w1LRKsL2jmldHArvETH7gkT8i5+Ew7Fv0KggTLsW5x7hss1a4srgLQN6KjLyarI3+8FnKMFksxmjiSEp70e9hQcrigsIl6zyNnSRYaTWggnqQ/bBnw/qtFD3BJuD/bWjcb8yKEIEj3MK9Z4W1n92GZf2PjwoKVi6z02F+lSwREzAIzAj0Ej5hQE9uQkiwAAAE2AQCqn3F0Qs8+Vy1oJ1YjGzLmo4TDshJ0+wVheE2e/SnhjlS/XizheZxoed8CMbxd8nkfcU45wYoSw5S927UMUZEy5qDUQtwEfgZ216x/dxgVfijkX/HyAmmpBTwxUAgYF6zJIf2Jf1gkVIy+ayxKpFXgaqS4i5PkiTTths3wsHSsOY6ppBKqXq4QDTIpvvqL1VsbbB62qtgOxonD7kCIWjZF+rZOE4cRGbeP7BBAWVhhpitik9xwA1fODyvKjIPnpk6dfdkGdzD7xy45MCxKvwRnrAUu3fyFd6LBMzWloppaiqxb2p+KLQysVpCLtnHoFgVAg9sQh8IleX5CZ3NAQy0cUIMVh9ImwGjzxyDdq+YJjf4JTXuQI3n+W4pQlX+VqukCQRNfQ5WFsoDmio/G1gApKCH3fgAAARsBAFUn3F0Qs/8/tCLbP+MP3uLpoawjMk4116ggXdAJfbRvn0fBKfEGPHhNGdyjkDDDWx6PS+KX8jNv5yohHCYGwdhTV0veryTEUJyNkz6ogmtbpyorFtOiSb0rav4qr4FwZRsXMH7YsVVyc/otfTagI3UTeCAl9BPRzZOyvG2H/10VpDWV4YWOl/6JaUNa0pAV5nDMM5QkSa70QYY2CvLU2mjEdhY0WZdWF7khInfCLaFIWSERITpHOfl6y8wPNpr8qkrWYSNtawem6VdQMd7AAO2OeHGGNGDesNxwQz8J/njvCKpUi1Apwnl2IXFdUHez5pQlQZ8WL2GcsS/yV4GeRpsl4/9y2VBhA4aK+OqNheJkvbOkV/yO49YaAAACcwEAf6fcXRCz/0Eygcamk1cEY+7zLf/8Xb2bZcmqNqtF/skMvO4bjVIesPUgFBp4GSf25v0KOQDPHUdHi4paenU5y4YMGMiQzDPac1h21cF6xwVc+VaygRwkx11z2j+Fw/EW9gzypMPORbBD/GCw0DVhXei4iW/wh72Z84EQzpLovrRW9AUmh7VvfmEosyG+T0+U12igl48HhLT4t4f25nvl9m65mIPGzVweViSUL8Bxcmdz24Q9RZiEXlP3c5MJBRIhhUUOZwdmxmvUxZE1LRtTZ8ztEt7EFjQJhJH6HH0qxlvv86IGW/yPjFyAwVq+zm3JTfL6OysNIb3wqICVjFswnXteoLqt4MY7urZxjVfdMcyg0yTllVv206s5KeenZLYX0UCi/IAoSnIyxOP4LizUNURBVBJbqf8Odrg9IOiCsj7+7re63V+GsBaUh9AjMLmPkyKO3HXXx8lwq3v5qzj89znX9L5MhBlEUO6ntUnA6/ZLDlBKVEKlW/eZRYhQbQPSmkP7EqNPEZPi04WeLYrggCmjhuLPpJTRww2GfKFy9ZhXYQofeT6WZ8Z9mSEj6eErWj4hukFQBb41AAn9DHkWjnxd5Dgh4qMHUsOPAETz6zraTmQvECqjCuKNiiLmoKAjI7mVIwkvAs/6E2mZMD51TiYant4HlqPPPP/VdL6H4JWhF+yPmf9LdqJ8cnOgplDtHt0ZuBio3Rj7xOWLakgnup4+LPhEUCi5Cyt1aducoH7lYfjfSRn93Gy6Acx42N9II7hdGsCTOWWxqSzrH906F1sF1suldBgTLvIblPSS6dAFDg188pZK3mBSahRsy/YCSgAAAPYBAC0x9xdELP9Apwq8GNgWq4WjQosilGcl33uj9uZvTyJ61pBMnKjmWUAzRioHFc5arFvOyrzOUcO9OAypS65bCzaBeeH+ymRHT+V5iEbvm+8fr1NKMu2pxEL6tQYMJLEGIa1CuhLgMmh5AMneBGSEP7Zg0+Jydp1u7+eVn/fhcMYZjwXn3jr+swsxoFa36TCpwAodYoTsTRZT05fZgWv6pYMAp5tUiuvFtkIfA/RTj4d+0Pz2+rcjhVnBN5LDNybRfOBhOH7azg/mZrJHjCByFQvv1dWCVwa84AOO93m7EhoqHXwKu6WfoXbq/Uf8wwfB+/pG9aAAAAGCAQA30fcXRCz/jlws6IDH9hIo1eQr6f/+AX9fQifjWszAoPj6jBm4slz9T53gv5s8hsJHn7NpezPFKCggsr2HTjXTFRRgZ/3N3A84DdRJSfnngkHWLfVlUVZVQ271wSGmty4VP1jeThpJGS6HaBDfx/+IGmK4oZGQMTvs6h9S9Rmn/RAbSFUnwMBdkaUAnhbb92qhDgcuU17D49DqwBMZdzxoMyn1OIP+IllQeoxhqRf0myJ7uigbFbgsQofz9/Y4YUIkKS6/T7MNUIJfjEOQy1ilJL/9TrPDZgqYXUI1PiLjygzWYx/lo7cCbePP+mAAgskrSfdHFYDYJyLReuXi48urLItNjIJ3t0h+Nc55qcUjXfl+s/fcU7OFxN43oQPbRKAVFrt867ofImP7UiFJ/3Psnn3o33cgxLjArEDNaAHcnZcU1Y/vEiwQdXWXKpazB/KvzQ1P33j0YRHV/LLjz1b53rB1nKidvIxjEpmLqSdEzRVnakYwN2ojYAtewelgfnAAAADnAQAQnH3F0Qs/kOAWzjqo3sauwqUJSby1rsqeOR17kkKgiM5WX/u4GrgzLFF0zaf7ZbwV3Fx1u6V70FU5DDiPPpXcAzKKSxN87H605i9s/r/Viss2EBwck035iIjKAOgZdVjd39ZydexfPYR7BPsimwoPTjvbyWqCMkapVWDGHOIrpvaKiY9aNUgXUMTpQlGoSM3RxCEdLe22YLOWhLzjs8Z2MM5kPjVHPekowOjsrRu7al97e4Ix0d/s2JgBBUv2vB9MyS5wA0MPV91KmGP1A9BhFAghMPxvFmwUE72L+dgPp8+z0vmAAAAAnwEAE0R9xdELPyjlgJNAMVg9s5iSJop+6rZi/65F5WZWhnBaghQoID3XzKGjGrCVP1GZIpkX8or+XADyvJUAI9mTtavQcJ9UyCtpuRjpyner/UA8SiviyY7mK4aGm6P/NWZK7I6rTQYM4EIpPMGnEX/Bt1Yc8VJHCCtW3VWHCad+ICNj6rBl8cLWOC4DEn1tc6kfGzPOuhsMrHPUr0Ar4AAAAXcBn3NqQs8eeTKXL9l3L/vh3kEgt8BTx3rl4xUJ9mr74iQIiMR4CFU5ks/kynSx3vehjFW6Z2c4jVTiltxm7UclJzV2PYjgGZWNTfmJ5S/fQawXuGCAzkIAvVLfxseFV3dxqCXUdtD9HNHWBVtdZ9sTG1xPTfYA9z4J0l7IJ2QIcxFccSPoirX2061UmNqaZ1/3imfRtAZb0LwxAd77x6k9DvM6sG1vYifiL23Kt+RPjrygFtJPA7WrcVcmyIf9OW6jBazPUqIfNh7f+NjIwiESilNzy2c6ti+EKEtoPkka7S06jJsVaiQf9qVljdwU+h/uVR3DpcB+rbjBZYxLsGSCyMs4MJ6JHY8OrJPymiKdz9Zk+JDHMl947ubC6Dj6EQ2zKLHM/zL84EoJK+XySpBKhq0wDaHXq3YihZ5SBeYRKTxAyu0HU+2D7OTGth5JBiPLqzX/IVenRBU5dE4jZMiW8HXk7o/ptM7lz05EWqE/RNBf4j/RXcAAAAEbAQCqn3NqQs8/Q+Ix2Bq2+HQ33x9Rg+gBJ+RDRindGm6Z5V19dPw3tcwnIyAcS/Wtuf3v7GVimN1Z6wVeHM/e8B+LDIkLVfx+LpzxT3aY0S3nN4OSELMaPj8s2gJ4VeRbwdoCcvtfS9jLG3m16tYWmWnIwAnESghto/kAY/mynibmCRnB2C4+68dfN1hnQN4thbLPybPF5ouhAuLIilUZqG08gy3CndialsZI+iMX35cdKOIFOf4RuyB+/LdRQZAFITEDEmU7WzZ4N2Ne9GIiNyr5PrFbdTT5+8LAsHvWMoUByYw7zC8HNCIBFFuQKura9CZdRn0gtTKhKs6WA0vR+3bvZX4dmSMYyL6xrpjhytUe/HE6T3w7C8CGpAAAAQcBAFUn3NqQs/9BMpS6/scEbzMSdfB+ys4zSy6a6H03rjNESiQCsj1njHpqU2M7QGjQd7uiyMX4b6ywY8fDusDAH/GQT6Mc+a21/OncskOFQpJ9KIojLPM1kGBlOX7mQdBwB0iLy28iXj2xoHDX+J6x1JC26jMtH2zJZJLCKWesjYSBuLJJ2SbsEmB4fgQIdhUTtW/JaGvgEfCV2IJTdsFMaA4thBYEyVc++JMEkAulrYovynVGoCgXPzqe3t8s1Kwz6SMYiQiwrKK8WEe1szjZbfPP3b+vsh5oiwvvD4OxBhFc8GQh0yR/1fNqEYsDE02QlgwzMTP91XEsTDRS1iYJUb7uaQhT8AAAAigBAH+n3NqQs/9DMpiwKpRoAIqP6Rc9Uee38Nvd5KBkVIA+nO/2CvMKpPeWBMRhw/BQ/pPELMAQzoWJegiwGVs33uk2PTpmTHdXkDfK+v+V8/Zl7eNCuGTG8G2JbGA21svsmSVByzi0zxNfuy1+jzJ+Wvc2Gsgt4G5ivuQ3YO5q9cxgR/DEToDPicuvPyYL6ZQnlbLq0fK0ZPG63HK4Hd41PtA8AmO5WNhuLn3bQQNaWpzqtm4E/pCT0HI1KMKGOfzPQpPOkL2fWGKVDETdxLolRX6Y9kIJwUnRBI9vyWWZodg4cuXWNeqB1AIBbbQ2uAX0KPZsgRiqQaqRNJdl5IEDHNm+cDV24WXrXs1qYY3PgRFreHg+ZwngWL0H9X7YjCnIVdWwdzkqVsALqcSiL7OdHyyzxrgMt1nYrbVderCzzLQmLHgbyvEGZmB75w1KmbT4uUV7dDO2DRr5RSC5xHKW6pXP1e2g0+punWUEHpGWfV6jN4jdmUPlhLu023j/aijKGQ84GsplWokbhVB9CIz2BfgnNXOHtrKKP9sVRywWlIy1LdbZ+5FNUC9T6fteyEs2+nW5HDOvFd+LPz6k2FWYHPA94CHd+urwT7HWFOhH7Fuxxn67qZ41BaoJ8NTDUfy9gTgRDwos66DnJRz+Mwb6FEdiZidgEGIipUU02csIVAIHVEdzFUPuHQDWkhblPb38IewDnr9Z6CU082pkF3u5voXAIO4whdIAAAC5AQAtMfc2pCz/QSURkKQ14AyOu25OFuJNO2JNuPTi7OOwcWIYeTZhNlQUnRv+1GALYyWHmGcA94S3pxr3Cc1N9fHAX6kOswMjHwcVW7eqlK6ZSvkrHhOUWL+0EavbbE8c71zywrpzHeyosASpeQAIaglSbQkEI0BYC47ZWD3kVNCZngH8ZJIphaZz0V8W6rCyiyviQlmNO9AtwepNVxcIMLj9ULUsGR/4PDdZq5apRWpH5pNWbA5TcWgAAAD4AQA30fc2pCz/jBEUaSLZiAZ2sw1FTwAfJ/cdy+b3s70a1tQTowl2f/55kNFs+eeB2LXJUw+pVavr+6e1oCim/Vx9QgbCdbks2ufjJmeAN/5fwmqLhSgsi8lrOWKpuwla9sL+ZKPV+5qvvh57rrBsdIgeE/9y9orPUpT9gHLnbS7sMEzuKq7lDEqYr/jS6D0GYGSTkogEHkuYW2Ydde43H1B43wsf3I6z6JwIfCmday/3o4PPuhROe10vJpgTQ0pwL/yqXWkzgt1FuDNyNKH6mmfPEHr1lija5XZn28UIr0NYyIp98TCE8fm1m40cQzGKp7sC9Dk7KGAAAADIAQAQnH3NqQs/QySkr6v2A+CkTxJxRc92bC8afE3eX9DkXzJAznST/hexuTMcAoXjEmbutuXk3PH2qwccBHfwUvfghEQv7YrSb4U9iQinCQvUxGDIZ0qnLcOlBgnTVkVyMCYNqjl2wQ5q39ogMy3fnfelteYpptVS5tB3tIhCJsiuP4oWz2YGJnaD2Iq/Wv/wpMbI6YMHdb9pRvvMITcHL6l3/GD7K0l0sqDkWc3CsU5LjL7RJ5U0ML1UtnVNK9mYHrZWmrMtGfIAAACDAQATRH3NqQs/KMtFe6ArKLBkRZLjNtPf3VDG6vtSnfpnXeoE0QArZTgYxybNhJAMnqiawu1qeMiLJO9v03szija6EoNj9aKVTM3GHHYtA9JxIHu8TXFNjrAGABQ6zLBGgooUiPRSxNcXK6qrrhARrk3QwOx9ff1FlP2XnhsvHdA5wrIAAAMpQZt3SahBbJlMCFn/Bu8epkA9hELlP/Lr7z7M9jXmhjeXOg77K58YXGZTTbShWCniTKEypeIP6+x9LI0/RT4CmpkVR8Ht+fIpXPWkzN1lXNuRVcGO7+4btCybVlTmy6vx9oXX4pU5yOn7UgglorUVFIRzT/KbLNAAAB5Stjq2lR40oHCJUU2vtIGoH9YDCkCeLRkIs3tdYtCtH05bcJkZm6stfXHEs3PSHo5w36NY1tHhydFUUpL4lYbTcqGkDKuoH3s9IxP6nG/jhv29sRVkssXVeH8pdYFcEsmTMSRinvtGdvTLX3WkhEqWckgytm50kOScYPV95ImUu6N9WL/19uBc1Khi1kWl1e72/qLNgKIdQYvIYjT55HYstg1tZ+Lr01GjtarRdXbIzXZyNxTlSi3TYmH0buoMOsEiLX7CT3qU9uLKwfP0ktoNZHVIxU1eX/qfvkY2BlrxJkALsX8o0bgM377BHzUwtU28SHlDIHXXx5DcD36xBluTCSLZwt4AOYq/8aFOArGvEK4fkzswCY5uXF+XkVk0AwHuePfANOddyAQ1GcQlAHYgcCBuaoDIih+77cl3Q1fuPDl+aMdlGC1p2ZwHK6dGTOpBO0lsuZIvMlQpdPTrGnJdrKj4Dec3NdYXcLXzes8wJfWIgx1EjyVCf4dR3WZopGdVoyIT+KGJQzhZQdm2qOJLRGlcN9Ecaua5Q0Ov6dAGXDii9e0mgFi9Fwgq+R4aqBR1kOwTlBlN0OmPoL/Di8B9ssKv3dYnZ7awRdKlANghL2Lo00zTHRAKnGeoxFx9R+JbZ8EX/JsZkzfAEfJ5gXyLkvOPdYcA1NU4AQEcQISJsF7jO43CSJuww+zDHScwEsXRZqoTPUIh9uIc7drRnIWGGTjn4zLDM17TatoHDmW8PEWVxKGPE6c033cN/sapt/gIS0GBSQiUkOfJctnlFMDfiofcUBE27WSjeRZM3RoEhjDHpBarYsQ5LFGWkDatp/SrGFVwemfWga6FUPvPKYiZkANDAr8t4BXDnsQ0gDXR/dYvn7Zc7EaZ1FfMyToWBMTgNITAOtscbDoNo66EukEAAAKvQQCqm3dJqEFsmUwIUf8IfImOPHgfeaVZJfJTamtxGpAOj4Q5BQzdsf5G9JQCnilPv6wR47DjgRnQO8TtJgAAAwHOLTlTwAQH7d6UDkh+pFltKemnI/IrH3AePRt3Wcj6Fh54mJQKbDf8m+bhJaf/hQawlvBi2PveDY6bPR5xjxGLsB9a3axaGU4FMOQG5F8tCt2kaiiueTVF6D5nwFSoGOtQvzAKyv0ulDHAac8eMjiZfXEx4awBMaXZrknvm47W9SGlFol/B07XFaPdMsrtMYQoFsXhmrSo4XBq1KPdUEO9v86lv7cxk/mVaoIYzNo4fwougmUb4JO46dUmVvmmMpIUxIazsDVVbhLxSrg6eZjXxGVBmw9BwGK8sL62fMAt1u4XeUkv+KDqBufOA6MPHEaCIU217XVS1UFRPN+JqSh8QiJkYbwWql/ioVKOWWu5cdwt11AX1KyeVO5sygHKQ5jgIgWWzdC8ZtS44FRkBjG7QwohN6FfOKDsNtipxLOfobjNfIQs6/QfiqU2o5OHhWt8q/a2rkXDEUuqqTLtA7eTc/kVam+DAWohnyksnIK0ZiUti9DYKRddswOQfuseat50PxqVrS2sTkFbMiXoQZ8ZmANIz1YwbEE5Fsc9Vh/KvHNFUrzzQPX2t+BCS6s1DX1HxNJ8vX/hhud1LSu1JHsusIaHYfuIX84juvIwAAwPf4IGhBdHGFuAbJ1pch+UA6k9dVQxtgOW3WkzGNKVmIllbMMIrKkK7dMHEDpQCboudWrn8gyKkqXbaoUsMPdgdNQNRyTTGz1FgDeRU9XKCPCO0oiGxngixZFuodMo+APkXeOuGWrlKfa3HZkpV8pg7DmbEV+TgLWbXR5CpF6dxaJd4ZQuFO/Ut7ERpgclBwAdvofDNvOddLYPTAH2idx/AAACBkEAVSbd0moQWyZTAhR/COfUag9NuLYHOQywkCAH9YQAgAIW8pJtfUh0YtRE00AAAUf7kxEGCObnkTLaUvQT1ECl6qCYJkKzZ3V6fTbYHw55IabkSghlYZrB6/76ndy14y/en2rHeDlA1F/2bnLrBzD0KE9Pi8KXWZnZFc5KivuZzcNsejOUiCip/ya9g2MP1YbCcLQMW6lOe9te1nhuuz4pSnMErBYGQfGhecgxRBV/QhRqKAPv5Acj05Bdj1YXDJqrov9ptNYIuRI3D6/E3b3AX4Mt5da8/PwIXJVlWVU/0POcfy4Ylo3UtGeVDuPsgiQWOE8f5+khadi8fyjoFMmEYwmUbiNWmXvMYNcgm3U71nEdSVzgh/3wdqN7oKbu1SUrMLShnlD/wQSNm2ETJZFp5/TJdmwE4im51mugXu9qNt0iNtf4HyZEgoxjEyeExiBdS+OtxP4aczNEbuBiOn0KcEcRcmkQyoZvQra8nbUXolX00rpOsnDC47P9pd1SAmUVIcIPuvmIA7J3cezIvY/GiVFXzZl6e5tkLuwl+7SG0rdiFWLdygKrwghbwS+m66t9syhqjf+dv1AIiaMYIh5CADMxiiEjwwmLUBgeUQbzN5JjihVA2QCye87kh3G1hV7uP/9pIXLgQO1szV6oNLdn2EUy6PDkNumY67nALXHNhGxPZhZnAAAFWUEAf6bd0moQWyZTAhR/CUvMqriqAFZyhE8aO2hBQqNPx9fX7SaJ1e9Ril6tvRShlayrnJXmeyBoYm6rGX5LCiG89xAgriYJkJy8++jGaXKl2naBkHfxFyyVCI4AAAMAAAMA6S+XHHP4CiVnQMVT9D470eKm7AyhDgW9VxU256O4HQJhUNy542hT8zQJwvzjf9L/VJei/vzRH8phupdU9ZRWz8rs2PDTw4JjqCwBFFv9hW2wp7TbRWhik0VOJ2612uZUa36wJ4l6V6PwYipuEV3GsBkMy7W7tCnskF+OOyJWkv0trmuamJY6XyhhyVC1RDgvci6+55+bXb3oif20XPxUSVOUVfAu14tZvoJz8oyGW2kQyHJagaWtsMToV2BA6E0OG6XWCxwpE99pW7VyW7so6Par1CS9K5XzpzWsDg00E892MyXKEz/qFIlOlWRTdBwgMgemfSTMKgOnuNAw5GIT3ulduOyeqftTCt4bbqM7ETYfCFXJrCj88MLimQKsRYEawSIED1cH/UEtSzclhsNHR4djrQmZZZp6lYOHxBTUqiG7jCnmMGlC0Gvp06Ugzt8WdaoGJEwbYsD22DQUZW15sYenpr+JvD97W0tlZS/ZtJY/vtt1OR77Yf2fhQgLBMZEboFn3FhmWRKl//HtHz2NiMfSehqEG/HDZeVeEs6HoGQjggT/n3Yorv1UXiRjxyM1KpcGbuqRBVvn3f8XyB9UE1al6oSSxf5STGlgBND+55dSw3uXbgo5zWn6zHFxjWFs3MbCiF6IDaJ+nfTMSARdmWsjnuncoCfpbagsqv6a8v3Zi7m5huen2FeaRtT83tvYtrj81TAW5UOt/oUHbUe6n79vplK2pN2qoSQkxfowqyqmEKi1nnHZ84XftNi+RoDKvQsopqR76TU1GRrf3/Z4jgSeICBR4fBgfhtHsDBosOfKcUrL7O+Qk38o9Cu6GnE0z7uBj/8L4StNTyJXl920oHlGjaL3MmvH/JVhsr+NnaXlJrJriq6hvKe7d/HfoJLeuJLttu1Mt8684Gklzh1X7FjWBFiQlcu7bkfFt/WGpMec95I7VkxkMJB4ztyLRuYnM1aBDcJmP2lBmyYYtRarYBRjA5b5f9Xi9IIdfA2X5LA/qaPI5NvcSLJTs96dV8NtHXuVShYGmxxP1qgrVlxdRPpCy82ZpVRfhngsQzzYQqtbUtxK5xLYKLYa44ALf/XOF/tY0ViFIhf3piTnfk2s2ZgEHtwhyJmuRYt/GfE+Vh9WN3azcb5G9CBES3Hy/DpqadXwdbm75J5e/Lie4H+mSmfV3zH+0+odwncQPlCUroxMohWrfZCMgwo0WNyJE4RdhYhBlyMgLYS8gyHZvlSgMHGeSMG11uLTMiW4Sbldzg/Nl8/6HIzILimAQM5biafWlcxpbomD7mdf5mNe5AEyUVWX1KQ7PKndHzWwNFZycKrkoer+jMSs21waAa+uucs1is/PSpaq0/u2ruzG98oC70MEEeKjqdQy/Q1T7MlRv6K7FeAMH56+tSOL80kxKFXC43XwSp9v6ZUeixawZBCgg4DMHshxb6iXTrYyd+BGM27O7UCKC8/p5n6D/Di8px4FpPGRAYhzTLY5OTP87S7JBxLWomIsfPaU9wm/EOzi9vfRFBEm1zlwNPjlM4apWjtnRPtRMxrcGhEPxvrHTZJ9fxW7dSioZIBsLvMs+WVG4tN7g9T+N/oWn6MkUo1YYIqEbqJE5scLkh3l6Prp+Kowq9O+8Qnb70zBQ66SIZZ6XF2v7X+1K1lyJEmQzgdoNBYVgTsFGHSnv/S4pVLn8ietUcdARG1vBfEAAAIbQQAtMbd0moQWyZTAhR8Jp5ISjGVB/CCznP3JmM7M4n7/zAUf1DAJr8EMoojuabscm1v2ey9oRZV87TN1lbbkL8043qV5kx22Dd5xWwNcTt+CtbmFwM/uWlS2eXB55dejIFa3i/0ZXTNWI8MfE1X8p5+foL6ubnHpCTAk7cNm3/tPS2AUva2kBDzt0N9HZq9gvbUwu0RojqTQoou525qXl4CU/pNkf0Szzkx0maz0cPZ3vVWw7brMHB5eX25ENflDGByLEbXPFX3yzr73qAFkKIs2b+GOWSnS3i5opOrkk3OS0LzrdSiTuIO+oUjYbRpNwW8n5u/1qDJ9T+gUY8tohZyXKdoXjPFLjYaWHHTgptWVxTCkgdQFm/ryuWv5EkqbbnwLdRWl+RLkErh/LQ9BT8H1+oOfRO6xtOZYa7zcqAuxDzR06rTKzSPp1ujrm+BFR76lEs49jOtQw6rMPO5GRpQiPhiSl/UTPKB3n1ShYhVniYH0hZb2EOKfdEC1KllQ6r5/DtP5gC3AdXuyOE+2qpnCwYg149ZEwYdd2zESlsdebGDOpaZVAuphAf7wG4luym9eqn44Dg34FvL3U0kEuaj8gsL8r1Uk1FRXZ2ma6g4F5kxj8QhxQFueU0T6UCUTXz9dvtqkBYUZjHS+ceoJDzz8sM92AYZpBuoKYNs2FL89RcQYqCHNSUtUC46ft3FhFAJFhIQIEhmWLkEAAAGmQQA30bd0moQWyZTAhR8JJreBf3P/a9/N5/FFitH6IwEZhH+VJovJAAADAaSyY5dvWGG9h2kTS5nORI6KCNeL1560TDWBlAgVnkO7MTRynW2u42E/kfr1TKyqtwdA1d5ZOTqo7rb3U5jJAW8yU31ON1KOj8l+oBCFQdUb7HsUvK8nXlPRX8EGWfKGW5if/n73yP7zWK6KnCEdX3np9XVX2hAX+I6NcqqLXKz0p+eHjNOz6Q4rRk/2hUlJxTqQQLX07Lhi5b23+K+aJFGY0EUC/XAGEcPsalSCopJM3jEuP0LFw7B+OEBrjlbo+f/+a8lJsSGsc85//IHZ9QgFblvqj0GJUoYynHXY/pAgtipkcxdPoRTUiyb1F1+YeGF13JQOiSKhanpLglvhZbJNPmDu48kAFjp7DzSIt/l6pTNXyzTRqy2EQ456t13b8p59zxPYigeGdunqkHbLtEsnPGouDYlX4oUYdYXO7yUuQX29GJ7FjMQCVAaccGaP4waIgoq3BviuGgpFxPCBikswOwhSxPGfsi1hWRSPH48oUgGnlctHLhyoC18AAAHCQQAQnG3dJqEFsmUwIUf/CUvoxOg4EDLSwOjI6RwINM9AAAPjznhYTi/FAyCQMJOdqK0PeEwDgngBifK5KsGN6tSkO89kevSQTe5kZocgspE5Gypdzz95rv9a8mbE/oF2i2T2tRLDPFaHYf9WuLTecT7FywreFYHer2pTh+o5Px1/mptJzaSiphNGVSMvghHpzmIqeTikP08CDQVTHGHa4Q7cVM6eTTTCSjTwI5Yb7/LzVWJ8aeho5673nXhqrEei2/uKncrehSedJ2ULggyf4roHwfndZukPm92a2l3JruIgyy4i6FwD2EuDFtMO/GXlfrKJ6ohHsaIKt50nVcj7uNV+iVKcQ5RrEbeweijZ49Vaz6VtRT5yCYYYIl9TRZMXwJkLQXwZauxbvs8r6rFqbIIjbh6t5tSL+bK5Mz2KoOTVL8Bv7a0zVJBb4cZYi3QvlktlmSRSRyuDfv9jQQOX0qmQ9dK5OckUDsNQR5apzCRMRnLPgl/TS1A7gi+oDLE+NubaDPp6hyS+4oo1vGgQ/U/ZunG+UInQqW1XBJS6QoJWhNhq5Ed99RVd92sk16LFhbgdFZAjYRx4PgcKhthu/tOBAAABR0EAE0Rt3SahBbJlMCFH/wyNbP/QCqIcoAOfM5U0ycS8BM6+/YBynYaJpRwsY5XE7K1yZGUMnncPO1i5dICrPVVpcmtysUu0FVjOciu4Ry/IGqZpPvIhWBEwTjFLfo8P0MTKv9VTleBrWkt7nz6bqONg8MMhxzVM7lyo4OBtutJ62sN0GbKrBO9IyuSVE7DSRgvv32n4jQ9xgU0nJZEaLz4FAzoUJNa3tuPXVMEs49aocRnm26ZVZBMNIaGFePhwJ7DNkqIiHcy41mBaSpgttsOaUU51QD5MMtzIaoWpoUg2SWqB0JBZb+jAJ+br/vIgWXvmzUq2Lr1Xhm3AYK/10efKKWd4RZKWhTo470ws+LBIIoGGZtJfg5vOCPrsVXOj70DxzV6lz0kYFBp4NV9tXdncqB8f1uVjKY7r2M4a851Wp/ddxcZ7IwAAAnhBn5VFFSwo/yM028Pcfh0+5Dt/xTHm2VrkH7tfjBJzTXdnwsmPp8dOhg/d7b8x+S5lF5+O1aLdjtgmlspqTBSsQUb7zAGip1Uiih9Uct8ddHPbWvGyh44R87JuGyrGU6ownoj09NAHbyBL63uZQKeaee6xrLOiio4sbGuDjlnkiwUd6YJK6oPv2CvTr2I4vD0fz1Y6SikG7LpWdJy8zRgOJ7CNAN7DW8n+Bt++eFchDmALJn2abqUIsb7y4FjxG5rmR1VYg6FBzfzwdxzbcR3PXqvIcISRSptrOk+iUXm9sLv0939iOHH+CZFcmDLxDzzCn70VP/jXqjO/vusCfJlWJj7Q7t66ZZF6OySSTOSpq2fQkSUD+mFJ+ynk5Q6arKbjKEhYYTyto3+ENqJDkXvD/nPlxau0NcZt6w2dRC6zjC9jp4VF0l8bkBrSMlE1KI7IN+UL8j8RBTXepDeh6qsKjO2u19O1cy1L3eWBxr23zFquhHbe0a86mh7jq+UsE1SlPrfZb40cuMYhsVX5Sg3wCNhXtlK3bjfoxEZWAhyVPbdAqd3eICJT3ZaHT5l+1xDxFXJkXvHF+AkbNB9saMhXsl2JETXMNDH4HnWOqLwGE5dG6dDJMDX1I4RFwEgdDtA/s9MCqHg2Rm2DskgcG8e5W6+uOVpx+/hfTsOPN4WpJeqxy2ytNe9xTbhUTmUJB+qKV2nzcLYwuM5/Dlbi9OQCiFjIcU8YTWVOHBWF05PC6A7nJPVFceDcfOJP2QvyUDi7diAGdWBbuOmr/zKsbSr1dV92K7H37mO5DirT8durh81tNuXwjwkPeZuDMrjRy3jmhdpy/mCfGgAAAadBAKqflUUVLCT/OqmYYkPPKBJa7LJep5Dx3dyQ4+kacduATt8YU9joojRQH8Lhwn2cJnVYSMfMdagdQfi8XDTKTSeYrYOwVujrOECKbQRs8Du6YXJ/kTrHsEuRpE/yYgU0IIGzjkWmqybo/YfmOlSVUneMu8sK0L5H76NgJmkNtFs8v1v0inOGsjwUTJPGEexeLBNGw+EZWYZ2EW+Z9oS4ZH3t3aog364Mz5EWGY8671oIv3fd8yMMPWWGbXPLx1o+h95ulVXbNjvEg1wdC5SSd4bHtaTtfXQ+j3V95JuiADg1znQF0IItggyZKheGM5PIGMZoRX9jFVhGU9UtwQ2ry7pVtOmQXyNuKe24VhhKwKUesaJQE9u06kmRtYGzv2ATM0Hc1lYRRUqFCtlLhTTjWWr/sEiFVTqM4/uQDfnsO5k1OEuc+2W73IcHPtVMbSylao0KSkHrkQJu9ohA1rhKfV+G3X0MrpMtTU6Mi0BHVsmTMUmPkHD3BHyukAMBQ/ShChJJ02fxkaVIkQnsuQo0iUEeTWOmKn5hEC9Py9I39W8ZHB9hXiAAAAFMQQBVJ+VRRUsJPzwJvG+AU7X96Z5tvQ+0gBghfRp6Y3yBVFf1TQ6CYYkC3LzcZlYle0azuTlyjhTlG76QCcux8QAFxU2zfK6Oe8/A86eLibCKn4UENm2FCFRblBA6f26tOO4o2G198LQM/DuJmydTGmrZ66UOtUWJecUl+WE+MPGjTtSMm7/otGVXpa3AXqTGnNGZb5MioplVvGIaHe7mpefMem4WU/as4HOIh8E5+k0Fqutod9GMazl8m9J7H2XSIMqIMZ3YBbQzrBC9WPNVrNvdpV8GewpaxwqVoZHOqOFTey+yxTfdm/QmgTu9dtXDeHukU6FA9VrYcHZWrX3mYJN9OihO8yHxRSabzUsgfbGNqVfzV28QFDYGvvUkNKdUhohBSs0jB5dyLF9beSK1CM5hgTrVKDm3mykBAE/BHsNOuRv/LNnS/G1O9+AAAAKXQQB/p+VRRUsJP056SFAIn+Td82QaZN/n1TNamt9+0z7IAQGUVFCk61OmdXT6k30MKFCkGdQX0VUlp65QjtbCf0WDRaCbkNdIp35PYjPGHThD2gnd0aZhJyrDk+qTCL556MkMFtMyEPig3VqXvs7k16Iqm3tQsexRNDZl96Lfj/ZcVFjvpUtOuEjHISB2cSgzS2WEsqDzLOCskCPwcJnYCtzshsr3RSLFdNT3Rko13tMJ1VU2lAjVaospZqhAibh8sXgXP8YvLduVQ8oOpfV7Goevk3ze8DVKcZdVtlXLxECnSKrlASQREc4ez8E9IRI/TEwQOHU4LdNsG/ogwnV1AUNDR8rsKhGH1dehNUNb+qrEAusBf2TVqQ4X7B29MNe0MLxgbqi49Mp7TvPwBD9zfwwFzchNwGXsffUBN7KKgHaoxk5YijsuK3XsL6A9qQn4Bo4dho1rwU7hy61D6gIUonnbWxQWj4mULfJUgHaIBR0k9EEerBmqtoTrESe7QEPX+VjOUFt+YHAurjhvOJD2mMjpvJOY2yktjcS2v4rzZt8VBbo/sO4qltROL9AD/OwOHrRfLpwOvif7mdXkY35TSDbUTIVLNws5MYxTKAUOKBRS682e8c+5PiRFnsMlLqgrKoLgnjnEp957U6UZZE2fgaxE/CJ8Mwi+JI4JLrNz72zkUzy3IbtjJukI0hBKVEFM0Jaetbnzzh2vy1W7lGym8NAi5MuC5Nnr6H/TNNxPbQ4m+HMLS8AmsAUFzIDK4CIouhhToU3E4dvOdSUN6Sui4ZIBMASOrX4218FBw/Oqr50g3xcICp/wlYeEr/OyjO83O5y3wa3pdo1QNqhKvKMid5a/oG5pIEaiHx9IG1yQB6JRoCo8BypDAAAA/kEALTH5VFFSwk/XtQwIb+P6TlYXb67ovS/gt6wzwV36ZTdwvbsJwVneCsf+1Tz4KyhWSC2iusfkDrFPTCrXqMNqXMlBs4Xq/TbN+QJR/fTmF9QUS4nw1Kxbs75NAR/wa1hKomThf+YgD3mA37W+GNXirAESvYdYUy0KXRpfcLK/Fgf5YgshAFsIEbSc1d+X6qQ0i90pHGVatFpaa2qkGYaVGsrIOhZXOI938A1OL4GJy3XzIJy0T4rWTXc82n98w/DaZccj06FEo67+YWY5ibxfZuUvTZl2Da9I2bFhzXWMayvnAl8r/0WVbglikgsj16zhOGaGLp25bE85epvAAAABCkEAN9H5VFFSwk+DosdCE2lr9Vt05BE5kZdJnBZznMcqEThR0IkwmbaH5XxDXw4wd9Ta5wYYrk4yAVPgq0Gpm5iLataIDsh11MZBtf9qW9Z4WINh+zCJ/3wE8TqjZzHj9yZhVUBPIDjBCrFIby5W3cZ6Mw5KbrRuVsgocGWnJjM84QILVamlZdKw9lUQn8uV7LaYY4/4/CLDnSBwqheAgJxEQgfmKGtLgBfqNY4uPz3qHSe0et8jh5ybg5h/eRDUipIzXo+5ZE5GnXWPS1/RkP33D6CCKuTBgnlveSe47I9sNFO0VIRmA2Ft0nApYkAE/tFi8/9gLmI4DuB2ejpLEQp3wUywOV296skOAAABD0EAEJx+VRRUsJP/jRRoAHIZU/pu83SRQpZbsj7ag0atBcN6tKrHFlELmxhYdac+vwa0+3q512oGIe8ZgvTQGx2Ujg6WHmLG2e61GI6ihZ6BCssa+M1/exyG15LUelzQ64BFEyIWNoWaLMaw3RkkYiSbGtxmqU8UXp7PivoF8ygGB8e1j3Yr8lsqtneD03sTa7wjYnVbT9A+exTusweTHGaOI2sgb190VXQaW1SDINTE4Ep24scVdb0ikkPEVpgRN5qxoH9IK8c6Nb/VxzVFe3GGIaoJwBMJhsO2BMYSFrDQOZLHXkx/2YorT9WD6DBGApmQJp1dT+w1hNWnm0WJSUTVg/WSM2LKuxOnnDp/66QAAACyQQATRH5VFFSwk/9LXY4BH4dVOe6AGKhri+/FwiLRnTjtOoRtkJ2qwZmjbHs0VbTGhet/a0u4Wo7WiNL8jndovS0pMajalJAEWHBioH2FTFzxYMPsb5oHNx9NgbzlgRi2Dfare6yi/jgnakiyM++UFDhRSr3qrjXAMVuLhrgyrie4zvOVOQJpoDtFX+Mdrv3hnJAPiwDXUlGwmYXedZKUkDWBuBRPyWpNV2udnGeQKEBmQAAAAYkBn7ZqQs87+l+sv9iOBCHQK2Ao1b1qRodZu33Dwywlg2+C4Cnw+GEvfnN1oIEHd03ebwidJEQjq5nqaLWcpOkuAA6PMWhgWtQ5rgGyq9RCZbfoFwt7qcmueuYj7vLiKiFGASyP8koFiEL5zRcGARK+pqDIbLPyDwfV4ilv/dxNz7UxohplJmTeyP3Uj+/4g4+w86A4xnsPWZjLVICBthvxX+HHGyXFlBqFXfBncdGc9w5OGwVUARmgF6PWPE7gPtpciMYsTgrC3zyxSmpYAiWNFHnNsbAZV5BDel4TshYDBuKxpOR6qvDQJC0XD77Tz6ePlYQv/PvfEwBti2Dx40Lm6yLc4cLwDxrFNyb+d/usfkP4mqQ1mRB99Sj/LKNCJ1dTJMDqEveqW9rljKf3u84W83yhsgvYbQ+luGxfTsZE/cNb3PVLR1821XpDOMdH2fPv/Pbhs5Qlm4MT9Jvk3LFyyVLWAQpiodIHdFgHPqF7n2kuLput6PhuN9rVyBVEsHyIG7wNPwvEKnUAAAEuAQCqn7ZqQo88rLK+VrvP+gPGJNPw1BI5DwJUaLFdgalAI0cSSDI+ImcVZMjO1VIZR+49sFrobKWow74XQFbqKF1CQHh8Sj0hfSuBzRlU5QB+QXZZ5ojtdVRXe64J112R+n+3VWm/Rz6AOtlaL+0lS33DEdrjRjJkSwRJVQOkgcUnGHJZ0khEEOymuHfe+YSsOw4VaD5ZI47VcNzKvlZtIhq0Dp1EqJSu3paK8q6FNhZFQA8lWph1Aa2tHp7YfqRcwoyMx9arohUe4TiLhKixeJdZnQ9XRSYKRRhwOOcDSMSJ0W7ThlokEXmTnq7Jm0xbldapp0Q1Rc8wXtMXWG8VIt9B42H1d+FDTNrQYMSa0OaZ3ZInhYY3rAmXoqc/XzWv85D8mB0EiFfmZ0VawhEAAADVAQBVJ+2akKP/PXZJo8ZaGIuwUFFPqPsmoE1OBc6l05av7+Q0SaE0+d/G3kYV6cx5Fvrz+HZrbexjrsB/milxjBY1SpVsNoF6ii8O/yd1CiOIpaLqvbr5MJUnpC9ELeC5LbSbNAQohtoPjB9B3xqoieSnkMTf56m7Wy57gUJwMv581v3KlwcJSkd1/wd6r9JmR1INb32lMQfwX8MvDQRWbyQISomRui2OhxabPLgWHyzJk72XmNtttFcK9/ae7ZuV6b1Z4dNX33RC8X+GPZ6Ybdl6mma5AAACAAEAf6ftmpCj/0FjRPt9wqg5/Pr6yBcrYk0zxUMtMkgkaaLoGUL8+5sciTrzebZrxsyLza42fem7INe2XHQq4EuaohT4504HOFzpJ08SPMatT2iPhy3/ySg5w79HZrEVmSXGDeJpOj09x6BiAl3/TpJHW55vyj0X86K4+w6de43oYzy3TTynZcNnpxiBnBv9dCRuA9yiwEzbpH6Rz6DCKCBMn0Zy75CicrPn4vF0X9eauekC5tDc+rmkajSbpdMMCGceJ2AsjJXiSs6kschiR8rVEecsRvr8as/vcYe/eLwOB+pftehMQPNlDv3uEN2sUHTfAdtbwcbQpQNcz2F6XYoqV8d6vphU8RfhAarelx2MPpHtqywXSde77l1/kdXskAkjR7OAAf//jVzEUivW3+7QtPWooyGMYsBGyu7OTFNdVEz0cAv3jQVkq0KkqDn88CSsmSqJh2leHrWu7SaPRuw2QJFYFws/zA8HoumJoINteMB4k/4fj9YqUVfHzjNSC8MArlR91uAG+bwfIAu7t4F2DG1s5Cy6xDVkxQM8rT6YxTovPZDI+HVRtIaYA4xOB84HTO9/8GB1qERNjAaytvQWi/LvBxYGTo4fRFNxFPJrj6XsUoxtbDl0uixamUSuuA5OeuYxGdBEsxtbgKIFEdmfT3XxHYY9mYNGS+pZ3ovBAAAA1wEALTH7ZqQo/0FElT8sf7CvXilflzgC2Zst4DhB2/DjnuPApahkOA8VbQAHJEME7TvZsYjHzCAiyDHNORjUbgFTxvf+TUbuDxFmqESktCG+1XNGBFL9xoUXtc5TGvufbsEHtKlCXguptTuKWciR+2Cc5oRcq9XOfQqWAtq/0K4+c/ZuJv4Lznymy4GI1DCZDrwYiwPbwOKnn+pzQcRJ4FXAVP86TJq3ggWLKuMnt0KeyMYq7qDy5FcILvM13Czd8eZkRfA8OBK2kVeMnIFA03x2OYqE/5d9AAAAigEAN9H7ZqQo/4eD/KXvqHswAQfLIf8spYh87/xcwB27u+aHwOn+zOCvM0Ng915ekQcn6jda/AGpTTQq+BoSv28/1cY2IiRSKXqO9T3FF1XIXRNdsvXc9OD4tfNun34hUUwobYwrNwb0ORN+DksHiRDYO2d6BSAI3ypCRL4dUubXgydGWJOLev5ILwAAAM0BABCcftmpCj8+8aEDM9xCRZWM1tCl4vt9i6hI21skk8ygh9kd/P0j32+CBBWLEoZdXC9l6BDddajsdINjfKOuXbAP/tB7QDWs3CxreILD2GHGfdZE4LAZCdTpwkwAXOV3k0O9DttVZT5FHMFFFGveMqJL1IimThHkLmEvhd4xd1jVPstiFByymp8pDgNK2XOSmmwwnHPV59KqVnEGq4+oUILLr9RsP1z1IYpIUHGxmuU0JRgm7NguMFjP9YDWqZaDpmjkkgVBok/vSEP3AAAAngEAE0R+2akKPx+KeakFXk7xoIYbIdZ7vwrsYvBAv1qEU4OQ5eEIQnZuyAz5/9FSxRhw04rARlC2mAgWdAHKAfN1M5s7Qnoqc88j7slT3dOiC96OSAdAWMaYZQfg4BqYLB60q/3HxjY2+jDlSOxC4aH59M7UI4mMtSh0bjEZVKYkVK/lQWBvwnXVix5wIoRLE957zhExynK9piFtDOaBAAAEQ21vb3YAAABsbXZoZAAAAAAAAAAAAAAAAAAAA+gAAAMgAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAANtdHJhawAAAFx0a2hkAAAAAwAAAAAAAAAAAAAAAQAAAAAAAAMgAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAVIAAACBAAAAAAAJGVkdHMAAAAcZWxzdAAAAAAAAAABAAADIAAABAAAAQAAAAAC5W1kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAPAAAADAAVcQAAAAAAC1oZGxyAAAAAAAAAAB2aWRlAAAAAAAAAAAAAAAAVmlkZW9IYW5kbGVyAAAAApBtaW5mAAAAFHZtaGQAAAABAAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAJQc3RibAAAALBzdHNkAAAAAAAAAAEAAACgYXZjMQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAVIAgQASAAAAEgAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABj//wAAADZhdmNDAWQAH//hABpnZAAfrNlAVQQ+WeEAAAMAAQAAAwA8DxgxlgEABWjr7LIs/fj4AAAAABRidHJ0AAAAAAAPoAAACzqcAAAAGHN0dHMAAAAAAAAAAQAAABgAAAIAAAAAFHN0c3MAAAAAAAAAAQAAAAEAAADIY3R0cwAAAAAAAAAXAAAAAQAABAAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACAAAAAACAAACAAAAABxzdHNjAAAAAAAAAAEAAAABAAAAGAAAAAEAAAB0c3RzegAAAAAAAAAAAAAAGAAAPbgAAAW0AAACrgAAAd4AAAIPAAALGQAABCsAAAIVAAACqQAAET8AAAcIAAAC4QAABOYAABwgAAAOcAAABtcAAAi6AAAaWgAAERIAAApxAAAI3QAAFCEAAAvrAAAIeAAAABRzdGNvAAAAAAAAAAEAAAAwAAAAYnVkdGEAAABabWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAbWRpcmFwcGwAAAAAAAAAAAAAAAAtaWxzdAAAACWpdG9vAAAAHWRhdGEAAAABAAAAAExhdmY2MC4xNi4xMDA=" type="video/mp4">
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







