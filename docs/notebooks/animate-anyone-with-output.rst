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

.. container:: alert alert-warning

   ::

      <p style="font-size:1.25em"><b>! WARNING !</b></p>
      <p>
          This tutorial requires at least <b>96 GB</b> of RAM for model conversion and <b>40 GB</b> for inference. Changing the values of <code>HEIGHT</code>, <code>WIDTH</code> and <code>VIDEO_LENGTH</code> variables will change the memory consumption but will also affect accuracy.
      </p>

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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
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

    diffusion_pytorch_model.bin:   0%|          | 0.00/335M [00:00<?, ?B/s]



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

    pose_guider.pth:   0%|          | 0.00/4.35M [00:00<?, ?B/s]



.. parsed-literal::

    motion_module.pth:   0%|          | 0.00/1.82G [00:00<?, ?B/s]



.. parsed-literal::

    denoising_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



.. parsed-literal::

    reference_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]


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

|image0|

The pipeline contains 6 PyTorch modules: - VAE encoder - VAE decoder -
Image encoder - Reference UNet - Denoising UNet - Pose Guider

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

.. |image0| image:: https://humanaigc.github.io/animate-anyone/static/images/f2_img.png

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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4565: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
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
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABG29tZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAbBZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0VHZia55nYBODXH5bs5XPcb/rnN+t4HEly/zRT1xA3+a/EQnTOKPH1B5FL/2VsxmReQ8s1wlisqe8gkwstmOODmrg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" type="video/mp4">
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







