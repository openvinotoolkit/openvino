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

Table of contents:
^^^^^^^^^^^^^^^^^^

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

.. |image0| image:: animate-anyone-with-output_files/animate-anyone.gif

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

    WARNING: typer 0.12.3 does not provide the extra 'all'


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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
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

|image01|

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

    WARNING:nncf:NNCF provides best results with torch==2.1.2, while current torch version is 2.2.2+cpu. If you encounter issues, consider switching to torch==2.1.2


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    +--------------+---------------------------+-----------------------------------+
    | Num bits (N) | % all parameters (layers) |    % ratio-defining parameters    |
    |              |                           |             (layers)              |
    +==============+===========================+===================================+
    | 8            | 100% (32 / 32)            | 100% (32 / 32)                    |
    +--------------+---------------------------+-----------------------------------+



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



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



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



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



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



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



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



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



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4225: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
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



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



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
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABGl9tZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAbMZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0YckLNMaV/mo7eHFEHmeadh2eReP9mR9bguUgIizCo8P0N+ZIyqww0PoXM504qduz5WCsaWL8gvRuvZN/rYXQua7G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" type="video/mp4">
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







