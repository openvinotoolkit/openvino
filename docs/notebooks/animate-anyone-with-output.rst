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

.. |image0| image:: https://github.com/openvinotoolkit/openvino_notebooks/raw/latest/notebooks/animate-anyone/animate-anyone.gif

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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-681/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-681/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-681/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
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

    .gitattributes:   0%|          | 0.00/1.46k [00:00<?, ?B/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/6.84k [00:00<?, ?B/s]



.. parsed-literal::

    Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.52k [00:00<?, ?B/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/154 [00:00<?, ?B/s]



.. parsed-literal::

    reference_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



.. parsed-literal::

    denoising_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



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

|image1|

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

.. |image1| image:: https://humanaigc.github.io/animate-anyone/static/images/f2_img.png

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

    WARNING:nncf:NNCF provides best results with torch==2.2.*, while current torch version is 2.3.0+cpu. If you encounter issues, consider switching to torch==2.2.*
    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (32 / 32)              │ 100% (32 / 32)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



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
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (40 / 40)              │ 100% (40 / 40)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



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
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (270 / 270)            │ 100% (270 / 270)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



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
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (534 / 534)            │ 100% (534 / 534)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



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
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (8 / 8)                │ 100% (8 / 8)                           │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-681/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4371: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
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
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABGJdtZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAbPZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0VHcKEycpIcq29VV6TmoAGGQ3SGjKsrzn+EjETRiQO94Kq9mY8t7gsI3sM8WeIb+FURjXBJkky/+wTQvLJ2khxFMe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" type="video/mp4">
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







