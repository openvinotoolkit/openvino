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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-674/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-674/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-674/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
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

    README.md:   0%|          | 0.00/6.84k [00:00<?, ?B/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.46k [00:00<?, ?B/s]



.. parsed-literal::

    config.json:   0%|          | 0.00/547 [00:00<?, ?B/s]



.. parsed-literal::

    Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.52k [00:00<?, ?B/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/154 [00:00<?, ?B/s]



.. parsed-literal::

    denoising_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



.. parsed-literal::

    motion_module.pth:   0%|          | 0.00/1.82G [00:00<?, ?B/s]



.. parsed-literal::

    pose_guider.pth:   0%|          | 0.00/4.35M [00:00<?, ?B/s]



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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-674/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4371: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
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
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABGPltZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAatZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn5TPBnJwhbw7lJFEHm4IXS18bd69kfSysnwZIc4HSBDWp9nO70W490tH/62cwANdJu8K50gHi09aK4IOyy+tZy0oXO5H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" type="video/mp4">
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







