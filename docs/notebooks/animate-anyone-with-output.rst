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

-  `Prerequisites <#Prerequisites>`__
-  `Prepare base model <#Prepare-base-model>`__
-  `Prepare image encoder <#Prepare-image-encoder>`__
-  `Download weights <#Download-weights>`__
-  `Initialize models <#Initialize-models>`__
-  `Load pretrained weights <#Load-pretrained-weights>`__
-  `Convert model to OpenVINO IR <#Convert-model-to-OpenVINO-IR>`__

   -  `VAE <#VAE>`__
   -  `Reference UNet <#Reference-UNet>`__
   -  `Denoising UNet <#Denoising-UNet>`__
   -  `Pose Guider <#Pose-Guider>`__
   -  `Image Encoder <#Image-Encoder>`__

-  `Inference <#Inference>`__
-  `Video post-processing <#Video-post-processing>`__
-  `Interactive inference <#Interactive-inference>`__

.. |image0| image:: ./animate-anyone.gif

Prerequisites
-------------

`back to top ⬆️ <#Table-of-contents:>`__

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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-707/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-707/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-707/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
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

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

    README.md:   0%|          | 0.00/154 [00:00<?, ?B/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.52k [00:00<?, ?B/s]



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

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__ The pose sequence is initially
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

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

Denoising UNet is the main part of all diffusion pipelines. This model
consumes the majority of memory, so we need to reduce its size as much
as possible.

Here we make all shapes static meaning that the size of the video will
be constant.

Also, we use the ``ref_features`` input with the same tensor shapes as
output of `Reference UNet <#Reference-UNet>`__ model on the previous
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

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-707/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4481: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
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

`back to top ⬆️ <#Table-of-contents:>`__

We inherit from the original pipeline modifying the calls to our models
to match OpenVINO format.

.. code:: ipython3

    core = ov.Core()

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABFVNtZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAaAZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0VCiPRyeljiRkXixNKBkQjfILGBd35eTm/k119RZ4CmiL0R5wkQnNIgguj8QJm0+aeSF0BwaqfwL2r88qWl7+Ji+j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" type="video/mp4">
     Your browser does not support the video tag.
     </video>



Interactive inference
---------------------

`back to top ⬆️ <#Table-of-contents:>`__

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



.. raw:: html

    <div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>

