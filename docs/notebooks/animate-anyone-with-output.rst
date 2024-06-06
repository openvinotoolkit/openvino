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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-697/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-697/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-697/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
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

    config.json:   0%|          | 0.00/547 [00:00<?, ?B/s]



.. parsed-literal::

    diffusion_pytorch_model.bin:   0%|          | 0.00/335M [00:00<?, ?B/s]



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

    motion_module.pth:   0%|          | 0.00/1.82G [00:00<?, ?B/s]



.. parsed-literal::

    denoising_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



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

    WARNING:nncf:NNCF provides best results with torch==2.2.*, while current torch version is 2.3.1+cpu. If you encounter issues, consider switching to torch==2.2.*
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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-697/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4481: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
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
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABGhBtZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAbXZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0VHVyOdonpY4kRfW9w5BCOG2Zy5eQKZGZRXebl3sXU8EwRiHen79QO1sXMDb4Os9XfkXrCp43yo8HWHFtIXb6OdcE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" type="video/mp4">
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

