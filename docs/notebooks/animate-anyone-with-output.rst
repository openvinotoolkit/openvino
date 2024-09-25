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

   This tutorial requires at least **96 GB **of RAM for model conversion and **40 GB** for inference. Changing the values of ``HEIGHT``, ``WIDTH`` and ``VIDEO_LENGTH`` variables will change the memory consumption but will also affect accuracy.


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

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)

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

    from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
    from src.utils.util import get_fps, read_frames
    from src.utils.util import save_videos_grid
    from src.pipelines.context import get_context_scheduler


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-780/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-780/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-780/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
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
            repo_id="botp/stable-diffusion-v1-5",
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

    config.json:   0%|          | 0.00/547 [00:00<?, ?B/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.46k [00:00<?, ?B/s]



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

    pose_guider.pth:   0%|          | 0.00/4.35M [00:00<?, ?B/s]



.. parsed-literal::

    denoising_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]


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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-780/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/modeling_utils.py:109: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-780/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4713: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
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

    from notebook_utils import device_widget

    device = device_widget()

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
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABHo5tZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAbDZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0VHZia1w9L1x+WzEZkL2VKcu6nG1VP6tfI8uRk1Z+smbWM2iQ25fqMyLvnQUIfTN5LQJYbfU09T2XvTCtb6nO/Hzc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" type="video/mp4">
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







