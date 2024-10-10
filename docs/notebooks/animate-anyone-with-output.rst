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


   This tutorial requires at least **96 GB** of RAM for model conversion and **40 GB** for inference. Changing the values ``HEIGHT``, ``WIDTH`` and ``VIDEO_LENGTH`` variables will change the memory consumption but will also affect accuracy.



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

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/790/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/790/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/790/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
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

    config.json:   0%|          | 0.00/547 [00:00<?, ?B/s]



.. parsed-literal::

    diffusion_pytorch_model.bin:   0%|          | 0.00/335M [00:00<?, ?B/s]



.. parsed-literal::

    diffusion_pytorch_model.safetensors:   0%|          | 0.00/335M [00:00<?, ?B/s]



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

    motion_module.pth:   0%|          | 0.00/1.82G [00:00<?, ?B/s]



.. parsed-literal::

    pose_guider.pth:   0%|          | 0.00/4.35M [00:00<?, ?B/s]



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

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/790/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/modeling_utils.py:109: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
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

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/790/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4779: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
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
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABE6RtZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAa1ZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0Ydw/My5ovSxzhiHtnrK6KJ32pL48pCylxFYMP7Q87NkG2J0Xe6YQplfz9/VpE73n06lPbMIIExsZsWKKJWkF1lPJ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" type="video/mp4">
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







