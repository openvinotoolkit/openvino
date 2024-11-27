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


    %pip install -q "torch>=2.1" torchvision einops omegaconf "diffusers<=0.24" "huggingface-hub<0.26.0" transformers av accelerate  "gradio>=4.19" --extra-index-url "https://download.pytorch.org/whl/cpu"
    %pip install -q "openvino>=2024.0" "nncf>=2.9.0"


    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
    )
    open("skip_kernel_extension.py", "w").write(r.text)

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/cmd_helper.py",
    )
    open("cmd_helper.py", "w").write(r.text)


    from cmd_helper import clone_repo

    clone_repo("https://github.com/itrushkin/Moore-AnimateAnyone.git")

    %load_ext skip_kernel_extension


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
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

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
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

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, openvino


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

    .gitattributes:   0%|          | 0.00/1.46k [00:00<?, ?B/s]



.. parsed-literal::

    diffusion_pytorch_model.bin:   0%|          | 0.00/335M [00:00<?, ?B/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/6.84k [00:00<?, ?B/s]



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

    reference_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



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

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/modeling_utils.py:109: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
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

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:5006: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.


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
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABGKFtZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAbaZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0VVvO2JjrlJDq61jcMRge3C4Mozn3e+LiVbqyP1wdlHk9mxmV0hAxfAwUMQ3jq2vs0kQ3hmafalO9+jePFxLcUW+JInW4ZLyik8XC37l8vCXa1r/rlJZv+2+Hl1TqCEj5UDTISll2H5MNPLiF6rJXVawGHdddG1dySAPLgYrGYsq856RGX0GaHlMFDmKzQMi4z2OaDQPriv+8qhfo58pLCokBXVbuP8ZV7tatQj1W8mouEYG5qQi2qlnxIATkTBwnIP5SIt3HKN1j4lXH7F6ObXgN3h/3jlQQjwcj4J9wCB0iGorSv3mHqKwl8AAFNvTMKcWeTBFxwgIC2n85ieFEAAIz1cOPvfNnLIsWbgLX5hDGHz5xdJK4MRTL2VlRAOf2fqIrGgsJapfp3vZTevw/91vEhauqPrSKp+1nwH55Ge20FUg94vFqZ3yy3M5gTE3nptpio5WWtvxzjUATcgUc3TTCNlw4xpVq2cb5jBdjWOVQE8vaIDL7Bw7Df33P0IpuO3oUWbbolVgt481iHlnY+t52aCHdCNlnDsfrc6BF32vaa8UmR2qufo03xSiKHWzigyeFjw0C3gi42zZKt1mNnTuPmCZqCq80WL0hookYbb7HxWUs07aZ+rvMb9tmjvs71O1k55rlGBp4eBFFZjeXiQklnQPG+1lLQGOo27WsTXExa8YYJxQqEgAdPFWSTgMVbvAyMJCjRzcMVj2zbJdH9scckU9lV55FTMCL3JOH3VApSPnJKwlWumah0obHzj/MNU4AbOPfKMPRIHVtT5+dt/2/skMqF8oCFiwlhbW5Fe311bhd1bB32ri43m0ii+HG7vWCgiwVbBCMAnnQuKeQEIXIBKHCMYKPQw932gcyb/dSzIfrNnDKOJcf2vCvvDEmL0QRETK1H3IdWg99w9ey4Qvtnne6z4iXi3dnUbwwF9jWrW3I23foQmrvybd75AyKAouoWW5+YE6JTF+HvbpxjXkdNaFjW3QxC9+JK//a+dfclJgMaUBushbSJ3gnxTg5BdoQc1UTAR8PzSDprWwo+zucs/GoAAGyDL7Wqt/xuHGPbTypRa/OyeaEbInrRO//no4G2nOXl2WMDb7Ar0CX91g01faDbldNASVGrUj2n2Cqw1ZbE9AE99CA6dm4tRs7Z54REi91F/UL7kcDUmni9T0/E5FCCRlAuuutUjeqAp4TCDe0CLKOiq7vtEkpxtbK/4/zTRQgg0of/tWO0N0gMdPGysCNC7h5T+kNK8JcQ/ycdDU+wQPHg1GCMt123+wHVrglJ2GN4dCsOyvCPeOpCTCICss7sXbtCEnNWiv5SHDCr0g+tDWmj2QzOoaYZV02cAqrODIV4Ff4zlvDAudevQLwxRR58xkAiDxoDikCY8wyLwcOZW8HJzAuX6yq7R+sgK4LZx3sAk+c7JAYO0NRQc9tN+UfqAhcMTGKVvkEYH+gu0sayE5venAqkyI3ElIqXTEd7m9flXcr6jS2cohjPDiRVGXLh7gXq72p5hH0DO7cawQQnZKYJ582EJBniaKslWSw0FiJ/u7R+t+uByM4cgQQUbppF9e7aCIDPbuyYRjd9zTOjAtj9gcpzb7MagSkOxnvGQtPIHE41TxzphNE5w00Fl4+LN6FX/gFoGL3MKwep1eDcZUrPAUjVtrRlW7dsfcwg3VsKBUxfl8kdTcnQgZfWkw2BfLmo8Do6DccEVOaFZbZzyHIdJA0yR9PvUellDNMwVNora7mj9Dl49A8I8F67Oj9jH2DjgrbIJgxf/qt0tfGUZrenXGfH/Xrnxpgc9yXBU29rfZOUll7q9vFnXBmvltgNdiHcJ6+rEAJULsyGJzGISeYXABh3EGpxdJK4EEBgtFDgwEwvIAb/nl4EUVP0ybz+GQoVKYcn6yZ+1wBqXr2eg683KFOx2BgEzHv39Uk3uAm2/mBC5HBlLtLgGPEfNoGG+V9hvHWu2itQTDy40nbLmYLsqbFXtpqwIGLl2tOgiblWqydB+W8XIOHQ/s4y5FhX9cT8sQHtnqz8YxVLM9AAAANHu9tYREsmqPb5Mx/DaSsMFC6jwk9d9nBZQy4lXqaITOidb5arYfjpM2AZ1VG0tkwkHeZCANlgGr3ar0WWjcaNItqUN8pC7FMGEYll+7aaVI2WCjoO/r3BnKOfEO5M2GrusV9sgwNI/4GliTUUfERvQYvIDx5xgddd8UuoScL+jrgZ/6ys8DAkKYOq62etwp8ogsAAAmtZQCqiIQAc//RaeJPR9/+mopk2uoAmIETYEDrtY7YwgZScxjrkBb2imJABPTzsaVAhCwUESMzsL06p4+zWyDNVQByy0hp+jzCW5/WxK1wDbCAaHCly3aq67YSPAnO1PdJMRWVhC4IDqxAsfLRVH0jNdtOgkcie2vb2C5ED+6otkZo3ow7TArSBqO1PMJd82YLh8ClnGc17AK3Mj/CcXE3/DrLUPe9Q0x/1Zbde/4lf7LgGJ+JLBEU3AhLBGHOX44BeX8NMi3LzoDx7xtuUz+CA/L6zPD6mXqYyZjCYu9FJQuACDlpr6fZQ1569V/2CNPf4IebUFkCJXKWa4dBkSPXmeMWt8x5yHGTg6lUKThjmnd7L4Jtli1+RMrhDV2r4YKJL3010OygwDfHnz8PRg3FECgJzEBXtcFcf5n8y37cE/F72tUQ+lrakEpI0xNFjxGGNdzMHKK+Mryn8+dM9p7IItH9RsyAyiRAXU3hzZihgdf00ScAgdfBqbgUZyz4WExURmuFt6+r/8XAp9QCHZEw4AmrcG3QIMWsv8Yepe4kjd4z/YFHTyhZFXGr6wshhaL1Fr4Q9lU+KssNJMLafHLbMe78pq846OMwl9hS2LjOZik8YY7C3zdejQSjBUP9z3LQBUE7PIP3WAAf7IuMiFP88tHheEc7N/J7IG0yLiynOO8zEexpNmWiMdzyhas/ZeTMxXquNgQJ2jCQP014HxiLMu9TzQHdEy8Wpyuku5XObpq5pVdNxjPoKTeiFgAAtJ8DgNz/pJJqBoQWQzHKN0Ygly3+lxxnthnWc9cNS15Q4m/GWPYDikeBTphM4BAU1EkE0WGVayRrRjXXZcRjDE2zKWh9BSOB+R6LdZWNKzrevuEgjUFfEy7NNUMuosEubSCwVqFSlw4BzmKiW8PHj+2+2eZLRmC7awhyM4jTFOj1+7GQUzvxWiWAZiZFpa8lHQpln5Y0+Dy9zPYjgDW3qUGlE2UAONNE/lZtTEbL5XPp/AEzcyFHcmqz6frmsXOxUOKtP3rvRhC/XPqL/e8WwnY3JMeRbtaQXjHC5tp3btgcrjUoA2lq2VJ8eYW4n5uNwsheU6fT3AaXZI0K5FupgUxcUsMvyihQkw7mFkOnOBlD/jZeB3A2Nv1aFkAM+2TRB968G1Loft/bFm+KgdFkqT6ZGfu2qrXdzqayoxZ9iihBb0xhZSphIj4x2WkR07q2wiULxBoTVU4xp7nVevlV3jQUcY+C6M/ymMixydFDFnkica1HHN+FEs33q0SoGPllDeQBAQzLpLMSe/gfYgGbilHAbYBiARF1bijud5qJIbGeSDI1pMoIpwZNnwgDYY5dENegblhtBxYZ35hHDDRlaCp1DJcWBAQvlqX1RnxdmKODf9UOGMpLWGYPtjC96yjfwTYnEblJe5Ke51E/314NBXNg/vvu+I8inR1zXELmdRKx0wEJS6hlD0mzsdNvelwQBcYjmD7JTqzgA4A4carh49T/jDkKm0XOSO2mPQA1C4Ur1U6SaMkKcz2caiDKMxFeXmTcnXyPyXxtyl0wqvcRbX0cIPZcbNqe9JXg320GtOyuS4pXJue9ohIaYc+qCzen77edcYd0C2roRyhj0lUgQmyA9SVd+m6XNqcDcsScUnhYgsYCU0bgjfqRn8QIVMf8aufj/1R7Z/9D8sure/fnjaB06n/MvFsZByOXkJSiTNQAkZk5ZA7lQ8Npdgc/K/TJLQ1RI7yHVGjTASF2vMe1TACzSI9Gjl0dj+XDIhlt+iqqktvrmYebcPTe9Tz7vjP+oYygq5NKhbv+IdhhjptuV/hrMDr356+wGxoleWjVbOv488yzWM8d3p2L9Rulj+JLYnJ1r5hHe1zqLGYSHkn0kVMlh5ygI4UYGL7PacgWzV7JPjbz/sdzRJs6DBdg+UXAc7K5ylNTTBwbmZkP5roXbtVdoVCdElMAzGZyrqLWnpL8cs/4+39+F+O0yyIx06BwgTfik6XctsJlj8ScCWsVa6AjWxWDeN+DKo9KPLayiI+GdHKrUhbheTpKdPi3/gSU0DepMrLwfQq6fSHRmDeITUbqBhXrIFmPWKaJ/UJBu/HPX4A9YxV+TuujRNrpCwkNieWUa+6cvkdI623FTXN7dVWKRK2oTz9JMoNjkF7pKqdIrmoNF+DSk9y0hoznlRGy8nuYfGTMKdQGWAbMM0GOS1Q2PZk6cCcLTtUALrFZmllVwqIiD3vcyPnHgz4Vi49Ugqrxaa6VfqT+KLyfRIRCF7zE6RVpF1Wh44+QlFRDVhW0rr9lhLdK8icpv8XJZo97cgikuYM1Sj7mcO0YUyCIjCWpt/N5ximdhMk3rYueOjaLIq062ILWIG/0/3l6f93J0hzkERs13x4H/IBJGbt9xS3J4i2sKR5LcANUeY0F6T+rZwuIAhaiS7CHOZTjIb9BqRgxw4vU898Uj19B+y3f1CXqr8J780mTassoTKjHmUrSyyQgsIK3n2oKmj0UAuRj9eotPlsUP8KtH5h+0vYeO9pwP/5e9SETadsRflGjTLNOL/+5fJ+ioXUWXrEIX2H1S2QbmZgZ/mKylLTsbnX2aIPrDQR5+6eGt/FlQo741jXEdTlzSCFBIr5OkCGeHJ58yT4m4VXD4s59do136/9H77fXAuTxKxZOYTVb1kY323uarGwH1fZaW3nS9UpLaV63VdOvLnhD+aRpfJOWgCJWryUHOHIue35IziLW8SBUdFrmbzAn8oDPwAEPTdxR/DiJLFTJNLWfIaFz3g+ViYzH9jELCXfUnilHNqt9rhcT0ThJjubyAIq+rXvzPUqbkZb6R5AFDeBneL6i1ACjsBDCnNH7WNBzh63VkzW9VpNQZAE1Tnq+sDmk+Dpdyp9RZfMkKxhvtvghxe5sjKezBPk4NcN45AY6EA9umf6G2+eOv/qXSfg5cto8DxexZlE0SePC9Ytln2GznxZiNcFdKjb8lVGU7SzxyXFf2+9g398wqq3AOEULarCB+aLfRJw+aKbV6W03l9FetFB0aGfK0lvo5+KskYwKe3ww9e5rXHz42Xax1LC2xn/Cp5Ab7TGejtAEhVs3wi0Z0wqfcHkeE2bWDD3zFRtVJ2BNhZTXgo66XwmFAm9MH2HJ7XZoDkkPJgRmKnVs39nvTu7wzIv/UBO9Q9gazZ3YY8MahIRMkjqPAHHXbo83BJKkv5C+AEgkrs7N7XrJG4T5pfA/r1JNVhd+AuEimvHKitb0zAyJ78VTo4T9TNViRg0OgxXAN9MvGpEKv3bqRDjY6bABEP2/NRsFWyq7ZB9gf1M+yFjRDrqvorkAAAsCZQBVIiEACj/WmilrFC/GV6dEKACcq4PmyVVj7kWjMxkq4uZD8dJyflZLRynem3unZHlcLl5ZOMEOGGPUrEwUl7MCF2OHSSrE65V03iMtgrTvnOd31cGwamFuWfFYd/ReBmeIiWsR2JSDUlxAJKJcN0ZJUb1Ayrgx6LFG616dPBCzQHFwxztSm1F5/3tktB5Ogi628P6Ia14pvKa7aLpS+qhQODaueTB1+9hPe9KkHB6B/u4dh9MR8EubQzWKfARL1iIXDF9dQR2Pf/U5naRCSbV7YjEkpy2dI1s0B+Iedvx9KiL9WEt4Ej8s7QmnuNRxXSkWN6Ax9GQnJokTiENNWBneIcxuIhJ/lq2/JzDrGnwQOYAqYrFBZ+nvDXapO+Fm4vQIN/0Ie/7qqvnOkuwrbra7z3Amukz+paolvPKHkqZ85/qJqw9OF/3l2knHP6r2C737gpZdBc6xvMI7ncA7zfgsaSs+9bMMZKABenB1cf8snjC+6Pd7Zji+HzjjXKWbSAJQjJpAekzh2Mz0JQM/4WNQ6bBKRrsM41TI1z+1o6/aIuczZqMulH/Cb6nbE2O7XqtjwNSjnBjDybmUnFA5oOwGXW88fu6/xZpxbCbKTy20inQ9SaYUnowWmxZjdg5xWgwFa/PEMtlMzglZyybpChIrhrICq9PQn9WpVVzAGy41a2TjFGOnh0+Ox+Pr3ZIiwUaYVpQytN9VHssxcohmkOZ07Ppt8hzwAJEGj8gU+0dAOPGVq7seXX5KBLOwcyNKwIWrusNYVTNtZfckMUwU5lCzowEp4yN0WPnPfRubnKJbl9Gsc4HD4hMPfDryJ36pxaIxjN84mwqdz+hx4A7Yb+/RXZY337yBRwWyEdErkWF6+3UpEuRCy+rwz7voeakB/2gTb4GqsDBvd6WXIxe0jH9KBalQsViCyWd7O/iZkekF2s9omGJKNlZW8rH6kuMUJQ6c+3m2SOKhbcxHCSvW9H4SOw1mDIFGeU4nP/70+2UkObw/NWgHmYk0uq3yN23pgW/pyUxDLgxQo3dQLVv+89HO1dR8J//TF/xzjMTjQjauCbMzxyqSCe9wcIiLI0FL75wIAHZs658RXxKfyCVwL9En+yk9hr96JqLdqKCIfYwX1lDefBKj4dTUgxzM515QgoADyh6EcBOcXe/PJVHEhpvdKsqElTMzFsfX3Has2UobAJpsnpMGqBuJeJZysxHo2BsdoigABi9n4uiXTl8YvmlKl2t8IccJlCFAzWAYtIU7s9YB+VVTf/ICrlKvfHFgYxeiaIriLAx4ggtfuEvKqFzEz84N+hscukDicGI6WZwvmFzglivVfHXi65CcKnLyE4tm2a2StSOSsGK2rvbkdGsolHzmRaZJPSHJkDxO3XU3NQik3k452upSIHW6Dy1Z3UQFIFnEho0Tx7pZglwUglpjZ2zl77CqM3wAAPzcY6nI2WtDvGHwyw30Xem4Pe/8iuGuWKKsn30tAD+BGh4I6pvJtfms40I/SlNMYZNx0zi9jNURMUOkMCFDeeAeFsmbTqkYtxTShCxgvBZxSxh2jtL68tfWVvutKYCumZksnoxc5RBC2Yup5cV75ZBYCkbTjfmKrywBmeaqaJFdTH9EDR9FXcuBsYCW222Q5Awhv5jawxUZ3UjLS0EGjt25Oie24PbnL2xOHZPl0P9gAXF4+zRwBC8K0gJNGJmCpx6Y4I9TKy9o4jxHz5eVVdfHmd3wSUWIt6jpKLX80T87QH+nVwoTiBJsARVIuOlixWGQhPOP+wiFeB0vgGdJ8yh5t+iURIC0ibELRp6YMSqHOY12VCEoA6RpTZ+wS47cMxMugHwtWySJ4B6TVontgWHCpXCm8ZGNKWGyCFbxzhzmZO+VgtVlpafmtvAx2OrQH27Rty7QCqffQdqHdZWAA+HCe0LPvYwkOHvwZb9KmXGi2YxDO0dh4d+SJIfow9i2/enPQZyazcJYc0AQ02lx9P0Sg6xX2JpdGSLSePJLYUimwV7E0WnTdyO2Ly5us8xoSVbfknqJIykGUykZ1Yyxe5IyVdTw5jQ0vR56099xPu0ZO8b0ofCjf048enVCYdZRT/t1PEzY0spArZjdZcN/bsceaHOoN5nK3jmnHtnF/rM6kOBySP7N5+G+12qFfyS+8DRXZ5dHT96uSh+AX8GxYrRN5xAX7QSD5azy3VBGjFcbVbfRjh3kJ/ATKu+4qtdurBUvHcQKdEYFrbW9UFwEvYBbq1k11r8fVnTJ5Urdq07LooMr0t8dtcNvROhQytrpc7vKvSfRooPmdM994buO/cF1WRDDNthvS7+jydNcXmPbmL1MVnnBuT7n8Fmkb8zjWc0AInzUIOAEG5R+L3X20FGCxGv0psfeQ0zEE5i6nX6edxt7W+kKzIxnb3R6qZY++TdUkwCcElvk8L30L9jXk+qG0O6OAg0CvsL/33papVyYuypIbUsQ+Q+aMZxTnCsQcA4NvwidWh2jq066HmNN7xAJPMo8+5M73zUfPEdvyyxEtUkIqAtUO+SV6qMnpjCJ1FPAlmaX/1j4za+d0w2ABKTZQsRuMfHe40nDXkQe6kllukNyD9u+JbLaQwRNbiH550HFuqZ2Bsg385zY+YIHU5c4kAN9a0/rOuAdzjdSIE2MoxRQ9s5Xiw/BfCyvzvrRBUeZxf+14XlejaVyjsOiZtHpyoBSl9DHnE+PvzsIy/Y29KE6v0+7WpJffHsyD+XVRQhLEz5+Yk4vQTFyv6zqLYsvHkRHE/Uo1RDVMHyaGyA2u2KgGB/pgvZO/9DTRpDG5gXmrftaXpw/gfbnD5Buf4h4N4bwQ1le1jh8Z9hdQuVOms0ZXnZe3luc6y1JoBmRO+Jb6Gho5H+CRn0D7KCBs9UeJuk85eHt4OgF3rFR/6BCWxEkgGqERpm/6u2WT48L9dpQzMMbt4WdWRXU7iwi3A3yhIn+8XkRG+kvJ12HqVftGFeLkocWCd0OICz1UzAN+DBJWYAiv8H5MC8iWHghgvNsjwX07D+xesOwsZOVZy4qNOuYRPC1z9n79HdmnWanpmpxpz9G47Ypo623dByqmAnVHhgTaqkG6ve7UL/Lvz+7y7YNAQvi09uI4XiCdz6Ph0mdNXHp5uXCAQhF82mLF9RNA1kDCiDnMjMbFXQb8q8IAb5m/N+qffOgOg1W5NxZBLoQqUbtWYoUUbT6dcMExqi2k43P6T4xTReH+rTSGfToiUBqtDaSuj2CyiMVlZydIWW2G6ZFhUj8jpfV2XrCZ49mUKlEm0wbdhVP+3VzWOfN4xdVjTyK1C5jmjGXJKfbZzQ9ygmdQy1lqovbfWHryKRZA/SFOHk7taTIJ4C6UvA4EQ41+cQodpuQv7I+CdYqIv/4ytEHAz7C0D/jef1ImmfDSmQQ0UEUPMFCtUoa2FvOuQ4LNtyWVJnfsynzp9I/Iio2+pb2PSDrpJTy0aCAs+mZ5QlkXio5Q6MNLArJkhE/IMZobe40R6UJX6lrx7xS9uifLhRL+5sCfjx9GXl66mhGSTHvvOysQ1oPErjh2N3ujaa5Ix/NOz8pySeqrdIO6SxxbMNxv629nAEyrmo7/be0K2AxritEErbuPgbvMl2KSp6Z34X+A+QSql/jC7+lJ4+pMnPdMGF2sHTZ+eMBJzwas4FsIxOUMCzWAF8IEkUYnSGP8RQJ2VFdg6RWFt5Vdiy1Bn5n6FZt3KzTT08uuIp/bG70bU9pYx1TOE8eMgobHMPBOKxZSwqm6+ImSwtiEfDvpuCWzwAACkBlAH+iIQAKP9aaKWsUL8ZXmpgcACGOqwl7r5okQfQ+D0kmihhn35duf08YCyoxfGHmzuLc8bUDrPtMdT1053y53z14BvOqlApmTqvh/7owPfvq3ho0BetIPtDJ7fGLL+gl+KabkhHnBt64+Duwbg1VY0KaMTDo1mC91/E29yoHIZf+IwMbN3y4W+Ikd6oL+UsQZnVOK23xyAtLVa3+tMiGJT0qHnNRSzpkhF3cljTUMMgKY/CvwF32gjJmyyJt2B/J9XwnEuhRMXfJvd+AOhz1yQTkfF9TB7dghwuyh5pxfaZ+J6m1jvAYyCUoze+CuFsGFPinGntJqtyD0oYTULPUdsDZFDBtYrARQwoup7gAJICO2Hk83mlbTCYPTixa9kQ0O82FKisWu/nFdHIwsV/ZCiAEfEDNhEPuMTahB/NJmeJlorRXi67ZtGp7dsiP188PGNeF4QarL/NVwm5pfjb/AP5Ku2s4c553O1+ugboIGRu8FQ5VJHhfGVWUghLN8Eg6oXDr3FkYvjCRsr5+FkFZIFEfrvquLXTBnxD5DKpLPz/K1fK8/0BMAZjqDGtG+rZD8kUEPemMD1ihlH8KhlWebQZeaZA+uF4ZrdLZ8eSkMkiMnKXLp5wuztv0AgCx/yvKLxSzbLxhA1EfVem7i7qiMf9gIAY7kExSbmwAEYwRyHW4fAEvFjj/t2BJqwa15wV2zrX4SQNjNrnqMsNM7F+fjxsQjReuonkdD+EL24GgNGCctTrw8g849p9oRPBvOG7AhvpxF49wRPq6acdf2I/S3GN73LguUXrOX8ESkwsQIy+/6fXsJrbkr9NdRV651WvE9BWJ0+k963VDIvABGJcwTnjphlUDXOWdaWDyZgbdA2t2JMDP/kyg+G40jRSPTNIH8XIBp3HiEdOYSzD67w9XyBkJvmqjQe5l7TwCLmj681Dr7AoKti1sLWDoy4NU64UQJL8tdBbfHcQQG8aXWw1MOcixklHYk0ZtRcIbzQQniE0AIpp3GYb51BNJIPEX/YMaNj6VTTVmj3cQl2H3vfRkjG6aW4705y7h9/SLQGqkgrwCGX3zNmAFhL1jWuouhmH2By1wQaMGYOpEKoLZGbjzQPdsVBKPRbu0FD6g1kZsNfoDcR1AhAIZCwsFQHdT4pILziyCpZUMIlWyFpPiRqNLvYF/gnoQgAOsMRImncWxdn4j4FJ8vO1GLb4q1BUzmpK09Gj0Dk1LqNo27RC42JheukiBHuwhaerZnJX45WrmHVxWHONyRMFe4pkNDo/clhZWIBZ0U3FKSSulprpwEoQ445//4Ntk49VEaKTm4xuF80Bk4vVxusDzG8mmgfr9n3JPtCdYFvQN+gG1ab8mYAPLvTLM/G+PS5+oyFQdI0K0umA6hXKIv8gerAMsnzc9yzV0UG45rtUsHLR8WhN+pZFtJ2aM/zoUKMDAsrvftU677ZlPWUUmmrr1OhHe/e/KA2qFueeKshEtcfhov8Nw+YS9XK+kYDR7ivEg57YHXRzaNve/BdEBXK8wuzKSlIpPhPlHDueW8VHu/gZbIFp/yRruchI+TUg+DI6OCBK9bNcji8ZyUAEztM1vbIvqX2iSkvk4YXa5LNERlrZ2t6QttuaMwcWolG5ETiEHoWSZg0dm76A1cpAF7kfIU2k9GizSZTsG/ZcEznYE4aWuiRAW94LSu3bzoSRw71YKwyTSPo14pZUQMtiDMS4HQLc2qo0B5U4072jpFjwWdMhXsGbUWZqq3N5ELaRerlRsEbrBnEySoGhQ/dldAidh1Z1/3Ok84aIId1BKTD0O47spxpimjgcogoqKw/vu5hCIVuemHNJ26Qa8gWECBJsmPljp/LyEnpgCjox4xalVrU8Ny8FPxHAHA46k4oiXZO35KUBes4O1huxEZWgpGDY5vgSx8oTtBxo7AeUNDOJXDOslyP7KQGnY2lDTHVPc8lAVhepIEWhJRkmSGwj57Zd+IP1uwPWBe9JdEL4MDE/v95p8nZ71GxsSYNLO79AWHADdOOtayLCvUomqfydJr7MVdnuHFOc1HIPb4cj8tIw1+EZp7C+CT5qAmCR7HgqS/m8XmqTR1o9/Isi/44KoZqrI0ZoPe8JN9gzEjuQjoq44t1tmUtGfaQXGpAfb7wPE6HceqxIAx9LIr5/cPst64Hy9RDfqj2PSKC+Voz1him8iNADrRMhw4vnSVQu5kTy6hVLjylpvhUEGVwxPGsPwKfuJlfizm8N0uRIbjKXB48ZgFIuji8jJYGEqwh/W08WlwNxxms8Jx1JnikAbxMXTjZio5Jan5MepokxEBAff5IbnDcoFcrHzJ/sojF5vHrqsUfjXV/gwIrbtLFGuuEpViDjMEoFjeTPP6LAgeSXdOgOgmhyeLf9RnrDEjct3yYX9OkJRQlQAQOlWe7n4zdMPeSS1KWLBKrQ+kl7ucaG8NxxIVlCxUeKqdcX9axrmnXk9xQ/aHF6U28cG7Yp6GU0i4LJx67BQ7G/3EGK6+ywNLmvwGEvzgKSVZDEpQOjsaZPRibUFNgtSejNbB863+XlWE7rDWmbzA7w3J/jV+0O+xAGjhmcqBQ7wezVzmY+C//vM00MiK9AvqRQEmOO0H15qsTob1lve7dios16FuAv5XWT2BE0L7aB6AOU7yxNzDM8pLs40WQwiRCELrGcduwn7ghh6ZWY0c67i9Bqq/Z9NhWK0sUedrmNKquyj9LYV3zmu8CMG5U8tVzeoaoWpHw1ljI2acXXX8jRKcYDI9SIE85mxo6a3clZj+HedVI1xXQ1j1cCliNDWeSbQrwny7znTavIKntQjTrCjXuYYrcUKSUjEzR2rlQgL4SQxbVEgjObSam844qCmsMBuZHm5XWnNIWeHRLF6hHa4EkPJwfWLaqwPKjHv4gAj1x/j5Q7U2VDn3MO0twud2xR2WdaEyx0s+OVKQCmx7Mjw53sWym+dO1mnxrzVPS3SIig6HYg6iWwzPyy8qLFmCD0WREWCQyAKb0UpVKLWjVL6GuYnQUIiWAEeiNfzcDUdJeU7VTiuF2S0jILIjzm0H9gsn+Mub4TD4q6dz2b1QEuXLXteZuap4Q6HIZJJExm/Mp3hlESlTbIWRaawc4Gu06s00blGBzSdqN2HgzQmDzClmoYZm7oAanzBTVomVcdNlES9RrfzdZTtqUcD9V4gaWEKM5FErdzTkksoG3kyfhxDQ9rzjyyrDoaFqHTRMTDYps6ensvwnhxmL6RtBkkJO6ZD0mzvgr1vZcqvIkopv17kXBTvIN8RQVT8X9NyyxoVYs8zomQjegWljrcDeIqoblTHkfdAIIAdj087jl9yaJneLgI2XdJVdqfiN2lr4f9IybZ1psywLKoG7IgzsMsDK6rVuGprad9zaX8fmeJ7jCE179rLAwGc76gZ5i+oAmdBsU0vErFDyH2KP9gjr7IvnqV7WiPYXg1mZIPOL3FxjNqXNczTVWzkPAZYaZVEBfIZAT89cNGO3PEq8Av8Vp9nyBhD0wAABxdlAC0wiEACj9aaKWsUL8ZXlkOs4Av/kcA5e8WCjH7t/cficetssv2ixu5xCm6RYRknMhHGH4tNOo27GBUwmNpo026sNScN3GGsSoeapmeEWZ67IKz7Y0g50glMgx1BkSli5rX5oExX46GFxrUwoPeF2UEugXJIfCYd8er1gOH4YqMZE21iz8tI959BsgPZRbhpUzDJC/BDssbdygSFNEFTfhbg2OkWJ0e+jS/89nO8skcLYBA/asdy1YKvM4aFgNZu8woqzf2XqtsUgnWNbD603qG2Az9ZXH6QhRsLpj9XpeF4psAEBG4BIquVbMebD6Egu/bwHOkQba2vWjTq4yVNE2UzGvyW1XILGh6mKB67eLmEkT1dUWKhzVWkvEa+0yAPNiRD2Mz248esxdph53NnIZhLDG1qgAlYk67wtaPyraXCiHyR3eHJnSqWgWYv7Hc8nzJO0g2CSOhmONUKuo3Sr4+g+o6IUkdUorEIx1TTJfSchIi6/XLyUCPFojHIamOK6UCAHJsIgCnotzRyWDuxuxXKIavyBn7GCKmF4nPwWolNidDJ4YY3QA+Ti2qkl1Xx4JZ5IOr8RPViFG9GbWl939CiYTHFla2/ePfdAsoCtbAKZ/+5hoD8smtOtq/63jmv1di1Nfto1izL04gUUJNxY/k8UleHH0dGp/nb7QnryVshQGOU6kBialr4miUxjY5x8hZeQyVcVAH3TmPbugfhpXsNnT3meWKQ/xPyuMED0k94SsxSh5rXfK2CSyCXFWDqC2Ag1H62i9ld5xoYsShuRknaUbEg4iptbVN5HRJgygOmeRux5wq6qpcPBPHrbw2QpqR+JZtywkQQTkuR35aJHcAhZOcMkcGgqP8ozACjPOpU6GfAAxqRznWn3NsMg0OHHXgmHBJqT5I7AKO/IrQ/K3/vjTUrSKh6Cc54Mz1bYd/OhLhxS4CCYgcrRIzAYKUDsoFgqN3r70RYgBYqJuhBcZeAFfOh5lfhJCSqr0M1cbUdxkDQom3AKCBm514Fhf+yF3Ix6Ro1MdEDKGPWyMT07no2nvFOknOtJ0fFeq3SeZFkmUirn7GzL5FTq6yDImXB6fviRgJAXsS6LXfrCLtBDG523FUU3HjqsJIzggrOdAggEeA4ayeJg9bG4Bp/DxrhNp7r0aRN0r/hbjvnASAW/HOKBEWdya/bY0TdwiFO0iTldfDclpMKhRvtrZ+r7BSxHT9wPivHc64cJ0mjrDVe+en47Q0Ig0Z4mdJ/GJWrSp+aQcOqmzMpyTfGmDkhHrG1QW0HtpX7eY9JfJyJdW+xegayQgVjjyum54nmOm4MhBUynV8BwKCvTmnrQTBLPGqoC06fTcYvmRVxp8GLSJGeGL+LfdLajeidwNn5fGSk77TVgB7ImruWQ3wIRRFZq9mSWYgZEipX8GDiF/dt/2paPGEfGlmbt8P4oZ0quP9fE0wc2w7G5idQAfBjWpW8qHzYNWjoV7UTCkk/ck+wzgIKg3bantDHh9Umd6nGhl2qRiVr65Ni4jV2kRfXd6Y5tO2oLX9SHJbbF+Mnu8vwMGoTUEqjRBSelNhNhZyNQ4dQG6vGQkYOaNqfAIK8tztuYSeHQc+7AbeHZLUo6gCDEuvNXxM6wzoWe2/QrLVLxo8sf5jHlSyap0HMaXhOMSLc/kDi8sUh4bMBwjxi7MADK8rt0iMookeI2/o9m2BP/i8QPWNQMoJLyDLtCnXVLOe8yBbO/vELHHNPA6DeI+cuCGpCTtTO3bGI5Hu9H16K0gL13nF54T70mSgRf5NpEsTc4Zo+qyw9Md3iqPLHen7hDo6A/7FOalk2y+2zMbUJ/qO66Pq6yurDdy59riggRttLvMakd7Xm+BtsaFGAeviQmRwW/p42z+cG4mOZ/UQCWlR9PZN0/TnP6GycHjl+Ze+tSQ8Ya65jlW3YeWLjSjT3JYloKOXrTs9aMYHkpbc1Nzu0Ohn6B0Ip+cnHQyU2TYsB0l/kRgixDFccpJqyaIEypOxedTpLShVGeuMrqOxZdhoThm6FR404fkDYheSRkQzEaWjLlllCkOKsy7CevyzhZc0Q7gAW/NnSEkb7whux0v0zcVTswu7cVc6zjefDybtqa24iZ5J/SbxwpFwOMBzQ/I6TdlOUXmKY/bSRtmQd1ALCIAQGylYE5BxPWf+xYRQAXq1KS15R0E684//3Ey6kaAjtCZdKhUy//yvFIB3QBFpu/1vH593Zy23G5N9Nc4VySoyjv49WbDF5rLC3a3FYplv3dMB1e6H7NrEkBPlHMPlmr8emn4I+1lHK7gNs71+s7BL55Ejz3QihYYVO9scvyHcJEbQ13uc2h93m69EzWsTfxVPnN/DZyg5I5zP6sxTqbMI0jurK+BWs9c6asba6yhjZR+YoCypSEFELbB0zE0xhPLjljXiCj7kAAATVZQA30IhAAo/WmilrFC/GV6dEKADHpzEkf7dzeZuFVJaY5HNelkPWl0O3lLbmwtMguQMh0MBQgP1vn1gXOkUr3JpiHMlXgKTXx5fQ/hLCQufQz2wJrOra8aDmTkFMDKz0VHUGIVGXPC0CmC2aDbvibyXNLY9Oa2sh8Bc0zKuEYaJzMfpx/8ns9+OkMzrz7Pn5ax9tKb3iiZP6ozfLTTHK+KWm/nlX6/4ObHbcGIg3EMw/ezNKU/8VVg1TQaVuq8Hpn66tY+bI5UWHlCwnQBgwYO5wHMitQ/GTr85dlPI+Ci4KbuEyPPytnBnqcwAD+/uVXeo6YDpFpLcV7hxoHN2mXR3nadZwSKUQdnuB+LnXJry2Yna5n0Pqag80P4DIdiopUlgZ3LDk4EekNyrWSxrZV/R62GqMLDYEi01Uh7a9NEx0U9qs7/vvQZxyRJ8YtCNwUjYeABvBWOacOP3Vj1U3ZURtVqNR7VuR50K0EElxo5UQ5gJZSlffOTp1oQztWm1LQCnH02UUGm0h7rg0dRZQfG+zFgT9Lyoeh/UDvxMPJzClrG181Denrs8PaBOAwMBn5nq8Mu4xueIOaPPuU8RrNb6KAkt+spFExHVuUVg9s9H4VdzEAO5OZCCRY33ixShDyfKde8zYm0ekM30/NRYAlCKp7PMbcS19gMtr72GewwLmW3ic5dhj4GNEDsqRUogWUPHLjkolUrYqkXL9Gbtd9FTeLrDvA/yIdqBhc7YHjaVABFs71szhYodAJ8RfZMzOZgbIFkRvDzu7dzWdVhpj7SLTgTOBMmL1w4COlvhwjXoiQFyYMR36xwJcW0cYUP4oZqna/6VvLAJEUHMocClqPlqbosU0ph7MpM1RplWBMlcfqtc3/aFpbr4uJDNNvqikuwPLbi8diYyskIuV/C0fAV/NpEqEY0EpjLXW68Iit43Npsx5EpHY+uXcH4kYccUd1jJXZqk7ymxo6eI3JAkY72E3vFlBFkWRQKtAkcYZwz8mD9JOjyS5gcucnhOOFlwvRp0ItS4PWzQ9O7i07UyRtxProFepvMg8MCmgYLbX7gTJjj7V3csYxCsn7LRQpcHfMjprtSf+3vmwUtSKtDZp8o/eHL/I2/LMHl54nXzvEIK+LZ8RS5PkCJAF0ABDfHOb1eCn73FKkcXQVjokcKl3Xp06WX60DRf9fxTsX6lMrEBHg0iqhxQQnvpUTYvAYUYEvAeAfpnelS1PhH/UsPlbdEAHT5U7Fcss2+QGsHv/I8oVd5CZDK8MrF4tjdGVu5bIlfAn539yOMrXzTqsEp1UW9pVEraeos9SuAWXqHQmaz2WWeGeVpqWGRldZboORt+21SfjrqCjTDmfISrVMhNYtUnWBkJr4emksn5LxcyWLERlGdK4LIJGUTWCdcWDvNmtH82+Shm5Oo7qAGnyTNYReD2MrYw+lFOJ8A9wFASeakiBEwi6IbS34Kb/N3Cx7/4Gf7bQGvFAfU6mYJKp4dImlNKs+JTnieFSd+ILWLdeYAkPvpCWOEGCH5e3OzZErd6lkUzKLuMSNvjjeTCPS/qXgh1VNmHDUBeSvO+w15z8HqiSRQb4PVNt2EMqq7+nlGD79vQzUDbLWJgEmheQhEdfy+bvflLz+euH+/ByWszGljwESTENQwAABlplABCcIhABj9C54lBmL/5NA2LHZ4AELNGYW1ITK9cxHO5ia7lbvhsP4OBeOx8CXGUp0P9pCi8yKTyB4GiVcxOThAYaU6WIZG5dj4oySWgbulV3vc7Gjepmi3r0tB905SYGgc+UJJfZGVBo6a9FQz8sAMR3Rn2NsBpSawT8pakNOeBfZThcdtnqPZBPrJO+YmslmvlNSjlRfsVt9mcI/zX1d3g3lWMkVMXlIZli8eRGdORnky7/gf5pPdgNnqylKrFT6e5+/exKEZBCoSZCct8pG9eXmxeo0YxA3d769MqWJF7wkwrpgNCBcP3CpwYMkBzdpn4kErKwpl+zKLoCTsufVM+6usg1BurZK3hmAAANBG1s1lP+rgKgcLGa+P0LhiPxWoY7DvNFU4riceVCL7QkCeM8EH6xieLpsNlkp4OV0lMtnTTQBZMxeQnFNhV8UY9heoA2+ifa1jUb8GYwRuVdVZTJTAGk+yfg28S3r9GMQ14ubCzZq4DuvZXLhZOk0E4URDgbjG+ntWLn1hIIXcE92xYeYLrne13w3gxoltIsJMXqzPw869Qm677N3la51KNhZfG/duN+ryywR+FdzwEbSuBms6PehJajd1JrbqjLlHCEe2Ca0eOJ4iHzQ1UKpgxZvs1LqiTAQ6c75xkoEixLS0vu9IOsjF26YdRcu/3F3MEM6rF8PsejjJ1bScNm76V/0ljFDIZc2AFdou7tT2hj32K1ntBpkr2WAbcsudDo8DcEj4LDVqngo+/yXE29jGrOb/MFnYx9le8CoSnB78Bohija3dC0qHEzm8217u6d9RSvJcG8jNyUg+F1MUXW/J18EPrvgsRpKgkDIL2i73doRuP5X4TLnbIFKd+vnABNuyyjGcdMO3t3VyL+D5HrNtzbbbc6LMRLIymNTc/9gvhsm616cT6KYr05hp9cSA1rIrub9VZtqyyIo3A09QicLyVeUikJbOsQl/qTE0/mdWpCl0McrCmHh45SnxcZ1Bhy2p9E8Rv6iaoYr8IZMwhtv8C6MKMoilbg4MNCHOS2fX+JVubMgE67NJewl9JVR8Q3vIbu8miYmhJ3Bpx5YaWQtOdAKMxNbmb/1PvdLazMuBOd93MgOIm5XLp9ccFagT8gw4UsgL6ftkwtokzcM29AIcm/zmv/r0O2EOxf66t57DfIImVYbNtyQXYZ8TdxETn+M+/vGaaRTUytnu4vKNNwmFJQfBULk9H2MkQWw4K6p6d+5Es8OsiVxKP/c9XC6mas9tV85o1M1Mh+igM3qEvZmojj6avpvTorieCjgFXx0T82KUaSsomtZrNiDn1g+Hr625LKQrWmG0pBL9vgfnaB+AZMeumGBiDtMEd8MNtSnNcsm2F9ge6dnIAdSoPJhW9RN/ipIFb1PD0OBkHWuS9tLvPTx4UDhKwtLRWy4CpFHaGwHwtbtwHRud3EjlAJ6rOU3pe7THvP58FrnhN39Pj+UmqIsLiXsq3JeyZB44C494a4HuMhflxBrtmlJmE3tyhluTOpupnt45aPey4UDFDMDmAIK0+kh9cIbEuPaHx/pkZLllPNMtEkrDCRwopvxm/VjfIGZlGP3zJGrAVKumH9P0bwewrUxymHC6PzipFwYM/XB7SjsyPpJN+1J9WhMngzLMTYTSA/YI2V/tFlEotHGfrUGIMKgbMbskrAGM/WJ/zgRR5nqlkdpEFJ9jMjgmxYv5AglfBs2PG58m17ecHnUutZ4ktgDjXqpTuaOMnmcpp9e0P/tW6tB6AyEqftXynrNp53qerQXbZ6GTkCXTYKGOpgEo9xDYYe9BepTTqq+9WC0UJWcTnR4tyW4sm8jrROImwkanNjMZsQdHvw+36Y744hJ7vqZeVuYArq2eG4ZIX0nm7KQvSa/RxvwAnr5cYcp7sI3lagZoIn5vxhofpiPqngIEHaybxJisKL+AWTKY/lbTdmchIpW2IxcdXqr9NKd1XkTd3RKgfSdQwg/NOORXcGhP3+CsQn6x/6eTzhYaHgcrMq3H05O5xrXtAv+he/ZlcJrqAy9kmnD7G6uzWxMwlO27GWKXljtYHQJQAmIIoGoTRZH/MG5Xx0xmaIFE1fTKMbB96IUS37hd4F6MmTUbDu85bgLlErLkVuSJCcBJ9SGMwfqmkofL+qIPgP4NbGzkLiy46Jrpc8AeEAAAOHZQATRCIQAc/RaeJPR9/7MSvZLruAJnFWZK0i2Hm4nfGxq9bW33hQ8oe8wC2yaDqVDg6dwg79X5Vsk34rQDmj7RhKBQp+yunaQ8yv9UE//VHhPRZT+MEDpvyKd7CvP1HOOO0IXgSQelDsoVUAgkyF8xzseouUCGAUxjN9XdNCMoQ81oXJO8xMUgnBJaG/O6jt1K/WSICav9Cpic/clrjx5kzj3K920yJzf7wBDSz15PUfXAiEfZLkDXlRWhGLXP09q0XhAqdXNrrI0Gi7mX7dlswdjS1Njo9JKrv/a2AjQVLPrM+frPNC+zayBpU1SZ9UDd1aRPwl+CbmMnkhKFxVl/MyLCrvbvB7debITLlvCk/L4yS0LsTWtg+TB/DayxPr2IL9rLEpDXLgiIQtfY6IQ5aB4p6g6jVniURXTegJPmIrckNVGE7arosdKRlPFS4difWCoAAbQc8mkv/lkXZB2Rhf//rj4IAD4I+8V7zqMg9IHM/d5J6DUu+4AwWAH3eAdG/zTa5SNChHggp8/BQXHp+uwlAZ4nC3WUYqnRPR9mURdQih/ohHVM3VvNHvr2EfHE/DKHWcHoFgl2CA2+3GMMvFRcLqqPP+JfYCdlXBJwjxJA/TCFU6pu6C9LDH/SvVHuokn4WE5bHkwxcGr03E19VgOIniE/FfZv6amFOHHohLW5j1hdfjiMC61nU8AC9yBs8eLy9qjujbptdLdgRxR62tQb+Nz9jcR75IdfhLQWmwEn1siI+4bKpepc9QxP9z71IqCMLVfy62Asesd0WhFee2rUgKKhfee2hesXwJbLvnT7P6DjCWLcwjeuFKf8x9dHDxQzAjP1RAItEHOnz+V7bvACCSkKdolTyQNcDBW298HMB6i+zdOFpF6qeuJW5vCDgDq8Wmr9HaIyUR7vGC3Io7/gTAVUMcEck2BrkzrqZgBkMFJ0aGuKAROFR6bw5Ek0gffpxLPmrbDe/6oxtCeo6fFcOXn0f9++kCLkK/gO2mARDg4hsz6J0CiJSlqg86c3WYwNftyWYLbsyz/K6gj2o3pOEfX9EoNghsLscwH1N+eYqIAM+DkA3uBC86R5WEVPlg6WIX3NvEGZHDWbnd+smVsaKrAign1StlHOm7tbKDbAAAAwAbvlfr3vFjcRbAMnj4A6NZ25DAEO577x0iUZKE7PXBdb4mIGA7/p2mhPgYAOsWDFXdAAABCUGaJGxDjxr0ijAAAAMAAlvuENLXjYBlGaATCvl4CFvG35SooOr7pX4QUaAlqmWNUxQlQGb02Or2IQdhHto/QhOA7xCAFpV44Qar+dbEXO6MjQCpusO1BHeyuwJBtybZPDywK0eaP9YCwjmo7iLj9+h/C0wZ+Zl2rq5MKIWzDxraULYMUMV5NisfN+rAbT3kFvCBzA/6qwQAfQc/NTcZ84l1XEAI4cfuQKyi6ATukIY41IlVZKswFLDFWSezXCofx3XqH2OaUgFUtwJE0y5qF7ccoVK51/PxFS2ypm3TBOuFVhyMKy5tCCnLWqnNQ0XBGLj8zIktUAcxh/SYI1Yot67pxeFx4pcI7UAAAAEVQQCqmiRsQ48TviflRQk+7x9xdWy+ofMgPYlW33KREndB9ROEJjJzvjKOb1QKjfF3FKetlbh1tS5icf7Qqjy/r5j6RpaRTsgpYKt+7n3jnWTvrQoxyCmy1G/iw5jVKVZdPUQ7x2xWahFdAABIY8Pwde5Zj0DsmFWQNv8s7cKSQ7a7xyCPwuRJPJJG6AF39aOoYF5oOYGxUpQZA8h8TNo/G3WGytMiFabu/M2cKhuFAWpqj6J6qOjUjwH7hAz29JUvF0W32aaYCoBjddjbm5eTkuWF2jBffdad6Oovgbd4A9Hl/0uIA7CHNsrS7NI3WnBIwPrI17umZAMa5CqLTK1V6MV6tVKrCfbJgT0VWZqhXBYjgb58VwAAAO1BAFUmiRsQRP/X2/gA4u7oAaDhOV4h+wRGkywvVvlApk4QFnFji1shmHQ2sQ/aYy+w2ycpdOr4suqp/mePoe1iKUAzRNLdE907UpJ+9C9tKEICLWfsbkn2Nqk1Jn2I+Y14g/AxtgHbeiYfsIygMe9+YooIfo0H4DZIApDJf8J7rkJVvqdporXPZu9PMdT0BCV53anOEry/v1R+FVuByqODmr+KNP8yFZ0gefpzUzUJSaOokiz/8IQoEj9KodG7wHR9xYmfeVjGiehfLNQxjbC5PsczTPooIKNuJEMaKk84ef9Bg1kTsmyDM+lBq7gAAAEZQQB/pokbEET/IQ6p2A/hoLqpHioS5upVASLUHvANJEgaqaSu60j1k+jra1afE8RY7OzR4S4mFKg0rR3cosuQAafk6x2jvs7URiO4cF+3MYvGNbgVGn6J+daTeX4Kz1qWyN+34gpUYzQ0lTnt7M6RDy7b4inTkbme49Sqm0wH+tmyrS+HXlmvQQ0GqiKgaU5TqTqGbD6VXJmBDro4hmufNvQ5oxtXhVYZNWEQlU15NFZSncrkgz1NeTCyDBIFauS92ljVN0YAWreauNQzxPiyRI0a1/0o9JpGJWzoi0rlAaIXx0jHvyu/N2blPXFn2yu6dNtRJ8UikNdE4C9mgoTeAXCdxWiQQ88FSkQtxGR1odNs/wj6c0IZbYAAAACdQQAtMaJGxBE/I31L+wW9Bw4I2ue1Ji38e/dlCceTiqTD2Jk0zHF+AFhodCBnar79tdr2FX/Qosg10D22+ab17DN6NS79jLeFKP/kyMcY9Wj1apuyYM6kHwwlLozqAEQdJpyvPY5lRiqe6Q5655WMce1HsOOuL4hXpRsaQBQ29bpG94PaO4tqVORN9eR6Ak3SNTCP0OhYiTnVd42UUAAAAIlBADfRokbEET/X2/gANWeNOZuAA7OSsgnjtQBCTGOAgMA/bfB2JHQcuXBDymC/3ZpAfULeNlCAAbkAWaraBBcvQZTaSUYrYjCyCig/Ft9Eo4NrEE2JLKd78hLJ7q5KNsa1U982IgbLHtgyLx8JT3yMrMsSTxcTYD4Hh7vVSPLEyhrH1qf15t4eMAAAADZBABCcaJGxDj8T7k18R7TEDjKvEc8+xPKUSiaxDlQalH34p9xDoqceFMwnRTgTOQBwwtD3ESwAAAAlQQATRGiRsQ4/Gwg0AAADAKFAPHdqM7/PPm3yQ2Cplgwq/0RUwAAAAGpBnkJ4gof/KYVUPgQ1l7fhtgJ5zkmN/2bbuVj+CQzA6yXF6a8eqF5qIwf+Lohhok2vFt3HbZOcPjZ64e0jEasMLXmrYwG58mqpVQ25X2tk9OtVNW99+60oWZWCUdaweaV2xpaJ4s6PgdxBAAAAe0EAqp5CeIKH/1Q5xIAsOjBfgKt0oi1YL40JtlRQhsnOoSMvHMPLfe8qgRPKYZBmOwR0g2b+ZxJGrwaJYcV9Hwr619oVCYvXBENpoOwMVxhNed110oHl5wtrUuSW8PadyhxfwXD6hvVq1FbCZ5MhEyZbWEkC1kPZEKnbQQAAAIJBAFUnkJ4gof9s4fP9YyK6ctKbAWTeE2DgpuayImvpYh9cdV5MQWeEftc25pLEwFFQ2UU4WlmdbRlTMbpjoKCbPb7AigZOld1IOmrUNQv0JnKAtMUkd83fDaHpNW/G77uDVcWrWtG7scYnFCrL+lKgVJ1GOsdfszIdBIOxlisszSVBAAAAvUEAf6eQniCh/1r3phWSVDgBkoYEFuHkJjEXBsTjyE6FBVti5+yEx+0m+EF3hJbva7r4akMVC8HYwLxxdQW/X5lE7cR8f2342oLQmwGtVsY6zno9MZzF04y/uVK9PPEfPrVKbM14NESc9ID/Bo4lRVB8MDtBuCJTtjHogepiCJgt4OhFdfBDKp9VlUFacrImGqxAKnfGeuZlLZf+xpc7aLIMk9mrYnQtPw4W8WIXu2qp1Dw8kGBLvra1LjTCwQAAAFNBAC0x5CeIKH9eSHKwfZg5AQRjjq9DwJvmj3sXYyYP7wKdauOdQx834vgmMMJaKLzB+HGdJU6B0C+ufLcOwy8DobblHmh2rVxU7IbPPuEnssrwBQAAADVBADfR5CeIKH/tuzNPs+pEje/sfk/XwsX6biW1jJc3snDfIoMJ8awzH4Q8/nbSedfrPoU/cQAAAC9BABCceQniCh9tzhRM7XGlrlKb8BIi/Rzofy+v4tPW/rW0SLMaHncfK8HyfhHNzQAAABVBABNEeQniCh9ueQwBq2cxMDNACPkAAABTAZ5hdEFj/yxZjFW5fSVv7uMAvaSdul2t0nFh26psyShRIlnea2C6gSMpSBHomiGax05v8LvTKtj8kWPD3AAqN6tSrPmSgtlCs9zMewvAjY7SRMAAAABIAQCqnmF0QWP/WLQQq8KcxA12L6iSiA53ftoj86z6iCfSly6HjGNWYAuKHy1mw+31dHJ6r/xhqm/QIjSW5CGQGZAcrv3rbAffAAAAVAEAVSeYXRBY/18hIKvy+gUOHo0cE/dUa8PR26AH05pvVr8OB7XwftjybPdGVVwuf6JQwI52VxGwZuQqQ9P3ARuR1I7/g/UD2PXfGjqFBhmAYx8EEgAAAH8BAH+nmF0QWP9c9GoeqthELz7YfncLFwojxB/xl+oqLmWH2rP+FhElRf30PDh5vgZWNPxDEOuSzpiIqsyeCNcK6VXkwbB8zByXPophg8mSqZdXeyW+sZ3FmHu9/g917EUXtyK2ajmosJ1vL2e2H/W684vDB0gOT1wMxQqXYI2xAAAAPQEALTHmF0QWP2GLsH2Dkkk1bAUbFhqdd8KfEMeRDve4rhAQGCCf+tZ1j8XIxYsNanCaoaQtyaA7I5/XsYAAAAApAQA30eYXRBY/WK8keZBENmawvIgVpBARJv46wZAKMG7TFTwMNNT3/MAAAAAcAQAQnHmF0QWPU7XFkZ7e2X6ezsXdAJFwuG93AQAAABABABNEeYXRBY9x2Ui4lAP8AAAAOQGeY2pBY/8q3ij1dp6IP4xdqbrku0a1OuJwlftCYMOBWVujxEBHS9BKs9h90LfBxp9pOUN86VPGqQAAAE8BAKqeY2pBY/9YHkyZF6RTQO9gF0JwXoVhaKCiIWnbrRX9kHyBfb5j/o6gokzYFRvVx2JUaGUjfVcweYU7kfuY2mYotYiNhd577jVu+yhPAAAAYQEAVSeY2pBY/1zMu3Hsz2p4eIiq5cDRIWqnflSoaVajo1bP9RK9lAqRjx2P9JZjazWZ6SdKQZzPYcVO4s8fba1rWjpKrj6xkipAocqiyQCULQK6iKN5syecr9WwuiyY5KUAAABuAQB/p5jakFj/XxNT+z7oN+E7bIxOJ5oPFlYqWXbqF5NUNaMY1Hn+pjdcwffWSwBqeYcZkU2f7MRU1+5/0YIJ4EDzPJtZDD8zXP0cdQnaGni8tZo+hqOn3MwYBKmrdh6VpIyXdil8nEQzNoNQpzsAAAAuAQAtMeY2pBY/YVWhjrRm+v0DlboRcEV2uUVAl9a/bGi144jBIHqM1pD97hzygQAAACcBADfR5jakFj9YU3nGiZ9CovEYw6qBgehhIm+qa/dFUX+uyRKO40EAAAAaAQAQnHmNqQWPWC8SFOVfYG808L5+p56hvhEAAAANAQATRHmNqQWPcxYBJwAAAXBBmmhJqEFomUwIUf8MZ2VoAAADAAAI8PXVvrfa3zpGFwwZbSAd611hOJWURvzuRUO2TaVGKixWrDpF1qVRKZfGS/8IIVA5iGFUkNcPO0KetsoSqLfxOCvy4N/zs/GyJAikZ8Keb79Kghdoy5K6S19nkRZ8gn99lC6ZEpCRHdVip//Z17GZ+uE8kXs/+dUrg+c52Bnz0al+M7WxqLDlUKrvu0l7XFScuJbbkR+QATouMWrK2Xd8Mo0qfO1ltC+BsAvWQsEfJnLpFMqKtLoIF7yHOBvJ6fqmmUSLKVPtPuOwsBUr52Uiuouj8Vez4PSaJF2KW5U+x+UUzSsCHWT2m3kt6UuG0ahN0TyAxqwDtL5DFpCe0uGSKg/RaMIWFk2z4AKEujZ27HT4ZSBpzZEBI6rfxEBmOwmhhxtGxHJeTuu3+Tfv3ft6+uKAsTEC2vEHpP/3DDuBmuLPwmT1Of+QFSjlHgHOgibMgMoDNG3TunyOYwAAAY5BAKqaaEmoQWiZTAhZ/wv+6JuaYL3HCGdvraVz4NWbILUdbgPKlnlayxBQ4S9uT+r8Y5HtOH0oCFDUCZmhAgAABiGypPw6Hxeu7iCeVAE58wnNCnGVFMsoy8ZKOyMGebRWL3VFBJh4ZYlFIpIqZ5mfUQlTGb3OWXmFdKZHOB3yaOy3Lm3hxlhyXFuo82d+xYkKump/4qe01LoGlit741xSWqZtHIL1eE9qkmUy7MVC9deJ2fmUr9u5yMEWUXGk781mMODQ11387Gk+M+3rcC3PEglZe2NfxBNf7ck9+AE5WTWA7GAUIAp2k9c7LxG40W0LY7TUHvPUyPcp15TXLCuyt4pQ4vqWy3PwQQFHBx27w98fNz+TKwcwN/yeyEnpQTzWBRRQV0WJrHPK48otxK/6WQUdhdxlwGGJyM4b46ImxARNGQG6r8iNpus+zuHg+2Ph1pwOqPMxFyp8u/zyG9NSJgcMq66r+G6gMAC5D/4RV60lQI3kh6bWtAvPzI4BUjxTvs9ocPlFaAipFN7xgQAAAWpBAFUmmhJqEFomUwIafxJel0MsGX9JiDVHQEi+g5ZGOBnPOgKjLcZUwAAAHX91rg7EizyuWL85N8U0tJjHGe5Bn1v9EEciVW4umF+QdINsrC3Bk46AYROpuru8Qj1wxS2mKiVNT38SOtaCpiUJpP4UDnqbpr89mc8EIFrUxZBvdPj6Op5ktYBWdJX2i8uec9KJiN6pvsvqwkSc+nPRTzJyzEpigWoYJusmERNhhg5cTREX/EHUc6Aq1FU80UA3nLYE9dfi8ih+n/FTGD8rS1H/Si5eZHEMC+o+RjxxUUQqZ0aYSyGIB3B5gmyviaosWl6+GPrE/xr7exXO4jo/5Mqn0KmyVPoYKQuxsFFsdOupQjhf6x/GrruLzDlnjyU+BAlShPXRqh3pPPmfPUScFBCBu5MZlNo11cTrwIvy+fllYUrCiMB2GTSjvdIWDsbhhByoS2lIhrQMpAOiTQqkskhuQgB+ZqxMmYJCgQAAAm1BAH+mmhJqEFomUwIafxJelzfPXFVKdvSleqxvArhKe2lkHs7Tzy2wcbTFONrSG6wAvHeQQlI9GaGCu9eiF92hyG92VjbqZjUXAoZeuXHEXxTM3MjFRuKOTdlzGFIFjM5ilikOrh1ImZkHtxrAASvrw9uwW/libqtK2xkxPRhMVwRYTgqQr+9FcJrO9O5uG9LIoCeeQ+hu7YepxA8IDdfNgcddVJGVrzqJrxIa0b9NbWvj5D2C1+AXZ1KkS5vnmundvimDKvEsSA5zkBiCbpO2GO8KH+s5UdH/m8zT7MGpEhPWPXnZNMsdXp+ACMOwxARcaPxukJ2R8xnpe9azsz55GYejGWMHknZQLdlbVf8r2krMiXief6bNSikZwrHKSdmjnNrBFcD89zbjRmlmV1Gc3tkEnAPwsg6Iw8jBu/pDHyhyB8v0TyCcE4PHikc+5vWAUpv0+WY4GIjRKf926DBgNEqFRs/c2i7PeRV3JmQS4PHxwWPG1vJxKlYSjFn52LPZz4PamDNb5os5fwhk/aI+J+rg3SO5bUW2zygH4mjJd3L/+eG75T47YjDXKjARr1KzLNMdm8o3o5wpt5bMMwH5rk7gX3T9D89bnmTCCkLf0yeXCWwi3JOUuD5C8hP467MY3z3NDUlPgaojqwYwPG2aexa40hZqnBcfPA/UTI1fPNYj/624C8DAcSQKUpJQgilTAMdLBhiKmSkioGnJrUuEc6tZqYZv1kphv90scyM7Onu7f9YflT+N48t1DZMANFRruKq5McKTQHV79unNk3+8NC6dXt2EVL37HoxeFEgKqSTQXjwEbi4Ri3rOdpEAAADzQQAtMaaEmoQWiZTAhp+YOKySqh8xdQZpGsMcHD7OcwITuYqJj2z4O3ni/6wtE9PtvnhMACAAgfdfnaevv4AeLEeuMVH/2dPiv5mLFRiCXoDGmzRx6mNVdyLoWOc9bG/W6NuPbJevJo3+KykFP64VapesabnLST8/lREd9ToP2pc6+TY7vy7UXy40Q8RucUNSCB7/rbYkouqGUvhJf9/HmErXqrFdNLVZUxM+YIgoHuVBD/orfvMzs6hWo2DPekGsj6zBfE6zFfkRdh+25/4YPsSFUZdB0PRa7a+MAyvq6gU9Eb383ybjJC6qCe9WqVDXywPFAAAA70EAN9GmhJqEFomUwIafGXy7xXBAAAPhq0rb30OGam+2MAAndE/7MKAAAXVykzZJc5Exr9KW4ZF3rRJt3ckWmJea37ayubtr8mD2pbtZbCaCtrho1lSsSehNBDfI2ebeUcDL2J0W+K94/czQaJ8j9p4dZu+8zA1D6/V8ZFUdnDeOpDOWgJmDoBJdqPQTtryPmLRpWDlqopcvMdK8TsTaqw/8QlDXJFXBXyJNw4ZUPDDUo1T/t4XGx9BQt6WdimOmmKrwgxRatJbOklnHMsGvHFpFBaMrjU7zSLQsTQd/0yCYlj2eKK9Vi2+09Nlj6JNBAAAA20EAEJxpoSahBaJlMCFH/w7rmBNwAAB9+HWmkeJ/qUZUAArvT+QrPf5paho50Lk1g/0ZVGFGizWrL1aMt/bUFcqXBuQGaPccZpzZzHQvCPoCnk/j8/UVuMkm9lQ9WrvUaycmtFybW9cwXKZRp/KlADQwuZzVbJKWshn2ojPHE+6e03F6YgcgVp1A8+QG1Wndh/cP36K8BvnyvaYPeJ0D5sQGee3H5mRW2eVEAhqXEDw2c+kCtHCf1WurqyddmCO+npPpQUAkJNfPXLKFQiYKqqrKTU6Rx+Fgaszv3QAAAH1BABNEaaEmoQWiZTAhR/8scdw25GlAvjLcAAADAAAEGOSBrw2w5qj0VkiBPia1izWgCKFQD/u8tSpFxFIlZgGybDGspPHb9VANHAlz7v+3EBbvXnMJbngS1SfsUbUqihAWr0EXjyBJkaHsxxJeIHDOFnEoDsu/t0gnwRJvwQAAAIpBnoZFESwSPyd2w0dg2utcfcWri2um3m4q718zuYv15MaRqV0clbLuZovK/S/+7AVgWbqyGYfFB/RrEjLtTxY/jtSSTyx7B4Hna+XkiiFSLFmsYulaAmOqVDbNlld4ex75AFRWZefUWiegHid6FHwS1cpE4iv+kbL35jCP5Qikks//URBZE3kB2f0AAABnQQCqnoZFESwSP1JKgZbWBCILKDbcuRTEwSWDJAQe3jr5LO+7qcvZpUPO+0Md9M1I9jtee/Q6avXujk4mpIySCmqOgwv+o+KriTvCfwKJro0BUzPuxC6c/UuoZ6uhHHT7bFzJfyjicQAAAKlBAFUnoZFESwSPVfRHqDvUhIjlD6c3C3EqKBgOjVOsr1P59x2EvHC7FfmnUUXl5+/6HYwF8n+tRsKsMD2+0rNSTwevhREZNSC6Cb5u+zWjRAm7qcvc6/iRZyMqEcZnI3YExD+c5ho5bWYovOC7Mhc+UhyDibF1g9jVa4qjPydNQT5nXWJprxvWbgeMo+YCLJn2/KhUFbgOhXSdgQkvgnAV5o2d5eu2rV0/AAABFEEAf6ehkURLBI9ZCm6aTXDLDJ1pysjERLdQw2INbDpRLbWz5mkqsdmDAjBKYj5bSsERzzoM4ltvuT9XVBEg3h9cBmUxmKOCMoW1b9XRMop5clvfnYUZ1JDKso3ej0Bav/8u/4V58KDbQcHCe5DXMhgxB8CBTVMjjN9SK2cqcbydJrcalrkMF6789eWtJGtmgb0oNLpCz6dqlzE/G8mc67UBRNs11LLFEGoW6mUdipg/a0c53eZv3kE6k8NvWa4OEf3SQnvUUylRS57P2NQdH6NBygVn8jNZUidEpgK2IGB2bEhIIkelNEdsVIh4IZ+YAEZtBa6/FU1sbPjclJEOrKC+lNw8DX/6uf1hQTsM4DkoO8aFFwAAAGhBAC0x6GRREsEj/1jqVaagw0/I+RROrWh/xaZnCByydEo8zLDXKN7vCT5elCZLNDmrc6IxT6srvoyHvQh4MFwU7oH+JyhTiIsV9bsnRfcnXy4JdqIIF/+c462rvQlgWnv/gXD+Zr25UQAAAHJBADfR6GRREsEj/6lTSzV8AU4sorlT2PMWCdntkFXktagIMQ1RcrgocG9qSZMOWvItshdGi+HOAFASD5msMryTiOu+S+aZeQfRz4hCf52aHzUpSEImb8rAQPZ2OTeFZJzObnSt0BQ1VDi6W/j+pyLEYHkAAABhQQAQnHoZFESwSP9RXgS4a3N13SH6mh7R87tXUW8oeIBlcSoRHuRzeL02z2sSvwXXUPLGxknd2zBbTn8SMU27kJ0DqKMWlgtYiAXKFXHE+SgH7xlqU7ezV9G9WyBUcFdYGQAAABxBABNEehkURLBI/2pRHuj8dxgTJh0Q0vGws8iBAAAAKQGepXRBU/8qM4pLk1aw1fXi4n7FnO0u2IElUyvyFo73SoSxwmPsCB2BAAAATgEAqp6ldEFT/1XAwJk56O9aa3CRk+kH2iupsqTopfFzVSpx+KJa2GSdsWxD7Z7ZFNSVICjZXMBjLXfkYW7yfwpn3a/RV2nq6v7HZEJKwQAAAE0BAFUnqV0QVP9auB9siFsZCgIAe7TnXshos1tsgovZSnYldytPkharJslwBNINN7WkBzkG/KyprSpgY3K6gNDSlsBjL1RgVgZsOtINVQAAAGwBAH+nqV0QVP9c+B9sQs+5Sl4IEtE0lisrZK2BD9YPY2KYPkgp1xDK9yQ5r8PNJOTbFEmJkcpF3kOpjRHZ5IvHtA01tBULV06wW7wWm+zSyBnKEn7vfrpTbp1r9oY7RvHdXUZgDYshmcgYHoEAAABDAQAtMepXRBU/X1QSkFYjq6NjCd4sIs1kVdeUkbhYinueZtLNXFSwJ9HEIGYKuJRkVK3hqvVpejPMIdqwc5U2G8ZpdQAAAD8BADfR6ldEFT9WKEWRGmr6oqQMF4RFWl4+yvEyOzdkTfdRShjrhq/amqG0vKTesuIYs+w8eJwREaWStiOYvBsAAAAwAQAQnHqV0QVPVilwlIpwz9hx+ovrjFqp7P12860FWsujmVGgJRBbgYGmUPA93/L5AAAAFwEAE0R6ldEFT2+JI91Vj3+CcRtIwV85AAAAVwGep2pBM/8ob4wMY/lZRrhK8uv6MBleyCSNWdT1xen21iGeV9KpI1YMNXvOUFZynfuxAKzqaVUlbE5v9fXNUayttnWudJFrjNaXm4jfU2sK12j76eSr4AAAAFkBAKqep2pBM/9Soea+YhJjqafT8iEYPat61+qbGGg2zcF+gqWJ0Cohdbnl7Lwiel54SQsb8uuZF0QkV8SrizVwXZIADmDTImyB/PAkMoFUPYDwSQjEQ8pG4AAAAGkBAFUnqdqQTP9XF5wW/TrLL2GkEKva6JaYb+L4O0CN8IaYkwmTzQwMiTBJ9ZQgJQ3OjrqU/F0Jl6807EKEr+FkBeVZ7FozFClQFATRo/cpxTifTwdUXysO4xaVHBOuyAGequlsVmYk9eAAAAB8AQB/p6nakEz/WVsDMxf6+Ipvzh7hC2GNbIQ0+QaT6gV2P9ZxQebTsYQ8hnrkbbe1I3f78kORFEFAptXkVTRTlxgukxd3WSXwNInlrtwuySiKTmWiiC4SJt3qBZPLKBUrLdzUyONECvEHDnNRJNJ54HQTP0FU8YwGCx19gAAAAFUBAC0x6nakEz9VXHpfJTObor4ShB+AVUOx6SYggzSBrRTerfZ1kpJrtlqeWFBmIRHCignRAoj8fgN1jgK86ktZwC+AWxKVWx+WRISp2uSluDNQ8eRwAAAAJwEAN9HqdqQTP1KkWS1WpcLxSDAcT2GczlFh3p9TP5/Zl2895q260wAAADoBABCcep2pBM9So8bbSnBmrFbQH9QOlOo6byRhnYSR56uF2jX9u2m5jqw0lQpkfK5ApsxZ3ty6yJvoAAAAEwEAE0R6nakEzyYPGKQmO19sCPgAAAK7QZqsSahBbJlMCOf/AadtFC3z2hZ58yuwuz78cmxnR7eES0RocEojoIrXToObN6XAzcWDJTn1y8xmiE3Jb9L1dyUd/jSkomy01H1FkzLH5l4K8o/xMUegq05fbm9pBhLgcA4MSXZfCFW7Cz750sm/S7C803iX1E2koz0hqyfVmfPy5yJ1YvUuesZxCmtWVu/UWJZiE3nAMut2wsUArxonwtk1xpF/lMuxxQ1rQ04ZA30A8IRNhBqgTHj+yfHu3X7ZzSA6axq26gs8Yl134WPhzwjob9nYL1+SC5CxRjM6KKwafx4iSmVutLodWqgh4qSi3+NoRDu4ECyVGSuNPnKQiJ7PfQRcjMn+Qwb5yHBnbyjmGXRN4u+SorVWL9GM04q8j1ehMFD9Q3+S8Ui1GhK0kq72aejzBA/pLKQwdMET5UeVm/0PPTKxtpAQQEwDfRsD7hkakrWVH56sCAAjCheZf9ECm4zDGtrEUZFYbPLZJd7xfysbpVbmrxEgOOlgdUZ/mocuiNYHuodH6gNkJe5VgUx3A4Wtg+HNZfKVqp2BES+DQCLy5dCmYVHpAY7TlJG3nuuv6ZtFE6ixrGhwYr4ijTfXE/sZ+IfpRDWNhOzjwYy+ookCGFaMgZVzA3NGFkamOSyqH/KWglKrV/0Eud4ut0jFRpwWc/I0sqspn76wNoX6KsyESpVs1DM59tiYYXJitfi3VLwtzWYs0wepYo3WNYMy01OTcNl3gcHS+7aN3kqeYiC2TGudMVQVzgGEhsMZasRDTjOMhwW1/+eB7zVPnqjsIL5Tv9CqWvlllHSZcBfil62T1ieGhBIEPt5ml/9RUY1m/fBkwyEu73egwPreAk8bdkHZVdlhOLyQekCt+JSY8nWNbeG9yt6fSdCmEWEcnDqiATWXb40bRUvj4F/zk1yFKkB+FuPq3NxAAAACf0EAqpqsSahBbJlMCOf/pjRBqEAAG5eLMUIWO8x1GkzjO/zuzZuiUWTWmapbvcQ7uQJHEXBly1Er4nfcnyUyuM0YU2w3q1TGFkRtb8a2mnBn/op1MVx10xg5YtCgiC67D0H4mgWH04eP3lQoqB7rM41POIi9VejAUHPuJ7s538UmOY4dI9l4JNSZCmizE+ViDyRmF2OWk0mqlEpTS6hXwZube6BQq4WXWRQQpzAieuDRYLrBky5V00JiqjbR/lDjeYhO587MxbqKn+gjIoG0OmG1ssj32bmpQE21BPWJvv10l1rYm4M0CE1l8IB/JwHtTkEk8KprjRVTtBHrKgSBK11zje8hgXuMfah30KEMOym5ighxe2LfW0pLkAzkb2Ia068fCbuEutkfwhbn00SrkhuPwWXPBe4bKgC9B78kTaceuItQWamoqlNEYSPud89r5hENUWPzgUxqp7HzWNsGi04PoDt8kZkToqIP3wiB86jlkwMgyQjMvLU/MB1rlbzJRrrTcMybHUQiDIbMp9sxbMmDuJ6uJUMjccxa6yEApICd+2SLjNKQlLixdPNguJK2Z3gekTYAzMCmPkHzEhHp9VrlgRTZ+Sls5DLcPL4Jsbejy6H3KMP/44umH+jYAsXBUbk4Y+cSRBMcsVPOe4tumRichvJSvThlBvD6txGe/72M3RXfxNR/Vv+//hyOmrjkldSgfMP3bj1s9Nn8FuzT+aUNkk0eZa7woI/h5xZOoTfTDleM2pTPrvfTg0+dLVZNVdHcUXkPGH4AHTnpHTN1TKWmsKC0P8ViPfVOztC6Zox6OlH18iuq3EZpa4LjZGq7dLUdywd0uAR6yGy5bzmWEAAAApdBAFUmqxJqEFsmUwISf5cokR32jm+zUG4r96v/54L9OvO+lgTQEUxL56Vmr0VkXrePixd6KZyE4Ar8YuRd2Rn7Ng8xMneB/HxWZT9Ai8AKXftNOWHBZp91Yo215l/CdTIJicVmGQGWjYw7O8Rvv2juRem+OL7n/4ykxkD9XuG7qiNBDBWgKQbamwFII1m/7bFXifVQlWnHH4R2qgp9Xg+CBtAWPbswTdpr61EPgaFnxSIX+ZNqeVcElgZ+hNJoSjuaQfTyIMjEU8p3zZMC0y7FIfyxNqOfUV7Vm3SLOqqWS4cpKLorWDodtnpSl0lqJyDf4aRn1J+QRIf7cajWN8xR8tAG10I44Yzjzxt0brAl9TYpOW42e9WKiXz887/My8VzS7VBiqX0i/0enqq6UtmkFRcGbhTwq077/Fk4J66dGhX4jsWsLW+hB7V924G4uNglasRrH1tXqLLmZxO0qqeTWuB7mwPYL5peOWuvTdaxO+FGnINwsoeNBwVIlFcDdWFyJfjyH2LQFE4krfrKrhEjPqDy7NQ0ZKJjV8rC3CPyYfsiWnod8WYJjjMEIweoPpEl+hk89s+5dgAPmDFgNCk+TQxo2LZ71Zrqy3IJjfsc8e0ZNTZtr0znFiw2c0PohiYmjvo1hMOemkQ99T+0jOpryrF8ZFSRqlxAoxQFu2ZIyPSbvkOF8/pv8fqjVb8q3A+LoAoFCl+tFpaorbT48vtIU5uPvTvt1jPjC9Q52YZd5NdLaKqAPOuJ8o1Uz5nJJOWdoeHAQE1ynY6Rx4vgtsai0v+aEB5n0R0BCDmTqCnrdswgcWH2V4QR/BQu40weHRhJVUUJKur+mjn3X1RhXGfkyfnzHm0YfZbu0ZKfKsYIF7AFgIbMzRwAAAN1QQB/pqsSahBbJlMCEn+X7wkV2FpOhKAAY1wp+jyM6H7Pt5Q8vJCpeHTgWQ1A8I7t6AUiRTTCszJGAHAgWaC6a8KepjCVZuT5xOuroW9zSIhWG38jZHeUNBexlUtukH9TtXbVMtp63nLw9cfW9wt6ytOwH6xQL2BkYRGCpoxd+gc2hm3ABDKwTfd7eHrh6rhKHoh207avaNtssTJw/Y1+NSniEdBwxTNSQJokRo8S0BP9MRxzd9C+a+6HdftB7Lv2DWcw3tnA0r6NqGbziGmDvtEFBRoBIjW/Qvet9LyKwqPt2KHYg/vR5gW2Sm+S22FyvD6farRGy/VkyoXjwBgUnjQXhtz0w6+x+5bncCFi/55OJEfYARQ1YJCQYor/x3vsS3a0K/51YXzQa4KhIPjVVxFBbS/H9l8f/v5YMq7YuTmW2QhKnWXRoNaJKDfHh4Q7TzYdjGASLVqWYIDUmDBkDSbSv0r1a0bDxZNuxBLCwo6r8QdmlRR0btx0q+wWCHdmaG0h1QKYw4nx0Pv1w/1MJO/u+mk6PLzPZ6b6Lca/XgcXQc9ZXy5HrItnpzswWF2pkkbTsjK3OiHIttvgBU47HD1fyulvQPUhthKRRbNzbcqiGfWLEV7GRnqnW5qp+inw83Ng0frsAvc6ojZ9aa0RzU4lAcBFozaA7y12kSfU8uFzCZVRd7aGkl65hposc3DgXRvjuziNmPNtQcs5aWqC+m1kjItybINfJeNHsx6IKQh/iXAR1gEctP4pVheZexOvW+IgoaU3A/jvKtmaiIJBR5XCl91mwg1XUmxcoWtQlzRewn7EFOorq81UIjs0K4J9nbG2VqZjsZLbzcNedWzzTYWLa83rgDdICsoS7f3QqgL0sGYLQzjK6m4bDRQhNfinbiwGFHB7NI5HXqwH+fS3PExobQAvYXD+4exg8Q7uRfjxsMFOIjqcJchilyYetObJ7hOVrOCgQrUaFJ4hCjtJJKxeQFAgPJsbOcb9v3/DoTvo5x73FTw6AIzipyDGMi+rO6aDb1X5OGr7mjVw0amx99/nk75JFvPVRJyaMPLu/6AlwXO74rbs8Epyj+ECbiHbE7nIiFSUBQVvL6L2YwtT+35J3EpgV4m986rY4o0H5dX0FhJ8SI37UrE7dp4AQ6DYwZ45vYanjBTnd443jlKGZftX65pAAAABg0EALTGqxJqEFsmUwISfBuYSU/cR/gtagHcU7n+R/oKn7Za7vyv2GpPqQ4ANS/CxAVAwB4qoSb2iDXFE4tDpQAUE0t/mm6QmodQUobeYS65bUfeyRPqYyUZKFLuBDQ1EPmSaIMJBJlDw3fMEHOimca4Kh2liOmIuL86sSF8rJ9pF16Kimaev20GbTxsTqxxnp8utFNufmPcwBBRp4CVLuzIiOdIbq3w6TVd3YJq5FYmvh4Hw4/scm1d7Jq57g+6KXnMvzj5nl8dTjsWPLYgQPCKpOGQD9sons2cI+Wep5Gx1fAggET0bK+FGHu5k+yGLr8qNv1TI8w5csf+BKJvrNGvDHyqmMMgi2J5M7BgpMTMmdUuiavsXzIQDZHnDZG9es3rp+Q3gjh625NHUID/pAXVTecAHjwQ/jWA21zOk6oSWcOqwan/30xkFSDghzhGQSMZHcqYrhbPuH9xJkt0pl8HYisWrzhE8e4/8sJwAZASFWpft3iCxsBExvAtDl4iXYV55YAAAAXxBADfRqsSahBbJlMCEnzG0oTJQASI7qDJ2y17pGh+k1Lt0APkhOkkNfczb5Qd9+jxLO+oQAHxJ2GZDnVrJRPJEYPsIWYeyAVxl6K7gql6BpdsVbmf5GUAOSSvd72wiR6XefNYVRagJg1nZ8zwTHslMUMqqw7CadmVo38nehc+uQCLF6hL+m6qAfv1I9+VS1s/c86jmgdKbZS9dY1fUcOyszCOhmnP0wNdkw+mLzejTbYxh/hPq3GeBnMSfzFl4BQdK6bfRxRLw3LZwbkfCB0Of9wUWXEClBo6l0iWvZwSIvVirp3QoowQRKeYvsvYz3vdQ1O3jCuj4366FDKQiR/R1IqMKrq4hRcaTRFfjtaMPK0G+eSE2fL1o1AS3K6+uHkKUgIs4t0OsHuU1vAestGS75x2VttZeb6G263Z/WUEMgQQhhPq1mz1FWH1aiqzAgTyueQ6YY8N+t4jrof9P3VDo14qVLlj5A5gYokiKiWdPsdKx/eGrOq37nhMqwAAAAWtBABCcarEmoQWyZTAjn0ayDUWpgpUWMJIA7i1oYAAPqw1Nlm+eJrTcyvAt+Dk3LfwHi3pE2Vq4Lpe0tDnc3QWkXXEEYasNSLBtfC+MCEFMfE/INcpv/LzHIUJ5Ani3rZAi7CVdD8HabOj1HkOs8/w9aRWEeBFEnupIUlvdEWN33ZnSw8y6MkulndlAi32sHAjVJLsTT65kJJ0CKuFdgrNbYdadtzd3BvJIs/Lbr7lT7UK3IfYpxTGQcBF+ATc1D4tXiqqM2SvtJ49HQu4GriDiCGu1WhyyGjfEX6LiyRzvNRlSUP80koaXE68zv2f8K7oSOlyzseKyguYBfGq9gFY7VC53uFup2XhaIx9Zos3gOBdfhnMKeOVECkYG+cWv+LhapKMou18osZILQDtOfL9UiP6q02rIVQ56G4rm/9at7R1+A+WnJ/aV1W0uYzoGKcqzZcR5PhmwERu7FjPzRV4DAy0uQdIgCMBbVYAAAADcQQATRGqxJqEFsmUwI58wr7K4ikgAAAZVm71yVGA6o61dvKjNj1t9hVEY9k4YgI/gNnRJIlIPYsWzWQmJNdm/BDpbiP/xleSjLBCLj9mWsLfpa+g714r/o4o+A7spIJh95qgNoVh1xesXD0dGvSMLxvPwemTRB1EAAbi3b3meVVAi8Inc8eFKKo7Li/p608tMlC3Pd8ZXiTKDpeflQ+nlNQg1WJJETHVoR5SUqC3OiufLKDSUrZQKfgnlCnoaVKDV/WKvQDYLdJEqBUVyJtRjjf9+Ag2WcfP1ssA74AAAAPhBnspFFSw4/yN9w6iuCHckMSDt7UgSXIb6m8eY/6sal95aH3GAbcelsSgbLRibBQ4T8TR2P2XSQgKw6btvs7c821RqQOqFqDxw/6jh0YehH0k8kNH/PXZZMirLax3l1t8uMyfXebYG23rgVLXmVk+n40pNrJqSINIye7gZScgGWHsZTQ8GJ5+/+7bvhc0kWQoeiKZrR9vvIi/9iOfHDqAbprg5tKnFZ/Y8C927CLcvC3SGFzAJ4c4q+8jn9v9GJSLKkHIzP+7SyveYKTInReTYC/vKVmc2M7w0dOEtzUGmI/3e6seKr6TN4aFSyYH63G2zSlg2yFHnwQAAAP5BAKqeykUVLDj/552xt6Aro33YWm2i74pirGdJgnTWmnH3js7/ESsIV1JwTlL/sMXyrTM+uqaz6i7frVSXgoySKxlRyfoTHaQsKvuzRIKMomer5admkImoUedsejci1MV0Rsd3igOAldam9yofuukGDxX4NKOPYLwoxGKH74ClemzGdzES1Xj4zpsRgQe5I+fYFFomuQpK8QlfabkyxzIJ8nnxBzTy6EmYj/bgzeVqNZD/PMjJUSLn2+ofrA4rB8ul6OnBF4EYGrZBjQRyjzKhZlRLCiLjYIs8Mae4UuecK8e2e0F0LzC/IBadsJYVH+EwVGnMwW3AALpWS12L+QAAAQhBAFUnspFFSw4/THjFjGpyTDT3xpOy77c3lZMwtCUQIyHW20mdwUn8lFHumbtNf+RNfzvyKJGfVYy6mvJlR3iUgQhMYkzyYknN7N7afZlAL3P8tSBsV5x439vFRG2BCg+HXqfa/qG2Dr1oYYlMmYKy+WavamQRrVEaeiuPXxGOeNIOMt0YgsW/8uWMpXpnD6X+TvJ0iEAMC+3SsDYLwnSa9sRQunmGacDDe376h4V/SSqRcLazxwbSCwWdpxvrOE57kT/VXjCckBQLU6IUEII5gK03EEqvxLH+C4HsaqOg9gBxYvkCo2rv4rqiN4Dq6pE4TuYG0Kv5cMIE7P/zVf7rvPMRuajoOSMAAAH7QQB/p7KRRUsOP2yWC05vkTko7bvOhAGbFtlsv2vfxcPwDWlzvCg3onM3bDKuWuodd4fXYHdUkRttwWdwj0Sj7ybOz1z9IkppHrJJLe7NA/3b9QAWsG+PEpC59qBGA30J2OAU1QMldt7lyj193FSnxiLf8sqNNnLa4SXafyq4fbuOAfo4ZSPksOSxr/fu9r5Jm1MhJWa/N6DugmSg1LeTXXlb5OBaUptoI+7kEvJ8Zi50pMezo+8XGC6Rnxhc/CLdbBjS26zpd55VZ3g1TVKOr72sW4RRhtYswK2XL1eKviinVSalhlY+onvW7X/q/9S7kjTyudjnHnSNqAIQZVtzjokI8KSJrIgUbzmp92i6lNavg684zTc1alUvTVSLEEAPyauaPdSbDdIlUYx5bb8ej919wBlyZz94xrNJa/rT4QNlrejcJ6v38BZMnfAGg5z510/6U67cf4kaCFwOjsXM2m59ZXd5OvuBDqwSsvCbK7FUTZ+2gIp7Jiyq+nDstQGLz6+GQokKSGrxMYcOtWdIGqkxQrRmlWhwz5fazBoAZrWXSWvWwgI/MAe5Vh8gSVJkgYyWggqjPjISvUurei5Xp2wLCVXWxnDesXtj7HOVgzsc6Rfkyq6LW+f2rwV0OnEHlGrm5q7ZMw2y3Fo6fpb41WCwTkEcvcaEmviFAAAAx0EALTHspFFSw4/hdYb/WZAHjQ8dAx+BU7SAPGTHlbXiB96LctQ/1Sh/M/MwtFp+CRWd8cM7AN/2Nm8EMCDQNKDktKSEf2cNdY8yZoq3bmtO0kzECTUtQLCStQtaOBfP7Jpc6GoY37rUsjuPtUx69IB/V3N1lS5Ch5fgVfEaXCiPDsdRWSYOf02ugfm0ByKqPf4pYEO60CVZfRxqyZejfJCe5vJpI29P+B+JZDjlrGrsXoE3P/eEmVTd0avcIPyZcA7uPFQRFq8AAACrQQA30eykUVLDj6YJy2QaTKVHCkVuSKArsM5wreCoKExrAaWHz0e+IS6oC/aYjg4cU34QcJDzfvtL4yyiqczVbOiYZpSZ+QncSdaoE5qDmGZX64WY1q3XWrPThwPg3vcPON7iFPf1UrAtaSuS6TBJovOnz1GPX54HpMoADQBvsjCratXxIfhJnME9RzK+DEPnYUlRrCL/uV44/QhN0Tp14ubiT+ILHnTS5ARBAAAAaUEAEJx7KRRUsOP/ahZEDngLdQtHodT443Z1mFinh5Nv3i4ZBV8F/FhleH7n6g6WuDO7Kmm8v93G2PSsWakySxc8NqvWO5uu5s3LQVQ/9qXzzpDNc3h1r4NMnIbrLT7Bq4Ozd7TPafiDwQAAADdBABNEeykUVLDj/7GP/yZiWTNfKhAn6VvbqSHL8aNSgWAX9yxtkAl9VPXRUlzUiWR5dUgOSgPjAAAAaQGe6XRBE/8mb4baXMKVr1y9J6kB5eRp6OF8fNpEeOosvX3pq93BFu8bO+x8zahPpbph5nzieCQl9mXubXf7fKd06o25lGmhvi/swm2+pphWll3dX5EI8DFVXmN+LHfLxM6Bq6DqYPgv4AAAAHMBAKqe6XRBE/9PEK/vW2PWs50YlGzbOSe1oGWKJg12uVBgLWh+s7guri0MfsNG/LgvnoK+ekOaMijDyfysqD6D/zWtANFDa+9QZwE8Fd0krCPg1ZvWMIJsDV5AVCOVo3JRDh3/DiMX0ltyFcG+UAMQRamAAAAAqgEAVSe6XRBE/1OVF/hOPPJPHTZzt1znArgVJlSAmg7eVCgevbTqR5pwEU9Kx3kQXLKRZnL9G9J72lQN5FdDFFdImeuFQCLJK8qGIFY7F8RomByhbaWKBbkOWdP+7WNpHyJpMPiMLZ6PqGwPgTtRHb2VG8O0820pVm/9PllFehvFq0OLPNisBUNMChwm5rbcfO0r6YWfvjIPLjMaTr44jeQGoXktHyPYmG/IAAAAxAEAf6e6XRBE/1PirS99+gLtwEyBNuoSy4zen1F/9Q7tHwbp46YHvycmBnZ+5jjhTttWbfxVB4JoH0qOt91SGTSywlmT/0gPIXsrjuHa94iujGTef8PUnkGMI8ihX9SGs5VDfx55XujG787O8NNVpYxAiPa8ukUtWFSC4i6yvWUN9G8lT24KgOOaTqtj64GwR1y/q49FkWWFGWnVa1yiqMUV3zl09M1poy/pd4oSvyW8ODRTCYSRDOC+fT4pbLWzpnF8hcAAAABpAQAtMe6XRBE/UMEMS5Y/GAkRjX9edYImZNGfaRV9Jat2pibyL1ZoutlKFZ5dHI28WXPxOnQFlsnRH43rcB9COrgVuYC7o9p3QUT/kVZAS0VNM88fO4lzNAvVcUy825devvLOf8txOxNbAAAAewEAN9Hul0QRP6UToYWeumFVi07ImqUr6NB3Jojcik1eB7rvJdmscSXjj894aIakWx4+Lr9ZbBlKXNbrK4Q9xIbD7lWIPZyDXejjxJryGpNzLH80ZqsL3DhfjbcqFKtiE3AFgV/AgnvAe2zggg52s3Vr6WcXIGWMpIZsgAAAAD8BABCce6XRBE9PPjbVjXiMs8LEq0IW6hLKxHn/p9xjVsLLEdjZnw1/VK+/iSl8EhxnKs563lAJtqn/3K0OSYAAAAAfAQATRHul0QRPupFP1CJtlzKzHNrxggJFw4OC47ivgAAAAO0BnutqQ88k2UsxKMfYA0k0MbmP6NTwpobQbncRrpu6JT01pJVC3Mc22JyXIR//+f0Y9IgpLoyYfcy4zuOmoaUoUcJC+yRjNTBIf/Bn1Kra7gZyz5JLCzw9/mLmYyO7vtbdt/HKojeb+DdEas0Ie6cqQ6ZJKFf2NWwKIx53eKpSKOHVSERorj/jCLxM5orL3zYuS3f4PW7is//+6OiAqTWX9arPvmaY5M26B6SJs0RlJ4PVcoMt+E66dNYyiNMwPSMgwJGvwVCt67WM4ELhRWVHIvWd1eNZAj4z0tAkJxMw3moak1iIYLV6XwjaPEAAAADIAQCqnutqQ89LT0DnwMKh91FTgHELp6KYpacT/yiQFB5av9NPm7XF0e3FZYrlUi6vD3HOznlKAaetaY2twfF/TwMrDSV4xyoE8UC8PJcRlzl1OCf7RoxO2+NHEtBvAbzOmIfJkCNHf2AA1Qig4KAuS471DlisoDS4rL738u2EcmHFeuXSlmBuASZ4qZKSNpJxosKFYk1Oyx6YtGO7OfftSZjhiIMuEyUJ8krgktrHFjagEMLduiRgC3Eb3pKs2gAqBaJxj6tc5IkAAADdAQBVJ7rakPP/TnXtVOeE8RNTrJ+9VM7uERZfAqfZTe/vuu/dCk6uLR11Ef+6aXYERt0Px62wIw1bdBitwbreH6L055nGjvlJDGLNiAKm7YPXn+M1LX0sz9N9iUFFM+S0L1ecdNpbS6XAIccqGIy3KXr3IYC4TfZwoS1C8D+LHr5+qL6F82rClJ/ha6g+4MswP8lewArpYE3c9uebRErhZZ0TUJB1wQRtnRkoSMVzliEVRmLAs/hcVWFe5YSfOJ5LMUzvvaamah5fQyYxWsy+70/lnfS3T2kUFu5is+AAAAFYAQB/p7rakPP/T50oO8zLGbxJ8XTtlUZrUerieDR1HAgnN539guWbz4I6PsOQA7ReS9cCJoIO5eDfwefLSUS/GYd/BkH0Q1UXkEzj1KVWpfZOsQkdk8+Jnu/9zvWcYE1yyDmT35WkZRmXXznlpg0LetKI/z6nmOWEJ3IMstLFZYecUS11hwrV1tCNb2OazkTq/qB3Ff9wve1coZpwFJhXajdxfob4GR/f7Imp89LWMG7gbI2vZqzE0iDPAg/2fWs4SFWJhQMT2gctq4kbCnRabhMlrDWEQ3KeWTA8gZ+AoHGOtfaYTwN6VGOdRe8WgaktI2wbqHK3MvGtunMQqevZXp8CC3PyFmITEspPLrEVWa0r9RSwVRjXw0NTtk5vU5TxApnWzU9DW9/kb9rUR8FF6EZ5EMQAdpNI4KFywJXi+x7dm80l4LgkL/txwpPQy0kDmQx9IIWn6YAAAABoAQAtMe62pDz/TbJHrLpsSJMV1Ym9MSV8/Njdx/+Wb2nFIZUoQIEBEP5r0Aya/IY/KcoipOkx8tjVOJRel4GSAAaNJW8GHdV0eWb3Si0gqt1hmQ2OCoETgB2YCKw3lfx5gdAo3VdDuYAAAABkAQA30e62pDz/oBYQxGcthevEVJ28okSRoDP9xOQBByuXD30zLwAnOAafQikQLQksiLTE77SfYXPbBDeZXLoLDM9iJB+5cSl/iJh/dc/0kM9AjH076Iu6N7IAuIroTIo4yjKgcAAAAFQBABCce62pDz9KwkZZ0KPOvHTqv/y7+OSKnFpAFvckFrHwAPF0S3MafTjQbOFDMTVaim3WJH4fxoZrLSg/v8kzNYJp5sqD3n0Zif76CMAzIHU+gLAAAAAoAQATRHutqQ8/LUvIjDcoKmJh/JTkVTN+d5zI091Y/B6JNlcv97jhZwAABTdBmvBJqEFsmUwIx/8BYFUGUaOW4hIdGVxls47O+oaBp7GsNH12aSh84D+4e8hBu812WwSLXYP04oUhojN17GaQu1BBt4ELDyd+1mxnJAgy7YcO/og/khq8hxXwhN5Ht0oPvOGtB5TuuPQOoLMefebF364CxxFT9pX5SzWoUbYEqOkdkqsZUfhAjKzR1qDu413sbtj4RrtdDUCzBj9vUIMS56cS72MFC+a7E3KbfDTMZX4e1wZe38dDd8sqfmLz1FOqFTP11RCiZ2xlAcuRLCGn0N59NuRInfFhpUuN2HfiZXgnF6Qdmx0uXEOrUS1ACaFe8UuSwv9NZsVDtjaIzMd7kQGCN+1bBfJhg/C5tTaFKI1xVxV/JrNuHkn5fWHXlW9B8/tTU8FApgSfRBiETzTYoYINDZjCXLIQd5xAAvhn3oVhM/x2EofE8oHNSrd3Z8mDv4uk6SyPtT8aqTutnlBVB4e9pQbdJas7coaWs5fDMHPiBPO/whuiTohRvrvW5CnDZXbwsCZw8GKG4OMM19K5x7UoSEVwINuUq5I2mr8DkwHlVK9Gj2nC8xVIEsRP0jatsQfOicc1IPwIWykdJUlykSh2u7fQSDpqu2TDp/kpY8e8+pvNv4DheBk0sciCv+Bear5M9nl/O0/WIWpMt9XTmiOSd0MlU5WRkkB3sXqlbmLV+hqbNKTu4hSnre6YAEoOaX/FfezYevscFCRe1LVUWmCvnXXk+vWLMPus0/DdH5i9NbfyB1Oc53i4tc0TpDvyHtQUUt05GQrhcvBQeZtCd49XnDPW/O9Yq0Rs/45n6Zy6gJv6QW4aAUYTWqgUWnRnbZSWl2Z59yxMUgmuCOW2DOn/k88YU/tQyBGDB8DSu0pjHc58Hu5jnzyB8WBWpM0vWc/3AE76aG/FH2XbNzqVQ2UTtpSEkCakrv9iWiUluGVunA1+YMFR5AEdLHmIyYG92WpB3pd1u8UIgoRKh6F9B84GljrhPPbyqLiF4V/QvNNjLAT/rm+dsYauGtNLiasnybxf3NAUUeB4RsUiqvCL+QvKM/XLffUllD3rHn4Z2CSAXI86HycaWLUEd3l1aBXxILec2YMlPB6BarNzIEo9fLj0q31v+Of1S/AJy4H79GBo00F+n7yuADXNJa7Vl/S86tfNHF4m5j+NStjJ9MXshc8grHR0x99fvk5ruVtcuUipIR3NtO5g+CQFP4wQpE4TUUrTphMs6sqdnIcQDgehvstj9nmluEo0475DxdfMeQteHJ8eA9cl5A/a7r5YLOWal6w31fdxyYPBFl4uLVfslXh5pGXAcjlx0M99jMNPFWs7XxuymiYF1kpCnJVgcjtTxDkQIh1OPNu8Ks/Uo+JfcHgXVNTKgF1j1yB/E06hWvBJeUDr9a7fx87onHf0qabTUAQI1fA7emsMIah/XYGoJe2zJrwaYCY2RaA/Gl31pJQ03g75iSTQhNOkzATyt1p7EKTEY9fJ4UXZor+uAOF7n8yKv+x3x5OIjLdul1AqlSPvZN3w6y/IIJ+EFmCqYgfg/o6IfqGyFukAZCxNijaO5F9iQ4ZA7lA1kwofRrEJT3sYYM/rC4Mr+XLTKFwoFjCwdrE0Jc7eW/oLy//yIlVblma68jMopfXSIiWSf1oDqcBkQP/IiqKg/9JmAsXLHfCpEs8VVDOC2Yp4UQk5mp4C+DSRA4vQ9hqqBUtSSmhGZJKPHHFkSN88qq+tmwARNJCFpjx35gPx16sn0lTnI7rnfA417J64Y6Fp0UaBl6j3bfhP2nQG1VEAAAOjQQCqmvBJqEFsmUwIx/8NjpTdWyE5dro5IPQg15DP31nFC3RRpJ849H1zU4VcOf0YeR0PKI8ohbCaCvRZ9js7Te5dK1QlwE+3v/bnADI7pQAAAwCyE1qNjPaYk6afu4p/G4/aZoJkRMHcpkEXQOE4deBv9PNLz1IMst4ijdad4MyDmiAN3Mgj3JUfeZnv5JyzF30JJHQAAjWqrPJ2vGoYLEtMdmYP9j3FnDct6WlX1dfAkRUFPt/McfW8rwGfuts8D8bqIbGBTPxrBH6yF3v/xgZMshahA5qp9kFz4GgQR3KWibjrDZkkFfi812C5zmaCkeL2GosyJm++zCJmpKBqbG5Sg5pZlBtiiagaTiroSCD/QjFJR9bPS4GE+iuWRwMPuHeY40Ku/znQwiz6VPQfB3e2LS6bAxXvzA4pKbNZ4c+ghoZQ0njp1UYLuQN/lGtpTn2P82P9atEjsC+N6ttEQKoaYTasYWVvhmN70IUgF2jPw5d5h/PLerMvY+k8CGZFE0lTg6s8ilr26GTLgu6hNEFPADGEVNwINCgkZVPcIiQsHuqQ65yhqBfnDBxzHnTgaMAOsxrWY4+7F2Ouwc1e2UcYsUlgQhVpF+MTsb7amPsPJstlDWC/lfYReBbvn+SUu0SgpArYlMjWjwshUxdFGYCOVnJBDBo/sleDpUlOmS/R505v6V5S0+NxQTpwQ0uR7PCa3avdBe7LGH5aDvBdIWiD0OU5u9Jq5w+8szGOx1Ve/4iEC/27rJAuM8dbkL88lA74tjfpp8j9TPkGmKyvYA0vaGhkSRbVenTYeZX6YdhVK1k/QFUT75rLB0jF6/qkPSQ09N4n0KVdwGK2ZVsfQFHVDnSvP8lPhqtD/rudFBbAumw1XnPh+rVQ+Tz4aeMur2XHoAtW/yaNwvbzk3+OZES+24UrYwb6s34cM+4zMhnRKCi13i3R93wJpedUpGKdfzIjbD9nRRVWzgJds4PWtwv7LPzFFY4WVftTNRXStmX0MfdZO57e72SzOOrzwjIPeT2KBqRRBj10EO2/ILLGH/U/kAuca7Wb3LnL4CeKdejWxLAfO2dQIB4D40rmXYfuUzoEUOwVsN88KJ/5FwoU3OOqNWDsAD3KkK5V4DaQbXCAv5pV7Le0cZHocKL9n/qVrKaRu3WZ3+iMejp7Z7IYMC6Q9sCItLwWskAmjyeqZWhUkXrT7N8vw26UXWsLE0yl1Q7t3izLc+dulD1EXo/sIVd2wQAAA3ZBAFUmvBJqEFsmUwI5/wOP3DxR/QJnlfUK8AvuyUcMQH+/Bgcs1myTxvVQGsIxvebZPYyG1u1SwlIcoefqQrUiq0J7X3CMH1+3yysEZhvUbMPiHdItfHyQi8+V6M5tz6PVBnNmFe4IAAPlBmzgxVwX1rGo6v4NNxgp8D8Wrmkg/NbWa7dWaRcPzAVxihNc/84ywv4gl2zbAfyyrwl3MhYzklNiAkaw2lJ+A4Z+TSlphmExg3osiiGoCpP2K6hMfaWdD04OunCs4Nw3Q/pCFfSKurTwM82s1z9uM1CGNDMrjvkB47HKLDtGbBet6L+1HiE/rigpy0sTVBLRBGuuumEjsiwhfXV8ZcDxDquiCx28DRTnyzRi9I7KXESCipLB+OYVT3B2Vcl2lpEGyNwk9/V9LqVcNm/tZvUkj95Mp7gRU432Klfb27NVAhp2z43pOR6K4iJs8cMOCygzZfc9OnKab0LtA+ckH4+ACEVjpd6CPBIr/XwyDBvs04XZkonbCvy0VmgBzexv0hUxejg5YRMkf0+7hdAFsZL0or9iLJ+JBCqqezwMoPyNpXdtPpj3wxxs4Tj0pZXDsn4LAytLJ3wruQNwIrfkcU+ljTPrnXJIAGpV0zs9sLcc16e+koKtMQFQw+fu+eqbUsCGUzpn9Ebtwl9uZS/f47zrMqh+GExeHu/3VgpKRiKjhH8qi9sK36BeXU062UOsBsxUPlYelMjBGNa5swrLNKkxh3/zrhaCrbEPCKtXXzfdSu5CG6Pl1BQraS8rVxQi8gbWqrjlPn7H51BD/rUeq5Juk4AMR3OcmD5IOLpjCKtIuPDZyj+M23+yLGVoqef5iXmva4JEBGvRTg2+GyDkxQFS618xx2FiSltaY1g3VtCnim3lF1dFgNAvAcncddVk18NDihM7QKL359r+1jsOwrlk/+zhSG/dzq7Z9lUFQK/kp+HJhrLPmu5ppv2UKlrstCeGXFEZh1MRaY1Y0QOClu9i25duSKIcnDUX1umXFp83Tx9Nf9ZNXMhoyh7JhcJ1GoXZm4FdByTCQxPvEmbCN1dWGSOk0rEt97WwWOmmKgUMLKdaD1tZ0Gt27BVzdC0A4uM6Y+irxBX32dVCmDV544WxDzsALDHdnWWct7xfoJuvm4NgN6dK0QhZD7NwJ7vYCmaM3Jxrk95IAfGEjjSxAAAGVUEAf6a8EmoQWyZTAjn/HBi9Y7RCgVsnuqB+6PKQGsTf2KhragBiGpwQV2/Q56D1QG0SPC83Y1ncbMjdorARp/Ju0zdHterO3vW4l0Xp3jo/7iJw9VRKP/Bwon0OQzJu7KeHo5I21n5Gqy19sF095eIhr+DTLhwqavY0Zz/JaYSEQZZD37u2SYEL1bdcigf++ekaR7374FZalQe6EryJp4C8q/Sc4snbiwFZlVpBOmFrVzKi2UGxYK0JisfrXvwTOmfiHOp2ebuaD4LFjwdRIzqh0MRBYXl2GM9hE9abWV0wb/Rwf/JWgfNxynEepCBv9h1B/vD+Ep4iYZUQIXp+5DYvAQZnobyaHainElLTA4bccpw860ZhoECSc4ZAL3q9HMdrQvbo4NfjhxNEP8moidKjEFg5sET43kz/ZrupSQWIpl5Fo+rX8viJYj76dcwrjp1IZOK5obtysxDCdxOJfNmmwTumn4SIgLWx83eM54mDgv/bNmOHEZ/NCqSW848Dtdb4/Z6D8eHbWARH6Ecza4SWKWGKeRL02B1MPzZfIT0WSdLyZEsA58miRCB+6VmSqpisiwLDKxGfJLXfe7NlYL4HE9F3AuwkPhC/m7mKZRUm9AkSmq3n7R2UT4mvgS/AmAIM3jCSNOIk+KqQLoX8L/YKVVzRCb177SZfHGZ+Nw+6Cw+6hPR6uiHF0fJmsF+PhkMVBUJL6Tquz2GHsLv5zzTVaTtaIi7mL47aUKzrtBlPmsIHmUT8TpV4c4wkHAQYMQ3xepFhhauatsm/+xfcskhuK/Tqv/eDNvDxNfumtD+oiX2RkEcaoVCzrCSl4XXtXKFJN9YKsda/KaNRVphT+PiIA60JWsMcsidPKkiWaPvbOgBxCtF3oZ5onf8pCuNFnoSur7dG0OD3RY0qbCT0YC8NaxndATL3T8J16U9suxsLXAZegmvV5Nv+MzT1UU+bUqqYUkLkAjCQUk1FkmkV4I+seFCQ2LxmjKhKcv3FxtBMeLFwmvX9FV5+o1gscfY498ODQc5fexXwTDmY/yfkPVr09Y6Gwc9OI0YJuq4fnxgV+77bmiCmIr5E4HdJeyD8YnnQIg2jRmdBgM6WyAGMLiMmma4pvi56ZkvJBgGGzdtf756l4ZMtNcY9WSQ8a6+fHNDPT/KpE5q+8VYWC8XFZOh5z8ZroJfQOszprFpSYuM8E16YvXG472ouKEhWN1WJGPUVJtXRCvdWTW2zUeXfodFP+/uQ0EOyP0riB5y9xuYCpYowdD4tOruWJZdZx3xdrsblqP2sdpflwf6K/4apuQPm/K7RN561OKgkcd2EN/0uaZ+aNqoPUrvWO41nse1iq0V78irsu8+CrRI5Qo865cz2zYXO/oxL4BHwR/HYdBoJkWhiWAnLrBS/2aKJPatMOVNQja/yk3D5Yqhkwzyy9Kfga7q63lwbgW58YZhgo4yFuVoiCpAIzKWqrpw5Qh+nfio3ZOYpczWA1Q40dUjRghFCD2qPx6RFooPy3EoVJq6PGbJwC95J8guZMIJsrx1JXIUeBpZI3ngpdJJdr8DBfIJKGwS0xtEyYrrCsXK/0fjl9d2/82AzrFDnXVImHbLykaFMNUxiflYaN2dHr3e0RQwjQYILAoWYskSayCLSHtKKfRryg5kYjpC8Mp6eonkoO60dqzUqJtNvja7PDQTBu8A6xyMa4zgFU/N8gDy8G+QEScpEPsThtHajjcGonpB3l72cdTnctA4eG2/ECYGvf6i5Ca3HERipMh+zUXkrJPYqurBtdClf7eEQ/xIL/55EZ7nqV2Cwwb8ABSsNf6gxwVpRz4N6D9Lmom0i0jRqRk279+jF/6KEllTqECOteB2Xm8D/ShqHrsEoEm2aqamuVGtkIorQGVw/trtXBe1GDTFPEtAbjrJO6YXctH1ynN0p/E2y5XL4eBF5IgN42oWBLdavLp2PuHJFQ2R37tN6xQVARxTiu1Z4zQV3mnKOOjw4dErbMpkLoomV22oTAnM1LZwBU00zUdUkdJyi7BGoTqH8vVAinNhoByitlbzKoUZXXFq9ZHdb8TuZ1yI5kUU3+Gz5HR7/mon6dz5d0bU5Ta0jAXDCq8EPRvl3ER3Dw76/rz2GTH0Fsdk5RQ1qEoAWc1ZkN5G5y2TPuzU0k0wz+ByDChI6SjEAAAH1QQAtMa8EmoQWyZTAjn8DoGCQAR0cBApCxnofX/1B1XfGZsqFiw8fS+FV0g73QcDd8y1oUDwAARzvIK66mFi6bZuPAey33ABLYzJGTFCTSuWPQn/JpIhii5GXR/AX6rPPCZOv0BRaTi+9SUbuiPnlBwHtiAp3wgNJXSLcdSBTBza5w6Kg2ZFuAe+ChJpi8VxrIgnkOfhP9rnbrJ1KAkQb3RuXCuxA2m3iiTEY5xpfBNUFZFTJUg9Wo6mFt7bFNRR8QcIhOCRSvZtofQZMwvWxsoNOTsLCz1i/TJFzEthdAdK0kabpZ85mS5NWmHpjlFqA7qXWINdnj0uBhEyZ+P1S1B0c5BSHpM46saZQyoIb1prZk9jsBn+t3SWnBrSILNgMcJ2GDqwsHwVX8Po2pPVqjteFzocnWPb6C2YNsdMPLVieuBN4//nvUmntMWLNNTVOzSHmgFJfUyOOiJqQIaZZxk3bTyWM/URH7AsCWd491cPBsm9QGtht6Mue1ilH9GdXiLTE/YDLbLJumkcpFaeLYU7Bjb1rpA5CwRTLby8hBK+tykeNJxIiYzahQDgNqQ/icRsMd73ux/r1BN7YPCnLVYFjN8eSkvgKwsV7t5mBqOkMJDsrrIR/UC7bBC20OjrfcvyxTA/q//m+8dJg0hWXpkvnl4NZAAACW0EAN9GvBJqEFsmUwI5/CUSuDlpdxus98XwuHH1xjnizeNh/4OkP8inqUP2q5tigbLocqdFERNQrbO4oyzKdq2p8g2oDWc3Mb99chs4mCXp/OdHrMG2i9/fqYJvxnFX2BvhAAACcEaJ3dqBUbzeQNZf1yEki2VzCUSTgv/ABqYnDOJE6j4eYgeyOMmCAa3TpDQNTxTljaarpm8THZCJTM1WKMYjGdPwJV/5RxTVYfHzqwLSNyKbvV3zwHZTZy8t0A/hDAaZd0ar58sjMY2iMOYXYT+eYmPJB4u8UPBFlXix190sbS5WjhjUWTLNDltzLBv8wRqElmJM7CCUHa/Ifobn2nn4Wmao+azTI8Ryv0Li8VWGRTZB+fhKKDTA6MRZCZmefvq5FkDFYz3tClU/0b0h+Bn8fotASE61phFv+dM7Ths5Buz2EZF8rd9rIfGk/A/uu6wKWIA06chBKF5YraU7ceOt0KhEfy4xft7CKXSRq5D8mwj735IGro8k7ukAy4lg2W8F++QhRUouqTa2YIANvUlmkNqSzk5zzlRaqv9QNYc1w5PSSjWLYy1onI7d0rvupfEBFkliQfXf1l0FRK1Xfktq2hsVGDlHg1fQRSt9DwtpLjNOl7gSRm3wIwqbaEuUEDn/jtzhB3GT4dcMmNwLDoqFLeOP5qEq71atRDKtUFzs0NzZQNtA8jVxW7NR8DKbtdSUDcnS16SAN+Qxm35Xy9lC6uOxZpciZWAA/zRSwGnIXDaZZWsG1h4SWKODltrX4jhlr/A/iKPxOKQA8Q2Sg/j4y1rFapP7feQAAAqZBABCca8EmoQWyZTAinyzol3jy8AZjPsCWuYUPRqeougFOJ00PYdeJf1sWwlA0i3uHsW/TEW/D9XKd5CLcMXSWQgOd7EMOZuXRqFmFerej9d/4xajg0kSkAAADALhgiuxASFLC4edeDjqVidFnyrp0Ct2u8uW0OPHbZltBQz4Hsu/qQRrHRZoZ92ZTfaxJzAZUQIvQu7dZdOM0/yRYvxTRksJUKGpjpRXOKz3u1lsQYnjqnSdSaUnB3iQrquy4YRFChuXQuFI/IiNxAGEG+2jyjCMqV/Tv1NDxuZAxFLmRDCwZoFZUiTvsskTN9sF/qKjJidz1AVVtu1yI6KpF88QSWlImXOrzrR/t12DVCBx74tLhYFR10S6s2jY0hBCb3OIk7UIlnMl8gwPwDeHnDTy8QSdgVPZp5LvVfZfzs5p6KaKO21/dIQ1Ah8KsBnvTJF0TG+MpwVr2ck3FaYC8XStaGdORk5qm9Is+8RTapzObprLxvfh9jKDT597P4b+RfTE5zDad03dHkaL8QlQqb/WGv+XGhfpNG2u3p1sIxvkw9tbdHLg6NiJNlsdT/x5leNhWLSMe/Z0Hbn9iqjfC+U5ABsa6/Z0nYeNSBMZGlRVPe1UmUQMItiOkcicwFfrdq66HjLpY6Rd0lMNY7PweXe7Qz4S0vUUF7KDD9klw7rEPzuOuK/iq/BbUO0Op7scpMAU+VzN06lcQ+YxW9M+RJv1SexoiY7sdtFP1Y6qmUEdx6rW62zbluwsaGqOKncDKkYld89KjaNw6EjanJzH1+Enplii/sHmPK+aVJ7H6tX0DTW6RGeg3rOmtKHEivPKfUtATLa/POzE1MrnkekbgVnZeTKnmOKF9C3KgE0/Kfti4VoYz23a4DFjwP8AfQGqUz91DCH/zc4EAAAFWQQATRGvBJqEFsmUwIp8s6bz6zjBtIJXtHUQsTpqf3CedW+RObAFgmNW5Cy5pv0fZIESHdNTOFWzrLk4EW105QlDd5hYp/JcMyom3gT0/ofzcTIY9pVxeAAb30NSlI2MFAxIFw08PScYt1R5qi+sAlOZvUE2mhHCnUKuoyptBo5VLcLW9Vj9Xuw2cLn78PZ2huL9560cAKRPBDgYixeiJbOAqe8+mFXVHcXO0JgXHjpsuAzl2HKp+UXaqczQpEmoik6aG9OAQ8yYSdPLimS5jOUhsNzY8LMUvzVGFDcDSh9YbghE/Svykb5SNRxilFeuql2WHiMFkN7xZ0wmt/Hf6Wkr0V/MMUsp5yT4She6TtqF66y0EgDWlFoNZq9JRKVa7D3z98MWGzn77cOELxfyViyZK5DG5OGmEqDVYudlphoSmimrDG/D8FPdfXB8Xc2t+d12oNdRBAAACL0GfDkUVLCz/Hno89IaZsT5ekugH8TVPnMPLbAKVgtGh8sgLTDuPy3U6OGBdhJdjCzcH0+2fsyfxhSkIgAu7IUkMOIVfcUk/XJlzGqR/6ZFP0XVXadx4gd8HqeDAXowL1cwNo4smLkq6MLSftGuvphW70/V4I9KNymH5DCbqEQe3aZ4j/mqL/Cvwpif+FkBVusYH/WRE965P7amU7QrQ2RQkT06KBIMvvpo470Yq5fzWndklnsRYaDooDutadWvLF647W4WUnqBjVz/OsRWbMQvatyH/Z8UVZMTLhmiugEwvtaKHKhrH2mMFpSr443Z4C9qB3QKQ1ndopJcfqks1jwky9fsog/WMG5ZqxC+L7Axm9Y+L23bv+gKL453AEf6svfM+nEjGytvnEgWZo8g7oAN7tGvXD0NoGamvU5C1SNCk41Rl+/6LxcdZC/xGSMruDsEDffKW/a2RmJOCkCB2tiQd0Lup9AKizXbzeQ6ULb7urNqQhav/3k9DEyiZ/N9Bd0lwY/Aies82K3DNsW7fBc1SBOImXdjgRGuOfXFK8UrC3EvoxVvmlXpExv8r4JB24DyTzpv6Wu5NKxuG3YAbFJh9A55uLLk4scMT7++gI/xw4z1KHRcvcwSTBy47/OGr4Y5vbMsZZXU0qX1qye1Ss/Fk/RTpN6NsME1bJcRrkXQo71Y9H+iDaAQFEpypw9WuXYdeCYN9bk1UGDiHH4mBN5VYDuccd8HkMm8N2vGes2EAAAGqQQCqnw5FFSws/z2WugjOlgmPLxYxjHx6NzPuNgyf5RD0fwQ7sLPzABGu7AApRgy/gUCGsTNPctUJHVkx9STTTYqjzbgembtNcLIyeXPDTr9l80NogM3yQFKn1bGfhE6TmNH+4Qz2MMEHJ9ObbyvNXsj+JriVvhkb6Kw4f0SxFuJZ3WfTU81Vk8Mg+9UeSDqoGDanTqKXApE/novYGGYO38hvuXXp2NARtMPIYpZJGcIQj4hrDBpysVUXOB+Bnfn8YTUtWpINvouBxJ/Ov5g34QQqulD9SjispHyk/97jMgOU448EorRffzOvW4Q6FC67VGW/TtRC5apMMC96Wotv+v4jPav8915MN50dNenuLhaAGYStXoJk7kFC5LkN6BnwRwbuFxJG2esRVJYLa+2OGniDgSdiNxFwmftvXmhKNq9xSFvP1p5lKK4L9NZkyBjuvX094xQLJm9fH+DtPnQSlttZIR/RBmzyK1j0xxovVAZbDonZd+4ZXpH7QUIcs292i5itS0RCkZr1/jd9LWZVfGzRx6p9OlIxGaucJDJEDijxsYdyAhyzOlERAAAB/kEAVSfDkUVLCj88JaNvRuBd0EBcNu3U3Ud19pgCs3gxhRJG0BIrHxaRQEZ9udSRKPlXPw80AkaEPp10At6Sa+s08HDdSxeEMjwCM9jpGyQDOYw4VRF15IIgroSMHsQX7O/SBulXDhM02XEmaJb69iDCGNg17AeC/FERYuO/VhQaFomUNfN5f8IibkEkKsyHR8+4iQrG+qH/sMxvIYDC7iMduTWAWyNSG7n6VpWHq76i0fJ/RDGALwJRpTGHrrDk6abMnAjD2WsTtLf/o3riUdvqpYwZ1s90CsQEUUghd6j0LSowBstsq7f7bb77SXP+4aExUZ3Wkt+5RmOtzG4BspAk1+Nk1qsVYAU7Fjm9PW3By5m5Uh5MKhfTNH3svWnUcP/wGKptvv2oeFqbl3Jx/u0+NKrfRtUsfxaklTEblqHkSIUbFNPAkY2qU+OGwAo7N5E7dHgm0VGgu2+8IcQwgAOZ8AfPieWralcp4YsGtXWre6V/mjhyj9MjhhZu3ieCasbWMZif0FmML5GpokBnxFielqlYx0868cPRvq5Poyt3bapSi+8otu7/PZySL8Anv9zytN3do8zvWhTdzp6K7OVRyUXto77JkHF5e3xs7tK6Odh2y/JR2W/hJoXUOKFPd1U/m83M3dJaUIHwpg+BIS1ogMo2ExFsYhhveJw0qwAABBFBAH+nw5FFSwo/YQ8UAWOMLpEOCoTokvYEa1mh+ZHGNrIQvCCGwJrxkycL69Z6czqEFbEduGeeFlcAAfujPs+xGvQ/aJnWIWQx10yUvc7nyPgACdcaxcuZqk38jMoW2e19Z/JOVGIwFTWrLtErpGmBm7XeYZDbtqcENjuPQqh+zIXiMGKR4NnLkm0XnM/TpL7yyKeJ4B07j3giawZWthc3JC3bciR2l7GGfBIiK/nXGU61tcwRNu2X8Ao+tf9WIxCIHWkrOlb+J5nCFZLPFHH10ItR4d8JN1kbuq+s4j+ndqxwdYBQVOkDj7+XOxv99/aMR6QNfaMxHOjJoypUrqE4x8/31k+z86/v+LC54/SQC0THld4G/Sss7hx9KtWJebXvAOtc7pAPBC80B3W5HJX8MGT+3134vLhs8htZd4V6g36sUcN+BmhJn8uvcpjZnqpFGh9wRtlnx2G1z1VOAZJceVszbJ98M0/7y0xOLR13ny+z3qmQsvYCIvrWDqQYLJcO4iUCdz/SaVTxqyT8LNbulYT7UXOnlgnF72MXBnQ7iC2OAhGEfSUDCVRv8ufBs7y9TJNldsh3b3mQnxfmc30NbhIIg0j3IcBabQn2e8M0BZep5MKtCTFM0qt4CKmcsmUcVTfLoo02USOEeWbKbUkJ4zIUcPZKjwjL4zar/5XYp4fgIhUfE5TjDtmPbyb6yEmzBJRM0NFYyNtZufGxgIsM6BmnEZTjdrvT0CwtHPx7CWsBGj4VOKue8O5sNLSB9dxbEZPrcRtJlOaoforydBggZ/u5wRqHX0qzV9w3k6yr4xm2JmmsspDaD8wHHSwjDvxQjDf0O6cygNbhHKRzL+aqB5UTMHBafqzbddAporwtqwLrExqZMAhsDk68WzdAgn52CMmD9yQnxLURkszZFP8OsUjYjsXM+dwCRGjc05jR7M2eUelUe9QsCn672bMxYpqBl46+eIk+bQrdjkwIfirMrnafaag/qo89LFN24bnvz+/4e8wXq2vQvd0Uo9Vk6EhPUXWILPRq3DDKNfrPmgLsOns4GKafMYNke+aQdPwPbeT9wYaJZTc+gCtOZH6tH94P72ETvPU+KArvRMR74xtA8jqQA4cHry+ZYgsxjo3iZmzasq8V7BXJ8ihq/Jflz2lZXNGPAOgGLxpByqZbVlT6PvoeWKGtPV+UFRwkhEVMagpCA78oO2TTJFGHi0fn0Voww24hRLcs8MCWtIqKCgHw3gPl05gb/p3vd5ElGLvve3hq/EyVmduAmPeNAIQddm1jYS3XBoNJ2tUE3c+64sNBBPcTk+E2pD9FEBvHuYymsv3jYNF7M75tdhKUuFI03XQBdi+qvPe9g7WMftzjd4TZ+Zjg8QAvjmHigztkgtD7sBsAAAFKQQAtMfDkUVLCj9iWy5Hba+TIrBskTyjUX0SH07NgIukjmrSK0l4BAf+vDYbyNpvocYG4QX9MCZ3h/RsGgNeNbLVNhdNmfXA07+TldsdO9yhoz4r7Zb8Ex+CRQBlHE0wy0zdOaoGdODK7jtTVL7lrOamxNAM+zNlf6xZDVTi5kDHlTRS+szlo64Jco4whu1aSjy9UjQK6Feqv/LKbtZSjhuL9FGZm1g0fpmUYY1w3tw23Wjos70tQDgv8pEvw6CrQfQReMi6XLoMlS3IIWDptFpIJUZhjSzOS0nfEuf6Lq3BWvz3P/q4RUphrmvtMiz2UGXi1J3bxgd21Rv26cDhkU94n63QrUKxnuF5yPuroE0e79MVNRt/hM8oo6f4mbA/U6PhD402SLUuq5urayGA5d2KxKC+8z+n99j3gvBVX7aqrAX0Ah3sbObOjAAABDkEAN9Hw5FFSwo+Hg5ODSXliv9pLCiMtmhs3cVp7qdGArOOPhST3+A385sMuy+Fx08dbweMAIaEKYR1QHOZvYlrn3wjWGsZv69wP0HngVB+8V+E4MCTBWyoPIOB6UDKyind8ycqAdBAI63EJIlNzzjXcOdxzLSmMMLnTyrq+ws8xhLzOS8PXCJ/Pq1WdrPzJPH7mpkNaEpEr4hwz4P6UJ+CUDoEDYiUcBFcMXLGsCMFLGHPEOlSx3CH+6RLV4O+OsFPGV1TWcb4HTSyM9yVMfI+oWmD/rK/0KDnGfFS+mCos4sSKRW7CbYyy/xh5cRNGfmgNs1OOhJv88J+5n1m3wOBHA+MGYczjUvuuWzP40wAAAOlBABCcfDkUVLCj/5l0RzGwSqUVEl9ws1iX9ctA/Tsib2lkujx7v45Js8itPVmIfKlXIf6UsIj9HTNBuHC0cPp7R0Tl+EXk/0g1rHbfKyHnmwVwGRXaUEbzOtoo/xRT3lxG2iN+Ldq4DGztLOGGduSgmC9mNglh0WZK+g8ojXIzvGAQKPrBKH8GMq0Chsul/B3iCSSYfiFMjdvgeP+kzvD4RVyOoOFhqW8hLucE9MSCliOuiMO9SSfWLJ0rTKZNB58kH40T5Mt38kL4jLA+X9lgZBUF+Q5At5nu3v/QucztNI1ByZAcFPexSwAAAI1BABNEfDkUVLCj/2l/I2i9vAElNTmGxMhleeaDwK9HfkvG09HT+PwnQDqIe8ACIB8jsgCZ34ZQWB/F78y6MZc2hX3lXS8Tt/Z0gQYi8rFk+jGssi7hak31Scqki2wJXN1ge0Y8m7QTMiQiSBbM3+RF8yK5tA5Fx0qJji23nTDQTMTNtuEXTG1EoOiQIqEAAAGFAZ8tdEOPI9n8RYaDmVeuYzVVGRZee9FqrEn0rACDQ+OkXAVuQTeOeYqxMjwBSrwZ7u86cpsFYXGnklrKorToCh8j9dcT0mACeD1fs9z9vm6HI/68CkU0zQ3DahU5huFdFsP5r4zupffkFAAYqjVdSBk/A1H8XWL469/MOEa17dVbc6+t0s3DYOyscSpJqvDAoI6JJXYlCWnDaCH9/ieBWvDsuWFNbTiqfhJq30c8nfeE5jwAy8rXl95lTBF1ww/8KKQLFcaRsWwAWq3ty/GcMONSrM57WRVe6ScozSefjap2HiAf8PmmDH/69LHDO+dnRCm/SrqwwWVqQQdLRWCMRk1HzTH5AdJJShsC42WGNfsEecxiyXDohVSaTSQvnUMByyIxmtr30rbeBvqjYMmFPQ63LiO72yOfD/l0G1FaCnK2g9waZ/lpGVXQJLXZOAG0X03OuemQRw78Hzt5B0SzXYI+SE5yq4YWm/RKhhlBEGuBXJSSCHTIRerzwFpDUpAYXFwjY0kAAAEmAQCqny10Q09EAzOk7izI0FmhSV+tt01PymDEdn2OgSlPjH9ZvLZEIjYW9D2RqrdVJLZw2JgSR8zR/wbrfcp+7p/JRaioPqPCgUPziD+pHsm4dN4/BMR+cXV1jDjWksxIBplxChXOp2okMBf4qFgGa0QDvYcqZiZqJj3J4VHsRwq8/PfVc1PtFZO14XbyToz169q29c8EFmveYS/UZ2+VxigusDzw+dFEapcRWXMB/0Cquhu6RdSRD9YglTiYMMVZhDrK/7iFOsAq2w9yuqOE2RueiyhCT6wJkUQoJpGv58Pexrb3ezz8asAJLpiLjweaWS/VxGh3vwbywBHy3EiYPK2BZX1UPVNrUV8sAWOs8Z9BalklvCUJMI+4ID8eTnfnLVYRioxnAAAA4wEAVSfLXRDT/0f7bjjbly4t/TTTec5Xo0APabiId5v0kPgMRNBLXhD3ZeWM1NPJuC73dsDewgU0U3O1tqPwnJqjtHo+9P76L4MI6Jvfdi4SLozxTWzwq1l2Lq3tbGTWDJSkXvH4qBp/AQ7MKh3IhFNEtIreg3CO6cGB/uhPNXLpSrRNzpEwdANDBSCOdvGDFSvQQJenJIvou0rdx2gl7HIOjcnMFcO+fpeloNesKn+UtQKqpksGuWQMqlgqJIhUu/6kKbd/6B5VEH9o6vuOXQYqdwjT2m86TrhlZtirvxgBlE4xAAABWQEAf6fLXRDT/0sh82Kv2ASX/9S9w0Fk+LGKlUpKrGjFTa7PJCUQASpbqIjgpoE/iohOrMuwYTI9pFz8dryeZdGrGfoMlvujRZANUbRwLwbWLjnIp7uA1KJpC2B/Vms3MPDvg+rb03j8A2Z8DgdcHW+zkx8pnAFbhzmbp9AiG+JgZ7V3iez9UPePhc0idpmXEXxtQ21KTVxf0u5KmOrlEgx9zgULBkL0g3vWKlIiKtFRVLSPk3yEZyd37qmmY5VDbK2yFaORZr5RU3cqoJOdRrbZJeowChVhxXf4HO5w+CC9/tTSnQFdoMPyFmbrF/QxWOWZC5RBpUQHbP5hhdh6SrFJozdTyWIXvcNQ6XsVl34jy9dDGfbYDe6RQpZIpEDbAYDXwR/g/eDw56fj92wF6K3m08msyDE45KFvvW4s57s/ImSG6mLzGDgFeXivw7Bq0mRElxFOmHP/gQAAAMcBAC0x8tdENP9F8RhlCK9JsZ5rDkGlyODD5BsxSkZsZtQEo7YIuOMtycXuaS2HKtUDDeCr8V1X1WG467d6atDT3dcxiORoPUlee38OVpInycSGrWZ7tdUwCGPqVMpKh4PxgqzlBs7s/mqRTK08xnsRybjMi2sTOjVJbmPecX7SH4Ok4GxxEz4bA9/QrdbLrxZK9rjB/ZLfE7T8eoyIqQ8zxi2MdOmeHvo5jepummQketQHmyzTOnpmTapEiuvfQQN0Dq4VLvNBAAAApAEAN9Hy10Q0/5KP4YKcL45ZvhkM8X3YoPc5fCD9QGjn9noEO5Q1ASNcISiKZsA7WGUB7/Rf0lcWlZJj8+RbVBaY4Hx3KdMp2YgD/dU4xVDyBGrrTPLtc/CP6LL1StwUqMbvv3941nBauElptyqXKPWW8PYCBJzR1i1/gLdVuG2zB+yoB5c0iC1kenifJGG2jXScwRHYgMKu7xT2xDKxiliyhCWnAAAAqQEAEJx8tdENP6SRYYMfWX3+AHOptih7wmDISbKyDLpCYA/8kfOxSF9gkMEnkdwAdFF3QEfkOt5DpcE70umovesLhZUnzNeeHzz0jmnnJQE6rphXfK4fafwJFxrf9/cnayE4zMp33bHhxoDSmT/7tE9e4TeLeB6SRDfdSZ0oj2/17HcRyam7CkTdmUNIjWj82JREA9SPbNHWpl6HuNK4bUZyPWTTHlX4w6EAAABNAQATRHy10Q0/KQP7tzuZlGjoOUXf2fY/VjrJKdA2UscrBu6YcavzHih8v//Q6RCHrxJjDJ9/uym3lXABy+DDA2lNvZJbfiaOpYwAHpEAAAE2AZ8vakMPH9lseec7hrSOzi0ShBkHNUAqhMBbxsv9RFBoo8GDbmG6ecZuQvUOhYUBSbhzlAeYTXmvI6CzisVUhmYCtVD2ckIq4REo6FPreE8dtG/GHVzKw1NNUAjXT0O8nVYcUsN7EHLsDx+wsnt1mVUJuhRLU1uF3E6Tv4eKCAjJJKohAcZ+3A9spwNs9WHdpbNXNAei/9bkXvwTGnho15EodW+3jtuamh/hKTFvGSBd6Jn3CnlqWnTRLIlMb0MFH+IA9rN6yshCVlFPqQ5SedtF3c15/NpQq22WMA4y9FlqXO3uJKPL0cUAfg9qOYd8/zmVx2AfriTgnBDLuMGmbyuhentOYGiFGyyTSs8sepQGRhZY7GPi9GlMQuA+fuVs/Y+SlCNygtdnmff6UtBDDscvjrPnXwAAAN0BAKqfL2pDD0Cs/FjiaMkfS6RtGXkEXvZgCRupttVDi8guXuoKkv7zXMIf4r38jVj8WrSqR8e1mOFacIXZf8DO26eTcz3q1YLILl7YPE2+W6QYmPjnwU71vrTTvz/dYlEL5T0enfnqL7Ecz5D8IuRC4HQ3L0lJ8lcHxTgCwQAbn6QIuDDKQkfrpflMDdm5HTGdipzwjQnELHTFfEJ13TWRI3+LB0pt963eQOrooyintVKRwAqy2T1uSQyDzIYQH49l7Tbogo5FKfPYk3O8XYzo7fq5EQ+VgNS2pHZ4hAAAAREBAFUny9qQw/9D6/iFSZHoAdMhEi1SRKdntWdKX/mANkS+xwYgrruYnE2QXVJJ3eLVxZ1sATtPj67rBkpxZShH7KHxNRFZRd4F1Uj2Ud07Ycitwx0w6RNVPywm7GkXbEfsJVp1g7Vx67nUykXeVsBOn0GPQwwVVomhRmDxZyny6Gk17Ffg4x2KQFyHqFRDbshaHMuCE5HcLMWWNxNjMSoxxyLNVlXLSY96DoA9Kb5c0SKWEe0OSbKFFcrb9BYnwZtKmQhaX0scApiIBvIe+UVBxy2tNHJCOCMHi8A+1cED4Fwc6PmWMEbFAvR4Gs2D+yZBiohB2gjUtJZUnLCI6FQV70kpeLjssn8nUodwRS71Y6sAAAH1AQB/p8vakMP/RMjPOdnjdc/7bRDWetxNDOCMh9jm9AN2HgRhpqm03ta+tLof0VvDmYBqBJGf+4+uZADYdZGzEZTqrBhreuWoJ9VOK6NUaJrqfyHkut//ufIKMe+1cE2lt04+NAJK6E+hBqAacEXOIbi7qRFE9bXNOU35iKGW2fh0BwJ33Kl3Gy/ELbAxshWUCXHXloE7HWg8Bsx9yxKOPVTwjwq/AEizWEu8eMhGLDC2SvZs9Bi3S0G36zLfvLCdeXhNFeN34NkERjYx57zSYEmgWmqMnpXVfbl2V+Y4fPLPoWAsZCLCaNlizchEe0xXjYOUzY0svmsM3BBPcVkAOiWngNn3HFj2YeWXH2lL82HdRTp8X+xu8rZy0mP/96ROz82ZKlJbsjRV13KXDLMfLtPwyP/c50UK9rqqpnR1nq44RYpmy+id5GZLjuFOlxr4qQNCnO3D8GOel44trhdTSFfiSNUUlOgIM6wZecXJIFOX+0ZktzjqozktonsAiY5F/U+DPV6gzTK26ULdjSRX+/C6jcaxeF//XL+nQiYF6IDCQCMl+lKiWTnGmmCLECZYIcLSZHJ7Tp9uHf4cZAY40K42DTwIcK0GK1dn3Ggiq46R5mGz5nRzYDsv8cDk6wH8ni4cHJj4B+IPpp2eRAaympyr7oqQAAAA4AEALTHy9qQw/0S1l61ftCkg7YZOnIdF8eA0gYcUBwrMxTBBF2xEgziYmhGmguU4nYxxR2goMfPYoJpDxKO8o2HmmaVE3NVKET8q6RPaJv3prtcRGVkMNF9VNCFm161D8K6W9AqOi+L/uYvyp0nnBPAsYro98nRAeirySrfOXm22A4i+nXbyR/CI5oOo507Un7wBsM0goo4Xfl369jg/TTKx2+FWZiMIr9ycm6ZPSfEgWbSI7gjjENzp1Jmkf52Rqo0KptsKp+UT2DX6SYp8hHKKUXLxGtWkpnGbUZeNOEFgAAAA1wEAN9Hy9qQw/5ECNoNPkd2gvWaUAC+V/dBFZkpxm4LuWmtQHxVZeclxG3aD+ehE3Fn1YNEFcLGwQiPBC0i2b+Uwyzowp/pOeIIvQ17iu/VEehMJ81ISHvMM/O8bXbKVih9BQYZSF72NRvdkNybA4TcqdVVltryhxmxpVCuknPTTsvAGhD2infArg0RzMo4D2PlrXkkzBgIqbRUHDiLo/qWGkxlqgDguPPG2hU5xciF8Zl+CiEJyo9p3Mop31Y8quUxXm2F087+rF7Rd8c0KY6P5IBKPSysgAAAAzwEAEJx8vakMP0SrVkVwcfwgJgDGBOJxPUSCnS47Ypwl8EBnWWfG9u0we1K53jzpj03Dr2qNfPOg3E1EMb4QMcAKEtAspY9LwvqzWKPpUGPCsX68Gh+lVhChN5XW6xELIBmfnF986EDjKUJD03mr4R5WzfUN9uqVeN3SyQNA3D2ngi0V1C3Ltw2UIm8XQfX736xWJvWKl6mVIcPen06DpSWI8UgVTdBOV8xurokLIsxpkr2z7qv6B5f2J0tYAswtreafS2NPn3ATI6QiXIQuRQAAAJMBABNEfL2pDD9X+HJsX5ZwAuorbSdcaDKRTHhx6PLUUF33ct9YcgktTgTMyBCQudSO/oBfDwvA+ZQppLziWtXixm1PSBTGdEQiyqCPIQds5+h3dZnEZYlEQ9qFqqxy1tzSjcK/GJK6Shg7GByXnBunpJ72DvZTP6gldSxGU3Wwvo8E1Mmhnq7NWqJCGe3kKV4AUkAAAAP/QZs0SahBbJlMCOf/AaaMtp6OuhZhTvamznvH1aznpHUNuByRQKO0FAjkRX1hZcMxZGkzW3A00Q90g14FSeNkuKb21k3618VX9+MkXmpmJY+9tsW0JIagCx7c4YwD4GWGAkD7Jeh67MsaIthDemKky+atdwZ+YbKYu0uK+ofp7/pTKfDD2LGPSsc4QD6TYW00Gu00L/xuae6b/UhSriXlUgmU8HHPF+i0TvBdC1P7QY4Cc0TUpQkqy2s235QUKQ2QIR//kNX++hdhmcc7kaQ85D4XKewRWUkKvb70o+JqPAPSeCDLI6vK1Tpayvnlgi1NufoNJtSHcv9eHAFR60gqLdH0pdP4j243v26nWg0qV3bGRS1UyfETtwjW77I4ERHKQlA4aL3bOF8KIM54Q5GX8S7XVKI4TdYcInQzUhnd9d/0K1GeePTN2R4ImZdUC2ZpggLEXXWL/bFtkfwutq4ZUAMujx61wt9+tNBPML6XikYz/svkyLUUy+53CkyWw5Xpem2WY5LfddX9QYEWEqW5icW/4Nc3/klLT6jDOE3TUSVaOCPv6mCCeIafb6ZIYZERn/c+jtt83QvCyBWEarlFmXJzDmDqi8mxBslJfCrJeYn50YFoYypnE2oGrX+ZiBuViogI1rgESYAmISIndA4yz9cS7V0eu4ODDHoP4eiO4iHnJeM5anPZZPRr25Ye3re/cZghZR1p2u/DbI5WGa7vBY/tI6zgiAAY+kFjSb2byrzwu7ImyjZzJj+A5QS0VkPEFj89z9rMDBgbsB57EpRwbzwvvVFnDRv4CiF2Wpks1aUBC+AZgg3wwIU0dK/fDZlEdfIz1TVqvNEjjqGAinOh5BuN5w33Y05eycTFO59PaGkx7j7PQ4lJLcgnU1W5nzpfI4b2XI9oQAjI29byzPqAswIkCN8WkTO+tZisB66GaMqorC7CuHPZYM7jz83bgh+0aYixYX85CVBl6+MIL5kDKCOu79AjRJcTl6jnSjKuXDzdPaFHPgoWp5g9nb42UmVeu4jESEtzCtgKX7p6ICylbULSoacpwNBwBtgy2SNd9EvjHQXxLLzz3TmTLwQhNKuRss+4P2jzj2NyU9Ydy4oNBraxWA8vrAnG6H7Rua4bVfMvaJRfWhirejDHvNYgm3CkM7utyZdLtwdJLXckCzycxTNEbur5ahwK6sjodwzvZY4535tKbPQ160h1qJxhaNmyCJkoN3NsCo1SuL3M/ZSdbBCBPRn2B/z6qqW+G1CJM2X7GIVdbNqrTG61agnfhKidFPA+FAKbqeIXbysnJF2RjbEiNpOPOjvDglXDDVTg6cDZwBsQRil39F7fa942zy2q09ZzeE23cKPRcTixp1/gAAADtUEAqps0SahBbJlMCOf/A2o9071Ka+ZB/N1S9mboKbcXjKaI9Uwvbx6tp4a2QU6GjcB5CfhscrI/hdj5G/nDrKczjm3h0Z5sXu8pvQEsdHrCTFZ2OoK4ys9DZqKJmgVOIp+gwieVDstF2yVnYLlGAuZSw7HiBmfj/5fTMz2otLA/nheNoe2MdlkSpWsuYEFWaeQegBSW55zvkAVcX4ew8wz8YqNL1OM4zfPycMMfw0dg0i/dM555aMT/lvMINQLjmPgS+SXiZumPopMgo1xEtu7GPvvgyX6z77261Sos79EbiQMxo5+VsM8RvRisV6r8JA+rAXYpkwd3sIgSpNKVvpIri06FuHuffvbfoFthHA8S3HWbFRxp1x2IBf8ZTN5SRzDBOKbxvl76IMXp9e9b+VfQiMXKSR94H69OGrvm9w1C52Z7sqUCCSZGjvyGGb9QwYnuCNH3OKxUi87Csdw40YhE/rgNEYyB8m0G2FHAolJL7EatA0afKuUN1FINCc72Oq3GNk+Lr0A1OHpabjPiNKCndKaegEub7S/EMgJWiUotaOI+5iQkkJO2LP6wrBZpWKQBbhFaeJeuYT4vIZ7rlWzpZs4cTacjUNlRGLA2t1bHU4+hoNj0gEqmqequ1Tqf3lu3DlNN+byGel6+V6d31OlsYed1knlZw183P3Ro2NCb7nE3kvnPzdAoFZ0S8y55YnbyZPkzXY9Dz7IEJ5KEhqf3WfGQMQZBTchBiRdDt0aVFdHrs6hlGas1d8aVAq+6xDR3LTm/fjIYnpG0osVBJwj4LYFx9jFKGbDX+z3xhFPXNNGLzKq9a48ymwnv4bqMjvn6m/79+Zyvv4yKTpPglcbP5H/aHtedrNvogP6nBieDY6sM9xAKVIiTCBhjr+ig+uQUOq1zB6Frnh1As1BHXIUeD3e500xMVPzUQBNhTq1Euu5qQHmG/6wvUWZ1m+MhAjSFFqIa484LqY/3FqPLcCD6REqrI61PF9zm2phUgL6h/U3eXgxkyMTjT8XI1MXDkc3cTcGjxNhSp04YP7RYa5jKtFaJB4m77uSTSDQymyOHeZ5r3sKVkiBYOG0BE8TeL5RVq0aGKrOmFN65j9cjzi5uCQLFuZRLuk/NKTrLCA+XYf6JiD71Fgmxcsp0mcveAdQ6dtCPStH+7t9ft6ZjISW1kX6OxAUsJj0+8HmwLbbim5JwZKb3oBOcD5FJheTgGDBz3U1lNV/iUkPXjJ+z6AEJtSc945D1lCSkDNog+kuiPGbbasAAAANqQQBVJs0SahBbJlMCEH8Ehyp8LvhV/XeFhWL7aM7XIufwVs5BuAqX0hVl20Ca/ta5aGCdzySSL93x3GQ45YRSEtrs/Awm4riGxF5P54nQW2C8j5cQ4eavNvcjw3PL9EGOR6UeAAA0cOHElQOOiQWGuvyzt5d0XJGjZIX6kbmJQp+0DUk1sNao9NwAbmDOHTwmBG5c8rDoZ/yFDmagak38p0MP9z9VTmhMbFfioL60UKaaecd2S9e7kzmmE5eCZ7AlauFr12/neP1J2hArUfGq+fqmyNsYHYkT+cv+bYplSJSt744dseYUq8soNvKQI1wjKIZvhXNYtURHfEkd0GLUAOgTJShYgXA4hh5Cm304Yzx6GF96MiNBYPJSRTYHTWt8UvpLFT5t65S7BfTkKVPzeJjiICskyG3WeZzTgTlwa5nICA3UMNIFB9yDMdEyzFGK0NcJ8Prvl5NLmZL14bqvyaFQM+mhJ5EugdBbLm/9HG/KtqYFAgXlDjNy3qh6ryvnCZykWfLRUGR3lm0lsEKY2aPzNuYdzxIeGYiPzM5n0fLqWoAyb+J8YayEmVfKV0V8qlCT3Gd91NfOYy6trtwYHafp480dmGHxvv8Y9VSlOfNLUDxbJa5z/ZxVKABJmVL1fUdtPAF18C4m6288ckIZu5xhann3GRhi0ldLEC66IItvnBCgGXLDZXB89rIcz4upu/i+Hp0XbI662Jy6FF19ntiR2AaqHr9IzvGgdR9/bg+XGGRo+fV4CkiqOkIf4wpjQiEhMmBK6UVORiRzRz304V5mT1y7YvTpP0YgxAoadg1aj+j2roJKICB5VpN29NIfo+lIgAW1X3TThndE7WWanRj2SVxzOYPyCwTNm1P4Wt3on+e05qibvQ8xo1fvm11T+LV1E0aWP7t1JnpauN+lUN/Fp9I1Zxrx0COkKgHrCGzF6EN1FMNeGvc21Pbmz3OL8TSLXP1A83mPBhVSWsN0y45L8KnWwG5AAbKPO6NPPicp/uhr+FsV9TRYkWxgFiEADmptrWE7AEFBIkCUk5bTIYVLMr8chi6GyZCTmBshOmztaJHIO0pqpIe3Smi/0SkGU8hSjIyHIJtN46OaAgFFWyriHUWihwNHdmmAjG5p60FdUKwk9CA0QVB92jRQhqp5mKw7L8GSgLnMGAAABM1BAH+mzRJqEFsmUwIQfw0UQnSCm/YMqwuGFjOJk2Kv+UddkpzbaKeZPtXHksyrRc5m+HZTMuVJA19ndKcfsdKOIstPBmOt5WxB8LAjhKpq/OwKaNbn2mUDaeqP6/5yk1fNwiz9LmwcCjYk8AO8l7e57FV0a3+ngfOccs7WJ4sFAJmTk4Hwb1aAcLiztWaHxkEmztHGOJLU8FNMECQlbbfA+3oTPx7JYAAFlD0bzpY6Y7t5dkkEd8a0w/5VP6RpDLuB9ydWreU0KcKpTnBgSA3t+t8pFkjRNj9oLfb+12MzKjqV2+z+mOqplfG8c+6bXozjOdMdeun35/qcb8ecY9O2UrGECZ3FeGf07JM+aMm4AWpv0jdBWIlTLDNBID3diEvfV8O4cZrioolpkvxOAA/5Oj0Kl0yAfCeZdmoKi9CWgo/f42++nWBdukq/PJcE7OVCRuGoCM9PYsuC1hGJoQ2Z7UplMmi/+FJeQHmcSFiXK4vK3DJh/XT2z/T5qiqkTg5q06XZAnavTCkgmvgJDdgUUfrflEQ7xOHRNTUcQQIjUQtuM0SU5mbAVVinaa4TYbvkaHK4DrFVTntA0np5dc45A+W5rDlMYMShzeRzkmSjib28aSPwe83uf+LTx1X2ehxBOSIg5HyVCpv0awssHs1JcwaKfN4dR1VjiT8CcAdgh01r1c1wvYPjUnzYp8VcuPbxecn+oK1fsKuzAVvsKBwxoh/0T0Kl3FwWl3vXzAbSwgnapY+ayH9wAvVQw0I0EQsnZFsN2BZqX3Mb2fTYfTDNINHnXHhigXaKkKNzTsjQgh//u/D2GR32dZ+8f3QtyD5Ez2DYNZjUjMItDMzx/J4Ts1qB/ELnwAHJBEnHhb+DyrIXW0MZqnx5uprmgRUWbaMhCUbhIDuxA10xhv1O1e++cGdywClTWIfwdbAfRdzJyJPzstwCBWSO0t+k2gC60nDpcZKekfM2J/MdAHHUDdHbuxCOfcwXKDzbRGFsnJ/Ru3NKz8oeKKj5UYLI0baZGpZboKoo+GOf6I+gQlfae4dmSPkLoOivFCo+3LxWFSqCNE3nUmLYwPBnqrrxlQNqaHPXE3Rm9QAgrqAgy6QxrUTsswA0fxRFZD6S23qdTQagkl247DBRTotKAUTcUvsenZasrHWBPHe0GncSBBz+rxLGj0fFSZETyp9QJrDRe0jT4c0HADUH+o3L6pVGzA5BRwaNeG5GE810JjNVmnTNJYED+TyDVIoR6RJPo4XwquOV50A3MP6NBPoB15FdrbTZYd4O6SmqNRZKSW12vS2JvGO+/IIy7S++kxa1/1uOElX+biYerMw4g5uR8xxBJm0kxoAD5bQZ928xCNXnrL0dGRFaAlkveWoiYp8vIGHfpkCB0RoCSADExvsAR/UBPnLQMg3kOJ/gKvwus90Z+xU9/k6iLXTfOnGOkFKPgrHu+JL6MfHaidngxpsHQZ80smPCUl+UPY+s7+85J9TxQgScqPb41l4eUtCpd6ZUHOkoKi39IlBjdoVuX8CY7bdmgbY9PaYeGL+Sgwy9yHHPC4NidHdLpd/jfFUC8ao75VXHPDd6Bh9ubgFoqbaFMnTtnVgvgw4qqZrraq7J4wGR9Q1+cgfQgwbD4ckKo/gyWCXt4AAAAiZBAC0xs0SahBbJlMCEHwSspiAlaY5VYJEs2cb0TWd2fWY/BVkg74Nb3RH0iaH8B3eAYNXGxRO67uqbw/0G4+QV5zNHwk81u4lxUIf1zFqoB8Ukvx3mLhusrxIWUdzOgzwWTiGaBudV/6SfHkpNv/7vjLhMTOOKuQgh8d8PCeQZm3ZQSUs2b5kMjUmF2jPByntbuJC5CoQ7MCv7BivdonJHFOgAjTDpAivAq9jKl9nnadArYMoO2VXJevk9zX0EdjXT4/V84bvLQbC2fkxg1AS2c+7A+z2XQa/dW6CM6BFEvhgte8BEglnXB4ctLUr1bTIdDXNALwP8jPiZXOYK63C4rvxDZeqr6Pdq1SNBRLf5pSOLR98g1yqz3cO2yA4yY0nDLs/aB9C2qO5sDfZSvGh8IweyZvEb1fW/hCy7N2huz2txiLNRXzIBoqlJVsqm/Co02dusE3B7+POai01q53zocmo8d3dwp8tYfRemAM8gngr+3kcre3JIROgNSBASnw6/5Lz/i0wYMXDDO4ft9Xerfm5qPtxawY/Ku2gkPKeuavY5XqysZPP7Dc0t4VwnjrkjMdf2GYD/TsEx/GH13aN7ejaDWR5rOUuLxcUGuT4Slem8C9/mIIJzZEh5l/RzokO4kioHpzhtwo0usCknB+gtq9e/fkEJgWeRJr+PRdmg/Ny3bN4IxlOvsQZ6OVjHuLYGa4FtYcRe6BwfaembhYwFz6JFffe4AAACJkEAN9GzRJqEFsmUwIQfBJwQ1zxckHrQCjoB+ml4GIhHXVCOKOEj5D4ClfDoE4gbQZfgAAADAGHTB5TwD24DL+YdwtW5uAiWSCTsWUMvwEooGWirzD9TwE9jEar89xTsaHELL+VBSv5e4qGVQDUdhC8dWwmWtm1+uzvgvC88JWkLZkzaQgpTYM/ZjZFYy9Z4qQ30E7ZxKpTpckUxnXL/UZYHWrs8SjFIzVsrkv6qeYKfQlBLDe84tX7UceTNTTwcQFBiflX7xoubUyEdAEaleFF14jC4wJ4uEqmUTAMiEtJ6zdnm80Jvc179jgkn/gXDImf9tQI1h+S6WtsKim7Aa1N0AHCdhwEWsULG+or3TowgE7AljvGeGntUq9RdG2CQD8V86aW3sQFIxgJijXABOZ9+t2DJcGuYnyAEESSJnAQISEtyldX+ZzUhGY203SbJFrUBgEOZA++qFsbbPB+Sq94OzOL4ZgK3x5alBlMTONFwoY1I7kJsfI/HaMtW0nfYGJt+vVgXOMouiNUxaKFY+HW23uHlTMoVjUrrqiAXUNllgz3cJJZLGDZ5/zjoDVngDyez32rT86Jt9bTMpJ0+xQ4TYV0VTVIE2XotFFqc6l/cmvsPcgHBYZ5Luv0nM1ct7MkicRQ+0kYScP3gcvoTGAN5DrzmeETvhRJqdbwLl5ICxBpCNn7VMWgWWwiQa+pOeJDNFMJNiSql7Ef7XgDCTHaCy+7vO7AAAAJ+QQAQnGzRJqEFsmUwI58DxgFGMlpuY+DIdlCNItSRavrfHvziPnOmOFd+qwWLSSHjNR0O4ZL9i/QAAAMBWeW27BZZpsKWv++v4lY5StMOrdBrqre4qAZXO+uIcZItM0/sRD2VpDCD9aq0Mao6WfFSgHK/CMDjT3d3gbsHPQ2M71Wc62vjOj4QKw/lQOwdt5zA8z9ekla88AeLy5/qRkBZJXloTPmKNNhQqxoNUoKC3z3iNuibfVHqZR0Gb+C/FLwTUS2A57RrczwN+czxo7l7HevpV4FqsVFvw22ZG3xL3n/aOqDpw2WlSyKLhMWZXFdz2CkhMU0H046chMnptwgH9LeNZM6bgorCH6yeudoSNKdjAukNooLAsWkFb8wV7qolfc67UiyHmzIXGDa3TvEiNDp+XXAZls/G9Qqdnh1JpYqb822QdH3wzlTFy42kHeQBogIEz6p97EL4DpWiho1pIqxXxa/WyMjaQ37fe1Dh/5VlXtxqAbpBpfKIRCO923bx68+vl+Cb/ZPKopofkNQ9Howq0N+WVmrE2FDW9JlLGZM5OeUQvThiBeW9SGwKZJKGMeOW27EHOfDOqXsvyoVaZezHKMkFJGbElcCmfIvR0VBgeM2UPDVL08pZ/RYmxUdQaqFXxXGBpUv17vwmHcgmRWzRjSl718AA/nRIi/yTEXWwJTpstdZQLy/wh09YMn0XtBK7g3tH+5GFL5zskmj0d3EsLgv2qEP3D10uEb1NxiNKb7cDXKID1Qp1jLT0wP4lZuZE4pRa8zF+kRnOOpLGbSBTlZ+rqrmuWGh9ZYcsBiCeLXxPi8uOAG3/PV0evoz7V2hVCOOrXItXio9vrsAAAAD9QQATRGzRJqEFsmUwI58Ffg1IAAADAAADAqXJQugEFMMZ643LFNHlsymNqDIbayGZPLUl4dXsup5jhVsY9qk62mTJ54jpWFRMpDyFC7BJea31xsHACRsNSI+5ZGBqeng15D7+SDjMO4QH7mbWy6QOTUhd1+ZibpvNhZ3o1us0Zp3gaxwoDJWamImaenF45PytYF+GxIoo2fD1ZnYkhDXf4/bGCs1S2qZY7/CH2DDq0VxOt1KgpWmRsVr52nnb7EWSETYMAcmAducmVAHs6/m0TtXS7fsWIYh82BtQRuAZ4jEqWdIYdc/BN7455inhSnEVELGyRAEb82MEcvAGBAAAAphBn1JFFSwk/xwH3VX1wrFtTG5oBh7Xo9Tj85yGF0IkarCtLBctler8W1/APRaAQQ3MNbJQTMb03KaLzZaev0MCpyz4SISxiQSpjFK6YQROCovaXd/UyW8Sm+yufmjseaHaeyH4Ak85ZGClOAp1HllOCzL5rW//uglOxR3mMS13LxuEKCST3yvMG7iFhjCyTs9DrJpHN7+7/PzaNxYMUnslb2bhqNvyEyACVj1pMIeFJBLUUtBv2kw6I6IxPiv6ncndEZ3PozoAoW6yF+xBB4xXbsEMQn+K5GaW3rrvFVoSkUzr8jZfwSY6xyUgcliCWOO8S55O7AN4/XXS8/LWjGDuj2M/LpEFxZ+9dxbci3Ieh02Xp8Hof4ye/3qfIuj92Sheefrd3Aa35vQcibz/c1wUWR0Vbt/1Xm5r00HioLw2NMNYtVA8Zd2fA8Cbr/zBrbYSf5BCgyGpI6vnxmIfDcOiHozpc5zvNZbOOqtU/GDZZdHSJ7ub3096KH9Aw/NEM/6ifqS9ca9DDhBoqYfthSxQFDR+CS0Y7k29mMh37Wet49xBFLoN/C94sRMBqjOwo4K7q/+82wgoec6zilLtSME4th3OwenuorxK7ksuqZk+wbyiC3ywDh7wx8MOJgJrPuAn7zQ7xzl5frVhqbz6D51N8x+6M9/0cMm8a4elGoyqMXL7NNvMe7hZAd9syaPNTknMWDmTs0u7QMtOnF2E9KCf5PugUtvxViV/2W1W1yvLwe6FcKu0rP6YIqAPIf4+24bK1auooGe+ygBmZKU38ib4xUv7mzmeHoyYK8u2qTQ4nhnH5OmKiMoFesqtPYzS8bCTN4q1FCHr/f6Nq3IrDgXZVBkJ3j6hoEBZJQZXPzS3xgvQ3jzeXjmlAAAB8kEAqp9SRRUsJP86W74ZoMTuYrqjtUmQCB8xzD+EkIt4GPcc7c7wCyayWkK04PiWR8IgZBnkxBDRUoJ6DMl+4+mbwqICKui2UaL0/msWliE9giUp8Vx6HnCVsryOG63kWMNPVJqlTp4kDg+fOHykKpghKVa7bC0SnNX1wUnvb4wD3+1DikDO0M/5d0+7orbmne2k9Xmicw36El5Ox/0nOybnO2y3yakOjRyOTjiyJ+6fm/yudoIQS/l1vAGpRCXk1x28kKGN9hCm7VQQi5GW97JoENm+LvXuxcgPtXUXJDtJhqRKKFT+7C1BWMXp6C5L5hi0hQT5YhYceowDLZqG0FREakJHLL4q7F3+E5eCQ6nom3A2T6RyUTyzOVNanYsNFmXIjnBCGDoKk6M+KwAXujdazeoBBLNceUBit7ndec06ytN985zkv3F5CvKhbdkfopuoIQODLHQ2YeD70ctFjz1OvhQlEl+gGtPc8/k5fpS2GvoI8OHGarSeu7oFv+4//dcjBfDk5NDaU0N0tVoRRUO9wNvGeSP/QpuV7+MVbqR4CKC4FN66kY5pI8jjFC2wkRk9LYEOBXNE0xojPRl7N7qNBXas6N2FKo4LX1JwHK4COixpXR/kCVCbX+D8uU4tDUUr1F8QDm0WJOV952EirtGtoQAAAfJBAFUn1JFFSwk/O7ekaE6D4YUd8Lur8UhNs1eX7NHkCPdmfhudcskrF950RA1aukAsVixg3N+jBGXEz3y/upWj/2kWlvLGPxVvGascJjyCCz51fNd67331yA/XFes8fXX2loUkcCC9vXwIwXd2Xq0d7jmz6aJyTesOPV2dxRLNIztmhuycOOUSzOQmZO1N9886G7c6PIbXBjvwtWniHtGJudc+pLIs+ON2tIYWv8hL0xfdNaaGOuj6+3bP3KcmrCm3TqIKQLwzXkaG1FpqEnYQJ9EkRCEFe5CjlRpnDtkluVmWI1+I6CQ4IID3/ZPOUxRGu36Qth8nQKSw8usS9L+rePZ8xzobal9uYjgretdQNpLTlNh9smGecwulLerbOQxqYXDvdVDC6A8D8gHyoX6OV8oAXpzgXcA/pa9vqVHttLh69OUVl37Xw3/ku7c3qFUD5/aYVB25RXsp6X2mkE9DXONbhcdoKzYyICTXqpZe7US6ImPmqHA8g6xTmeZ21w5Jfmn8yrZySWcAm9aCbM0M8htpv1yNTbaVq57T3yqPVbcz1U93of3X9UnUzelT63WP3fv2T+qbE7JTV4v1hEXDgcgNX9jAVooI4xsXG2vtSwVkEtVm4fYWmS4VyLaa6P72L1duebe6ByQWGBsZWOwjzTEAAAPCQQB/p9SRRUsJP1z2NQnQi7rz/BQr6QdKUxf8G+A1Bk7hVbZcHFdF34jeOeCwwhjDspXuhxOGKatL9z1yyhnAzdQQU2B3R/oSFgbECvxm3ONJX0grVrXvNOc6Wj2/bWQNh/bF/DK+aGZuIqDEQSXz5yQz24Vb3WV3mANSWo3dDekODx4XoI0HpsrMePv5+0v/d0owVqygDMlD5dj59kzz7XkUmmwdgomPfYrvkr63YOpDZSmVnlN4ZZ/nB+yqLS/w6uzeMR4/SXUf7Cf+yRbj/7EcTvihHK6xT+kEvqFdwzEUM/tVK+Yvm3u9QcJCojhC9kZ3IhUxJnfElEmFbBji2EFl5Q2P4sFCHxaKaJVpLZX53h4GPjDzsidUFrnJSEIvU1X0tVgRa8u6qKAvDNFsJIcuiyyiyAh4IH1zitHm+YhdnmW9wWcmOYE2g8wUhRhatUd71m6Gbq9Xfd3qWo15JxTJCUkgsvrWYLXL8y/IXkf3XJ2HaJdNyHUBZDbISkzcSdHEbGt6ZU4rD0AGpC/sKkteJL9g8q+Ts8Cma1hXxV3YnG6aedoFX6i1kJ7gY10bn6oPpAd4K0iT4OMmMq+vZdV3JAulNOH4Fbg/pcdsw52+fFkoPJ3mkHHYTXPZT3+0niTuIRy/SVrENd70+wWG9Mgn27eyYHJXiA33d+IsP3hDGJ21BQ2v9nhoV4+CpFerNk4Z9Gs0wxTZOxD9hN1uI8jjwnUSHjEACk4GrOm4AJFMMm4alQt/RDrduW5CoINkO9iHZVFgnxEVVUcxB56g/5YIEvxAOuAJf8bTqEjcwqDr0mRQ2nu/4QPoTG6D6yS6mQpRt9Qb/WYVwhsUrcoxoEBYQ7Q/rkcTOr1wTi6cFtb8z2MR8Kz1Mu6+a5B/rGxYix+NeA2kav3WRaPNqvOuiCujJrm9Zm7OWNduSBpXUTh3QNxsBEbMQeEEdv1o42YOJ2AWy89isWMw7699LX/C29mpFaLrn245CovAIpp0onvBTfMurHUCsr/C8l3XGvZ07gCnBSc5MsrBRJw8y5RZIE4XVQiQSCQvcoZs3xLdzqswFuJymKaHLu88iHZiTw3nVgzBwHLL2WCbI6kEix88yCx6DW/F8il1HDxkBJF0YxNk3hNt0uOrmrUvBiP5dBEaw7hVIB8afHNjXMQFKBUhD9GYCs8QRoRJ+TqkPCTWEVEE1j3f8R6Hsmxidlteana/HMtGzr9ss727HJrx1fLV8FfVANGGYHh11esznE9TD+X8Dm6qrzte4ET4PFnEiAdBiLEAAAFAQQAtMfUkUVLCT9e1C/436pRwTH8sK5+5gCytFBRbsCKdS58I+wJD2bJFDSJ5Tff5mFTHemkVayaOoyimnNx+F1E2vFMb3zIEDL9dUOyEjwmtJ612BpQ4SpIbjWl1x7hQbIj2Ja9orgL3AO1GQhkfdS8YPrS76+OrVGJt8P05Xe/kTgR5KiFjtX0e6G9UlR/8+vegPbIakN7y5Kig7tg9RzKe4/ursyCWBSxroNoyxw+j5sfMRucE0uaf7ldbJAr6VRs2WJoS4Gn91uOyuCsC0ZVrZCWce8Qx9TJgKnKWES4TF0pyV1Hbeemi1Jh3Gt+7oXw/ZiWDagPk8Cnwb3E56uF3mQZqwaDRM8SnDAqdnjLxKuWYLBJYewockX75xVdlR6Wo46OXs7/BpAWEmmEO2p9Wt3fzrnJWeGis1oSLHrkAAAFbQQA30fUkUVLCT4Oix7QZSG+7i3SBDC/kI/M05fZdMF2X1rnGYQUElKAdil6nMZe4LfcK3RcpY+y+Vk64G6UBx+sJbH/3wFSPBXsyMoQqyS/7E+DPa6J5QnwWlnXIphGOvulQHsCqfxH5TEioORLjqXxZJYbDSGHHyzlWOOOyCuRZZjUl3wQa6gmgJffDdfncp3dlAFKhjks56hjr+IOSlehcXFlsRExCBOm24X0RdEqtsf0woJVo/Pu2wAdkpyg6xospTv3yPICTCKxBztWoxQKbeuCyv0XhF8/6SnAg21wP0s7KhQcJmGSkIhnrbDS1oji5kf5tRbfTxT+KTfeXsWhQd7KBgnT6MX108fu8fDAWRlIlzT7UVKgCIzAtmoSFwLDFdYOsamrKsZYSS78xCzg9v0yRyWTLs4k12GMHlUpdtv5NBEs3ztAm+t4jI+/h3dwSYAojLjfCSWEAAAEyQQAQnH1JFFSwk/91Aw8IumLx8+ir1dTyEUJb55iHWtYINlbOM5eN9PRTLUBAOzh1QBqjodpncfpZGhD51Z3OX8DnItqtx2lquNvSfiV9QVgLfR6Tbg55zHnHX8TuNVHLxo3B4d8uBQeLEd8n1GREfzWGJd5s0eaeCbY8NFYu9uivLS8Thj9AhTzhGEr1vJ8lRw5mubisxBCXBtDmQ+t0nbZY3y2z2Yc7DA/dkrdwHyTQWF/1V6b10GYwixzfbCmORWKTpKwbF6kKEqhfA05EqqrwQbnnwNK5zICaSzBt78U99yBoUwO3ysKLRBIMlinyMz8rx6HRq3QLup1j97EupbLdV52jwbvoIzAhINEbx2bRaV702HM+1Irh3D4zdws87tNWOUaTjeIvbALPE2N+vy3ZAAAArEEAE0R9SRRUsJP/S88H2SgAHlIKPfh2R/tvcvw3f0Pj+2lStgCMXWqB3k8ov3OmdFqaVDDXr2U7iwpvp7Rl7iwxCTEU2St+fck9h3e6YdT/H/rbDjtfPct1r6JKdogTSvdhQPGr89SlY2JWcPWyYzn8lLFx1d77x2Am9y4HANndYyJ7Cv6tVezasycfWN4JLuB3LfhZIlazyXg3WR+3oxmpiFJWwcOnQgjAZMEAAAEZAZ9xdEMPH4RO0UqOh9P3/m0qa6S4i+gDzk946sGPwSTDd4Kvqi1oQ/Cr5yGuCWX/FUdx/1F+oFQMCuSl7E+LjgiXCStdb/PCZIu+qyYZXdymVDHXpuqYK2zVPt88xGGev/h2QDNauPlfzjjKy9Yrt23iRlRBgpYQ8/4qgtA/VTLm9JUcC6bzE/0CysPQAQGY3xpvUOj7j1fZHEnBoOaJFMrQEyd3+MGZO/vVWYOkESVx8qz8U5+PBodTVzUzhEDHa6sf6sRVqjtwDHZp4miySQt8TQGLrXkAQHzLfJYBHJ9bg0+vYjzE7rkxU5EN9CEmijNmMiGdB3E8yEYUmwvOn1I0NQZAg4t7ERjXTRRQa8KUhxMVdN11kYAAAADpAQCqn3F0Qs8+GqKWLPoTyTH+bYAB0SLp5dLVaXpf64fyXNd23UDCYXx33+7U+PE2AUhErbaYJrKezffbfm/ue0VfpqUfQbx55lDqBL4GXW2bsCbpQY6Wp7+1MOw4Or/YtorZTx2eKe5Cb547xNDoDg+Nq2AUm4Llr4QLzez8/pCTFaWbtKVvzIUbBHlWRSEnM8ZUrQpa4CBMP9MvjFVsTQ26hR0PVIzpu/76dL6KEuGzi5T+zGljrgg81iKvWF7hMiHxGX83oQGDNWYRMKaCKTjJSY92KYfAG5mU6N+LWnQRqtHS5ZclH4AAAADqAQBVJ9xdELP/P1TRXzWSDMj1SVuiZwAZryVArx3fGATPJc9A9mIxWhs13W2DfdbKBTvn3QCF6n+VEwAodbHpoF2VnZH660kchcB83ww2e+LGLtyWyHfhJPi52Cnnh8mu2dhhHynWg38NPKuFl6BhAZt+LL8J1RSj5l8VUGZ52/Vm4DpJsyIMgQiU1NdqGA/Y4EQyx3iHTpQrDy3MIwSjqWenA/NvdfjRRwpBalNyec0cMpB+BuKvD6oAKeD7g0OooDP6dCYBsiiQhdRzXIaOoLdDfZQ6S/6J6hhkszxSJ8wXqFo02U+rHhigAAAByQEAf6fcXRCz/0EzTAlABpaE+R3jzqTQabdq45F72T8+0SR0MwOytZX3GHyhxbFeil3P9tb2yZdh60pyqwTy39UF+fz9iZqtDHTSGapDYYWsfc66p8CSjL0P1xih/khvQGthDUEvojziOVcor/BFGU20HZWt1nNd3OlRSsnb5knL82nv8CAcGwi+ElcKauwwu2SqESIsQX1qseDJylG6IcyQ74IowwTChjgQ1Ut2Cz3BDCxR5krBUFfGLmtKxOfYDK3b26uRqHaS+6ZZ4NpfK5XhuaXsWtpETFG8oYdAuCJobOqBqmyh0BJIP2zijPt8H2d3J/dj7Pt1NVk46zXBlKIcvGu7qCs5d343r4n3/oISUZjRWdTsj+bXiXf+XmXPBiyhtPBLQKZdsUW1MlEpcJEONE1RuzTYwQZm+QbACe0ykGJy5tcofHap6Jqx329EzK0a9QOTD0l0qj884w7JEibc0tKmH6PyPC/FYqKZ3a4KODSVdjX2ZGVTPgLJcgyRwC6YVwEkFRZTCAkwYKAFMlVvL9IG2hluUg4M48vU2fu+j0HJIQYTYnlbgXr7AzbO9zaP0EB+3HM5YP/It94S+GwYzq/Zr+uKd4AAAADmAQAtMfcXRCz/QKeyK3Xp4dsJxKG4pSL+1ER/EjmyYcN3AJfyWKs5lmUs1xqwJSvYZd7MWpqc0sJRWC6zCG+W+EdYv0zMyv9kG0OlZX11OxO6J6DrYJ2LCmHWHHJFPwiDSNxGPj7uZisQT+2d8RqMKk1m265BJNpG+DyAWZ1iAcrNVNRvjSimNTp0wegT8sZKAk4X0Q05K99pMtJ+DsGr82gWfoAsWTN0SOAziBl6nyyc3GOp7utdB3QqP9NBp+7NYfsZJ6Bk8YvN5JvxIYE3hrSZk9KWefkWxQ21aQY5s8CWZIqMYRAAAADBAQA30fcXRCz/iuHO2CjudhgaAC55ilimUdvqlCjOmQKoZAJw69028IVStqZm/6a9FYz1b0jKlWb36AlijgZx4QXCThUWjdZg+bMdUlp9MBhqQ88XU6jCLAFFGyBr3cpw1sBJYT8YxW7XL0Kdo12+khVgqn3WCIvs9sjtqoAVp++SZIp0PbygLD18mrs2VfdA5GKKEu+awSp4uJN3juJb/ncg7BflS6XIGb2rw4ezSpMorW0PXKlnQzmOU9BOl3112QAAAMUBABCcfcXRCz9Bejz/gz8ACAtBdf97Rn+QWNnFSYjfXxfj77TQsRDKAWZkC6b4MHCombq59mSakFqwDYYd0Xj26/MQyzoCArduqIgoej25VGnTtzcxC+4iCi1vKnQy6NEhCHvX7LSNaSw9NINv+iSI1Ep81EhKbg3pcr7qCgtBZDP1u1robwnyRQaKMpMoztc+4SD5AwqUHWQjjT9xdJSVfl9SO5pUmAe0a4xwXYeziZ/Q/knrIzA8J6gdFz7w1ZcRmTUAQAAAAIABABNEfcXRCz8le1TTOYmWOYCpll1KOdTehw+N8lVaxcnbpv+/CDFb8OCA1sHv7i8Cs8Qrw5h7NrJVkJKONTT8DnrU24SbeGS6jtAsEIUS2uKXoAWXp/7LT/HsbC/kSeepQzJjKNSQ5JpruSuiGl8k9YCQyafUvqmAfDaQ4gBMwAAAAZQBn3NqQs8eQx+xt6sjbf3Fxa4fUQpO8h04AyEVB/wZiiZAUbLAp7GQMRiOq8OZfn1MRIbrxmgg6i0G4KW+Y9GlHHxMl4Pk+wLKs0AAUjq7kqBaUD1t4Dw2qmGcaQJgEDzbLlYOpLM8nIK53BuiZQHn2O+AVDBdZ64aCzRhZfmdJq6RkXn7UWGAtpvBtTc2mwVXXGu+jKq52+U5wzENfw8AMOObW7Qv2LLQOyZJMJSpB1R9xWjA92hgcuVKEyVTlN2vDp0TbFm7+t3JzXMydstCuxG1KyBMASzBi71wOCs6J0XCi6x+VedhLtPiGcDlbrbqRGM2Y41irqH2CIA8x2k7Wc0RHwQ9Hu2gstE8a2lIu68vrXQFPUgMAdlQhoVtT8SJAHI/a5lej9clqhKZVjdIjhrMGi9vnoXKQofPddfsf1PJgic9+Ec2lO6MTtCCWBSZgR6IQOMh3eR1KrgJlzUqrL26MyQt3nTndZuDluz3iymbqx1KuoTQK8Jy2W3jRFFG+/mMMktk9DY2PMtKsk52MksieAAAARYBAKqfc2pCzz9FJAzLdi9DeLzlLfOBAASLyNDAgUinh5Fmg7Ycw3maBL+EXwZ0p28LxdrsC7xib6iULpPHebNQKYkJ4E9Gi7G+YDSKJxcISCoPPKrrBJPPmdytXM7YT7u77GyM8yo1UWPfsK9wCYt8dKHYP0dsFx1vdKLM90lb22e6P0ht1ATmxNiqgnRjxRosHhqj52+6EfjDFKsPdVLW33j2x1mfUV6XJFMY1FDg+Jgwb8VkXbeIVgrw0mv9DUZdoL4CGPjic1iqgRVdItcPmtYAl5HNHdYdEP5xyS6mY1ecTOXtG/K5wLt5ZPbtsr/A0LZ3VGdxXtKSrdeYNNdtSUbzrwFci9DY9Xh132otRFV4sC8digAAAPQBAFUn3NqQs/9BJLGRK6rCLNM+Tx6CCGRhIc/+jeAa9pFVU4OUslHif934rd4J8Sez5JckPSWX9ZbzgFv4mhGZRyznUHQ/67pUUiYLke6hxBOzInTLxKK2lJrSv0nddZjL4wiP8LKUDLxswVz8qz2YgcmqO7I4loSCrF+mTRcBl4bcmejvWYcPtoGcgqLJv/YAg2ObhbXieeLlaRe5xGkxtgap/x2iPguX6+S6l/VdXpU78B5EzZ1U+9O0skGWJDOdupqOunPukldhb7jFH1UMEwpTD7AhCet1KN/0ji6q1F98XZyKtUaGAHi9tPNPqI7gNq6AAAAB4wEAf6fc2pCz/0OCAZ0poIEg6Vs7cJm8FxWpIjyFnHhSb357rDVJxYiYByMRUrUEiwB+q0qqWlXucr/L6jUi7xkHOGyJ1u1TmwrEmGRefdq89jM/lVPxQsWb6+u750d6VtvQGs/hyb8uwcYIXapIZzS/7tY26hEp5fajq45nuvnwjLNBaiFDjNKElu4U0JFC1whWZlbHZSAw45zMFjY5ezLVqi0alpAJwEwTv8FxdCEYNsPXJGSEwTuzQNPpP4XM/SveEX7LQaO2ehuJjCWZMjNrqwBQ3+8dqLVsWA08Ra6q9gK+UqzrSWbMexbAq2PK7WYcgEoZghLL7QDT6zeOZ5bYduWWS7TEZx6kPI8DAnxjU/poUTEqBQ6UH7I4BlwRiGUZTsjShNhShjqu6TsRW+pVnoqv7lz6hEOlSvtqMrC52xz7LTTfLHg7+2cCMId5j6uFw0qfCKvO1L0FszSrYbOIGqS5xG0o9PqZe0f4IIpYQKW2is47h1IQFiiSyX3BoXV04oSH19P+u5jyx32mQ77Xr8MvWkhwcq7ru/oVArcsHRh2BpIyzNsp3x+Q0W8cwRFtsAGkwLvwdvk8B4imfbzInHKmUa7avTYsDlRUiXq0dOGUBlgmm1dEOEZwhO/GJEFuaAAAAMwBAC0x9zakLP9BMD9S6GaWddRGqQ7w9ERMS4LsT4SFUDCWxjdW+3ikRsmft7A345kE3NhoD/USNO8w/iw5ViGpwIqxl0q1KqbXfr1jQlGyZXNHu8vrOqd7eu0XG9kvsuco48DwFkZE1Gjvsns2k14jpeEqTINpuW2wZLIclKIbj4nhBcZXf1/4otJYKBjc6m0aARyZ1TFHSsuC3icPXflVrUhI2T9pXV5FUbC3G542c/sXU7K7nF8Pmr2vHoebIzU2qbm2mtgUDwps78AAAADVAQA30fc2pCz/jZ9uHq4drRi02/YqcKtkZbsTwMFs16hlh38kcexwPZr4RH7kyitam7cM59vl4BJmvHwUKFAThr/522eoHKU6gACl9n4aHih8oLWU2colWMPxpBFC++HG5j6ng+qv2m3F6WruSpuBW1yMdUeJOjv5p5RettJDxszF/LxmPPmLzxdy9yMnsKWp3BqaPP+6enghHa+bqnwWKbVEJ1HIMHRyvL5PaKjXrVbCm1ESBwrTZw0f/o1bDJvegSNAF5Zk+KkqUgzed6ymfNt725lsAAAArQEAEJx9zakLP0MkpHLFEVWZR8Jn/9N56QKDcKcwfN2d34iVR7TAwRxtORZtYvoCAwAQCII8aLgR5x76aLJ8k3BPnET5I1mEVajP7+ewAZ3gCU16uHDSzw3j7PM/AHvYS+la1JiV7GlDsp9dAffO9pnlH5OHfa9IuYtJSwUaIaE85oWMJ4gXwbjD9iEgjfSxVQoqyldypBiM4SEuU0y1ROTCPrGz6xHy/NdK/8KGAAAAeAEAE0R9zakLPySZ5AAOeqw8uBvRGXvJK/rjgk/0GKyz5noAJ+pY2dhRke4KXCgbo+tKgkiHDNOn8pdAGHinmu5dJZBNZdJp/Iow534j7VOn6+4m3tTuLSmkJEb5InlvMqaE4E07OM3IvxEmzseyxqPy3TPR5VA4YAAAAyRBm3dJqEFsmUwIWf8Hcx8PJuYF7jj+bWWJN8V13odsKfQLTNqYaPYOjTe9w1d85lwZ9/3bfA4teMktUZr1WOvzrX3yn6pa4Ycyh9jmp9M7MzxoB76/TWxbrGNIqGkuYDEyueS3o15caf8c2veeHpdP6ibqeBnbQQ+Y3zMBHfYAkcxYrqzQ802BjVsH+hT/uO+QyNqKLf50HLhnozHWHZbsdvhjIsGCUxUxnv15021ctj3ZOZpq26Z20et4jNzLzjQTrpndHz2rZFpgrD/eO0Ibtj6327rA7LlffSkRDxzaV+IThiS0J/zhO41AZT2vMy+ptFWmzrrTgjKGSUT1U1mYwRnFC+80C3BYH9ZWxsaX03GGXUs6INTmW+yC6j8nllCzCcXfMaNEG/PjHbsi/Y/YtUEk/37OrEyc3ADAJzs4ZJtWtwceAQoQlk6vEuHJUcwQj9f26lFZVtmDjf6+ChhkiljERj/g2fyCF9z2XvxICZ0CL/5Uq/I6XtMHNSnFTnlcBznVkl4Bme1o5XmjxyrjIECqOfVkoG2Livbl/gA4MzUmcxxijDfsha9/y908ZWzOuHkBNxTjT+2nYSL8TajggdfylohyzaKTN7UTLWnBkfe1+KYHN2kJus5nrmJYm4KLenbgw+KL57EFfxOzBjl8eJzC6dFsuBpy0wlAT2OrwAlVrVOXozmHM4vOxeYPVzVCc7cxxMn4bGbJ4+qaf02fUF2gJRZOymdFz0cK1zmHzyUBM5HBnujvtjv3sOO8jN+d92Zh602b9PVBJbeuD+KvFdPRc97pGty7Rs8uxhSt+RYwqleX0EozOrppl5LAT32MBIiO5xJoyyGX1+RLhzChc2/5EthO0Qsvprwy92gKuvMUv46x9yeNPhfl9F08/v3Q0EHpzooMMBuC5Hyp50e9ovqs+Y86zUY+fwwcfBNwvx1/wvB2khKrH5hj1Q7u4AwGuSJ/s4G/KPHp6qng9Lk97aJGiLj2D1/LtleNPrDrY5QFH5BMKpoCHv2KxLPx43QF2UEuFr9h1UdOUhaLemH9rGQmHs/n4h7QA5tcIoqK2m96QZ0AAAKcQQCqm3dJqEFsmUwIUf8Id9Ep5323LaCUmhkxv7s4X8Wt7b1iMJ7/wvCAalRluvOns4Az4HgaB5hY6AAAAwHCqKvCILmCyz+HyF4xX8Q9KbVNZYMwNNG6Wxs75XmXT2CLxVYx1vg/wRM8/vgxSuyzb/WIpLgxg8Akd+Rzw4tEnac62ofoj51xeAwSED4zbD2qK34m3MSCCxBmPgZs5F8l2jiTpppEVD0gbGPMdxrRG9EfTaMPypGjqYpmrnYS0DxGzLj0WhD+OXnf3i5L86nlOjcYeoP+ddr7dkdXhB4B57jCM2Cw6QHuP2ec/1dPkdV5Y582H9pb/A9QGusfbC7ao2c9vja1swiCJexpwqH/HyY/75r7aFy8BGnXCi2ukTS9Hk0sWXjQqe/tWeNBvfKvaLSnK21lhUzdlH9M8JZvH0QRCPhRdfWx8bUA50zz0l7cJ1ld9g0qmUsKQl0J0YuSHjGXiUiRmYLxKr9It21gvPIbTFnzoIe/alTUdP6v+LR3IEHpOpjvFAgYnG0dX++QxTCXJ+Za9KwiXK9U4dxKuYw/3dnt9fzvb4D+EeWT5zPKWJpC3uyLoPytzCWRfAIpty7M73zjl3eHZPkcyAvXO2vJ2vDwgdrHpV67qaqb0SpZWzO4L5NoiDKPB/bxb87yhDA7tde5FgvmwRCSaNcUIT7R+6zyHAmexGAoSv4hR2AiQ2I2N7SbhIPNHSh8nOZmtFb12sCg0wnNI1LIqTyGOvzbPZOKJVPwzhl2sYyet7ivW82SpKk/o8/junj199W/zp69Uvh0bvxihgg3UqqY5d/jq3Snkb5jB4GhKThfY1sS9mJ2RMX3x8ZtfEOZ1nKtB9sHpRYBnwiB1WgBTanGqHUVLp5PETtw2f7tGj0AAAI3QQBVJt3SahBbJlMCFH8I55NZz3Mh3F1pf2hB99XFLPqghRYmTaJjJZRN0cumBBAabnTtGgGeFEqJe3qkClPjXFzz4rx6B0kd82/2x3TKzzXz992oYksHX2teJ1BikMLHBp/4icMZ6M0y962MaRqntctbdwVKOem+2Sqw2ipjOfv5n6ZlwJH2ryu1jvdkVRuWIWKnGCu7bq4iyTSzm4FMIgJQAPOaH+42yF1HtAjyIh0lBPCcx4YpkTX/5WzQlCQsdoPEbC38sXB++n+Qv4KA8eJV5UbaEp9qgDTBiBgwa4RKHQosb+lF7b8h1tGwb46Xw1+MWrV28W0MYcsguYFfhd1U7CEOQWlmPPVVYqUiKzieGGpHGPpwA7EfrV9Hiu5U2CqwNJSC8LMPWqCHqYidhzIRwsff9EO9A423sIuTjsojN2Ut4ToyRpzlRCO5RNuzr6UREHCe/WEAWjJDAEXmx/Kf9xkojRGJY8cLHz2Vyn2LG1Q0qoV3g/RKLJ7xxBpGn79N0FAMWEHxrbsVqBeap6FYmf2VHaD5JLMmerVieNDf5rZ2gmZvB/NcRazZIKlASzy6BxK+vZvFcLFSS3YdIAi4lLBM2KxrIAvxuL/jbnXYRRhxVY7aCeOvAdpoOPe2ym3PaENIJ7r7tSDq3zdN6Eu4AedV2YSSmr94VGHJbdvaQ+STA24ud/92+J5Buj7ZmIhYE97cthoBlnHDj4byv+3yxMX8rlVfIrRupqMhXCnOPCLpgBZhAAAFBEEAf6bd0moQWyZTAhR/CUvMo0EiXXUTWpndX0+axemourzywBMcEnfV71cigGkHjDDtvwvHsk7ZTB3UxY9WeZfKonGzUHXUXECCuJhhzz4iqVRdk5ZRlk54X7ce7xh2OPXZlLuOYCtR2Zz5Vlu3Zxz/9FEdXq6bgBiqneOUOIsq3fg3+Oo5ygoMxo9owOfQYWmstrlAFTQBkhgbgvxuCjYaFtehMZxaBWYAc7I4GvGXHNpSvKb3ISotEzz7VU5Hwagx39pHPe+CkVSsBWu/2pE3n84MS9hv2Ia9VCEGeeRbCdJTCbvWRU7rbZAYcg41PSbbPK9kY8TWj4UdG3lJVWsjfcJfTXKqF5E5wpc5+iC/LI94gQ4bF8wsvY+lU62ay+xUv7yjtfoauNbXMtkWuVDAjyp/eaNHKXVaU89poLHmpJPqAw+uXFK/de4n71CUMCYyGDjnXkCNrHTSjwReBgxU6u0UDamhspzGRpEpf8rsEir/ClJbKY/8UGk19qeY1O/rLCRkmC5wuccjs9u9N0S/zioCjFTw0r7VU8VMnhmXf3xNttzEAhrKoffF3vDBlyG9OSQLRZGmuMmEKSL4nfQ5dEOpwDJaTf+eQjUi5gl77E0Dr6wDLHo7hdoxixkqnytOq9h/FnWmd6Z7ktb2B8AdS6FXJbM+wZ5seZwmL1AtjRfIIW+Wgl9WtflhVKviyVDOgNzuoNaxNub0XamfzgPsZHFwtW1uncj3cVvbVlX07NTGc8XShHtzwaEZWPtlALc8iTBz8SZq6XCCS1Fpri7dGlmaJMghbS6q9RD1LyNL6Zxy7cnCTt9NosTfxNUCv31EnxU3/rQ6wPvroJmKmSD6vJD4ZudisSOEUDQwtRYFWYOcp/WC3oggylhmT3d+JbVMExz82p730GNHs9fJ+sscxo8FhqznZCwFqqPuXFTJmgCiYV7REfEfcu2oIrTy/ajprHtswu5Ji3IYxl0AZWW1NkCJS7ZCFLckevrMiBFO/X0XQsrqF0xcRTyOQR8I68Vrvu61tgN6D6MsRorz4P/5qLc0kaANz3e9OriTgjt/roZwQVSY7/6fxcZbLa1eZ+v94ov7JoYSuSbQMjP9d7e1zXwRkiKKi9XM2JFyK7iDflNB17rJ5MYYDtZDIL/gUPdAT19bKxbmzGSLzI7ZLgH6WEEG27L/3dIxt4EKVKV6UWAJ5wvXQq43LeS7n9Sj75ucUWubSbCNqPwdKjm0I8JwCQB+Y3P0Vd8k1q4/+lo2pw+XTcNdwgMdr+soWdY0PQwQhEYs+iG3qWzj6FTwxU3wV9XYmSLtIES3tgx5z0v2nRVvzz52ERBZxLpQXiPbT1wCUK7gNGoha/0ugkkZOnoE/dysNPGfo5qkfZ0MXeaEExaVK0441y246aojfymSeyQ7nCchS4XJdP6XWyTZXhGeP7pSgeiDXfL4GQNZpULchIe/7S/pL9aWp42B5GsqkRJf7I/mWHzcX+LO+Ihx7l1o/0/iaFPWE4fPXY3qx2HamkLszCaPZrvnbe0sekEtD5hxT5ToIGD5QCV0HL+vQ12y8NFPk1Zt/p+NS3HwQK/4tpBMi8CFpamcdw1NOonZk8lNGNsVxSE3lzI+GppVv3HXMHHiPT/pe8OvfrK95SJwzwds3gSnaMIBTeUzxHMYkf+8+295wNE7k4CgchinrJrFkiCyATdlSlyWZlO6ITsv3LDCEwAAAjJBAC0xt3SahBbJlMCFHwmnkhVEW3f5OWEv/xKlf1l9htf+MLLLzJr/JGxbezHc0q9NDvf0Pp5uJ0Gc7sWUEPC/g6JcClpLp47jzZ1tiAbewDszvyH6uYaH3RlTyUjmL0qUDJTWBRJ8aXS2fLwWgzBbH9EtbeoKak4M+vqpDrSdnSQUZAVY0DyrJMsSmaHg/TN5uYAAAEYzOxgrsNgEjdNS7xfGBlgAr2tfiStnRE6xRvn2cKWAoCDZOYGYoWRzsCY0UbDWvrp0Q3IWBuqFz1hXnAwDuUSfAzZfeALRMvcEI+fvuuwM6yUSJtn70t+83xP0kaUmG6sY4jzWo/NpZw5MOaOKgIR8XmTQyDArVN+FftAfXTXYOGp1AMDVrQJpFBNAy1QpIUqVyN7B9JLNquc2eWZ6/87gJGCO+55/b0+8y9ShAqfM4L0euyF/GHw0CNP6HW03y7VjcVq7qVzzyxR/1vm+1qt2j3qXiB8dUEUI0N7mJugLZo+w1z4gMXW8r+QIxFSMqvU9DkSromP1/XRFdk74JTVNbybWAeM+2s6FZm9nLBqNHYL9dmfjViVrS9am/2YlD0rjkFiYY5t2U0+2pi8QRIjGGQvu7YBU/3IoxS21bt4sqQPWdYyV8IQ9s5sD2UO3IcHX6OnX5jz/9GE90CvgcugVpMXk2ZN8/jZyZCP9q7qTsyUdCg8NZ1lrG560I+jM3MHgGCvi0DqBA2w4Fp7rL706Ao1pCRiBWx2POEppAAABokEAN9G3dJqEFsmUwIUfCSa3gX9z/2634uteqDUlb3MeRCjJjST/gAAPdHUT2Fh/PTUkvJ8vWmFQzGfYNSgxs+S1/QH7mbOEco5Wipf3j0dbjRyoZ1FbqRpkEJb4OUIxeWKYGEThPmOXFDEn/so5DekWqtbyjeGpra5zNY46hUiRVAP2hlkFkMUp88aAeyiFyO8Hiz8gG3XtWhIt8Roue3Af3OsNd1qi71yJoIWFlzpK8j8gts61GCdYPCFjfgBY5V+lMNMs8UN1U6dTBwuTQ5eK9G/c10krVy+VJP31HKKXmLpwZLc1gxFRhmw/tBDoIBLtw++roF2xP6JhvntDWos2Jl/r4/Us5DleXBE5wMO2ygfEG0nE0WFdfyBuiZqjBN2Y/nt6EJegWiirDY8azF6Q2QDYmGXZ1ecF34/YBbC3lNFbEROfNQeHt5KXHhVj+0TZe+rZd1P19eyX3ilPelsJeEcCmXbumIUO7KAZmZK2YeWIP6T8jZhN3frsYL+p909BNqv4AiL/I95lDxWbMwgJP54jNPV9clZ7F3NhIzSULcEAAAHPQQAQnG3dJqEFsmUwIUf/CUvoxSRhDnPIWrgBCu3i4AAAfK0RjnLqxQBS6qiu1kRWY3u9FtkGideU8234on8GHG3NR+uZW4ZHFBb5UyTeJxzCgmtNnFTPpuhZJ4ofJLl7+9INLM9oy5fBJRlQmkueq2eWlgGmCehvJSH9mYNIT35v51U0xHBjHrRAJtTQ0bOL77GaMvRNFsiXyjOE949uIwQZlI8j9CRktaTt+kZdFnbM3CwLaeFLinURPkg0e83raAKsZWZsgg43xJILWgqWLc2Rli8yo1nklvY9nhoFu+WdncWhzAmeSOhwmBV7k4pLVwbdG1EozcmAX6tk6vDFyvVl5blPi1t/jlxn6PEF63EWlsz8pkPTGs5kFyF7REdjt/qsc93L9IXF0Lkz11hjkY0isBES9g/PO0RnN2NOsDPz7D8tbfgRiGKm5qftORt6sKuHz2W4zIQvLEFyurLBVHCg6A7KB6oUcVO9c/z2LEDqhIOFHBJHpTrCh3M7hBQgtiBeH3ADoHRUelXb4O0Um7reVQgqOpCb4Vh/0szdD9IHkxxos1NZGdL8cVrQC+IheNgoVP55ZMKsR2plEBUV4m+C2ky1qm6bmPlghLyHswAAAY9BABNEbd0moQWyZTAhR/8MjW2Ry7udboQZou7a8n3QqAFrQg7HmX7zi/7WeMVRGFDtof1nOXEWyeOGQqYyIqx11uX2L4npCallQAKgY5SicnC6K3/OgojDJfPPJ74PWmqmmCluBfvpx0wXU41DHXFD2U1wcE4k0cdun/UU7CYAq7jyQMdAoLkEWqawaMo03IYHvMEWnjbPPCxs4udhbUSYNllmYnWIT3UAKvTgrjuaRww2sXn5KUedXuPGdchcjK/X5SHny6FLKFGD8IR08Je32D347dh2o6a83wm3NGJeYXMe2/VgJeuHXgv/jU/8hfM1jaZ0M0GgNZnqsXjMbjbx9t1jyWxxRt08ugsn+mYr7QKWHy4QTwwgTw3tR+TaCCkYvVwpt2ixUUgxQksdXw7W5EgSZM4nhlflpWsVkq9ZJEZd8ntQXBRNTob5s7qqam4EbkUwZNHFJdsn6oeMBFpXIqS7pdDaIP7gSvSZU7TGAnulVe4I4hIKGdiAO+lZsLN2RHl+pDeW1Mtrfj9jiO0AAAJiQZ+VRRUsKP8mMxVnRAM24IcW3Ry1bHCaYJ84geCI36Yh/qapzltW/KdE4t4jY8uH/MbUqSj8JQHiNnBM6ylj90Zm0jD4/+23/d/0JDIzMRoZ6MeZMO0B654tEvCNwMoSvXdJ9Zhuf8rZ2jcYbcUaHKSRxSZNOuIil8Sma+Vr9DD9VIAJdwCgh0BO2Dp3OTCp86rYB/rQ/AQ7FODQyxkdB/5wor/5kJ4rh0FsqNjUObMFw2ktRRQmBVexhgP2FdC6lJqoHzU5qI1paEfDfQe/45PP+kihN+O1MFBws3bD3i9HmGEq62Be1uqgjSC9kP2G/B45Itqog6Wyj3oEYfu18TQNMtD0LmmJsacY88eL113AC/QzhM3qZ6EViP8V3L88IvpmfCH1gReY6aBKhQy/XAzqk08ox3rT619opXySFW5AryDCarxzM4rIBMIhcHgvUqutBxxcPGIJt/2ZDdyRpSWNPwSiIf/FZuq7RC3JWgHUKK2hi/aXG6wbXp1VpbaOHmg74K4wQqachBBbduyIfDS4nVOuuZjDshTuafax4QMy6bwdXgLtp1tL4HBUmu7/83dw9BbQh65hT1QdO4oL6whDMXixbILAprSq2SHWv2iIf43ZJ/1JcB8l0V9Wdxx1015K6mq/ICF3+XI1ti8dHF009rm3wjGjBwSDTkhBD1lNM8xp/xAMZHCVc+tJUf3bpsYUDEq4hQWWkEj80LsjrfVb0pUnfecvn598E3m30iLR2tWHI05+fAOEZ21Fzf9JZAi2Fl2Up+/6nMbCDiQqlBOKhdjwnzwojhHTsB8EPckxYAAAAcpBAKqflUUVLCT/OqmTcvtwcentDKo5N7XIDNgCeuV015S8SXh2wBR3iHo1PuH0GcwjrVePj/dC6Q3L2si8hP2J9GYb+Dv161KoxCoBf0mjBfpH/UUlJ8CxtnjboaDOaxClUbGgozYyfc/KDo7JtYCvjP6v9sDnb7zbfwYIIx+Q7GULwy9ZVLp7zqzINCMZ3AmMFAhzdrlohrVg2a+apinW055ezS3ubI4Nvz2SDoL6qqzExhuudEYQXTvUuW++7gYUbD92glNaxsOsNyZitrQ+FV6Jc3z2JolAGTiMsEQHj2K/mOPbb3ovZqVHIivDIjmQLA/f6c+NdD2AQk06NCQBVPY5xPd2iPfN5BAmp6o1sdQpFDgysMl3l2UPiFxVlovz5le/KKs6HxnOV9rJB7rEjwT6/TSRR4rsbjTAgHqnEBVHgJusTApJ06JwLZGqlyNpPtvoDNOiAtJLploYZr+UfKyzYcYbUYOTZrv6F+R/LfqyRT7pp0X6KdKnZ8xAxArOuWB35wVBUinQKsDkMWsFREgt/5tqi9IaWKV1xxEdIuJpDX+H6Ufl+Fl1+0j29RmdrNJ8LsGv7RoTlGFnASUsqifAFxg7JNBCIAAAAT1BAFUn5VFFSwk/PAm8cnzO8XDT5Q3V9htnFgACjwCOe4FGFxA3DqfeICDNCD/QvrPiPSdP2FX5R80DOjS4Imk2nrdncDkZeqHlAVeA5Rj3rcv+p/mprju0g7jsyeJoCCmhjp5YEGvFmEprVYBd3bHMu1WkAzVYkZVUPxcerkXeAnpr23oIOHZCxcdkPCrDDGl/HdfKOX0AqzhVzi/4KUjiNRKXb4TIsVVlZsTDRP0GfzU3Xno+RKTC60xcgKgpmOqeqmWOYGeoOgDoX4XRR4N+ud3BqHfJDYH7BktwQliqiSTf0iFFY6xlVIZCe2j7TUyGWwJ+J2cd/Q/5wokaS16yGnxtFo1U0VL5+ESIJGl45tH08/vPqATW59lMcZDZmiF2nGq814jQ452haUE51Xf3lWRa8CBt4VfiZp6fgAAAAnBBAH+n5VFFSwk/TnpIT+0mvli+w9imsWC3KEbLGed85hALKWCLinMlxiD0ipu614iKu4qOvp2wfuxyYok+axAkxr9XB7RhZtW/KQg4uw+L3zeex2LPgWCSimawLoBWmBnuNBQZbnj1wLwNp6Ho6h3eimceiozNAfR0L+3tEXuPB2DZWYMQA/UxJB8QceUg7V+mEwCCblfg3k1IP9+ln0+D8Hi3pr3s/ebGKAYYDjXHqu+KBXpv62MGpfm/j2Lcw179B2CdKN6SFza1GeCTLBuC8E8rk1b+IaLTTSZ2ipjm47jJ9DkAIMUAsi89sLfykqKO6U+pQzSqLB2QTdTjP2huW+yov1Rv8QFRqaRNlqZfYxL0Al3mv3yUh4hJoFFwjMWGTB2ZyVpfQ+yLwlAdRfZoiwNtkwu7L5a7zbTMev974c6d32Puwji695f6cdsKUoSG1VI9stqM+ZbfXhmk08u7I3vxJPLecSuYAUNJLuUpLcPZRJ/mE6zCoShBj0//3u25AZ35PlZe7KMZEm1X96q2Q/Xd2ojvAUS7aAnh8vyTL8a1ClrwyWrSBIlv0LdAA0j9W4DL5GwwKFn99oFpTo9vZYd5B9CyTajbyF2KE/hym2RnP9tv8uQgPd9C0WUIlfv/Rbjh2Xdz3hR3eKnCPNuOGAgFqTVhC2ENZhJmFLbEDnYfN3N1IxHgaLJ5BJWciar2ik9BA2NtNIj24tY0ANu0VlH53on0Di04ohdm6x8Cv1V/U1EsGcWqJrxET12DVVfTLuE6yD8mcKgEx9ZR9KSsrJlko0+GVgnbR0BuHiZwRr04f40Sq6lt3rKescFsNIAAAAEPQQAtMflUUVLCT9e1DAhyadC/cHRAAhkT+aSoZX4hC2RlvYaq8goD9jRpZWKt6IqoSIIX800kWJCWLtU5pqGga5DUpY8UmmeU3CuAQ49o3rrU1L9cgpdXmjoAoGrE1TfUFjHoonwZuyipHQFz+nQXY3QGCE8FiKfLmCWDQySXChAUHMR1hhNFeqKLw9He0fn46WF11nr1sVn1YZZ4EAfXUK/I29ZwVqep9gQAeB2tRKr9/JiVS1N6Oa5TG2PhC9T4yR9AuKeZjM8UKOqOJHlUKUs/aBmPTOal3AMGJv0lDi+PcX7LE5eMu1Fo82lJ9o/C4PCnNNZ6VXIgQrppn9TXnFSZZfB3c2rwAZKPnWqZOAAAANVBADfR+VRRUsJPg6LHtBlIKVr/AXnfTVfrgFEglzUDc9Y3JRFos3TSOW6YcgM3k4axrFz2ZTU5OLn+MzUdzgEEgRRGDGfQ1UMoHqa9XAv1XJBJ5CVzTuFRoj8W5mPwoe/iv/oSAOGkVClYENmmqE/HeGwv9H3SsNomJKMvXyRSzLPKbBMTA9/+nnvyPAAAAwGixf05S+Qxz8vZueW5uBkCT8ZsYie+3lCx98MymRfLpY/irnz3wtmtiyfLhcgB1xHErOWNQS8btlE46CS+cQ+UXOqdy/UAAAD6QQAQnH5VFFSwk/92TzcaRFMAdS6P+vaiyhv5/TSS8Ib/QdweR0P2MwCp/z0YgID1dhCFRx745CfAmimcUnGXe6JufVVbuZZvXBVs+vlRROvhZEwwBoNKsLhYxzAm2ddV83kZP/9BLK4gwo6qE3ZzU8okrW59b7DvmMk/BBLJo8mQq6qbA9HNdGLuYtGUulAN9hIuaDqhKAdmpL/zgi+dPZ9S0fvcXLuEr2HxUiLUPJyodmPHXDq/UcPSmnGjUryFvgBi2idQMftcIsRZJgs3uqfmF2OaIa/qFWWROgHcHfBPZHE87iMQEZKE4iOYcJXzjVJGKRapuJqBwAAAAOpBABNEflUUVLCT/0tdjpIiKaybhjYtoFFaToG1Qu73qYLRoFp9xJvOIOvV9w1oACvsiZ0g0m2apZ/smFIVH3v6+/pr6+jM98Y6aff5yPHis/RBYPhWXY8Kzco5CkyQBJGZg7BRb0OljANyqg0OYVN1H2bP3ci9L7i4D/eh2ttR/6AleT+6POjBmIFr3RMHTD81cljy7dgA+37bU8+Snl1DfzxRFBomwr7s73iwgJqWAgLLNBg63/fMjl/1IhYlbQKD94/sODwmpCNypHbhllFMk3Vfy+yKcvCHc54tznwsfUo85GpP3wln9oAAAAJbAZ+2akLPHnm1CmjK4OhTO9zZWMVL6xJMpN3/Pe94a0fFFg/m7c3qFCfM+nq5AcB3ZdT+lx3ncJztu7hcksF4gk9LAOaeXIfjU2gfmdgfRODn3iTDr/MqWJ7t7Gm0rXuFAGASWaMVMXARTYiSo22Xq8krf60tOOhBjkQXhpMdtk6pzGWiVk5lu7SLlnjwEdODEjwZAFsIO0CoGwXh1ssMw/TRIKQiwLPgMLDn+qvXraTMp1YzjQ/ebGmqng8R8/R3BH6UEg/7nChKnmpvi1vGYIWhkjVNeq2EjWeuu6WyB64ImHITmWcUs++AWWteL6ACGh/yphNe8lXnGV0Ox+7m9gY5e+I9K/qyZqr7q0B7lZh+05aq1DoQpmInLIk+i1c3V82QBrorPpnCP8hFzHgQB34Tr3TzCNI29X0vLc7acXa+7DkRV6Tqzw8pe5ZTAmr2khSHs8mhUqN4IF7rbP4uqlxS62hM4CbayRNd15W4dXoO493jaxj2VwRNYWe6iAuFefE2DCro9ax45Op+WjB4PhHnCGqb5e+9IWHX2DWws3qstyxH+f2aJDXg4vu80DWS9JeWWOkeitX5kb0yRxfDj9YBSPaMUT6YjAfu4uZuXwZIjo0Era8Xe1LYd8CUYOvzKRUn/HeDP8V5yzp7QtUWU0pNejQ1r5L7Z1dn0BjM6Afhft3akeYk4EdNCvO9aFQh3xFl+2pevn1D2WC2b6OCazob3EKVa8toAkDQz1qMJYWcdl2wQFKYZsazcKBPw1rW2832X6AJ+lLfg9W0kdwwul/Vfg2ScmdeMquBAAABUwEAqp+2akKPPFpeBvIMZQIsovmBhfQSK4ARzp3QmEWmZjVukAX2AqJYGV01XGX6ytQLyDiLc75Pv/TPo1nDV4sz7Ea+lv99pqHY4nJ8clGjsldccSV6sewZ4qYE+w8iXsSHo1eJymFq8Fwf0Fuzmo69DA7nwVu9rLRNZ10jPDOBzriILmjGxKKG0thL/J+EgiVzh0ZRqhqEVF6qWPwUUUdFs2Ro7nEMLDJl96bkNAgWDkyvUrch2bd7ejpezaW4OE0d0lreTpoh5VebvrLHp3oH0vtaDst4z/kGTC7AsXE765U71fB9dPp7vt30Y22ATgqfQbu9/L7j1qy0FsWhnzGSaCPIC9sHNu8aQCmYVZSPE7hiFJF7YhrQFDnXGLjvLbuks838cGi5jHSAQs8jg6nQv9t6LWD5cBugySvuT4SFL9YXiFiipfhLnHng2VJfo2YDLwAAAOYBAFUn7ZqQo/89RJqIQ8r/AB8Qt1dPd+3+787Gkl+bH7qA5tso9LYkEOBFEVcqv2VDJQVsBwNXh/6vHLBB6+Ghl924HjrvLALsqivaQ7Grl3PLrIAnSZzgplOxIZROip3uks/nsqr1MjMcZcAhFkBSKD0bE3oUym4w51GDla4wIZtQtRMbTs7Xbhbl3J62BjD/86Wxoj4Aj+6IiK1UvdqDxJ+nZ80enxKPmT2tNZYgvMlJg2jQ9QNmVh32dbBnWKb025GzWQIkQq63OQ6JWmMzgum7Qksr83ulZwrUIumTYL3MdBYFgQAAAfUBAH+n7ZqQo/8/P6qTqHl5IojTYcG5kvwwhYu0Zs6myZoADsLTYrAqDl7E0H9q9lRPASJV/BgdELm/Zb4WLZLWIXgHGKyPYrVdDp1dxWYLqrB6hLpHOMAZyxEdg1AZRBng6VJ1lAMTnjvQ3/2x1V9Rm/mGe3PS+Bdmbuk5C2UIYCoyGjFuA3RJL3EB/fp0b5TCK0ba86P5Ah9oTFyALDA3+AiEO2G/SDh0B2QKWXXKms+wP8h2u5rSi+Uv7DeCETp/58SWpKq2tDt520WX6SgEP+VtBBVfUyck3v2Si7wgR8ZrTDwyC3Hrctdn4puu+4AHwbnXiACCCHTPu4DuEC2hBoq6DKj6FKnSutiKw3MKOLUkHXWe6CmFKd1/RHCuQVmiAiRIDDaGq/Ke63zL/DteJ/oBPOkK7ihCOFoGY0Z+4ipGHok7zKQJUqDl5CD1qqaZV0Ltfmv6E+tcU7ObsBpdByZ3HD+EpODqjAiCen0zPf2IRX0O2Ert3ARp1e7XHNIpnMTQv76Gq7FuBufNIieQI7jbUeCj23MgPKcTjW1dMlgpaLsmaDaYNtk4HZXtSw5UVkohnOP+Oc5x1xA30jA0h/hZlzi/3kvd4OgJ2f4sRPVF+QNvrwWK3lnOBK2bKOndvq67SThdgwPwyJ3MSv53dVPMhT8AAAD6AQAtMftmpCj/QUSVPyx/sK9eKV+WBKOfkdSN9Hsx8z4etcitpg4aDxZuAZTB+wO/W8/rs3AVBRdeCjSE1AEqCGG/Dznl/x7xf9UC2z3FenWckUXDKnIwjUP5wrot17D6MbaUAZ5Dr5IOxpXLRZ8luYoo4iL4pD0WNbuEkBo/B0m7xRAWA67kTUuawphN7nYpNCibAlMKfKRaezHYGmKIxX09htmWUJQxOy3Zwd1Kv0r1WBAF92sUOPtfZdczs53JzUZxG+HA6MnZDHNR1tA9i5rQbHRi0ubIvN2v9TOj1ciuVVBANXZBvJShyBkqHFgXO2D9bSo0gY55owAAAOUBADfR+2akKP+JCtLTi2j3qAFreDRMOe+Ad7v3gGjO2jN3Vj+32QzDQQOUgJ+hKmd8+/lQYCUWIo89adwYMqyYnFnRtktPgt9xqgenKzg9VqFA1Xi5BTp3yOHCJpLv2XZ+sL6kg1dPhuBRUEy+OiCf/84pn28jGYrmQlUlj159VTqDivN/d7jpzMZ2YLVkt4C/Pc88Ae9+cl53rYfT5zfQ5l19dylOl6DilIMdgfLnLKmsEGWE3gXZNJzyS3gKDAFs/OXVDqeWNU5HjhO9J+WB1hF/yX379CIXfdFVXbRSm3HgdJthAAAA3wEAEJx+2akKPz7x01NRFEcEMwIr7wC8onaBH/7/Vu5r2FI7SM9MqmpzSFKMeI4jr5JbbON9HQqorokPG1mki9ffaJ98zz8lFPKNIgqV+r01dq/mMhNNV0J3nWYRbgPqnbmbgFFXVZvWITYWJweVYf95GTuqst93L2DOjK88GkkJsZRma04WD1Czo8k4kHBOmyS/yr6ZCwJHLcAjvT5v3Gg0MYYdPn0l7S70bUEA/VUrD/R2XMm3tUY4J27cy7mr3D5YwVrZz0tqdp62IDwifHRQ4WXy6vi4UF4Et6XRU8EAAACXAQATRH7ZqQo/JcBe4ATBN0yYRimuX/71jr3iVyfDf81l/LnWOrZiBfeA4CYiPo8Ok77m5Gav7EnRfzojGSertMWtTubnYtwOddo06cfMdg4H3uiqXgV1dea6xWGtPbS03i6IBpY1SotrSFxG1V7Vor8bB8tPRI3WSRjRcB096+Hs4bZvAPzqDaFjyo85KvSOiCoOC8ClgQAABENtb292AAAAbG12aGQAAAAAAAAAAAAAAAAAAAPoAAADIAABAAABAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAADbXRyYWsAAABcdGtoZAAAAAMAAAAAAAAAAAAAAAEAAAAAAAADIAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAFSAAAAgQAAAAAACRlZHRzAAAAHGVsc3QAAAAAAAAAAQAAAyAAAAQAAAEAAAAAAuVtZGlhAAAAIG1kaGQAAAAAAAAAAAAAAAAAADwAAAAwAFXEAAAAAAAtaGRscgAAAAAAAAAAdmlkZQAAAAAAAAAAAAAAAFZpZGVvSGFuZGxlcgAAAAKQbWluZgAAABR2bWhkAAAAAQAAAAAAAAAAAAAAJGRpbmYAAAAcZHJlZgAAAAAAAAABAAAADHVybCAAAAABAAACUHN0YmwAAACwc3RzZAAAAAAAAAABAAAAoGF2YzEAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAFSAIEAEgAAABIAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAY//8AAAA2YXZjQwFkAB//4QAaZ2QAH6zZQFUEPlnhAAADAAEAAAMAPA8YMZYBAAVo6+yyLP34+AAAAAAUYnRydAAAAAAAD6AAAAr1+gAAABhzdHRzAAAAAAAAAAEAAAAYAAACAAAAABRzdHNzAAAAAAAAAAEAAAABAAAAyGN0dHMAAAAAAAAAFwAAAAEAAAQAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAgAAAAAAgAAAgAAAAAcc3RzYwAAAAAAAAABAAAAAQAAABgAAAABAAAAdHN0c3oAAAAAAAAAAAAAABgAAD5zAAAFxQAAAxAAAAIgAAAB8wAACi8AAAQlAAACGQAAAn4AABCsAAAHKwAAA6wAAAVSAAAbEQAADdYAAAdoAAAIUgAAF9IAAA7XAAAHwQAACGcAABRNAAALwQAACf4AAAAUc3RjbwAAAAAAAAABAAAAMAAAAGJ1ZHRhAAAAWm1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZmNjAuMTYuMTAw" type="video/mp4">
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







