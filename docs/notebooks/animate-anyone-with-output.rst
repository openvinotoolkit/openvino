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

    helpers = ["skip_kernel_extension.py", "notebook_utils.py", "cmd_helper.py"]

    for file_name in helpers:
        if not Path(file_name).exists():
            r = requests.get(
                url=f"https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/{file_name}",
            )
            open(file_name, "w").write(r.text)


    from cmd_helper import clone_repo

    clone_repo("https://github.com/itrushkin/Moore-AnimateAnyone.git")

    %load_ext skip_kernel_extension

    # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
    from notebook_utils import collect_telemetry

    collect_telemetry("animate-anyone.ipynb")


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

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/875/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/875/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
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

    README.md:   0%|          | 0.00/6.84k [00:00<?, ?B/s]



.. parsed-literal::

    diffusion_pytorch_model.bin:   0%|          | 0.00/335M [00:00<?, ?B/s]



.. parsed-literal::

    Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.52k [00:00<?, ?B/s]



.. parsed-literal::

    denoising_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



.. parsed-literal::

    motion_module.pth:   0%|          | 0.00/1.82G [00:00<?, ?B/s]



.. parsed-literal::

    pose_guider.pth:   0%|          | 0.00/4.35M [00:00<?, ?B/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/154 [00:00<?, ?B/s]



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

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/875/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/modeling_utils.py:109: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
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

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/875/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:5006: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
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
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABF4dtZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAcAZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0VBBRQTe8pIdWcY8EDYH9Y3WZcSsVqkEC9ZbH4skM/gryJNff3aLkgRiyQ4RhCGwJKu/CNAVbzp//O+M42zDKfS29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" type="video/mp4">
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







