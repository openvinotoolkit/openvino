Animating Open-domain Images with DynamiCrafter and OpenVINO
============================================================

Animating a still image offers an engaging visual experience.
Traditional image animation techniques mainly focus on animating natural
scenes with stochastic dynamics (e.g. clouds and fluid) or
domain-specific motions (e.g. human hair or body motions), and thus
limits their applicability to more general visual content. To overcome
this limitation, `DynamiCrafter
team <https://doubiiu.github.io/projects/DynamiCrafter/>`__ explores the
synthesis of dynamic content for open-domain images, converting them
into animated videos. The key idea is to utilize the motion prior of
text-to-video diffusion models by incorporating the image into the
generative process as guidance. Given an image, DynamiCrafter team first
projects it into a text-aligned rich context representation space using
a query transformer, which facilitates the video model to digest the
image content in a compatible fashion. However, some visual details
still struggle to be preserved in the resultant videos. To supplement
with more precise image information, DynamiCrafter team further feeds
the full image to the diffusion model by concatenating it with the
initial noises. Experimental results show that the proposed method can
produce visually convincing and more logical & natural motions, as well
as higher conformity to the input image.

.. raw:: html

   <table class="center">

.. raw:: html

   <tr>

.. raw:: html

   <td colspan="2">

“bear playing guitar happily, snowing”

.. raw:: html

   </td>

.. raw:: html

   <td colspan="2">

“boy walking on the street”

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table >

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#Prerequisites>`__
-  `Load the original model <#Load-the-original-model>`__
-  `Convert the model to OpenVINO
   IR <#Convert-the-model-to-OpenVINO-IR>`__

   -  `Convert CLIP text encoder <#Convert-CLIP-text-encoder>`__
   -  `Convert CLIP image encoder <#Convert-CLIP-image-encoder>`__
   -  `Convert AE encoder <#Convert-AE-encoder>`__
   -  `Convert Diffusion U-Net model <#Convert-Diffusion-U-Net-model>`__
   -  `Convert AE decoder <#Convert-AE-decoder>`__

-  `Compiling models <#Compiling-models>`__
-  `Building the pipeline <#Building-the-pipeline>`__
-  `Interactive inference <#Interactive-inference>`__

Prerequisites
-------------

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    %pip install "openvino>=2024.2.0"
    %pip install -q "gradio>=4.19" omegaconf decord einops pytorch_lightning kornia open_clip_torch transformers av opencv-python torch --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    Requirement already satisfied: openvino>=2024.2.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2024.3.0.dev20240627)
    Requirement already satisfied: numpy<2.0.0,>=1.16.6 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino>=2024.2.0) (1.23.5)
    Requirement already satisfied: openvino-telemetry>=2023.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino>=2024.2.0) (2024.1.0)
    Requirement already satisfied: packaging in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino>=2024.2.0) (24.1)
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import sys
    from pathlib import Path
    
    
    dynamicrafter_path = Path("dynamicrafter")
    
    if not dynamicrafter_path.exists():
        dynamicrafter_path.mkdir(parents=True, exist_ok=True)
        !git clone https://github.com/Doubiiu/DynamiCrafter.git dynamicrafter
        %cd dynamicrafter
        !git checkout 26e665cd6c174234238d2ded661e2e56f875d360 -q  # to avoid breaking changes
        %cd ..
    
    sys.path.append(str(dynamicrafter_path))


.. parsed-literal::

    Cloning into 'dynamicrafter'...
    remote: Enumerating objects: 320, done.[K
    remote: Counting objects: 100% (157/157), done.[K
    remote: Compressing objects: 100% (93/93), done.[K
    remote: Total 320 (delta 95), reused 88 (delta 64), pack-reused 163[K
    Receiving objects: 100% (320/320), 72.40 MiB | 21.86 MiB/s, done.
    Resolving deltas: 100% (110/110), done.
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/notebooks/dynamicrafter-animating-images/dynamicrafter
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/notebooks/dynamicrafter-animating-images


Load and run the original pipeline
----------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

We will use model for 256x256 resolution as example. Also, models for
320x512 and 576x1024 are
`available <https://github.com/Doubiiu/DynamiCrafter?tab=readme-ov-file#-models>`__.

.. code:: ipython3

    import os
    
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    
    from dynamicrafter.scripts.evaluation.funcs import load_model_checkpoint
    from dynamicrafter.utils.utils import instantiate_from_config
    
    
    def download_model():
        REPO_ID = "Doubiiu/DynamiCrafter"
        if not os.path.exists("./checkpoints/dynamicrafter_256_v1/"):
            os.makedirs("./checkpoints/dynamicrafter_256_v1/")
        local_file = os.path.join("./checkpoints/dynamicrafter_256_v1/model.ckpt")
        if not os.path.exists(local_file):
            hf_hub_download(repo_id=REPO_ID, filename="model.ckpt", local_dir="./checkpoints/dynamicrafter_256_v1/", local_dir_use_symlinks=False)
    
        ckpt_path = "checkpoints/dynamicrafter_256_v1/model.ckpt"
        config_file = "dynamicrafter/configs/inference_256_v1.0.yaml"
        config = OmegaConf.load(config_file)
        model_config = config.pop("model", OmegaConf.create())
        model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
        model = instantiate_from_config(model_config)
        model = load_model_checkpoint(model, ckpt_path)
        model.eval()
    
        return model
    
    
    model = download_model()


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/huggingface_hub/file_download.py:1194: UserWarning: `local_dir_use_symlinks` parameter is deprecated and will be ignored. The process to download files to a local folder has been updated and do not rely on symlinks anymore. You only need to pass a destination folder as`local_dir`.
    For more details, check out https://huggingface.co/docs/huggingface_hub/main/en/guides/download#download-files-to-local-folder.
      warnings.warn(



.. parsed-literal::

    model.ckpt:   0%|          | 0.00/10.4G [00:00<?, ?B/s]


.. parsed-literal::

    AE working on z of shape (1, 4, 32, 32) = 4096 dimensions.
    >>> model checkpoint loaded.


Convert the model to OpenVINO IR
--------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

Let’s define the conversion function for PyTorch modules. We use
``ov.convert_model`` function to obtain OpenVINO Intermediate
Representation object and ``ov.save_model`` function to save it as XML
file.

.. code:: ipython3

    import torch
    
    import openvino as ov
    
    
    def convert(model: torch.nn.Module, xml_path: str, example_input, input_shape=None):
        xml_path = Path(xml_path)
        if not xml_path.exists():
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                if not input_shape:
                    converted_model = ov.convert_model(model, example_input=example_input)
                else:
                    converted_model = ov.convert_model(model, example_input=example_input, input=input_shape)
            ov.save_model(converted_model, xml_path, compress_to_fp16=False)
    
            # cleanup memory
            torch._C._jit_clear_class_registry()
            torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
            torch.jit._state._clear_class_state()

Flowchart of DynamiCrafter proposed in `the
paper <https://arxiv.org/abs/2310.12190>`__:

|schema| Description: > During inference, our model can generate
animation clips from noise conditioned on the input still image.

Let’s convert models from the pipeline one by one.

.. |schema| image:: https://github.com/openvinotoolkit/openvino_notebooks/assets/76171391/d1033876-c664-4345-a254-0649edbf1906

Convert CLIP text encoder
~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    from dynamicrafter.lvdm.modules.encoders.condition import FrozenOpenCLIPEmbedder
    
    
    COND_STAGE_MODEL_OV_PATH = Path("models/cond_stage_model.xml")
    
    
    class FrozenOpenCLIPEmbedderWrapper(FrozenOpenCLIPEmbedder):
    
        def forward(self, tokens):
            z = self.encode_with_transformer(tokens.to(self.device))
            return z
    
    
    cond_stage_model = FrozenOpenCLIPEmbedderWrapper(device="cpu")
    
    
    convert(
        cond_stage_model,
        COND_STAGE_MODEL_OV_PATH,
        example_input=torch.ones([1, 77], dtype=torch.long),
    )

Convert CLIP image encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__
``FrozenOpenCLIPImageEmbedderV2`` model accepts images of various
resolutions.

.. code:: ipython3

    EMBEDDER_OV_PATH = Path("models/embedder_ir.xml")
    
    
    dummy_input = torch.rand([1, 3, 767, 767], dtype=torch.float32)
    
    model.embedder.model.visual.input_patchnorm = None  # fix error: visual model has not  attribute 'input_patchnorm'
    convert(model.embedder, EMBEDDER_OV_PATH, example_input=dummy_input, input_shape=[1, 3, -1, -1])


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/utils/image.py:226: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if input.numel() == 0:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/geometry/transform/affwarp.py:573: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if size == input_size:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/geometry/transform/affwarp.py:579: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      antialias = antialias and (max(factors) > 1)
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/geometry/transform/affwarp.py:581: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if antialias:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/geometry/transform/affwarp.py:584: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      sigmas = (max((factors[0] - 1.0) / 2.0, 0.001), max((factors[1] - 1.0) / 2.0, 0.001))
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/geometry/transform/affwarp.py:589: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/geometry/transform/affwarp.py:589: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/filters/gaussian.py:55: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      sigma = tensor([sigma], device=input.device, dtype=input.dtype)
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/filters/gaussian.py:55: TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      sigma = tensor([sigma], device=input.device, dtype=input.dtype)
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/core/check.py:78: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if x_shape_to_check[i] != dim:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/filters/kernels.py:92: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      mean = tensor([[mean]], device=sigma.device, dtype=sigma.dtype)
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/enhance/normalize.py:101: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if len(mean.shape) == 0 or mean.shape[0] == 1:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/enhance/normalize.py:103: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if len(std.shape) == 0 or std.shape[0] == 1:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/enhance/normalize.py:107: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if mean.shape and mean.shape[0] != 1:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/enhance/normalize.py:108: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if mean.shape[0] != data.shape[1] and mean.shape[:2] != data.shape[:2]:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/enhance/normalize.py:112: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if std.shape and std.shape[0] != 1:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/enhance/normalize.py:113: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if std.shape[0] != data.shape[1] and std.shape[:2] != data.shape[:2]:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/enhance/normalize.py:116: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      mean = torch.as_tensor(mean, device=data.device, dtype=data.dtype)
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/kornia/enhance/normalize.py:117: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      std = torch.as_tensor(std, device=data.device, dtype=data.dtype)


Convert AE encoder
~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    ENCODER_FIRST_STAGE_OV_PATH = Path("models/encode_first_stage_ir.xml")
    
    
    dummy_input = torch.rand([1, 3, 256, 256], dtype=torch.float32)
    
    convert(
        model.first_stage_model.encoder,
        ENCODER_FIRST_STAGE_OV_PATH,
        example_input=dummy_input,
    )


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/notebooks/dynamicrafter-animating-images/dynamicrafter/lvdm/modules/networks/ae_modules.py:67: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      w_ = w_ * (int(c)**(-0.5))


Convert Diffusion U-Net model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    MODEL_OV_PATH = Path("models/model_ir.xml")
    
    
    class ModelWrapper(torch.nn.Module):
        def __init__(self, diffusion_model):
            super().__init__()
            self.diffusion_model = diffusion_model
    
        def forward(self, xc, t, context=None, fs=None, temporal_length=None):
            outputs = self.diffusion_model(xc, t, context=context, fs=fs, temporal_length=temporal_length)
            return outputs
    
    
    convert(
        ModelWrapper(model.model.diffusion_model),
        MODEL_OV_PATH,
        example_input={
            "xc": torch.rand([1, 8, 16, 32, 32], dtype=torch.float32),
            "t": torch.tensor([1]),
            "context": torch.rand([1, 333, 1024], dtype=torch.float32),
            "fs": torch.tensor([3]),
            "temporal_length": torch.tensor([16]),
        },
    )


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/notebooks/dynamicrafter-animating-images/dynamicrafter/lvdm/modules/networks/openaimodel3d.py:556: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if l_context == 77 + t*16: ## !!! HARD CODE here
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/notebooks/dynamicrafter-animating-images/dynamicrafter/lvdm/modules/networks/openaimodel3d.py:205: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if batch_size:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/notebooks/dynamicrafter-animating-images/dynamicrafter/lvdm/modules/networks/openaimodel3d.py:232: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if self.use_temporal_conv and batch_size:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/notebooks/dynamicrafter-animating-images/dynamicrafter/lvdm/modules/networks/openaimodel3d.py:76: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert x.shape[1] == self.channels
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/notebooks/dynamicrafter-animating-images/dynamicrafter/lvdm/modules/networks/openaimodel3d.py:99: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert x.shape[1] == self.channels


Convert AE decoder
~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__ ``Decoder`` receives a
``bfloat16`` tensor. numpy doesn’t support this type. To avoid problems
with the conversion lets replace ``decode`` method to convert bfloat16
to float32.

.. code:: ipython3

    import types
    
    
    def decode(self, z, **kwargs):
        z = self.post_quant_conv(z)
        z = z.float()
        dec = self.decoder(z)
        return dec
    
    
    model.first_stage_model.decode = types.MethodType(decode, model.first_stage_model)

.. code:: ipython3

    DECODER_FIRST_STAGE_OV_PATH = Path("models/decoder_first_stage_ir.xml")
    
    
    dummy_input = torch.rand([16, 4, 32, 32], dtype=torch.float32)
    
    convert(
        model.first_stage_model.decoder,
        DECODER_FIRST_STAGE_OV_PATH,
        example_input=dummy_input,
    )

Compiling models
----------------

`back to top ⬆️ <#Table-of-contents:>`__

Select device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    import ipywidgets as widgets
    
    core = ov.Core()
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

    compiled_cond_stage_model = core.compile_model(COND_STAGE_MODEL_OV_PATH, device.value)
    compiled_encode_first_stage = core.compile_model(ENCODER_FIRST_STAGE_OV_PATH, device.value)
    compiled_embedder = core.compile_model(EMBEDDER_OV_PATH, device.value)
    compiled_model = core.compile_model(MODEL_OV_PATH, device.value)
    compiled_decoder_first_stage = core.compile_model(DECODER_FIRST_STAGE_OV_PATH, device.value)

Building the pipeline
---------------------

`back to top ⬆️ <#Table-of-contents:>`__

Let’s create callable wrapper classes for compiled models to allow
interaction with original pipelines. Note that all of wrapper classes
return ``torch.Tensor``\ s instead of ``np.array``\ s.

.. code:: ipython3

    import open_clip
    
    
    class CondStageModelWrapper(torch.nn.Module):
        def __init__(self, cond_stage_model):
            super().__init__()
            self.cond_stage_model = cond_stage_model
    
        def encode(self, tokens):
            if isinstance(tokens, list):
                tokens = open_clip.tokenize(tokens[0])
            outs = self.cond_stage_model(tokens)[0]
    
            return torch.from_numpy(outs)
    
    
    class EncoderFirstStageModelWrapper(torch.nn.Module):
        def __init__(self, encode_first_stage):
            super().__init__()
            self.encode_first_stage = encode_first_stage
    
        def forward(self, x):
            outs = self.encode_first_stage(x)[0]
    
            return torch.from_numpy(outs)
    
    
    class EmbedderWrapper(torch.nn.Module):
        def __init__(self, embedder):
            super().__init__()
            self.embedder = embedder
    
        def forward(self, x):
            outs = self.embedder(x)[0]
    
            return torch.from_numpy(outs)
    
    
    class CModelWrapper(torch.nn.Module):
        def __init__(self, diffusion_model, out_channels):
            super().__init__()
            self.diffusion_model = diffusion_model
            self.out_channels = out_channels
    
        def forward(self, xc, t, context, fs, temporal_length):
            inputs = {
                "xc": xc,
                "t": t,
                "context": context,
                "fs": fs,
            }
            outs = self.diffusion_model(inputs)[0]
    
            return torch.from_numpy(outs)
    
    
    class DecoderFirstStageModelWrapper(torch.nn.Module):
        def __init__(self, decoder_first_stage):
            super().__init__()
            self.decoder_first_stage = decoder_first_stage
    
        def forward(self, x):
            x.float()
            outs = self.decoder_first_stage(x)[0]
    
            return torch.from_numpy(outs)

And insert wrappers instances in the pipeline:

.. code:: ipython3

    model.cond_stage_model = CondStageModelWrapper(compiled_cond_stage_model)
    model.first_stage_model.encoder = EncoderFirstStageModelWrapper(compiled_encode_first_stage)
    model.embedder = EmbedderWrapper(compiled_embedder)
    model.model.diffusion_model = CModelWrapper(compiled_model, model.model.diffusion_model.out_channels)
    model.first_stage_model.decoder = DecoderFirstStageModelWrapper(compiled_decoder_first_stage)

Interactive inference
---------------------

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    import time
    
    from einops import repeat
    from pytorch_lightning import seed_everything
    import torchvision.transforms as transforms
    
    from dynamicrafter.scripts.evaluation.funcs import save_videos, batch_ddim_sampling, get_latent_z
    from lvdm.models.samplers.ddim import DDIMSampler
    
    
    def register_buffer(self, name, attr):
        if isinstance(attr, torch.Tensor):
            if attr.device != torch.device("cpu"):
                attr = attr.to(torch.device("cpu"))
        setattr(self, name, attr)
    
    
    # monkey patching to replace the original method 'register_buffer' that uses CUDA
    DDIMSampler.register_buffer = types.MethodType(register_buffer, DDIMSampler)
    
    
    def get_image(image, prompt, steps=5, cfg_scale=7.5, eta=1.0, fs=3, seed=123, model=model):
        result_dir = "results"
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
    
        seed_everything(seed)
        transform = transforms.Compose(
            [
                transforms.Resize(min((256, 256))),
                transforms.CenterCrop((256, 256)),
            ]
        )
        # torch.cuda.empty_cache()
        print("start:", prompt, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        start = time.time()
        if steps > 60:
            steps = 60
        model = model.cpu()
        batch_size = 1
        channels = model.model.diffusion_model.out_channels
        frames = model.temporal_length
        h, w = 256 // 8, 256 // 8
        noise_shape = [batch_size, channels, frames, h, w]
    
        # text cond
        with torch.no_grad(), torch.cpu.amp.autocast():
            text_emb = model.get_learned_conditioning([prompt])
    
            # img cond
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(model.device)
            img_tensor = (img_tensor / 255.0 - 0.5) * 2
    
            image_tensor_resized = transform(img_tensor)  # 3,h,w
            videos = image_tensor_resized.unsqueeze(0)  # bchw
    
            z = get_latent_z(model, videos.unsqueeze(2))  # bc,1,hw
    
            img_tensor_repeat = repeat(z, "b c t h w -> b c (repeat t) h w", repeat=frames)
    
            cond_images = model.embedder(img_tensor.unsqueeze(0))  # blc
    
            img_emb = model.image_proj_model(cond_images)
    
            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
    
            fs = torch.tensor([fs], dtype=torch.long, device=model.device)
            cond = {"c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat]}
    
            ## inference
            batch_samples = batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale)
            ## b,samples,c,t,h,w
            prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
            prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
            prompt_str = prompt_str[:40]
            if len(prompt_str) == 0:
                prompt_str = "empty_prompt"
    
        save_videos(batch_samples, result_dir, filenames=[prompt_str], fps=8)
        print(f"Saved in {prompt_str}. Time used: {(time.time() - start):.2f} seconds")
    
        return os.path.join(result_dir, f"{prompt_str}.mp4")

.. code:: ipython3

    import gradio as gr
    
    
    i2v_examples_256 = [
        ["dynamicrafter/prompts/256/art.png", "man fishing in a boat at sunset", 50, 7.5, 1.0, 3, 234],
        ["dynamicrafter/prompts/256/boy.png", "boy walking on the street", 50, 7.5, 1.0, 3, 125],
        ["dynamicrafter/prompts/256/dance1.jpeg", "two people dancing", 50, 7.5, 1.0, 3, 116],
        ["dynamicrafter/prompts/256/fire_and_beach.jpg", "a campfire on the beach and the ocean waves in the background", 50, 7.5, 1.0, 3, 111],
        ["dynamicrafter/prompts/256/guitar0.jpeg", "bear playing guitar happily, snowing", 50, 7.5, 1.0, 3, 122],
    ]
    
    
    def dynamicrafter_demo():
        css = """#input_img {max-width: 256px !important} #output_vid {max-width: 256px; max-height: 256px}"""
    
        with gr.Blocks(analytics_enabled=False, css=css) as dynamicrafter_iface:
            with gr.Tab(label="Image2Video_256x256"):
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                i2v_input_image = gr.Image(label="Input Image", elem_id="input_img")
                            with gr.Row():
                                i2v_input_text = gr.Text(label="Prompts")
                            with gr.Row():
                                i2v_seed = gr.Slider(label="Random Seed", minimum=0, maximum=10000, step=1, value=123)
                                i2v_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="ETA", value=1.0, elem_id="i2v_eta")
                                i2v_cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label="CFG Scale", value=7.5, elem_id="i2v_cfg_scale")
                            with gr.Row():
                                i2v_steps = gr.Slider(minimum=1, maximum=60, step=1, elem_id="i2v_steps", label="Sampling steps", value=50)
                                i2v_motion = gr.Slider(minimum=1, maximum=4, step=1, elem_id="i2v_motion", label="Motion magnitude", value=3)
                            i2v_end_btn = gr.Button("Generate")
                        with gr.Row():
                            i2v_output_video = gr.Video(label="Generated Video", elem_id="output_vid", autoplay=True, show_share_button=True)
    
                    gr.Examples(
                        examples=i2v_examples_256,
                        inputs=[i2v_input_image, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_eta, i2v_motion, i2v_seed],
                        outputs=[i2v_output_video],
                        fn=get_image,
                        cache_examples=False,
                    )
                i2v_end_btn.click(
                    inputs=[i2v_input_image, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_eta, i2v_motion, i2v_seed],
                    outputs=[i2v_output_video],
                    fn=get_image,
                )
    
        return dynamicrafter_iface
    
    
    demo = dynamicrafter_demo()
    
    
    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(debug=False, share=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.



.. raw:: html

    <div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>

