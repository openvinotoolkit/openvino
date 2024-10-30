Visual-language assistant with Pixtral and OpenVINO
===================================================

Pixtral-12b is multimodal model that consists of 12B parameter
multimodal decoder based on Mistral Nemo and 400M parameter vision
encoder trained from scratch. It is trained to understand both natural
images and documents. The model shows strong abilities in tasks such as
chart and figure understanding, document question answering, multimodal
reasoning and instruction following. Pixtral is able to ingest images at
their natural resolution and aspect ratio, giving the user flexibility
on the number of tokens used to process an image. Pixtral is also able
to process any number of images in its long context window of 128K
tokens. Unlike previous open-source models, Pixtral does not compromise
on text benchmark performance to excel in multimodal tasks.

|image0|

More details about model are available in `blog
post <https://mistral.ai/news/pixtral-12b/>`__ and `model
card <https://huggingface.co/mistralai/Pixtral-12B-2409>`__

In this tutorial we consider how to convert, optimize and run this model
using OpenVINO.

.. warning::

   Important note: Please take into account that pixtral is large model.
   Its conversion requires at least 50GB disk space available


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Convert and Optimize model <#convert-and-optimize-model>`__
-  `Run model inference <#run-model-inference>`__

   -  `Select inference device <#select-inference-device>`__
   -  `Initialize inference pipeline <#initialize-inference-pipeline>`__

-  `Interactive demo <#interactive-demo>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. |image0| image:: https://mistral.ai/images/news/pixtral-12b/pixtral-model-architecture.png

Prerequisites
-------------



.. code:: ipython3

    %pip install -q "torch>=2.1" torchvision "pillow" "tqdm" "gradio>=4.36"  --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "git+https://github.com/huggingface/optimum-intel.git"  --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "nncf>=2.13.0" "openvino>=2024.4"
    %pip install -q "transformers>=4.45.0"  --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    mobileclip 0.1.0 requires torchvision==0.14.1, but you have torchvision 0.17.2+cpu which is incompatible.
    parler-tts 0.2 requires transformers<=4.43.3,>=4.43.0, but you have transformers 4.45.2 which is incompatible.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    from pathlib import Path
    import requests
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w").write(r.text)
    
    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/refs/heads/latest/notebooks/pixtral/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)

Convert and Optimize model
--------------------------



For convenience, we will use OpenVINO integration with HuggingFace
Optimum. `Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__ is the
interface between the Transformers and Diffusers libraries and the
different tools and libraries provided by Intel to accelerate end-to-end
pipelines on Intel architectures.

Among other use cases, Optimum Intel provides a simple interface to
optimize your Transformers and Diffusers models, convert them to the
OpenVINO Intermediate Representation (IR) format and run inference using
OpenVINO Runtime. ``optimum-cli`` provides command line interface for
model conversion and optimization.

General command format:

.. code:: bash

   optimum-cli export openvino --model <model_id_or_path> --task <task> <output_dir>

where task is task to export the model for, if not specified, the task
will be auto-inferred based on the model. You can find a mapping between
tasks and model classes in Optimum TaskManager
`documentation <https://huggingface.co/docs/optimum/exporters/task_manager>`__.
Additionally, you can specify weights compression using
``--weight-format`` argument with one of following options: ``fp32``,
``fp16``, ``int8`` and ``int4``. Fro int8 and int4
`nncf <https://github.com/openvinotoolkit/nncf>`__ will be used for
weight compression. More details about model export provided in `Optimum
Intel
documentation <https://huggingface.co/docs/optimum/intel/openvino/export#export-your-model>`__.

.. code:: ipython3

    import ipywidgets as widgets
    
    model_base_dir = Path("pixtral-12b")
    
    precisions = ["FP16", "INT8", "INT4"]
    
    precision_selector = widgets.Dropdown(description="compression", options=precisions, value=precisions[-1])
    
    precision_selector




.. parsed-literal::

    Dropdown(description='compression', index=2, options=('FP16', 'INT8', 'INT4'), value='INT4')



.. code:: ipython3

    model_dir = model_base_dir / precision_selector.value
    
    if not (model_dir / "openvino_language_model.xml").exists():
        !optimum-cli export openvino -m "mistral-community/pixtral-12b" --weight-format {precision_selector.value.lower()} {model_dir}


.. parsed-literal::

    2024-10-23 03:13:53.907119: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-23 03:13:53.940806: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-23 03:13:54.509755: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/vq_model.py:20: FutureWarning: `VQEncoderOutput` is deprecated and will be removed in version 0.31. Importing `VQEncoderOutput` from `diffusers.models.vq_model` is deprecated and this will be removed in a future version. Please use `from diffusers.models.autoencoders.vq_model import VQEncoderOutput`, instead.
      deprecate("VQEncoderOutput", "0.31", deprecation_message)
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/vq_model.py:25: FutureWarning: `VQModel` is deprecated and will be removed in version 0.31. Importing `VQModel` from `diffusers.models.vq_model` is deprecated and this will be removed in a future version. Please use `from diffusers.models.autoencoders.vq_model import VQModel`, instead.
      deprecate("VQModel", "0.31", deprecation_message)
    Loading checkpoint shards: 100%|██████████████████| 6/6 [00:01<00:00,  3.46it/s]
    We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class (https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/cache_utils.py:447: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
      or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/cache_utils.py:432: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
      elif len(self.key_cache[layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
    Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
    [ WARNING ] Unexpectedly found already patched module language_model.model.embed_tokens while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.0.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.0.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.0.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.0.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.0.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.0.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.0.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.1.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.1.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.1.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.1.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.1.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.1.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.1.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.2.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.2.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.2.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.2.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.2.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.2.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.2.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.3.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.3.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.3.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.3.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.3.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.3.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.3.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.4.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.4.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.4.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.4.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.4.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.4.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.4.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.5.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.5.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.5.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.5.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.5.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.5.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.5.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.6.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.6.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.6.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.6.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.6.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.6.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.6.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.7.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.7.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.7.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.7.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.7.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.7.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.7.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.8.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.8.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.8.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.8.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.8.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.8.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.8.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.9.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.9.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.9.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.9.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.9.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.9.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.9.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.10.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.10.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.10.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.10.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.10.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.10.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.10.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.11.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.11.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.11.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.11.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.11.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.11.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.11.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.12.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.12.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.12.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.12.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.12.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.12.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.12.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.13.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.13.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.13.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.13.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.13.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.13.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.13.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.14.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.14.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.14.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.14.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.14.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.14.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.14.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.15.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.15.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.15.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.15.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.15.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.15.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.15.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.16.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.16.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.16.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.16.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.16.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.16.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.16.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.17.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.17.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.17.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.17.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.17.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.17.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.17.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.18.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.18.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.18.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.18.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.18.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.18.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.18.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.19.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.19.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.19.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.19.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.19.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.19.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.19.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.20.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.20.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.20.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.20.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.20.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.20.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.20.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.21.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.21.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.21.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.21.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.21.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.21.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.21.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.22.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.22.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.22.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.22.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.22.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.22.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.22.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.23.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.23.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.23.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.23.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.23.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.23.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.23.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.24.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.24.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.24.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.24.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.24.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.24.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.24.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.25.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.25.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.25.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.25.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.25.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.25.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.25.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.26.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.26.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.26.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.26.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.26.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.26.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.26.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.27.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.27.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.27.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.27.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.27.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.27.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.27.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.28.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.28.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.28.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.28.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.28.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.28.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.28.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.29.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.29.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.29.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.29.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.29.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.29.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.29.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.30.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.30.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.30.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.30.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.30.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.30.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.30.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.31.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.31.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.31.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.31.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.31.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.31.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.31.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.32.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.32.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.32.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.32.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.32.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.32.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.32.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.33.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.33.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.33.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.33.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.33.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.33.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.33.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.34.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.34.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.34.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.34.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.34.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.34.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.34.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.35.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.35.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.35.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.35.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.35.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.35.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.35.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.36.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.36.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.36.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.36.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.36.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.36.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.36.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.37.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.37.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.37.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.37.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.37.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.37.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.37.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.38.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.38.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.38.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.38.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.38.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.38.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.38.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.39.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.39.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.39.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.39.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.39.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.39.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.model.layers.39.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    [ WARNING ] Unexpectedly found already patched module language_model.lm_head while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/pixtral/modeling_pixtral.py:492: TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
      patch_embeds_list = [self.patch_conv(img.unsqueeze(0).to(self.dtype)) for img in pixel_values]
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/torch/dynamic_graph/wrappers.py:86: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      op1 = operator(\*args, \*\*kwargs)
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/pixtral/modeling_pixtral.py:448: TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
      for start, end in zip(block_start_idx, block_end_idx):
    [ WARNING ] Unexpectedly found already patched module  while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Export model to OpenVINO directly failed with: 
    Config dummy inputs are not a subset of the model inputs: {'input'} vs {'args', 'kwargs'}.
    Model will be exported to ONNX
    Exporting tokenizers to OpenVINO is not supported for tokenizers version > 0.19. Please downgrade to tokenizers version <= 0.19 to export tokenizers to OpenVINO.
    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 6% (1 / 281)                │ 0% (0 / 280)                           │
    ├────────────────┼─────────────────────────────┼────────────────────────────────────────┤
    │              4 │ 94% (280 / 281)             │ 100% (280 / 280)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
    [2KApplying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:05:12 • 0:00:00
    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 6% (3 / 172)                │ 0% (0 / 169)                           │
    ├────────────────┼─────────────────────────────┼────────────────────────────────────────┤
    │              4 │ 94% (169 / 172)             │ 100% (169 / 169)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
    [2KApplying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:12 • 0:00:00
    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (1 / 1)                │ 0% (0 / 0)                             │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
    [2KApplying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:02 • 0:00:00
    

Run model inference
-------------------



Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from notebook_utils import device_widget
    
    device = device_widget(default="CPU", exclude=["NPU"])
    
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



Initialize inference pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



OpenVINO integration with Optimum Intel provides ready-to-use API for
model inference that can be used for smooth integration with
transformers-based solutions. For loading pixtral model, we will use
``OVModelForVisualCausalLM`` class that have compatible interface with
Transformers Pixtral implementation. For loading a model,
``from_pretrained`` method should be used. It accepts path to the model
directory or model_id from HuggingFace hub (if model is not converted to
OpenVINO format, conversion will be triggered automatically).
Additionally, we can provide an inference device, quantization config
(if model has not been quantized yet) and device-specific OpenVINO
Runtime configuration. More details about model inference with Optimum
Intel can be found in
`documentation <https://huggingface.co/docs/optimum/intel/openvino/inference>`__.

.. code:: ipython3

    from transformers import AutoProcessor
    from optimum.intel.openvino import OVModelForVisualCausalLM
    
    processor = AutoProcessor.from_pretrained(model_dir)
    ov_model = OVModelForVisualCausalLM.from_pretrained(model_dir, device=device.value)


.. parsed-literal::

    2024-10-23 03:22:21.803644: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-23 03:22:21.838426: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-23 03:22:22.499374: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. code:: ipython3

    from PIL import Image
    from transformers import TextStreamer
    from gradio_helper import chat_template, resize_with_aspect_ratio
    
    if processor.chat_template is None:
        processor.set_chat_template(chat_template)
    
    question = "What is unusual on this image?"
    
    messages = [
        {"role": "user", "content": [{"type": "text", "content": question}, {"type": "image"}]},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
    raw_image = Image.open(requests.get(url, stream=True).raw)
    
    inputs = processor(text=text, images=[resize_with_aspect_ratio(raw_image)], return_tensors="pt")
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    print(f"Question: {question}")
    display(raw_image)
    output = ov_model.generate(**inputs, do_sample=False, max_new_tokens=100, temperature=None, top_p=None, streamer=streamer)


.. parsed-literal::

    Question: What is unusual on this image?



.. image:: pixtral-with-output_files/pixtral-with-output_12_1.png


.. parsed-literal::

    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.


.. parsed-literal::

    The unusual aspect of this image is that the cat is lying inside a cardboard box, which is not a typical setting for a cat. Cats are often known for their affinity for boxes, but it is still considered unusual to see a cat comfortably resting inside a box in a living room setting. The cat appears relaxed and content, which adds to the charm of the scene. The presence of a sofa in the background further emphasizes the domestic and cozy atmosphere of the image.


Interactive demo
----------------



.. code:: ipython3

    from gradio_helper import make_demo
    
    demo = make_demo(ov_model, processor)
    
    try:
        demo.launch(debug=False)
    except Exception:
        demo.launch(debug=False, share=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.







