Visual-language assistant with nanoLLaVA and OpenVINO
=====================================================

nanoLLaVA is a “small but mighty” 1B vision-language model designed to
run efficiently on edge devices. It uses
`SigLIP-400m <https://huggingface.co/google/siglip-so400m-patch14-384>`__
as Image Encoder and
`Qwen1.5-0.5B <https://huggingface.co/Qwen/Qwen1.5-0.5B>`__ as LLM. In
this tutorial, we consider how to convert and run nanoLLaVA model using
OpenVINO. Additionally, we will optimize model using
`NNCF <https://github.com/openvinotoolkit/nncf>`__


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Select Model <#select-model>`__
-  `Download PyTorch model <#download-pytorch-model>`__
-  `Convert and Optimize model <#convert-and-optimize-model>`__

   -  `Convert model to OpenVINO IR
      format <#convert-model-to-openvino-ir-format>`__
   -  `Compress Model weights to 4 and 8 bits using
      NNCF <#compress-model-weights-to-4-and-8-bits-using-nncf>`__
   -  `Image Encoder <#image-encoder>`__
   -  `Language Model <#language-model>`__

-  `Prepare model inference
   pipeline <#prepare-model-inference-pipeline>`__
-  `Run OpenVINO Model Inference <#run-openvino-model-inference>`__

   -  `Select device <#select-device>`__

-  `Interactive demo <#interactive-demo>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



.. code:: ipython3

    %pip install -q "torch>=2.1" "transformers>=4.40" "accelerate" "pillow" "gradio>=4.26" "tqdm" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "nncf>=2.13"
    %pip install -q -U --pre --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly openvino_tokenizers[transformers] "openvino>=2024.4.0"
    %pip install -q "git+https://github.com/eaidova/optimum-intel.git@ea/minicpmv"


.. parsed-literal::

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    mobileclip 0.1.0 requires torchvision==0.14.1, but you have torchvision 0.17.2+cpu which is incompatible.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    openvino-dev 2024.4.0 requires openvino==2024.4.0, but you have openvino 2024.5.0.dev20241014 which is incompatible.
    openvino-genai 2024.4.0.0 requires openvino_tokenizers~=2024.4.0.0.dev, but you have openvino-tokenizers 2024.5.0.0.dev20241022 which is incompatible.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    from pathlib import Path
    import requests

    helper_file = Path("ov_nano_llava_helper.py")

    if not helper_file.exists():
        r = requests.get(
            url=f"https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/nano-llava-multimodal-chatbot/{helper_file.name}"
        )
        helper_file.open("w").write(r.text)

Select Model
------------



The tutorial supports the following models from Phi-3 model family:

- `nanoLLaVA <https://huggingface.co/qnguyen3/nanoLLaVA>`__
- `nanoLLaVA-1.5 <https://huggingface.co/qnguyen3/nanoLLaVA-1.5>`__

You can select one from the provided options below.

.. code:: ipython3

    import ipywidgets as widgets

    model_ids = ["qnguyen3/nanoLLaVA", "qnguyen3/nanoLLaVA-1.5"]

    model_dropdown = widgets.Dropdown(
        options=model_ids,
        value=model_ids[0],
        description="Model:",
        disabled=False,
    )

    model_dropdown




.. parsed-literal::

    Dropdown(description='Model:', options=('qnguyen3/nanoLLaVA', 'qnguyen3/nanoLLaVA-1.5'), value='qnguyen3/nanoL…



Download PyTorch model
----------------------



.. code:: ipython3

    from ov_nano_llava_helper import download_original_model, converted_model_exists, copy_model_files

    model_id = model_dropdown.value
    model_dir = Path(model_id.split("/")[-1])
    ov_model_dir = Path("ov_" + model_dir.name) / "FP16"

    if not converted_model_exists(ov_model_dir):
        download_original_model(model_id, model_dir)



.. parsed-literal::

    Fetching 14 files:   0%|          | 0/14 [00:00<?, ?it/s]



.. parsed-literal::

    example_1.png:   0%|          | 0.00/200k [00:00<?, ?B/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/3.58k [00:00<?, ?B/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.52k [00:00<?, ?B/s]


Convert and Optimize model
--------------------------



Our model conversion and optimization consist of following steps: 1.
Convert model to OpenVINO format and save it on disk. 2. Compress model
weights using NNCF

Let’s consider each step deeply.

Convert model to OpenVINO IR format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



NanoLLaVA implementation is based on `HuggingFace
Transformers <https://huggingface.co/docs/transformers/index>`__
library. For convenience, we will use OpenVINO integration with
HuggingFace Optimum. `Optimum
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

    if not converted_model_exists(ov_model_dir):
        !optimum-cli export openvino --model {model_id} --task image-text-to-text --trust-remote-code --weight-format fp16 {ov_model_dir}


.. parsed-literal::

    2024-10-23 02:04:00.682228: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-23 02:04:00.715051: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-23 02:04:01.329449: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    Some weights of the model checkpoint at qnguyen3/nanoLLaVA were not used when initializing LlavaQwen2ForCausalLM: ['model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.bias', 'model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight', 'model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.1.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.1.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.1.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.1.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.1.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.1.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.1.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.1.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.1.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.1.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.1.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.1.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.1.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.1.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.1.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.1.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.11.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.11.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.11.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.11.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.11.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.11.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.11.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.11.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.11.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.11.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.11.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.11.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.11.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.11.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.11.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.11.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.12.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.12.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.12.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.12.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.12.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.12.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.12.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.12.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.12.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.12.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.12.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.12.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.12.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.12.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.12.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.12.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.13.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.13.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.13.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.13.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.13.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.13.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.13.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.13.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.13.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.13.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.13.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.13.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.13.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.13.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.13.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.13.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.14.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.14.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.14.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.14.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.14.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.14.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.14.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.14.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.14.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.14.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.14.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.14.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.14.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.14.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.14.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.14.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.15.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.15.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.15.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.15.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.15.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.15.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.15.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.15.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.15.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.15.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.15.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.15.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.15.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.15.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.15.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.15.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.16.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.16.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.16.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.16.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.16.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.16.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.16.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.16.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.16.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.16.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.16.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.16.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.16.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.16.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.16.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.16.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.17.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.17.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.17.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.17.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.17.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.17.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.17.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.17.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.17.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.17.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.17.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.17.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.17.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.17.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.17.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.17.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.18.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.18.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.18.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.18.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.18.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.18.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.18.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.18.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.18.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.18.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.18.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.18.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.18.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.18.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.18.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.18.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.19.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.19.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.19.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.19.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.19.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.19.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.19.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.19.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.19.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.19.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.19.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.19.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.19.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.19.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.19.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.19.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.2.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.2.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.2.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.2.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.2.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.2.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.2.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.2.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.2.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.2.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.2.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.2.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.2.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.2.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.2.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.2.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.20.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.20.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.20.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.20.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.20.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.20.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.20.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.20.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.20.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.20.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.20.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.20.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.20.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.20.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.20.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.20.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.21.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.21.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.21.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.21.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.21.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.21.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.21.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.21.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.21.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.21.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.21.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.21.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.21.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.21.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.21.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.21.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.22.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.22.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.22.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.22.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.22.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.22.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.22.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.22.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.22.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.22.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.22.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.22.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.22.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.22.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.22.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.22.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.24.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.24.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.24.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.24.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.24.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.24.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.24.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.24.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.24.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.24.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.24.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.24.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.24.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.24.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.24.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.24.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.25.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.25.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.25.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.25.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.25.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.25.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.25.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.25.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.25.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.25.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.25.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.25.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.25.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.25.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.25.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.25.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.26.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.26.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.26.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.26.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.26.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.26.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.26.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.26.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.26.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.26.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.26.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.26.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.26.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.26.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.26.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.26.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.3.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.3.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.3.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.3.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.3.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.3.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.3.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.3.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.3.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.3.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.3.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.3.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.3.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.3.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.3.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.3.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.4.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.4.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.4.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.4.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.4.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.4.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.4.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.4.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.4.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.4.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.4.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.4.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.4.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.4.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.4.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.4.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.5.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.5.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.5.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.5.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.5.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.5.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.5.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.5.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.5.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.5.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.5.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.5.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.5.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.5.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.5.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.5.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.6.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.6.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.6.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.6.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.6.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.6.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.6.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.6.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.6.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.6.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.6.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.6.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.6.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.6.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.6.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.6.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.7.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.7.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.7.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.7.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.7.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.7.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.7.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.7.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.7.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.7.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.7.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.7.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.7.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.7.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.7.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.7.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.8.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.8.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.8.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.8.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.8.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.8.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.8.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.8.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.8.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.8.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.8.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.8.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.8.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.8.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.8.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.8.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.9.layer_norm1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.9.layer_norm1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.9.layer_norm2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.9.layer_norm2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.9.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.9.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.9.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.9.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.9.self_attn.k_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.9.self_attn.k_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.9.self_attn.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.9.self_attn.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.9.self_attn.q_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.9.self_attn.q_proj.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.9.self_attn.v_proj.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.9.self_attn.v_proj.weight', 'model.vision_tower.vision_tower.vision_model.head.attention.in_proj_bias', 'model.vision_tower.vision_tower.vision_model.head.attention.in_proj_weight', 'model.vision_tower.vision_tower.vision_model.head.attention.out_proj.bias', 'model.vision_tower.vision_tower.vision_model.head.attention.out_proj.weight', 'model.vision_tower.vision_tower.vision_model.head.layernorm.bias', 'model.vision_tower.vision_tower.vision_model.head.layernorm.weight', 'model.vision_tower.vision_tower.vision_model.head.mlp.fc1.bias', 'model.vision_tower.vision_tower.vision_model.head.mlp.fc1.weight', 'model.vision_tower.vision_tower.vision_model.head.mlp.fc2.bias', 'model.vision_tower.vision_tower.vision_model.head.mlp.fc2.weight', 'model.vision_tower.vision_tower.vision_model.head.probe', 'model.vision_tower.vision_tower.vision_model.post_layernorm.bias', 'model.vision_tower.vision_tower.vision_model.post_layernorm.weight']
    - This IS expected if you are initializing LlavaQwen2ForCausalLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing LlavaQwen2ForCausalLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/qnguyen3/nanoLLaVA/1ea99cffcf50a27c5f06fe5d22a07046aba0bffe/modeling_llava_qwen2.py:169: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/qnguyen3/nanoLLaVA/1ea99cffcf50a27c5f06fe5d22a07046aba0bffe/modeling_llava_qwen2.py:187: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
    Unexpectedly found already patched module model.embed_tokens while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.0.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.0.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.0.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.0.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.0.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.0.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.0.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.1.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.1.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.1.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.1.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.1.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.1.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.1.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.2.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.2.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.2.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.2.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.2.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.2.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.2.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.3.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.3.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.3.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.3.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.3.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.3.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.3.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.4.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.4.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.4.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.4.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.4.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.4.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.4.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.5.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.5.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.5.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.5.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.5.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.5.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.5.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.6.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.6.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.6.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.6.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.6.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.6.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.6.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.7.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.7.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.7.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.7.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.7.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.7.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.7.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.8.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.8.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.8.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.8.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.8.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.8.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.8.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.9.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.9.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.9.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.9.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.9.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.9.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.9.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.10.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.10.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.10.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.10.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.10.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.10.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.10.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.11.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.11.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.11.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.11.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.11.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.11.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.11.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.12.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.12.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.12.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.12.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.12.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.12.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.12.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.13.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.13.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.13.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.13.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.13.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.13.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.13.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.14.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.14.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.14.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.14.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.14.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.14.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.14.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.15.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.15.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.15.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.15.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.15.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.15.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.15.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.16.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.16.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.16.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.16.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.16.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.16.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.16.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.17.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.17.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.17.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.17.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.17.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.17.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.17.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.18.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.18.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.18.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.18.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.18.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.18.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.18.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.19.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.19.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.19.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.19.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.19.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.19.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.19.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.20.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.20.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.20.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.20.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.20.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.20.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.20.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.21.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.21.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.21.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.21.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.21.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.21.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.21.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.22.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.22.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.22.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.22.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.22.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.22.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.22.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.23.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.23.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.23.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.23.self_attn.o_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.23.mlp.gate_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.23.mlp.up_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.layers.23.mlp.down_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.embeddings.position_embedding while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.0.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.0.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.1.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.1.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.1.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.1.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.1.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.1.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.2.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.2.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.2.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.2.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.2.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.2.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.3.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.3.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.3.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.3.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.3.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.3.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.4.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.4.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.4.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.4.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.4.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.4.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.5.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.5.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.5.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.5.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.5.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.5.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.6.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.6.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.6.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.6.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.6.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.6.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.7.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.7.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.7.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.7.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.7.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.7.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.8.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.8.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.8.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.8.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.8.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.8.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.9.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.9.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.9.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.9.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.9.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.9.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.10.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.10.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.10.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.10.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.10.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.10.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.11.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.11.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.11.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.11.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.11.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.11.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.12.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.12.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.12.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.12.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.12.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.12.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.13.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.13.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.13.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.13.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.13.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.13.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.14.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.14.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.14.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.14.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.14.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.14.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.15.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.15.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.15.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.15.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.15.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.15.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.16.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.16.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.16.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.16.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.16.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.16.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.17.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.17.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.17.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.17.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.17.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.17.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.18.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.18.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.18.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.18.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.18.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.18.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.19.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.19.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.19.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.19.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.19.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.19.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.20.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.20.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.20.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.20.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.20.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.20.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.21.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.21.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.21.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.21.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.21.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.21.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.22.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.22.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.22.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.22.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.22.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.22.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.23.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.23.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.24.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.24.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.24.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.24.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.24.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.24.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.25.self_attn.k_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.25.self_attn.v_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.25.self_attn.q_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.25.self_attn.out_proj while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.25.mlp.fc1 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.vision_tower.vision_tower.vision_model.encoder.layers.25.mlp.fc2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.mm_projector.0 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module model.mm_projector.2 while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    Unexpectedly found already patched module lm_head while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/cache_utils.py:447: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
      or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:116: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/optimum/exporters/onnx/model_patcher.py:307: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if past_key_values_length > 0:
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/qnguyen3/nanoLLaVA/1ea99cffcf50a27c5f06fe5d22a07046aba0bffe/modeling_llava_qwen2.py:939: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if seq_len > self.max_seq_len_cached:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/cache_utils.py:432: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
      elif len(self.key_cache[layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/qnguyen3/nanoLLaVA/1ea99cffcf50a27c5f06fe5d22a07046aba0bffe/modeling_llava_qwen2.py:1499: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
    Unexpectedly found already patched module  while applying ModuleExtension during PyTorch model conversion. Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.
    OpenVINO and OpenVINO Tokenizers versions are not binary compatible.
    OpenVINO version:            2024.5.0-16993
    OpenVINO Tokenizers version: 2024.5.0.0
    First 3 numbers should be the same. Update OpenVINO Tokenizers to compatible version. It is recommended to use the same day builds for pre-release version. To install both OpenVINO and OpenVINO Tokenizers release version perform:
    pip install --force-reinstall openvino openvino-tokenizers
    To update both OpenVINO and OpenVINO Tokenizers to the latest pre-release version perform:
    pip install --pre -U openvino openvino-tokenizers --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    Tokenizer won't be converted.
    Traceback (most recent call last):
      File "/opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/bin/optimum-cli", line 10, in <module>
        sys.exit(main())
      File "/opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/optimum/commands/optimum_cli.py", line 208, in main
        service.run()
      File "/opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/optimum/commands/export/openvino.py", line 349, in run
        main_export(
      File "/opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/optimum/exporters/openvino/__main__.py", line 416, in main_export
        core = Core()
      File "/opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino_tokenizers/__init__.py", line 53, in new_core_init
        self.add_extension(str(_ext_path))  # Core.add_extension doesn't support Path object
    RuntimeError: Exception from src/inference/src/cpp/core.cpp:158:
    Cannot add extension. Cannot find entry point to the extension library. This error happened: Cannot load library '/opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino_tokenizers/lib/libopenvino_tokenizers.so': /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino_tokenizers/lib/libopenvino_tokenizers.so: undefined symbol: _ZNK2ov4Node17can_constant_foldERKSt6vectorINS_6OutputIS0_EESaIS3_EE



Compress Model weights to 4 and 8 bits using NNCF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



For reducing memory consumption, weights compression optimization can be
applied using `NNCF <https://github.com/openvinotoolkit/nncf>`__. Weight
compression aims to reduce the memory footprint of a model. It can also
lead to significant performance improvement for large memory-bound
models, such as Large Language Models (LLMs). LLMs and other models,
which require extensive memory to store the weights during inference,
can benefit from weight compression in the following ways:

-  enabling the inference of exceptionally large models that cannot be
   accommodated in the memory of the device;

-  improving the inference performance of the models by reducing the
   latency of the memory access when computing the operations with
   weights, for example, Linear layers.

`Neural Network Compression Framework
(NNCF) <https://github.com/openvinotoolkit/nncf>`__ provides 4-bit /
8-bit mixed weight quantization as a compression method primarily
designed to optimize LLMs. The main difference between weights
compression and full model quantization (post-training quantization) is
that activations remain floating-point in the case of weights
compression which leads to a better accuracy. Weight compression for
LLMs provides a solid inference performance improvement which is on par
with the performance of the full model quantization. In addition, weight
compression is data-free and does not require a calibration dataset,
making it easy to use.

``nncf.compress_weights`` function can be used for performing weights
compression. The function accepts an OpenVINO model and other
compression parameters. Compared to INT8 compression, INT4 compression
improves performance even more, but introduces a minor drop in
prediction quality.

More details about weights compression, can be found in `OpenVINO
documentation <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html>`__.

Please select below whether you would like to run INT4 weight
compression instead of INT8 weight compression.

.. code:: ipython3

    import ipywidgets as widgets

    compression_mode = widgets.Dropdown(
        options=["INT4", "INT8"],
        value="INT4",
        description="Compression mode:",
        disabled=False,
    )

    compression_mode




.. parsed-literal::

    Dropdown(description='Compression mode:', options=('INT4', 'INT8'), value='INT4')



.. code:: ipython3

    import nncf
    import openvino as ov

    core = ov.Core()

    if compression_mode.value == "INT4":
        ov_compressed_model_dir = ov_model_dir.parent / "INT4"
        llava_wc_parameters = dict(mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=128, ratio=0.8)
    else:
        ov_compressed_model_dir = ov_model_dir.parent / "INT8"
        llava_wc_parameters = dict(mode=nncf.CompressWeightsMode.INT8)

    image_encoder_wc_parameters = dict(mode=nncf.CompressWeightsMode.INT8)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Image Encoder
~~~~~~~~~~~~~



Image Encoder is represented in nanoLLaVA by pretrained SigLIP model.
Image encoder is responsible for encoding input images into embedding
space. Code bellow demonstrates how to apply weights compression for
image encoder model.

.. code:: ipython3

    import gc

    compressed_vision_encoder_path = ov_compressed_model_dir / "openvino_vision_embeddings_model.xml"
    vision_encoder_path = ov_model_dir / "openvino_vision_embeddings_model.xml"
    if not compressed_vision_encoder_path.exists():
        ov_vision_encoder = core.read_model(vision_encoder_path)
        ov_compressed_vision_encoder = nncf.compress_weights(ov_vision_encoder, **image_encoder_wc_parameters)
        ov.save_model(ov_compressed_vision_encoder, compressed_vision_encoder_path)
        del ov_compressed_vision_encoder
        del ov_vision_encoder
        gc.collect();


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/quantization/quantize_model.py:432: FutureWarning: `CompressWeightsMode.INT8` is deprecated. Please, use `CompressWeightsMode.INT8_ASYM` as value instead.
      warning_deprecated(
    2024-10-23 02:04:33.280788: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-23 02:04:33.314985: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-23 02:04:33.946816: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (159 / 159)            │ 100% (159 / 159)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









Language Model
~~~~~~~~~~~~~~



Language Model is responsible for generation answer in LLaVA. This part
is very similar to standard LLM for text generation. Our model uses
`Qwen/Qwen1.5-0.5B <https://huggingface.co/Qwen/Qwen1.5-0.5B>`__ as base
LLM.

.. code:: ipython3

    compressed_llm_path = ov_compressed_model_dir / "openvino_language_model.xml"
    llm_path = ov_model_dir / "openvino_language_model.xml"

    if not compressed_llm_path.exists():
        ov_llm = core.read_model(llm_path)
        ov_compressed_llm = nncf.compress_weights(ov_llm, **llava_wc_parameters)
        ov.save_model(ov_compressed_llm, compressed_llm_path)
        del ov_compressed_llm
        del ov_llm
        gc.collect()

    copy_model_files(ov_model_dir, ov_compressed_model_dir)



.. parsed-literal::

    Output()









.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 47% (48 / 169)              │ 20% (47 / 168)                         │
    ├────────────────┼─────────────────────────────┼────────────────────────────────────────┤
    │              4 │ 53% (121 / 169)             │ 80% (121 / 168)                        │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









Prepare model inference pipeline
--------------------------------



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

Run OpenVINO Model Inference
----------------------------



Select device
~~~~~~~~~~~~~



.. code:: ipython3

    import requests

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)

    from notebook_utils import device_widget

    device = device_widget("CPU", exclude=["NPU"])

    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



Optimum Intel provides Transformers-like interface for inference
OpenVINO models that allows smooth integration into user application,
where you need just replace model class, other parts of pipeline -
preprocessing and postprocessing code remains the same. It means that we
can use the same tokenizer and image processor that provided with model.

.. code:: ipython3

    from optimum.intel.openvino import OVModelForVisualCausalLM
    from transformers import AutoConfig, AutoTokenizer, AutoProcessor, TextStreamer

    # prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ov_compressed_model_dir, trust_remote_code=True)

    # prepare image processor
    config = AutoConfig.from_pretrained(ov_compressed_model_dir, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(config.mm_vision_tower)

    # initialize OpenVINO model inference class
    ov_model = OVModelForVisualCausalLM.from_pretrained(ov_compressed_model_dir, device=device.value, trust_remote_code=True)

.. code:: ipython3

    from ov_nano_llava_helper import process_images, process_text_input
    from PIL import Image

    prompt = "Describe this image in detail"

    messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/8bf7d9f2-018a-4498-bec4-55f17c273ecc"
    image = Image.open(requests.get(url, stream=True).raw)
    image_tensor = process_images(image, None, processor)
    input_ids, attention_mask = process_text_input(text, tokenizer)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    display(image)
    print(f"Question:\n{prompt}")
    print("Answer:")

    output_ids = ov_model.generate(input_ids, attention_mask=attention_mask, images=image_tensor, max_new_tokens=128, use_cache=True, streamer=streamer)



.. image:: nano-llava-multimodal-chatbot-with-output_files/nano-llava-multimodal-chatbot-with-output_22_0.png


.. parsed-literal::

    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.


.. parsed-literal::

    Question:
    Describe this image in detail
    Answer:
    The image portrays a charming and playful scene featuring a white lama. This adorable creature has a playful expression, with its eyes sparkling with joy and its nose in a playful smile. It is adorned with cute, tiny eyes that add a playful touch to its face. The lama's ears are also quite noticeable, with one of them sporting a tiny pink button. The lama's body is covered in fluffy, white fur, and it has its hind legs visible, adding a sense of movement to the image.
    The lama is surrounded by a vivid display of fire. The flames are bright and lively, with some areas appearing more intense


Interactive demo
----------------



.. code:: ipython3

    from transformers import TextIteratorStreamer, StoppingCriteria
    from threading import Thread
    import torch


    class KeywordsStoppingCriteria(StoppingCriteria):
        def __init__(self, keywords, tokenizer, input_ids):
            self.keywords = keywords
            self.keyword_ids = []
            self.max_keyword_len = 0
            for keyword in keywords:
                cur_keyword_ids = tokenizer(keyword).input_ids
                if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                    cur_keyword_ids = cur_keyword_ids[1:]
                if len(cur_keyword_ids) > self.max_keyword_len:
                    self.max_keyword_len = len(cur_keyword_ids)
                self.keyword_ids.append(torch.tensor(cur_keyword_ids))
            self.tokenizer = tokenizer
            self.start_len = input_ids.shape[1]

        def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
            self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
            for keyword_id in self.keyword_ids:
                truncated_output_ids = output_ids[0, -keyword_id.shape[0] :]
                if torch.equal(truncated_output_ids, keyword_id):
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
            return False

        def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            outputs = []
            for i in range(output_ids.shape[0]):
                outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
            return all(outputs)


    def bot_streaming(message, history):
        messages = []
        if message["files"]:
            image = message["files"][-1]["path"] if isinstance(message["files"][-1], dict) else message["files"][-1]
        else:
            for _, hist in enumerate(history):
                if isinstance(hist[0], tuple):
                    image = hist[0][0]

        if len(history) > 0 and image is not None:
            messages.append({"role": "user", "content": f"<image>\n{history[1][0]}"})
            messages.append({"role": "assistant", "content": history[1][1]})
            for human, assistant in history[2:]:
                if assistant is None:
                    continue
                messages.append({"role": "user", "content": human})
                messages.append({"role": "assistant", "content": assistant})
            messages.append({"role": "user", "content": message["text"]})
        elif len(history) > 0 and image is None:
            for human, assistant in history:
                if assistant is None:
                    continue
                messages.append({"role": "user", "content": human})
                messages.append({"role": "assistant", "content": assistant})
            messages.append({"role": "user", "content": message["text"]})
        elif len(history) == 0 and image is not None:
            messages.append({"role": "user", "content": f"<image>\n{message['text']}"})
        elif len(history) == 0 and image is None:
            messages.append({"role": "user", "content": message["text"]})

        print(messages)
        image = Image.open(image).convert("RGB")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_tensor = process_images(image, None, processor)
        input_ids, attention_mask = process_text_input(text, tokenizer)
        stop_str = "<|im_end|>"
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            streamer=streamer,
            max_new_tokens=128,
            stopping_criteria=[stopping_criteria],
            temperature=0.01,
        )
        thread = Thread(target=ov_model.generate, kwargs=generation_kwargs)
        thread.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            generated_text_without_prompt = buffer[:]
            yield generated_text_without_prompt

.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/nano-llava-multimodal-chatbot/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)

    from gradio_helper import make_demo

    demo = make_demo(fn=bot_streaming)

    try:
        demo.launch(debug=False)
    except Exception:
        demo.launch(share=True, debug=False)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860

    To create a public link, set `share=True` in `launch()`.







