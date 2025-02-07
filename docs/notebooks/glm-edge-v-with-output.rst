Visual-language assistant with GLM-Edge-V and OpenVINO
------------------------------------------------------

The
`GLM-Edge <https://huggingface.co/collections/THUDM/glm-edge-6743283c5809de4a7b9e0b8b>`__
series is `Zhipu <https://huggingface.co/THUDM>`__\ ’s attempt to meet
real-world deployment scenarios for edge devices. It consists of two
sizes of large language dialogue models and multimodal understanding
models (GLM-Edge-1.5B-Chat, GLM-Edge-4B-Chat, GLM-Edge-V-2B,
GLM-Edge-V-5B). Among them, the 1.5B / 2B models are mainly targeted at
platforms like mobile phones and car machines, while the 4B / 5B models
are aimed at platforms like PCs. Based on the technological advancements
of the GLM-4 series, some targeted adjustments have been made to the
model structure and size, balancing model performance, real-world
inference efficiency, and deployment convenience. Through deep
collaboration with partner enterprises and relentless efforts in
inference optimization, the GLM-Edge series models can run at extremely
high speeds on some edge platforms.

In this tutorial we consider how to launch multimodal model GLM-Edge-V
using OpenVINO for creation multimodal chatbot. Additionally, we
optimize model to low precision using
`NNCF <https://github.com/openvinotoolkit/nncf>`__

**Table of contents:**

-  `Prerequisites <#prerequisites>`__
-  `Select Model <#select-model>`__
-  `Convert and Optimize model <#convert-and-optimize-model>`__

   -  `Compress model weights to
      4-bit <#compress-model-weights-to-4-bit>`__

-  `Select inference device <#select-inference-device>`__
-  `Run OpenVINO model <#run-openvino-model>`__
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



install required packages and setup helper functions.

.. code:: ipython3

    %pip install -q "torch>=2.1" "torchvision" "protobuf>=3.20" "gradio>=4.26" "Pillow" "accelerate" "tqdm"  --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "openvino>=2024.5.0" "nncf>=2.14.0"
    %pip install -q "git+https://github.com/huggingface/transformers"  --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    ERROR: Could not find a version that satisfies the requirement openvino>=2024.5.0 (from versions: 2021.3.0, 2021.4.0, 2021.4.1, 2021.4.2, 2022.1.0, 2022.2.0, 2022.3.0, 2022.3.1, 2022.3.2, 2023.0.0.dev20230119, 2023.0.0.dev20230217, 2023.0.0.dev20230407, 2023.0.0.dev20230427, 2023.0.0, 2023.0.1, 2023.0.2, 2023.1.0.dev20230623, 2023.1.0.dev20230728, 2023.1.0.dev20230811, 2023.1.0, 2023.2.0.dev20230922, 2023.2.0, 2023.3.0, 2024.0.0, 2024.1.0, 2024.2.0, 2024.3.0, 2024.4.0, 2024.4.1.dev20240926)
    ERROR: No matching distribution found for openvino>=2024.5.0
    Note: you may need to restart the kernel to use updated packages.
      error: subprocess-exited-with-error

      × Preparing metadata (pyproject.toml) did not run successfully.
      │ exit code: 1
      ╰─> [6 lines of output]

          Cargo, the Rust package manager, is not installed or is not on PATH.
          This package requires Rust and Cargo to compile extensions. Install it through
          the system's package manager or via https://rustup.rs/

          Checking for Rust toolchain....
          [end of output]

      note: This error originates from a subprocess, and is likely not a problem with pip.
    error: metadata-generation-failed

    × Encountered error while generating package metadata.
    ╰─> See above for output.

    note: This is an issue with the package mentioned above, not pip.
    hint: See above for details.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import requests
    from pathlib import Path

    if not Path("glmv_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/glm-edge-v/glmv_helper.py")
        open("glmv_helper.py", "w").write(r.text)

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/glm-edge-v/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)

    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w").write(r.text)

    # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
    from notebook_utils import collect_telemetry

    collect_telemetry("glm-edge-v.ipynb")

Select Model
------------



The tutorial supports the following models from GLM-Edge-V model family:

- `glm-edge-v-2b <https://huggingface.co/THUDM/glm-edge-v-2b>`__
- `glm-edge-v-5b <https://huggingface.co/THUDM/glm-edge-v-5b>`__

You can select one from the provided options below.

.. code:: ipython3

    import ipywidgets as widgets

    # Select model
    model_ids = [
        "THUDM/glm-edge-v-2b",
        "THUDM/glm-edge-v-5b",
    ]

    model_dropdown = widgets.Dropdown(
        options=model_ids,
        value=model_ids[0],
        description="Model:",
        disabled=False,
    )

    model_dropdown




.. parsed-literal::

    Dropdown(description='Model:', options=('THUDM/glm-edge-v-2b', 'THUDM/glm-edge-v-5b'), value='THUDM/glm-edge-v…



Convert and Optimize model
--------------------------



GLM-Edge-V is PyTorch model. OpenVINO supports PyTorch models via
conversion to OpenVINO Intermediate Representation (IR). `OpenVINO model
conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
should be used for these purposes. ``ov.convert_model`` function accepts
original PyTorch model instance and example input for tracing and
returns ``ov.Model`` representing this model in OpenVINO framework.
Converted model can be used for saving on disk using ``ov.save_model``
function or directly loading on device using ``core.complie_model``.

The script ``glmv_helper.py`` contains helper function for model
conversion, please check its content if you interested in conversion
details.

.. raw:: html

   <details>

Click here for more detailed explanation of conversion steps GLM-Edge-V
is autoregressive transformer generative model, it means that each next
model step depends from model output from previous step. The generation
approach is based on the assumption that the probability distribution of
a word sequence can be decomposed into the product of conditional next
word distributions. In other words, model predicts the next token in the
loop guided by previously generated tokens until the stop-condition will
be not reached (generated sequence of maximum length or end of string
token obtained). The way the next token will be selected over predicted
probabilities is driven by the selected decoding methodology. You can
find more information about the most popular decoding methods in this
blog. The entry point for the generation process for models from the
Hugging Face Transformers library is the ``generate`` method. You can
find more information about its parameters and configuration in the
documentation. To preserve flexibility in the selection decoding
methodology, we will convert only model inference for one step.

GLM-Edge-V model consists of 3 parts:

-  **Vision Model** for encoding input images into embedding space.
-  **Embedding Model** for conversion input text tokens into embedding
   space
-  **Language Model** for generation answer based on input embeddings
   provided by Image Encoder and Input Embedding models.

.. raw:: html

   </details>

Compress model weights to 4-bit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For reducing memory
consumption, weights compression optimization can be applied using
`NNCF <https://github.com/openvinotoolkit/nncf>`__.

.. raw:: html

   <details>

Click here for more details about weight compression Weight compression
aims to reduce the memory footprint of a model. It can also lead to
significant performance improvement for large memory-bound models, such
as Large Language Models (LLMs). LLMs and other models, which require
extensive memory to store the weights during inference, can benefit from
weight compression in the following ways:

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

.. raw:: html

   </details>

.. code:: ipython3

    from pathlib import Path
    import nncf
    from glmv_helper import convert_glmv_model


    model_id = model_dropdown.value
    out_dir = Path("model") / Path(model_id).name / "INT4"
    compression_configuration = {
        "mode": nncf.CompressWeightsMode.INT4_SYM,
        "group_size": 64,
        "ratio": 0.6,
    }
    convert_glmv_model(model_id, out_dir, compression_configuration)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    2025-02-04 02:31:50.386808: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2025-02-04 02:31:50.420435: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2025-02-04 02:31:50.973389: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    ⌛ glm-edge-v-2b conversion started. Be patient, it may takes some time.
    ⌛ Load Original model
    ✅ Original model successfully loaded
    ⌛ Convert Input embedding model
    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/875/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:5006: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.


.. parsed-literal::

    ✅ Input embedding model successfully converted
    ⌛ Convert Image embedding model


.. parsed-literal::

    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/THUDM/glm-edge-v-2b/2053707733f99ab52e943904f43c2359a94301ef/siglip.py:48: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      grid_size = int(s**0.5)
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/THUDM/glm-edge-v-2b/2053707733f99ab52e943904f43c2359a94301ef/siglip.py:53: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
      image_emb = torch.cat([self.boi.repeat(len(image_emb), 1, 1), image_emb, self.eoi.repeat(len(image_emb), 1, 1)], dim=1)


.. parsed-literal::

    ✅ Image embedding model successfully converted
    ⌛ Convert Language model


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/875/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/cache_utils.py:458: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
      or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/THUDM/glm-edge-v-2b/2053707733f99ab52e943904f43c2359a94301ef/modeling_glm.py:1010: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if sequence_length != 1:
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/THUDM/glm-edge-v-2b/2053707733f99ab52e943904f43c2359a94301ef/modeling_glm.py:153: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      rotary_dim = int(q.shape[-1] * partial_rotary_factor)
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/875/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/cache_utils.py:443: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
      elif len(self.key_cache[layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/THUDM/glm-edge-v-2b/2053707733f99ab52e943904f43c2359a94301ef/modeling_glm.py:249: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/875/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/jit/_trace.py:168: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
      if a.grad is not None:


.. parsed-literal::

    ✅ Language model successfully converted
    ⌛ Weights compression with int4_sym mode started



.. parsed-literal::

    Output()









.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 45% (115 / 169)             │ 40% (114 / 168)                        │
    ├────────────────┼─────────────────────────────┼────────────────────────────────────────┤
    │              4 │ 55% (54 / 169)              │ 60% (54 / 168)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









.. parsed-literal::

    ✅ Weights compression finished
    ✅ glm-edge-v-2b model conversion finished. You can find results in model/glm-edge-v-2b/INT4


Select inference device
-----------------------



.. code:: ipython3

    from notebook_utils import device_widget

    device = device_widget(default="AUTO", exclude=["NPU"])

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Run OpenVINO model
------------------



``OvGLMv`` class provides convenient way for running model. It accepts
directory with converted model and inference device as arguments. For
running model we will use ``generate`` method.

.. code:: ipython3

    from glmv_helper import OvGLMv

    model = OvGLMv(out_dir, device.value)

.. code:: ipython3

    import requests
    from PIL import Image

    image_path = Path("cat.png")

    if not image_path.exists():
        url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
        image = Image.open(requests.get(url, stream=True).raw)
        image.save(image_path)
    else:
        image = Image.open(image_path)

    query = "Please describe this picture"

    print(f"Question:\n {query}")
    image


.. parsed-literal::

    Question:
     Please describe this picture




.. image:: glm-edge-v-with-output_files/glm-edge-v-with-output_12_1.png



.. code:: ipython3

    from transformers import TextStreamer, AutoImageProcessor, AutoTokenizer
    import torch

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]

    processor = AutoImageProcessor.from_pretrained(out_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, tokenize=True, return_tensors="pt").to("cpu")
    generate_kwargs = {
        **inputs,
        "pixel_values": torch.tensor(processor(image).pixel_values).to("cpu"),
        "max_new_tokens": 100,
        "do_sample": True,
        "top_k": 20,
        "streamer": TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
    }

    print("Answer:")
    output = model.generate(**generate_kwargs)


.. parsed-literal::

    Answer:
    This image is of a grey cat laying comfortably inside an open cardboard box with its back turned to the viewer. The box is situated on a white carpet, suggesting a cozy indoor setting. There is furniture visible in the background, though it partially blurs out the details. The cat appears relaxed, with one of its paws stretched out, and its tail extended to the side, indicating it's enjoying some rest. The room has a soft, well-maintained appearance, with


Interactive demo
----------------



.. code:: ipython3

    from gradio_helper import make_demo

    demo = make_demo(model, processor, tokenizer)

    try:
        demo.launch(debug=False, height=600)
    except Exception:
        demo.launch(debug=False, share=True, height=600)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860

    To create a public link, set `share=True` in `launch()`.







