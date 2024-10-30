Visual-language assistant with Phi3-Vision and OpenVINO
-------------------------------------------------------

The Phi-3-Vision is a lightweight, state-of-the-art open multimodal
model built upon datasets which include - synthetic data and filtered
publicly available websites - with a focus on very high-quality,
reasoning dense data both on text and vision. The model belongs to the
Phi-3 model family, and the multimodal version comes with 128K context
length (in tokens) it can support. The model underwent a rigorous
enhancement process, incorporating both supervised fine-tuning and
direct preference optimization to ensure precise instruction adherence
and robust safety measures. More details about model can be found in
`model blog
post <https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/>`__,
`technical report <https://aka.ms/phi3-tech-report>`__,
`Phi-3-cookbook <https://github.com/microsoft/Phi-3CookBook>`__

In this tutorial we consider how to launch Phi-3-vision using OpenVINO
for creation multimodal chatbot. Additionally, we optimize model to low
precision using `NNCF <https://github.com/openvinotoolkit/nncf>`__ ####
Table of contents:

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

    %pip install -q "torch>=2.1" "torchvision" "transformers>=4.40" "protobuf>=3.20" "gradio>=4.26" "Pillow" "accelerate" "tqdm"  --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install  -q "openvino>=2024.2.0" "nncf>=2.11.0"


.. parsed-literal::

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    descript-audiotools 0.7.2 requires protobuf<3.20,>=3.9.2, but you have protobuf 5.28.2 which is incompatible.
    open-clip-torch 2.22.0 requires protobuf<4, but you have protobuf 5.28.2 which is incompatible.
    tensorflow 2.12.0 requires protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3, but you have protobuf 5.28.2 which is incompatible.
    tensorflow-metadata 1.14.0 requires protobuf<4.21,>=3.20.3, but you have protobuf 5.28.2 which is incompatible.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import requests
    from pathlib import Path

    if not Path("ov_phi3_vision_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/phi-3-vision/ov_phi3_vision_helper.py")
        open("ov_phi3_vision_helper.py", "w").write(r.text)


    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/phi-3-vision/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)

    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w").write(r.text)

Select Model
------------



The tutorial supports the following models from Phi-3 model family:

- `Phi-3.5-vision-instruct <https://huggingface.co/microsoft/Phi-3.5-vision-instruct>`__
- `Phi-3-vision-128k-instruct <https://huggingface.co/microsoft/Phi-3-vision-128k-instruct>`__

You can select one from the provided options below.

.. code:: ipython3

    import ipywidgets as widgets

    # Select model
    model_ids = [
        "microsoft/Phi-3.5-vision-instruct",
        "microsoft/Phi-3-vision-128k-instruct",
    ]

    model_dropdown = widgets.Dropdown(
        options=model_ids,
        value=model_ids[0],
        description="Model:",
        disabled=False,
    )

    model_dropdown




.. parsed-literal::

    Dropdown(description='Model:', options=('microsoft/Phi-3.5-vision-instruct', 'microsoft/Phi-3-vision-128k-inst…



Convert and Optimize model
--------------------------



Phi-3-vision is PyTorch model. OpenVINO supports PyTorch models via
conversion to OpenVINO Intermediate Representation (IR). `OpenVINO model
conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
should be used for these purposes. ``ov.convert_model`` function accepts
original PyTorch model instance and example input for tracing and
returns ``ov.Model`` representing this model in OpenVINO framework.
Converted model can be used for saving on disk using ``ov.save_model``
function or directly loading on device using ``core.complie_model``.

The script ``ov_phi3_vision_helper.py`` contains helper function for
model conversion, please check its content if you interested in
conversion details.

.. raw:: html

   <details>

Click here for more detailed explanation of conversion steps
Phi-3-vision is autoregressive transformer generative model, it means
that each next model step depends from model output from previous step.
The generation approach is based on the assumption that the probability
distribution of a word sequence can be decomposed into the product of
conditional next word distributions. In other words, model predicts the
next token in the loop guided by previously generated tokens until the
stop-condition will be not reached (generated sequence of maximum length
or end of string token obtained). The way the next token will be
selected over predicted probabilities is driven by the selected decoding
methodology. You can find more information about the most popular
decoding methods in this blog. The entry point for the generation
process for models from the Hugging Face Transformers library is the
``generate`` method. You can find more information about its parameters
and configuration in the documentation. To preserve flexibility in the
selection decoding methodology, we will convert only model inference for
one step.

The inference flow has difference on first step and for the next. On the
first step, model accept preprocessed input instruction and image, that
transformed to the unified embedding space using ``input_embedding`` and
``image_encoder`` models, after that ``language model``, LLM-based part
of model, runs on input embeddings to predict probability of next
generated tokens. On the next step, ``language_model`` accepts only next
token id selected based on sampling strategy and processed by
``input_embedding`` model and cached attention key and values. Since the
output side is auto-regressive, an output token hidden state remains the
same once computed for every further generation step. Therefore,
recomputing it every time you want to generate a new token seems
wasteful. With the cache, the model saves the hidden state once it has
been computed. The model only computes the one for the most recently
generated output token at each time step, re-using the saved ones for
hidden tokens. This reduces the generation complexity from
:math:`O(n^3)` to :math:`O(n^2)` for a transformer model. More details
about how it works can be found in this
`article <https://scale.com/blog/pytorch-improvements#Text%20Translation>`__.
For improving support images of various resolution, input image
separated on patches and processed by ``image feature extractor`` and
``image projector`` that are part of image encoder.

To sum up above, model consists of 4 parts:

-  **Image feature extractor** and **Image projector** for encoding
   input images into embedding space.
-  **Input Embedding** for conversion input text tokens into embedding
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

    from ov_phi3_vision_helper import convert_phi3_model

    # uncomment these lines to see model conversion code
    # convert_phi3_model??


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    2024-10-23 02:15:24.328514: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-23 02:15:24.364541: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-23 02:15:24.908606: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. code:: ipython3

    from pathlib import Path
    import nncf


    model_id = model_dropdown.value
    out_dir = Path("model") / Path(model_id).name / "INT4"
    compression_configuration = {
        "mode": nncf.CompressWeightsMode.INT4_SYM,
        "group_size": 64,
        "ratio": 0.6,
    }
    convert_phi3_model(model_id, out_dir, compression_configuration)


.. parsed-literal::

    ⌛ Phi-3.5-vision-instruct conversion started. Be patient, it may takes some time.
    ⌛ Load Original model



.. parsed-literal::

    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/auto/image_processing_auto.py:513: FutureWarning: The image_processor_class argument is deprecated and will be removed in v4.42. Please use `slow_image_processor_class`, or `fast_image_processor_class` instead
      warnings.warn(


.. parsed-literal::

    ✅ Original model successfully loaded
    ⌛ Convert Input embedding model
    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.


.. parsed-literal::

    WARNING:nncf:NNCF provides best results with torch==2.4.*, while current torch version is 2.2.2+cpu. If you encounter issues, consider switching to torch==2.4.*
    ✅ Input embedding model successfully converted
    ⌛ Convert Image embedding model


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4664: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(


.. parsed-literal::

    ✅ Image embedding model successfully converted
    ⌛ Convert Image projection model


.. parsed-literal::

    You are not running the flash-attention implementation, expect numerical differences.


.. parsed-literal::

    ✅ Image projection model successfully converted
    ⌛ Convert Language model


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:114: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:162: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if past_key_values_length > 0:
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/microsoft/Phi-3.5-vision-instruct/4a0d683eba9f1d0cbfb6151705d1ee73c25a80ca/modeling_phi3_v.py:444: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      seq_len = seq_len or torch.max(position_ids) + 1
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/microsoft/Phi-3.5-vision-instruct/4a0d683eba9f1d0cbfb6151705d1ee73c25a80ca/modeling_phi3_v.py:445: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if seq_len > self.original_max_position_embeddings:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/torch/dynamic_graph/wrappers.py:86: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      op1 = operator(\*args, \*\*kwargs)
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/microsoft/Phi-3.5-vision-instruct/4a0d683eba9f1d0cbfb6151705d1ee73c25a80ca/modeling_phi3_v.py:683: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/microsoft/Phi-3.5-vision-instruct/4a0d683eba9f1d0cbfb6151705d1ee73c25a80ca/modeling_phi3_v.py:690: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/microsoft/Phi-3.5-vision-instruct/4a0d683eba9f1d0cbfb6151705d1ee73c25a80ca/modeling_phi3_v.py:702: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/jit/_trace.py:165: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
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
    │              8 │ 42% (54 / 129)              │ 40% (53 / 128)                         │
    ├────────────────┼─────────────────────────────┼────────────────────────────────────────┤
    │              4 │ 58% (75 / 129)              │ 60% (75 / 128)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









.. parsed-literal::

    ✅ Weights compression finished
    ✅ Phi-3.5-vision-instruct model conversion finished. You can find results in model/Phi-3.5-vision-instruct/INT4


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



``OvPhi3vison`` class provides convenient way for running model. It
accepts directory with converted model and inference device as
arguments. For running model we will use ``generate`` method.

.. code:: ipython3

    from ov_phi3_vision_helper import OvPhi3Vision

    # Uncomment below lines to see the model inference class code

    # OvPhi3Vision??

.. code:: ipython3

    model = OvPhi3Vision(out_dir, device.value)

.. code:: ipython3

    import requests
    from PIL import Image

    url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
    image = Image.open(requests.get(url, stream=True).raw)

    print("Question:\n What is unusual on this picture?")
    image


.. parsed-literal::

    Question:
     What is unusual on this picture?




.. image:: phi-3-vision-with-output_files/phi-3-vision-with-output_14_1.png



.. code:: ipython3

    from transformers import AutoProcessor, TextStreamer

    messages = [
        {"role": "user", "content": "<|image_1|>\nWhat is unusual on this picture?"},
    ]

    processor = AutoProcessor.from_pretrained(out_dir, trust_remote_code=True)

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(prompt, [image], return_tensors="pt")

    generation_args = {"max_new_tokens": 50, "do_sample": False, "streamer": TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)}

    print("Answer:")
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/auto/image_processing_auto.py:513: FutureWarning: The image_processor_class argument is deprecated and will be removed in v4.42. Please use `slow_image_processor_class`, or `fast_image_processor_class` instead
      warnings.warn(


.. parsed-literal::

    Answer:
    Nothing unusual, it's a cat lying in a box.


Interactive demo
----------------



.. code:: ipython3

    from gradio_helper import make_demo

    demo = make_demo(model, processor)

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







