Visual-language assistant with InternVL2 and OpenVINO
=====================================================

InternVL 2.0 is the latest addition to the InternVL series of multimodal
large language models. InternVL 2.0 features a variety of
instruction-tuned models, ranging from 1 billion to 108 billion
parameters. Compared to the state-of-the-art open-source multimodal
large language models, InternVL 2.0 surpasses most open-source models.
It demonstrates competitive performance on par with proprietary
commercial models across various capabilities, including document and
chart comprehension, infographics QA, scene text understanding and OCR
tasks, scientific and mathematical problem solving, as well as cultural
understanding and integrated multimodal capabilities.

More details about model can be found in `model
card <https://huggingface.co/OpenGVLab/InternVL2-4B>`__,
`blog <https://internvl.github.io/blog/2024-07-02-InternVL-2.0/>`__ and
original `repo <https://github.com/OpenGVLab/InternVL>`__.

In this tutorial we consider how to convert and optimize InternVL2 model
for creating multimodal chatbot. Additionally, we demonstrate how to
apply stateful transformation on LLM part and model optimization
techniques like weights compression using
`NNCF <https://github.com/openvinotoolkit/nncf>`__


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Select model <#select-model>`__
-  `Convert and Optimize model <#convert-and-optimize-model>`__

   -  `Compress model weights to
      4-bit <#compress-model-weights-to-4-bit>`__

-  `Select inference device <#select-inference-device>`__
-  `Prepare model inference
   pipeline <#prepare-model-inference-pipeline>`__
-  `Run model inference <#run-model-inference>`__
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

    %pip install -q "transformers>4.36,<4.45" "torch>=2.1" "torchvision" "einops" "timm" "Pillow" "gradio>=4.36" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "openvino>=2024.3.0" "nncf>=2.12.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    from pathlib import Path
    import requests
    
    if not Path("conversation.py").exists():
        r = requests.get("https://huggingface.co/OpenGVLab/InternVL2-1B/raw/main/conversation.py")
        open("conversation.py", "w", encoding="utf-8").write(r.text)
    
    if not Path("internvl2_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/internvl2/internvl2_helper.py")
        open("internvl2_helper.py", "w", encoding="utf-8").write(r.text)
    
    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/internvl2/gradio_helper.py")
        open("gradio_helper.py", "w", encoding="utf-8").write(r.text)
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w", encoding="utf-8").write(r.text)

Select model
------------



There are multiple InternVL2 models available in `models
collection <https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e>`__.
You can select one of them for conversion and optimization in notebook
using widget bellow:

.. code:: ipython3

    from internvl2_helper import model_selector
    
    model_id = model_selector()
    
    model_id


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino




.. parsed-literal::

    Dropdown(description='Model:', options=('OpenGVLab/InternVL2-1B', 'OpenGVLab/InternVL2-2B', 'OpenGVLab/InternV…



.. code:: ipython3

    print(f"Selected {model_id.value}")
    pt_model_id = model_id.value
    model_dir = Path(pt_model_id.split("/")[-1])


.. parsed-literal::

    Selected OpenGVLab/InternVL2-1B


Convert and Optimize model
--------------------------



InternVL2 is PyTorch model. OpenVINO supports PyTorch models via
conversion to OpenVINO Intermediate Representation (IR). `OpenVINO model
conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
should be used for these purposes. ``ov.convert_model`` function accepts
original PyTorch model instance and example input for tracing and
returns ``ov.Model`` representing this model in OpenVINO framework.
Converted model can be used for saving on disk using ``ov.save_model``
function or directly loading on device using ``core.complie_model``.
``internvl2_helper.py`` script contains helper function for model
conversion, please check its content if you interested in conversion
details.

.. raw:: html

   <details>

Click here for more detailed explanation of conversion steps InternVL2
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
To sum up above, model consists of 4 parts:

-  **Image encoder** for encoding input images into embedding space.
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

    from internvl2_helper import convert_internvl2_model
    
    # uncomment these lines to see model conversion code
    # convert_internvl2_model??

.. code:: ipython3

    import nncf
    
    compression_configuration = {
        "mode": nncf.CompressWeightsMode.INT4_ASYM,
        "group_size": 128,
        "ratio": 1.0,
    }
    
    convert_internvl2_model(pt_model_id, model_dir, compression_configuration)


.. parsed-literal::

    ⌛ OpenGVLab/InternVL2-1B conversion started. Be patient, it may takes some time.
    ⌛ Load Original model


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
      warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)


.. parsed-literal::

    FlashAttention2 is not installed.
    ✅ Original model successfully loaded
    ⌛ Convert Input embedding model
    WARNING:nncf:NNCF provides best results with torch==2.4.*, while current torch version is 2.2.2+cpu. If you encounter issues, consider switching to torch==2.4.*
    ✅ Input embedding model successfully converted
    ⌛ Convert Image embedding model


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4713: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVL2-1B/a84c71e158b16180df4fd1c5fe963fdf54b2cd43/modeling_internvl_chat.py:195: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      h = w = int(vit_embeds.shape[1] ** 0.5)
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVL2-1B/a84c71e158b16180df4fd1c5fe963fdf54b2cd43/modeling_internvl_chat.py:169: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVL2-1B/a84c71e158b16180df4fd1c5fe963fdf54b2cd43/modeling_internvl_chat.py:173: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      x = x.view(n, int(h * scale_factor), int(w * scale_factor),
    /opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVL2-1B/a84c71e158b16180df4fd1c5fe963fdf54b2cd43/modeling_internvl_chat.py:174: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      int(c / (scale_factor * scale_factor)))


.. parsed-literal::

    ⌛ Weights compression with int4_asym mode started


.. parsed-literal::

    2024-10-23 01:27:20.642555: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-23 01:27:20.681421: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-23 01:27:21.329023: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 0% (2 / 99)                 │ 0% (0 / 97)                            │
    ├────────────────┼─────────────────────────────┼────────────────────────────────────────┤
    │              4 │ 100% (97 / 99)              │ 100% (97 / 97)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









.. parsed-literal::

    ✅ Weights compression finished
    ✅ Image embedding model successfully converted
    ⌛ Convert Language model
    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/qwen2/modeling_qwen2.py:100: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if sequence_length != 1:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/qwen2/modeling_qwen2.py:165: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if seq_len > self.max_seq_len_cached:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/qwen2/modeling_qwen2.py:324: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/qwen2/modeling_qwen2.py:339: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):


.. parsed-literal::

    ✅ Language model successfully converted
    ⌛ Weights compression with int4_asym mode started
    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 28% (1 / 169)               │ 0% (0 / 168)                           │
    ├────────────────┼─────────────────────────────┼────────────────────────────────────────┤
    │              4 │ 72% (168 / 169)             │ 100% (168 / 168)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









.. parsed-literal::

    ✅ Weights compression finished
    ✅ OpenGVLab/InternVL2-1B model conversion finished. You can find results in InternVL2-1B


Select inference device
-----------------------



.. code:: ipython3

    from notebook_utils import device_widget
    
    device = device_widget(default="AUTO", exclude=["NPU"])
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Prepare model inference pipeline
--------------------------------



As discussed, the model comprises Image Encoder and LLM (with separated
text embedding part) that generates answer. In ``internvl2_helper.py``
we defined LLM inference class ``OvModelForCausalLMWithEmb`` that will
represent generation cycle, It is based on `HuggingFace Transformers
GenerationMixin <https://huggingface.co/docs/transformers/main_classes/text_generation>`__
and looks similar to `Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__
``OVModelForCausalLM`` that is used for LLM inference with only
difference that it can accept input embedding. In own turn, general
multimodal model class ``OVInternVLChatModel`` handles chatbot
functionality including image processing and answer generation using
LLM.

.. code:: ipython3

    from internvl2_helper import OVInternVLChatModel
    from transformers import AutoTokenizer
    
    # Uncomment below lines to see the model inference class code
    
    # OVInternVLChatModel??

.. code:: ipython3

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    ov_model = OVInternVLChatModel(model_dir, device.value)

Run model inference
-------------------



Our interface is fully compatible with Transformers interface for
InternVL2, you can try any of represented here `usage
examples <https://huggingface.co/OpenGVLab/InternVL2-1B#inference-with-transformers>`__.
Let’s check model capabilities in answering questions about image:

.. code:: ipython3

    import PIL
    from internvl2_helper import load_image
    from transformers import TextIteratorStreamer
    from threading import Thread
    
    
    EXAMPLE_IMAGE = Path("examples_image1.jpg")
    EXAMPLE_IMAGE_URL = "https://huggingface.co/OpenGVLab/InternVL2-2B/resolve/main/examples/image1.jpg"
    
    if not EXAMPLE_IMAGE.exists():
        img_data = requests.get(EXAMPLE_IMAGE_URL).content
        with EXAMPLE_IMAGE.open("wb") as handler:
            handler.write(img_data)
    
    pixel_values = load_image(EXAMPLE_IMAGE, max_num=12)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_config = dict(max_new_tokens=100, do_sample=True, streamer=streamer)
    question = "<image>\nPlease describe the image shortly."
    
    display(PIL.Image.open(EXAMPLE_IMAGE))
    print(f"User: {question}\n")
    print("Assistant:")
    
    thread = Thread(
        target=ov_model.chat,
        kwargs=dict(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=question,
            history=None,
            return_history=False,
            generation_config=generation_config,
        ),
    )
    thread.start()
    
    generated_text = ""
    # Loop through the streamer to get the new text as it is generated
    for new_text in streamer:
        if new_text == ov_model.conv_template.sep:
            break
        generated_text += new_text
        print(new_text, end="", flush=True)  # Print each new chunk of generated text on the same line



.. image:: internvl2-with-output_files/internvl2-with-output_16_0.png


.. parsed-literal::

    User: <image>
    Please describe the image shortly.
    
    Assistant:


.. parsed-literal::

    Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.


.. parsed-literal::

    The image depicts a red panda, a large feline with a distinctive, reddish-brown coat and white face and chest. It is peeking over what appears to be a wooden platform or platform made for panda viewing in captivity. The background is filled with greenery, indicating that the photo was likely taken in a conservatory or wildlife park where penguins or seabirds are displayed.

Interactive demo
----------------



.. code:: ipython3

    from gradio_helper import make_demo
    
    demo = make_demo(ov_model, tokenizer)
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







