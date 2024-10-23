Visual-language assistant with Qwen2VL and OpenVINO
===================================================

Qwen2VL is the latest addition to the QwenVL series of multimodal large
language models.

**Key Enhancements of Qwen2VL:** \* **SoTA understanding of images of
various resolution & ratio**: Qwen2-VL achieves state-of-the-art
performance on visual understanding benchmarks, including MathVista,
DocVQA, RealWorldQA, MTVQA, etc. \* **Understanding videos of 20min+**:
Qwen2-VL can understand videos over 20 minutes for high-quality
video-based question answering, dialog, content creation, etc. \*
**Agent that can operate your mobiles, robots, etc.:** with the
abilities of complex reasoning and decision making, Qwen2-VL can be
integrated with devices like mobile phones, robots, etc., for automatic
operation based on visual environment and text instructions. \*
**Multilingual Support:** to serve global users, besides English and
Chinese, Qwen2-VL now supports the understanding of texts in different
languages inside images, including most European languages, Japanese,
Korean, Arabic, Vietnamese, etc.

**Model Architecture Details:**

-  **Naive Dynamic Resolution**: Qwen2-VL can handle arbitrary image
   resolutions, mapping them into a dynamic number of visual tokens,
   offering a more human-like visual processing experience.

.. raw:: html

   <p align="center">

.. raw:: html

   <p>

-  **Multimodal Rotary Position Embedding (M-ROPE)**: Decomposes
   positional embedding into parts to capture 1D textual, 2D visual, and
   3D video positional information, enhancing its multimodal processing
   capabilities.

.. raw:: html

   <p align="center">

.. raw:: html

   <p>

More details about model can be found in `model
card <https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct>`__,
`blog <https://qwenlm.github.io/blog/qwen2-vl/>`__ and original
`repo <https://github.com/QwenLM/Qwen2-VL>`__.

In this tutorial we consider how to convert and optimize Qwen2VL model
for creating multimodal chatbot. Additionally, we demonstrate how to
apply stateful transformation on LLM part and model optimization
techniques like weights compression using
`NNCF <https://github.com/openvinotoolkit/nncf>`__ #### Table of
contents:

-  `Prerequisites <#prerequisites>`__
-  `Select model <#select-model>`__
-  `Convert and Optimize model <#convert-and-optimize-model>`__

   -  `Compress model weights to
      4-bit <#compress-model-weights-to-4-bit>`__

-  `Prepare model inference
   pipeline <#prepare-model-inference-pipeline>`__

   -  `Select inference device <#select-inference-device>`__

-  `Run model inference <#run-model-inference>`__
-  `Interactive Demo <#interactive-demo>`__

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

    %pip install -q "transformers>=4.45" "torch>=2.1" "torchvision" "qwen-vl-utils" "Pillow" "gradio>=4.36" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -qU "openvino>=2024.4.0" "nncf>=2.13.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    from pathlib import Path
    import requests
    
    if not Path("ov_qwen2_vl.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/qwen2-vl/ov_qwen2_vl.py")
        open("ov_qwen2_vl.py", "w").write(r.text)
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w").write(r.text)

Select model
------------



There are multiple Qwen2VL models available in `models
collection <https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e>`__.
You can select one of them for conversion and optimization in notebook
using widget bellow:

.. code:: ipython3

    from ov_qwen2_vl import model_selector
    
    model_id = model_selector()
    
    model_id


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    2024-10-23 04:32:38.810057: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-23 04:32:38.845114: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-23 04:32:39.399370: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT




.. parsed-literal::

    Dropdown(description='Model:', options=('Qwen/Qwen2-VL-2B-Instruct', 'Qwen/Qwen2-VL-7B-Instruct'), value='Qwen…



.. code:: ipython3

    print(f"Selected {model_id.value}")
    pt_model_id = model_id.value
    model_dir = Path(pt_model_id.split("/")[-1])


.. parsed-literal::

    Selected Qwen/Qwen2-VL-2B-Instruct


Convert and Optimize model
--------------------------



Qwen2VL is PyTorch model. OpenVINO supports PyTorch models via
conversion to OpenVINO Intermediate Representation (IR). `OpenVINO model
conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
should be used for these purposes. ``ov.convert_model`` function accepts
original PyTorch model instance and example input for tracing and
returns ``ov.Model`` representing this model in OpenVINO framework.
Converted model can be used for saving on disk using ``ov.save_model``
function or directly loading on device using ``core.compile_model``.
``ov_qwen2_vl.py`` script contains helper function for model conversion,
please check its content if you interested in conversion details.

.. raw:: html

   <details>

Click here for more detailed explanation of conversion steps Qwen2VL is
autoregressive transformer generative model, it means that each next
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

    from ov_qwen2_vl import convert_qwen2vl_model
    
    # uncomment these lines to see model conversion code
    # convert_qwen2vl_model??

.. code:: ipython3

    import nncf
    
    compression_configuration = {
        "mode": nncf.CompressWeightsMode.INT4_ASYM,
        "group_size": 128,
        "ratio": 1.0,
    }
    
    convert_qwen2vl_model(pt_model_id, model_dir, compression_configuration)


.. parsed-literal::

    ⌛ Qwen/Qwen2-VL-2B-Instruct conversion started. Be patient, it may takes some time.
    ⌛ Load Original model


.. parsed-literal::

    `Qwen2VLRotaryEmbedding` can now be fully parameterized by passing the model config through the `config` argument. All other arguments will be removed in v4.46



.. parsed-literal::

    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


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

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4779: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(


.. parsed-literal::

    ⌛ Weights compression with int4_asym mode started
    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 1% (1 / 130)                │ 0% (0 / 129)                           │
    ├────────────────┼─────────────────────────────┼────────────────────────────────────────┤
    │              4 │ 99% (129 / 130)             │ 100% (129 / 129)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









.. parsed-literal::

    ✅ Weights compression finished
    ✅ Image embedding model successfully converted
    ⌛ Convert Language model


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/cache_utils.py:447: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
      or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py:476: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if sequence_length != 1:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/cache_utils.py:432: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
      elif len(self.key_cache[layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors


.. parsed-literal::

    ✅ Language model successfully converted
    ⌛ Weights compression with int4_asym mode started
    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 15% (1 / 197)               │ 0% (0 / 196)                           │
    ├────────────────┼─────────────────────────────┼────────────────────────────────────────┤
    │              4 │ 85% (196 / 197)             │ 100% (196 / 196)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









.. parsed-literal::

    ✅ Weights compression finished
    ✅ Qwen/Qwen2-VL-2B-Instruct model conversion finished. You can find results in Qwen2-VL-2B-Instruct


Prepare model inference pipeline
--------------------------------



As discussed, the model comprises Image Encoder and LLM (with separated
text embedding part) that generates answer. In ``ov_qwen2_vl.py`` we
defined inference class ``OVQwen2VLModel`` that will represent
generation cycle, It is based on `HuggingFace Transformers
GenerationMixin <https://huggingface.co/docs/transformers/main_classes/text_generation>`__
and looks similar to `Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__
``OVModelForCausalLM`` that is used for LLM inference.

.. code:: ipython3

    from ov_qwen2_vl import OVQwen2VLModel
    
    # Uncomment below lines to see the model inference class code
    # OVQwen2VLModel??

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from notebook_utils import device_widget
    
    device = device_widget(default="AUTO", exclude=["NPU"])
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    model = OVQwen2VLModel(model_dir, device.value)

Run model inference
-------------------



.. code:: ipython3

    from PIL import Image
    from transformers import AutoProcessor, AutoTokenizer
    from qwen_vl_utils import process_vision_info
    from transformers import TextStreamer
    
    
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)
    
    if processor.chat_template is None:
        tok = AutoTokenizer.from_pretrained(model_dir)
        processor.chat_template = tok.chat_template
    
    example_image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    example_image_path = Path("demo.jpeg")
    
    if not example_image_path.exists():
        Image.open(requests.get(example_image_url, stream=True).raw).save(example_image_path)
    
    image = Image.open(example_image_path)
    question = "Describe this image."
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{example_image_path}",
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    
    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    display(image)
    print("Question:")
    print(question)
    print("Answer:")
    
    generated_ids = model.generate(**inputs, max_new_tokens=100, streamer=TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True))



.. image:: qwen2-vl-with-output_files/qwen2-vl-with-output_16_0.png


.. parsed-literal::

    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.


.. parsed-literal::

    Question:
    Describe this image.
    Answer:
    The image depicts a woman sitting on a sandy beach with a large dog. The dog is standing on its hind legs, reaching up to give the woman a high-five. The woman is smiling and appears to be enjoying the moment. The background shows the ocean with gentle waves, and the sky is clear with a soft light, suggesting it might be either sunrise or sunset. The scene is serene and joyful, capturing a heartwarming interaction between the woman and her dog.


.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/qwen2-vl/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)

Interactive Demo
----------------



Now, you can try to chat with model. Upload image or video using
``Upload`` button, provide your text message into ``Input`` field and
click ``Submit`` to start communication.

.. code:: ipython3

    from gradio_helper import make_demo
    
    
    demo = make_demo(model, processor)
    
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







