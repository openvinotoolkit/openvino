Visual-language assistant with Llama-3.2-11B-Vision and OpenVINO
================================================================

Llama-3.2-11B-Vision is the latest model from LLama3 model family those
capabilities extended to understand images content. The Llama 3.2-Vision
instruction-tuned models are optimized for visual recognition, image
reasoning, captioning, and answering general questions about an image.
Llama 3.2-Vision is built on top of Llama 3.1 text-only model, which is
an auto-regressive language model that uses an optimized transformer
architecture. The tuned versions use supervised fine-tuning (SFT) and
reinforcement learning with human feedback (RLHF) to align with human
preferences for helpfulness and safety. To support image recognition
tasks, the Llama 3.2-Vision model uses a separately trained vision
adapter that integrates with the pre-trained Llama 3.1 language model.
The adapter consists of a series of cross-attention layers that feed
image encoder representations into the core LLM.

In this tutorial we consider how to convert, optimize and run this model
using OpenVINO. More details about model can be found in `model
card <https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md>`__,
and original `repo <https://github.com/meta-llama/llama-models>`__.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Convert model <#convert-model>`__
-  `Select inference device <#select-inference-device>`__
-  `Optimize model using NNCF <#optimize-model-using-nncf>`__

   -  `Compress Language model weights in
      4bits <#compress-language-model-weights-in-4bits>`__
   -  `Optimize Vision model <#optimize-vision-model>`__

-  `Model Inference <#model-inference>`__
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

    %pip install -q "torch>=2.1" "torchvision" "Pillow" "tqdm" "datasets>=2.14.6" "gradio>=4.36" "nncf>=2.13.0" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "transformers>=4.45" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -Uq --pre "openvino>2024.4.0" --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

.. code:: ipython3

    import requests
    from pathlib import Path
    
    if not Path("ov_mllama_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/mllama3.2/ov_mllama_helper.py")
        open("ov_mllama_helper.py", "w").write(r.text)
    
    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/mllama3.2/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)
    
    if not Path("ov_mllama_compression.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/mllama3.2/ov_mllama_compression.py")
        open("ov_mllama_compression.py", "w").write(r.text)
    
    if not Path("data_preprocessing.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/mllama3.2/data_preprocessing.py")
        open("data_preprocessing", "w").write(r.text)
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w").write(r.text)

Convert model
-------------



OpenVINO supports PyTorch models via conversion to OpenVINO Intermediate
Representation (IR). `OpenVINO model conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
should be used for these purposes. ``ov.convert_model`` function accepts
original PyTorch model instance and example input for tracing and
returns ``ov.Model`` representing this model in OpenVINO framework.
Converted model can be used for saving on disk using ``ov.save_model``
function or directly loading on device using ``core.complie_model``.

``ov_mllama_helper.py`` script contains helper function for model
conversion, please check its content if you interested in conversion
details.

.. raw:: html

   <details>

Click here for more detailed explanation of conversion steps
Llama-3.2.-Vision is autoregressive transformer generative model, it
means that each next model step depends from model output from previous
step. The generation approach is based on the assumption that the
probability distribution of a word sequence can be decomposed into the
product of conditional next word distributions. In other words, model
predicts the next token in the loop guided by previously generated
tokens until the stop-condition will be not reached (generated sequence
of maximum length or end of string token obtained). The way the next
token will be selected over predicted probabilities is driven by the
selected decoding methodology. You can find more information about the
most popular decoding methods in this
`blog <https://huggingface.co/blog/how-to-generate>`__. The entry point
for the generation process for models from the Hugging Face Transformers
library is the ``generate`` method. You can find more information about
its parameters and configuration in the
`documentation <https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate>`__.
To preserve flexibility in the selection decoding methodology, we will
convert only model inference for one step.

The inference flow has difference on first step and for the next. On the
first step, model accept preprocessed input instruction and image. Image
processed via ``Image Encoder`` to cross-attention state, after that
``language model``, LLM-based part of model, runs on cross-attention
states and tokenized input token ids to predict probability of next
generated tokens. On the next step, ``language_model`` accepts only next
token. Since the output side is auto-regressive, an output token hidden
state remains the same once computed for every further generation step.
Therefore, recomputing it every time you want to generate a new token
seems wasteful. With the cache, the model saves the hidden state once it
has been computed. The model only computes the one for the most recently
generated output token at each time step, re-using the saved ones for
hidden tokens. This reduces the generation complexity from
:math:`O(n^3)` to :math:`O(n^2)` for a transformer model. More details
about how it works can be found in this
`article <https://scale.com/blog/pytorch-improvements#Text%20Translation>`__.

With increasing model size like in modern LLMs, we also can note an
increase in the number of attention blocks and size past key values
tensors respectively. The strategy for handling cache state as model
inputs and outputs in the inference cycle may become a bottleneck for
memory-bounded systems, especially with processing long input sequences,
for example in a chatbot scenario. OpenVINO suggests a transformation
that removes inputs and corresponding outputs with cache tensors from
the model keeping cache handling logic inside the model. Such models are
also called stateful. A stateful model is a model that implicitly
preserves data between two consecutive inference calls. The tensors
saved from one run are kept in an internal memory buffer called a
``state`` or a ``variable`` and may be passed to the next run, while
never being exposed as model output. Hiding the cache enables storing
and updating the cache values in a more device-friendly representation.
It helps to reduce memory consumption and additionally optimize model
performance. More details about stateful models and working with state
can be found in `OpenVINO
documentation <https://docs.openvino.ai/2024/openvino-workflow/running-inference/stateful-models.html>`__.

``image_encoder`` is represented in Llama-3.2-Vision by pretrained VIT
model.

To sum up above, model consists of 2 parts:

-  **Image Encoder** for encoding input images into LLM cross attention
   states space.
-  **Language Model** for generation answer based on cross attention
   states provided by Image Encoder and input tokens.

Let’s convert each model part.

.. raw:: html

   </details>

..

   **Note**: run model with notebook, you will need to accept license
   agreement. You must be a registered user in Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct>`__,
   carefully read terms of usage and click accept button. You will need
   to use an access token for the code below to run. For more
   information on access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: ipython3

    # uncomment these lines to login to huggingfacehub to get access to pretrained model
    
    # from huggingface_hub import notebook_login, whoami
    
    # try:
    #     whoami()
    #     print('Authorization token already provided')
    # except OSError:
    #     notebook_login()

.. code:: ipython3

    from pathlib import Path
    from ov_mllama_helper import convert_mllama
    
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model_dir = Path(model_id.split("/")[-1]) / "OV"
    
    # uncomment the line to see model conversion code
    # convert_mllama??


.. parsed-literal::

    2024-09-26 08:47:58.173539: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-09-26 08:47:58.175474: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-09-26 08:47:58.210782: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-09-26 08:47:59.026387: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. code:: ipython3

    convert_mllama(model_id, model_dir)


.. parsed-literal::

    ⌛ Load original model



.. parsed-literal::

    Loading checkpoint shards:   0%|          | 0/5 [00:00<?, ?it/s]


.. parsed-literal::

    ⌛ Convert vision model...
    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/transformers/modeling_utils.py:4773: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/transformers/models/mllama/modeling_mllama.py:1496: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      slice_index = -num_padding_patches if num_padding_patches > 0 else None


.. parsed-literal::

    ✅ Vision model successfully converted
    ⌛ Convert language model...


.. parsed-literal::

    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/transformers/models/mllama/modeling_mllama.py:83: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if sequence_length != 1:
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/transformers/models/mllama/modeling_mllama.py:1710: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if is_cross_attention_layer and cross_attention_states is None and is_cross_attention_cache_empty:
    /home/ea/work/openvino_notebooks_new_clone/openvino_notebooks/notebooks/mllama-3.2/ov_mllama_helper.py:402: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      elif past_key_value.get_seq_length(self.layer_idx) != 0:


.. parsed-literal::

    ✅ Language model successfully converted
    ✅ Model sucessfully converted and can be found in Llama-3.2-11B-Vision-Instruct/OV


Select inference device
-----------------------



.. code:: ipython3

    from notebook_utils import device_widget
    
    device = device_widget("CPU", exclude=["NPU"])
    
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



Optimize model using NNCF
-------------------------



Compress Language model weights in 4bits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



For reducing memory consumption, weights compression optimization can be
applied using `NNCF <https://github.com/openvinotoolkit/nncf>`__.

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

In this tutorial we consider usage Data-Aware weights compression. Such
approaches may require more time and memory as they involves calibration
dataset, while promising better int4 model accuracy. > **Note:** AWQ
weight quantization requires at least 64GB RAM, if you run notebook in
memory-constrained environment, you can switch to data-free weight
compression using widget bellow

.. code:: ipython3

    from ov_mllama_compression import compress
    
    # uncomment the line to see compression code
    # compress??


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. code:: ipython3

    from ov_mllama_compression import compression_widgets_helper
    
    compression_scenario, compress_args = compression_widgets_helper()
    
    compression_scenario




.. parsed-literal::

    VBox(children=(RadioButtons(index=1, options=('data-free', 'data-aware'), value='data-aware'), Accordion(child…



.. code:: ipython3

    compression_kwargs = {key: value.value for key, value in compress_args.items()}
    
    language_model_path = compress(model_dir, **compression_kwargs)


.. parsed-literal::

    ✅ Compressed model already exists and can be found in Llama-3.2-11B-Vision-Instruct/OV/llm_int4_asym_r10_gs64_max_activation_variance_awq_scale_all_layers.xml


Optimize Vision model
~~~~~~~~~~~~~~~~~~~~~



While weight compression is the great tool for large language models
memory footprint reduction, for smaller size models like Image Encoder,
it may be more efficient to apply INT8 Post-training quantization. You
can find more details about post-training quantization in `OpenVINO
documentation <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training.html>`__.

Basically model quantization process consists of 3 steps: 1. Prepare
quantization dataset 2. Perform model quantization using
``nncf.quantize`` 3. Save optimized model on disk using
``ov.save_model``

   **Note:** Model quantization may requires additional time and memory
   for optimization and be non-applicable for some devices. You can skip
   quantization step or replace it with weight compression using widget
   bellow if you does not have enough resources.

.. code:: ipython3

    from ov_mllama_compression import vision_encoder_selection_widget
    
    vision_encoder_options = vision_encoder_selection_widget(device.value)
    
    vision_encoder_options




.. parsed-literal::

    Dropdown(description='Vision Encoder', index=1, options=('FP16', 'INT8 quantization', 'INT8 weights compressio…



.. code:: ipython3

    from transformers import AutoProcessor
    import nncf
    import openvino as ov
    import gc
    
    from data_preprocessing import prepare_dataset_vision
    
    processor = AutoProcessor.from_pretrained(model_dir)
    core = ov.Core()
    
    fp_vision_encoder_path = model_dir / "openvino_vision_encoder.xml"
    int8_vision_encoder_path = model_dir / fp_vision_encoder_path.name.replace(".xml", "_int8.xml")
    int8_wc_vision_encoder_path = model_dir / fp_vision_encoder_path.name.replace(".xml", "_int8_wc.xml")
    
    
    if vision_encoder_options.value == "INT8 quantization":
        if not int8_vision_encoder_path.exists():
            calibration_data = prepare_dataset_vision(processor, 100)
            ov_model = core.read_model(fp_vision_encoder_path)
            calibration_dataset = nncf.Dataset(calibration_data)
            quantized_model = nncf.quantize(
                model=ov_model,
                calibration_dataset=calibration_dataset,
                model_type=nncf.ModelType.TRANSFORMER,
                advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.6),
            )
            ov.save_model(quantized_model, int8_vision_encoder_path)
            del quantized_model
            del ov_model
            del calibration_dataset
            del calibration_data
            gc.collect()
    
        vision_encoder_path = int8_vision_encoder_path
    elif vision_encoder_options.value == "INT8 weights compression":
        if not int8_wc_vision_encoder_path.exists():
            ov_model = core.read_model(fp_vision_encoder_path)
            compressed_model = nncf.compress_weights(ov_model)
            ov.save_model(compressed_model, int8_wc_vision_encoder_path)
        vision_encoder_path = int8_wc_vision_encoder_path
    else:
        vision_encoder_path = fp_vision_encoder_path

Model Inference
---------------



Now, we are ready to test model inference.
``OVOVMLlamaForConditionalGeneration`` defined in
``ov_mllama_helper.py`` has similar generation interface with original
model and additionally enables runtime optimizations for efficient model
inference with OpenVINO: - **Slicing LM head** - usually LLM models
provides probability for all input tokens, while for selection next
token, we are interested only for the last one. Reducing Language Model
head size to return only last token probability may provide better
performance and reduce memory consumption for the first inference, where
usually whole input prompt processed. You can find more details about
this optimization in `OpenVINO
blog <https://blog.openvino.ai/blog-posts/large-language-model-graph-customization-with-openvino-tm-transformations-api>`__

.. raw:: html

   <p align="center">

.. raw:: html

   <p>

-  **Using Remote tensors for GPU** - Coping data on device and back
   into host memory can become bottleneck for efficient execution
   multi-model pipeline on GPU. `Remote Tensor
   API <https://docs.openvino.ai/2024/documentation/openvino-extensibility/openvino-plugin-library/remote-tensor.html>`__
   provides functionality for low-level GPU memory management, we can
   use this feature for sharing cross-attention keys and values between
   Image Encoder and Language Model.

.. code:: ipython3

    from ov_mllama_helper import OVMLlamaForConditionalGeneration
    
    # Uncomment this line to see model inference code
    # OVMLlamaForConditionalGeneration??
    
    ov_model = OVMLlamaForConditionalGeneration(
        model_dir, device=device.value, language_model_name=language_model_path.name, image_encoder_name=vision_encoder_path.name
    )
    processor = AutoProcessor.from_pretrained(model_dir)


.. parsed-literal::

    applied slice for lm head


.. code:: ipython3

    from PIL import Image
    from transformers import TextStreamer
    import numpy as np
    
    question = "What is unusual on this image?"
    
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
    ]
    text = processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
    raw_image = Image.open(requests.get(url, stream=True).raw)
    
    inputs = processor(text=text, images=[raw_image], return_tensors="pt")
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    print(f"Question: {question}")
    display(raw_image)
    output = ov_model.generate(**inputs, do_sample=False, max_new_tokens=100, temperature=None, top_p=None, streamer=streamer)
    print(f"Visual encoder time {ov_model.vision_encoder_infer_time[0] * 1000 :.2f} ms")
    print(f"First token latency {ov_model.llm_infer_time[0] * 1000 :.2f}ms, Second token latency {np.mean(np.array(ov_model.llm_infer_time[1:])) * 1000:.2f}ms")


.. parsed-literal::

    Question: What is unusual on this image?



.. image:: mllama-3.2-with-output_files/mllama-3.2-with-output_19_1.png


.. parsed-literal::

    The cat is lying in a box, which is an unusual position for a cat. Cats are known for their agility and flexibility, but they tend to prefer more comfortable and secure positions, such as on a soft surface or in a cozy spot. Lying in a box is not a typical behavior for a cat, and it may be due to the cat's desire to feel safe and protected or to explore a new environment.
    Visual encoder time 19374.52 ms
    First token latency 693.76ms, Second token latency 431.92ms


Interactive demo
----------------



.. code:: ipython3

    from gradio_helper import make_demo
    
    processor.chat_template = processor.tokenizer.chat_template
    demo = make_demo(ov_model, processor)
    
    try:
        demo.launch(debug=False)
    except Exception:
        demo.launch(debug=False, share=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
