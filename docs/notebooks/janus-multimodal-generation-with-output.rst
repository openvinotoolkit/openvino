Multimodal understanding and generation with Janus and OpenVINO
===============================================================

Janus is a novel autoregressive framework that unifies multimodal
understanding and generation. It addresses the limitations of previous
approaches by decoupling visual encoding into separate pathways, while
still utilizing a single, unified transformer architecture for
processing. The decoupling not only alleviates the conflict between the
visual encoder’s roles in understanding and generation, but also
enhances the framework’s flexibility. Janus surpasses previous unified
model and matches or exceeds the performance of task-specific models.
The simplicity, high flexibility, and effectiveness of Janus make it a
strong candidate for next-generation unified multimodal models.

More details can be found in the
`paper <https://arxiv.org/abs/2410.13848>`__, original
`repository <https://github.com/deepseek-ai/Janus>`__ and `model
card <https://huggingface.co/deepseek-ai/Janus-1.3B>`__

In this tutorial we consider how to run and optimize Janus using
OpenVINO.

**Table of contents:**

-  `Prerequisites <#prerequisites>`__
-  `Convert and Optimize model <#convert-and-optimize-model>`__

   -  `Compress model weights to
      4-bit <#compress-model-weights-to-4-bit>`__

-  `Create Inference Pipeline <#create-inference-pipeline>`__

   -  `Select Inference Device <#select-inference-device>`__
   -  `Run visual language chat <#run-visual-language-chat>`__
   -  `Run Image generation <#run-image-generation>`__

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

    from pathlib import Path
    import requests

    utility_files = ["notebook_utils.py"]
    local_helpers = ["ov_janus_helper.py", "gradio_helper.py"]

    base_utils_url = "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/"
    base_local_files_url = "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/janus-multimodal-generation/"


    for util_path in utility_files:
        if not Path(util_path).exists():
            r = requests.get(base_utils_url + util_path)
            with open(util_path, "w") as f:
                f.write(r.text)

    for util_path in local_helpers:
        if not Path(util_path).exists():
            r = requests.get(base_local_files_url + util_path)
            with open(util_path, "w") as f:
                f.write(r.text)

.. code:: ipython3

    import platform

    %pip install -q "gradio>=4.19" "torch>=2.2" "torchvision" "safetensors" "transformers>=4.38" "nncf>=2.14" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "git+https://github.com/deepseek-ai/Janus" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -U --pre "openvino>2024.5" --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

    if platform.system() == "Darwin":
        %pip install -q "numpy<2.0.0"

Convert and Optimize model
--------------------------



Janus is PyTorch model. OpenVINO supports PyTorch models via conversion
to OpenVINO Intermediate Representation (IR). `OpenVINO model conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
should be used for these purposes. ``ov.convert_model`` function accepts
original PyTorch model instance and example input for tracing and
returns ``ov.Model`` representing this model in OpenVINO framework.
Converted model can be used for saving on disk using ``ov.save_model``
function or directly loading on device using ``core.complie_model``.

The script ``ov_janus_helper.py`` contains helper function for model
conversion, please check its content if you interested in conversion
details.

.. raw:: html

   <details>

.. raw:: html

   <summary>

Click here for more detailed explanation of conversion steps

.. raw:: html

   </summary>

Janus is autoregressive transformer generative model, it means that each
next model step depends from model output from previous step. The
generation approach is based on the assumption that the probability
distribution of a token sequence can be decomposed into the product of
conditional next token distributions. In other words, model predicts the
next token in the loop guided by previously generated tokens until the
stop-condition will be not reached (generated sequence of maximum length
or end of generation token obtained). The way the next token will be
selected over predicted probabilities is driven by the selected decoding
methodology. You can find more information about the most popular
decoding methods in this blog. The entry point for the generation
process for models from the Hugging Face Transformers library is the
``generate`` method. You can find more information about its parameters
and configuration in the documentation. To preserve flexibility in the
selection decoding methodology, we will convert only model inference for
one step.

For both tasks, image understanding and image generation, Janus utilizes
the same basic transformer architecture in ``language_model`` and change
only components responsible for preparing input embeddings (joined image
embeddings prepared using ``vision_embeddings_model`` and text
embeddings prepared using ``text_embeddings_model`` for image
understanding and ``text_embeddings_model`` on the first step as initial
prompt embeddings and ``gen_embeddings_model`` for the next) and
conversion final hidden state to tokens probabilities (``lm_head`` for
text tokens, ``gen_head`` for image tokens). Additionally, for image
generation model uses ``gen_decoder`` to convert generated image tokens
to images.

To sum up above, model consists of 7 parts: \* **Image Embeddings** for
encoding input images into embedding space in image understanding task.
\* **Text Embedding** for conversion input text tokens into embedding
space \* **Gen Embeddings** for encoding image generation tokens to
embeddings space in image generation task \* **Language Model** for
generation hidden state guided by input embeddings \* **LM Head** for
conversion Language Model hidden state to text generation token
probabilities \* **Gen Head** for conversion Language Model hidden state
to image generation token probabilities \* **Gen Decoder** for decoding
generated image from latent token space to image tensor space.

For preserving original model flexibility of switching between tasks, we
also should preserve original model partitioning and convert each model
part separately.

.. raw:: html

   </details>

Compress model weights to 4-bit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For reducing memory
consumption, weights compression optimization can be applied using
`NNCF <https://github.com/openvinotoolkit/nncf>`__.

.. raw:: html

   <details>

.. raw:: html

   <summary>

Click here for more details about weight compression

.. raw:: html

   </summary>

Weight compression aims to reduce the memory footprint of a model. It
can also lead to significant performance improvement for large
memory-bound models, such as Large Language Models (LLMs). LLMs and
other models, which require extensive memory to store the weights during
inference, can benefit from weight compression in the following ways:

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

    import nncf
    from ov_janus_helper import convert_janus_model

    model_id = "deepseek-ai/Janus-1.3B"
    model_path = Path(model_id.split("/")[-1] + "-ov")

    compression_configuration = {
        "mode": nncf.CompressWeightsMode.INT4_ASYM,
        "group_size": 64,
        "ratio": 1.0,
    }

    # uncomment the line to see model conversion code
    # ??convert_janus_model


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    2024-11-26 20:09:59.629857: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-11-26 20:09:59.643309: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1732637399.658322 1754417 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1732637399.662894 1754417 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-11-26 20:09:59.679869: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    Python version is above 3.10, patching the collections module.


.. parsed-literal::

    /home/ea/work/py311/lib/python3.11/site-packages/transformers/models/auto/image_processing_auto.py:520: FutureWarning: The image_processor_class argument is deprecated and will be removed in v4.42. Please use `slow_image_processor_class`, or `fast_image_processor_class` instead
      warnings.warn(


.. code:: ipython3

    convert_janus_model(model_id, model_path, compression_configuration)


.. parsed-literal::

    ✅ Janus-1.3B model already converted. You can find results in Janus-1.3B-ov


Create Inference Pipeline
-------------------------



``OVJanusModel`` defined in ``ov_janus_helper.py`` provides unified
interface for running model inference for both text and image
generation. It accepts model directory and target device for inference.

Select Inference Device
~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from notebook_utils import device_widget

    device = device_widget("CPU", ["NPU"])

    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



.. code:: ipython3

    from ov_janus_helper import OVJanusModel
    from janus.models import VLChatProcessor

    # uncomment the line to see model inference code

    # ??OVJanusModel

``VLChatPRocessor`` class used for pre- and postprocessing steps in
original Janus model. Our model is also compatible with the same
processor code and we can reuse it.

.. code:: ipython3

    ov_model = OVJanusModel(model_path, device.value)

    processor = VLChatProcessor.from_pretrained(model_path)


.. parsed-literal::

    Some kwargs in processor config are unused and will not have any effect: image_end_tag, sft_format, image_tag, num_image_tokens, add_special_token, mask_prompt, ignore_id, image_start_tag.


Run visual language chat
~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from PIL import Image
    from io import BytesIO
    from janus.utils.io import load_pil_images


    input_prompt = "Describe image in details"
    image_path = Path("cat_in_box.png")

    if not image_path.exists():
        response = requests.get("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11")
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(image_path)

    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>{input_prompt}\n",
            "images": [str(image_path)],
        },
        {"role": "Assistant", "content": ""},
    ]
    pil_images = load_pil_images(conversation)

.. code:: ipython3

    from transformers import TextStreamer

    prepare_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True)
    # run image encoder to get the image embeddings
    inputs_embeds = ov_model.prepare_inputs_embeds(**prepare_inputs)

    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

    print(f"Question:\n{input_prompt}")
    display(pil_images[0])
    print("Answer:")

    answer_token_ids = ov_model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=processor.tokenizer.eos_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=128,
        do_sample=False,
        streamer=streamer,
    )


.. parsed-literal::

    Question:
    Describe image in details



.. image:: janus-multimodal-generation-with-output_files/janus-multimodal-generation-with-output_14_1.png


.. parsed-literal::

    Answer:
    The image depicts a gray and white tabby cat lying comfortably inside a cardboard box. The cat is lying on its back with its legs and paws spread out in a relaxed manner. The cat's eyes are closed, and it appears to be enjoying a nap. The box is placed on a light-colored carpet, and in the background, there is a portion of a white couch visible. The lighting in the room is soft and natural, suggesting that the photo was taken during the daytime. The overall scene conveys a sense of tranquility and contentment.


Run Image generation
~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from ov_janus_helper import generate_image

    # Uncomment the line to see image generation code
    # ??generate_image

.. code:: ipython3

    from transformers import set_seed

    set_seed(12345)

    images = generate_image(
        ov_model,
        processor,
        "A close-up professional photo of Yorkshire Terrier on beach, extrimely detailed, hyper realistic, full hd",
        output_dir=None,
        parallel_size=1,
    )



.. parsed-literal::

      0%|          | 0/576 [00:00<?, ?it/s]


.. code:: ipython3

    images[0].resize((1024, 1024))




.. image:: janus-multimodal-generation-with-output_files/janus-multimodal-generation-with-output_18_0.png



Interactive demo
----------------



.. code:: ipython3

    from gradio_helper import make_demo

    demo = make_demo(ov_model, processor)

    try:
        demo.launch(debug=True)
    except Exception:
        demo.launch(share=True, debug=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
