Visual-language assistant with LLaVA and OpenVINO Generative API
================================================================

`LLaVA <https://llava-vl.github.io>`__ (Large Language and Vision
Assistant) is large multimodal model that aims to develop a
general-purpose visual assistant that can follow both language and image
instructions to complete various real-world tasks. The idea is to
combine the power of large language models (LLMs) with vision encoders
like CLIP to create an end-to-end trained neural assistant that
understands and acts upon multimodal instructions.

In the field of artificial intelligence, the goal is to create a
versatile assistant capable of understanding and executing tasks based
on both visual and language inputs. Current approaches often rely on
large vision models that solve tasks independently, with language only
used to describe image content. While effective, these models have fixed
interfaces with limited interactivity and adaptability to user
instructions. On the other hand, large language models (LLMs) have shown
promise as a universal interface for general-purpose assistants. By
explicitly representing various task instructions in language, these
models can be guided to switch and solve different tasks. To extend this
capability to the multimodal domain, the `LLaVA
paper <https://arxiv.org/abs/2304.08485>`__ introduces \`visual
instruction-tuning, a novel approach to building a general-purpose
visual assistant.

In this tutorial we consider how to use LLaVA model to build multimodal
chatbot using `OpenVINO
GenAI <https://github.com/openvinotoolkit/openvino.genai>`__. For
demonstration purposes we will use
`LLaVA-1.5-7B <llava-hf/llava-1.5-7b-hf>`__ model for conversion,
similar steps required to run other models from `LLaVA Model
Zoo <https://huggingface.co/collections/llava-hf/llava-15-65f762d5b6941db5c2ba07e0>`__.

-  Install prerequisites
-  Convert model to OpenVINO Intermediate Representation format using
   Optimum Intel
-  Compress model weights to 4 and 8 bits using NNCF
-  Prepare OpenVINO GenAI inference pipeline
-  Run OpenVINO model


**Table of contents:**


-  `About model <#about-model>`__
-  `Prerequisites <#prerequisites>`__
-  `Convert and Optimize Model <#convert-and-optimize-model>`__

   -  `Convert model to OpenVINO IR format using Optimum
      CLI <#convert-model-to-openvino-ir-format-using-optimum-cli>`__
   -  `Compress Model weights to 4 and 8 bits using
      NNCF <#compress-model-weights-to-4-and-8-bits-using-nncf>`__

-  `Prepare OpenVINO GenAI inference
   pipeline <#prepare-openvino-genai-inference-pipeline>`__

   -  `Select inference device <#select-inference-device>`__
   -  `Select model variant <#select-model-variant>`__
   -  `Load OpenVINO model <#load-openvino-model>`__

-  `Run model inference <#run-model-inference>`__

   -  `Prepare input data <#prepare-input-data>`__
   -  `Test model inference <#test-model-inference>`__

-  `Interactive demo <#interactive-demo>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

About model
-----------



LLaVA connects pre-trained `CLIP
ViT-L/14 <https://openai.com/research/clip>`__ visual encoder and large
language model like Vicuna, LLaMa v2 or MPT, using a simple projection
matrix

.. figure:: https://llava-vl.github.io/images/llava_arch.png
   :alt: vlp_matrix.png

   vlp_matrix.png

Model training procedure consists of 2 stages:

-  Stage 1: Pre-training for Feature Alignment. Only the projection
   matrix is updated, based on a subset of CC3M.
-  Stage 2: Fine-tuning End-to-End.. Both the projection matrix and LLM
   are updated for two different use scenarios:

   -  Visual Chat: LLaVA is fine-tuned on our generated multimodal
      instruction-following data for daily user-oriented applications.
   -  Science QA: LLaVA is fine-tuned on this multimodal reasoning
      dataset for the science domain.

More details about model can be found in original `project
web-page <https://llava-vl.github.io/>`__,
`paper <https://arxiv.org/abs/2304.08485>`__ and
`repo <https://github.com/haotian-liu/LLaVA>`__.

Prerequisites
-------------



Install required dependencies

.. code:: ipython3

    %pip install -q "torch>=2.1.0" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cpu
    %pip install -q "git+https://github.com/eaidova/optimum-intel.git@ea/minicpmv"
    %pip install -q  "nncf>=2.13.0"  "sentencepiece" "tokenizers>=0.12.1" "transformers>=4.45.0" "gradio>=4.36"
    %pip install -q -U --pre --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly openvino_tokenizers openvino openvino-genai

Convert and Optimize Model
--------------------------



Our model conversion and optimization consist of following steps: 1.
Download original PyTorch model. 2. Convert model to OpenVINO format. 3.
Compress model weights using NNCF.

Let’s consider each step more deeply.

Convert model to OpenVINO IR format using Optimum CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



OpenVINO supports PyTorch models via conversion to OpenVINO Intermediate
Representation format. For convenience, we will use OpenVINO integration
with HuggingFace Optimum. `Optimum
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

    from pathlib import Path
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    model_path = Path(model_id.split("/")[-1]) / "FP16"
    
    if not model_path.exists():
        !optimum-cli export openvino --model {model_id} --weight-format fp16 {model_path}

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

    import shutil
    import nncf
    import openvino as ov
    import gc
    
    core = ov.Core()
    
    
    def compress_model_weights(precision):
        int4_compression_config = {
            "mode": nncf.CompressWeightsMode.INT4_ASYM,
            "group_size": 128,
            "ratio": 1,
        }
        int8_compression_config = {"mode": nncf.CompressWeightsMode.INT8_ASYM}
    
        compressed_model_path = model_path.parent / precision
    
        if not compressed_model_path.exists():
            ov_model = core.read_model(model_path / "openvino_language_model.xml")
            compression_config = int4_compression_config if precision == "INT4" else int8_compression_config
            compressed_ov_model = nncf.compress_weights(ov_model, **compression_config)
            ov.save_model(compressed_ov_model, compressed_model_path / "openvino_language_model.xml")
            del compressed_ov_model
            del ov_model
            gc.collect()
            for file_name in model_path.glob("*"):
                if file_name.name in ["openvino_language_model.xml", "openvino_language_model.bin"]:
                    continue
                shutil.copy(file_name, compressed_model_path)
    
    
    compress_model_weights(compression_mode.value)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino


Prepare OpenVINO GenAI inference pipeline
-----------------------------------------



`OpenVINO™ GenAI <https://github.com/openvinotoolkit/openvino.genai>`__
is a library of the most popular Generative AI model pipelines,
optimized execution methods, and samples that run on top of highly
performant `OpenVINO
Runtime <https://github.com/openvinotoolkit/openvino>`__.

This library is friendly to PC and laptop execution, and optimized for
resource consumption. It requires no external dependencies to run
generative models as it already includes all the core functionality
(e.g. tokenization via openvino-tokenizers). OpenVINO™ GenAI is a flavor
of OpenVINO™, aiming to simplify running inference of generative AI
models. It hides the complexity of the generation process and minimizes
the amount of code required.

Inference Visual language models can be implemented using OpenVINO GenAI
``VLMPipeline`` class. Similarly to LLMPipeline, that we discussed in
this
`notebook <https://openvinotoolkit.github.io/openvino_notebooks/?search=Create+an+LLM-powered+Chatbot+using+OpenVINO+Generate+API>`__.
It supports chat mode with preserving conversational history inside
pipeline, that allows us effectively implements chatbot that supports
conversation about provided images content.

.. code:: ipython3

    from openvino_genai import VLMPipeline, GenerationConfig

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



Select device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    
    from notebook_utils import device_widget
    
    device = device_widget(exclude=["NPU"])
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Select model variant
~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    model_base_path = model_path.parent
    available_models = []
    
    for precision in ["INT4", "INT8", "FP16"]:
        if (model_base_path / precision).exists():
            available_models.append(precision)
    
    model_variant = widgets.Dropdown(
        options=available_models,
        value=available_models[0],
        description="Compression mode:",
        disabled=False,
    )
    
    model_variant




.. parsed-literal::

    Dropdown(description='Compression mode:', options=('INT4', 'FP16'), value='INT4')



Load OpenVINO model
~~~~~~~~~~~~~~~~~~~



For pipeline initialization we should provide path to model directory
and inference device.

.. code:: ipython3

    ov_model = VLMPipeline(str(model_base_path / model_variant.value), device=device.value)

Run model inference
-------------------



Now, when we have model and defined generation pipeline, we can run
model inference.

Prepare input data
~~~~~~~~~~~~~~~~~~



For preparing input data, ``VLMPipeline`` use tokenizer and image
processor inside, we just need to convert image to input OpenVINO tensor
and provide question as string. Additionally, we can provides options
for controlling generation process (e.g. number of maximum generated
tokens or using multinomial sampling for decoding instead of greedy
search approach) using ``GenerationConfig``.

Generation process for long response may be time consuming, for
accessing partial result as soon as it is generated without waiting when
whole process finished, Streaming API can be used. Token streaming is
the mode in which the generative system returns the tokens one by one as
the model generates them. This enables showing progressive generations
to the user rather than waiting for the whole generation. Streaming is
an essential aspect of the end-user experience as it reduces latency,
one of the most critical aspects of a smooth experience.

.. code:: ipython3

    import requests
    from PIL import Image
    from io import BytesIO
    import numpy as np
    
    config = GenerationConfig()
    config.max_new_tokens = 100
    
    
    def load_image(image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        image_data = np.array(image.getdata()).reshape(1, 3, image.size[1], image.size[0]).astype(np.byte)
        return image, ov.Tensor(image_data)
    
    
    def streamer(subword: str) -> bool:
        """
    
        Args:
            subword: sub-word of the generated text.
    
        Returns: Return flag corresponds whether generation should be stopped.
    
        """
        print(subword, end="", flush=True)
    
    
    image_file = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
    
    image, image_tensor = load_image(image_file)
    text_message = "What is unusual on this image?"
    
    prompt = text_message

Test model inference
~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    display(image)
    print(f"Question:\n{text_message}")
    print("Answer:")
    output = ov_model.generate(prompt, image=image_tensor, generation_config=config, streamer=streamer)



.. image:: llava-multimodal-chatbot-genai-with-output_files/llava-multimodal-chatbot-genai-with-output_21_0.png


.. parsed-literal::

    Question:
    What is unusual on this image?
    Answer:
    
    The unusual aspect of this image is that a cat is lying inside a cardboard box. Cats are known for their curiosity and love for small, enclosed spaces. However, it is not a common sight to see a cat comfortably resting inside a cardboard box.

Interactive demo
----------------



.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/llava-multimodal-chatbot/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo_llava
    
    demo = make_demo_llava(ov_model)
    
    try:
        demo.launch(debug=False)
    except Exception:
        demo.launch(share=True, debug=False)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
