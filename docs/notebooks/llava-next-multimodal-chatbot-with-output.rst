Visual-language assistant with LLaVA Next and OpenVINO
======================================================

`LLaVA-NeXT <https://llava-vl.github.io/blog/2024-01-30-llava-next/>`__
is new generation of LLaVA model family that marks breakthrough in
advanced language reasoning over images, introducing improved OCR and
expanded world knowledge. `LLaVA <https://llava-vl.github.io>`__ (Large
Language and Vision Assistant) is large multimodal model that aims to
develop a general-purpose visual assistant that can follow both language
and image instructions to complete various real-world tasks. The idea is
to combine the power of large language models (LLMs) with vision
encoders like CLIP to create an end-to-end trained neural assistant that
understands and acts upon multimodal instructions.

In this tutorial we consider how to convert and optimize LLaVA-NeXT
model from Transformers library for creating multimodal chatbot. We will
utilize the power of
`llava-v1.6-mistral-7b <https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf>`__
model for creating multimodal chatbot, but the similar actions are also
applicable to other models of LLaVA family compatible with HuggingFace
transformers implementation. Additionally, we demonstrate how to apply
stateful transformation on LLM part and model optimization techniques
like weights compression using
`NNCF <https://github.com/openvinotoolkit/nncf>`__


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Convert model to OpenVINO IR format using Optimum
   CLI <#convert-model-to-openvino-ir-format-using-optimum-cli>`__
-  `Compress Language Model Weights to 4
   bits <#compress-language-model-weights-to-4-bits>`__
-  `Prepare model inference
   pipeline <#prepare-model-inference-pipeline>`__

   -  `Select device <#select-device>`__
   -  `Select model variant <#select-model-variant>`__
   -  `Load OpenVINO Model <#load-openvino-model>`__

-  `Run OpenVINO model inference <#run-openvino-model-inference>`__
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

    # %pip install -q "nncf>=2.14.0" "torch>=2.1" "transformers>=4.39.1" "accelerate" "pillow" "gradio>=4.26" "datasets>=2.14.6" "tqdm" --extra-index-url https://download.pytorch.org/whl/cpu
    # %pip install -q -U "openvino>=2024.5.0" "openvino-tokenizers>=2024.5.0" "openvino-genai>=2024.5"
    # %pip install -q "git+https://github.com/hugggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu

.. code:: ipython3

    from pathlib import Path
    
    import requests
    
    utility_files = ["notebook_utils.py", "cmd_helper.py"]
    
    for utility in utility_files:
        local_path = Path(utility)
        if not local_path.exists():
            r = requests.get(
                url=f"https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/{local_path.name}",
            )
            with local_path.open("w") as f:
                f.write(r.text)
    
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    MODEL_DIR = Path(model_id.split("/")[-1].replace("-hf", "-ov"))

Convert model to OpenVINO IR format using Optimum CLI
-----------------------------------------------------



OpenVINO supports PyTorch models via conversion to OpenVINO Intermediate
Representation (IR) format. For convenience, we will use OpenVINO
integration with HuggingFace Optimum. `Optimum
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

    from cmd_helper import optimum_cli
    
    if not (MODEL_DIR / "FP16").exists():
        optimum_cli(model_id, MODEL_DIR / "FP16", additional_args={"weight-format": "fp16"})

Compress Language Model Weights to 4 bits
-----------------------------------------



For reducing memory consumption, weights compression optimization can be
applied using `NNCF <https://github.com/openvinotoolkit/nncf>`__. Weight
compression aims to reduce the memory footprint of a model. It can also
lead to significant performance improvement for large memory-bound
models, such as Large Language Models (LLMs).

LLMs and other models, which require extensive memory to store the
weights during inference, can benefit from weight compression in the
following ways:

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

   **Note:** weights compression process may require additional time and
   memory for performing. You can disable it using widget below:

.. code:: ipython3

    import ipywidgets as widgets
    
    to_compress_weights = widgets.Checkbox(
        value=True,
        description="Weights Compression",
        disabled=False,
    )
    
    to_compress_weights




.. parsed-literal::

    Checkbox(value=True, description='Weights Compression')



.. code:: ipython3

    import shutil
    import nncf
    import openvino as ov
    import gc
    
    compression_configuration = {
        "mode": nncf.CompressWeightsMode.INT4_SYM,
        "group_size": 64,
        "ratio": 0.6,
    }
    
    core = ov.Core()
    
    
    def copy_model_folder(src, dst, ignore_file_names=None):
        ignore_file_names = ignore_file_names or []
    
        for file_name in Path(src).glob("*"):
            if file_name.name in ignore_file_names:
                continue
            shutil.copy(file_name, dst / file_name.relative_to(src))
    
    
    LANGUAGE_MODEL_PATH_INT4 = MODEL_DIR / "INT4/openvino_language_model.xml"
    LANGUAGE_MODEL_PATH = MODEL_DIR / "FP16/openvino_language_model.xml"
    if to_compress_weights.value and not LANGUAGE_MODEL_PATH_INT4.exists():
        ov_model = core.read_model(LANGUAGE_MODEL_PATH)
        ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
        ov.save_model(ov_compressed_model, LANGUAGE_MODEL_PATH_INT4)
        del ov_compressed_model
        del ov_model
        gc.collect()
    
        copy_model_folder(MODEL_DIR / "FP16", MODEL_DIR / "INT4", ["openvino_language_model.xml", "openvino_language_model.bin"])


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Prepare model inference pipeline
--------------------------------



|image0|

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

.. |image0| image:: https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/a562e9de-5b94-4e24-ac52-532019fc92d3

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



Select device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    from notebook_utils import device_widget
    
    device = device_widget("CPU", exclude=["NPU"])
    
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



Select model variant
~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    import ipywidgets as widgets
    
    use_int4_lang_model = widgets.Checkbox(
        value=LANGUAGE_MODEL_PATH_INT4.exists(),
        description="INT4 language model",
        disabled=not LANGUAGE_MODEL_PATH_INT4.exists(),
    )
    
    use_int4_lang_model




.. parsed-literal::

    Checkbox(value=True, description='INT4 language model')



Load OpenVINO model
~~~~~~~~~~~~~~~~~~~



For pipeline initialization we should provide path to model directory
and inference device.

.. code:: ipython3

    import openvino_genai as ov_genai
    
    model_dir = MODEL_DIR / "FP16" if not use_int4_lang_model.value else MODEL_DIR / "INT4"
    
    ov_model = ov_genai.VLMPipeline(model_dir, device=device.value)

Run OpenVINO model inference
----------------------------



Now, when we have model and defined generation pipeline, we can run
model inference.

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
    
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 100
    
    
    def load_image(image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
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
    
    display(image)
    print(f"Question:\n{text_message}")
    print("Answer:")
    output = ov_model.generate(prompt, image=image_tensor, generation_config=config, streamer=streamer)



.. image:: llava-next-multimodal-chatbot-with-output_files/llava-next-multimodal-chatbot-with-output_17_0.png


.. parsed-literal::

    Question:
    What is unusual on this image?
    Answer:
    
    
    The unusual aspect of this image is that a cat is lying inside a cardboard box. Cats are known for their curiosity and love for small, enclosed spaces. They often find comfort and security in boxes, bags, or other confined spaces. In this case, the cat has chosen to lie down in a cardboard box, which is an unconventional and amusing sight. It is not common to see a cat lounging in a box, as they usually

Interactive demo
----------------



.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/llava-next-multimodal-chatbot/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    demo = make_demo(ov_model)
    
    try:
        demo.launch(debug=False)
    except Exception:
        demo.launch(debug=False, share=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
