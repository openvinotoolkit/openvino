Visual-language assistant with MiniCPM-V2 and OpenVINO
======================================================

MiniCPM-V 2 is a strong multimodal large language model for efficient
end-side deployment. MiniCPM-V 2.6 is the latest and most capable model
in the MiniCPM-V series. The model is built on SigLip-400M and Qwen2-7B
with a total of 8B parameters. It exhibits a significant performance
improvement over previous versions, and introduces new features for
multi-image and video understanding.

More details about model can be found in `model
card <https://huggingface.co/openbmb/MiniCPM-V-2_6>`__ and original
`repo <https://github.com/OpenBMB/MiniCPM-V>`__.

In this tutorial we consider how to convert and optimize MiniCPM-V2
model for creating multimodal chatbot. Additionally, we demonstrate how
to apply stateful transformation on LLM part and model optimization
techniques like weights compression using
`NNCF <https://github.com/openvinotoolkit/nncf>`__


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Convert model to OpenVINO Intermediate
   Representation <#convert-model-to-openvino-intermediate-representation>`__

   -  `Compress Language Model Weights to 4
      bits <#compress-language-model-weights-to-4-bits>`__

-  `Prepare model inference
   pipeline <#prepare-model-inference-pipeline>`__

   -  `Select device <#select-device>`__

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

    %pip install -q "torch>=2.1" "torchvision" "timm>=0.9.2" "transformers>=4.45" "Pillow" "gradio>=4.19" "tqdm" "sentencepiece" "peft" "huggingface-hub>=0.24.0" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "nncf>=2.14.0"
    %pip install -q "git+https://github.com/huggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q -U "openvino>=2024.5" "openvino-tokenizers>=2024.5" "openvino-genai>=2024.5"

.. code:: ipython3

    import requests
    from pathlib import Path
    
    if not Path("cmd_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/cmd_helper.py")
        open("cmd_helper.py", "w").write(r.text)
    
    
    if not Path("gradio_helper.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks//minicpm-v-multimodal-chatbot//gradio_helper.py"
        )
        open("gradio_helper.py", "w").write(r.text)
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w").write(r.text)

Convert model to OpenVINO Intermediate Representation
-----------------------------------------------------



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

Compress Language Model Weights to 4 bits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



For reducing memory consumption, weights compression optimization can be
applied using `NNCF <https://github.com/openvinotoolkit/nncf>`__.

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

    from cmd_helper import optimum_cli
    import nncf
    import openvino as ov
    import shutil
    import gc
    
    
    def compress_lm_weights(model_dir):
        compression_configuration = {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 1.0, "all_layers": True}
        ov_model_path = model_dir / "openvino_language_model.xml"
        ov_int4_model_path = model_dir / "openvino_language_model_int4.xml"
        ov_model = ov.Core().read_model(ov_model_path)
        ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
        ov.save_model(ov_compressed_model, ov_int4_model_path)
        del ov_compressed_model
        del ov_model
        gc.collect()
        ov_model_path.unlink()
        ov_model_path.with_suffix(".bin").unlink()
        shutil.move(ov_int4_model_path, ov_model_path)
        shutil.move(ov_int4_model_path.with_suffix(".bin"), ov_model_path.with_suffix(".bin"))
    
    
    model_id = "openbmb/MiniCPM-V-2_6"
    model_dir = Path(model_id.split("/")[-1] + "-ov")
    
    if not model_dir.exists():
        optimum_cli(model_id, model_dir, additional_args={"trust-remote-code": "", "weight-format": "fp16"})
        compress_lm_weights(model_dir)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino
    

Prepare model inference pipeline
--------------------------------



.. image:: https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/2727402e-3697-442e-beca-26b149967c84

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

Select device
~~~~~~~~~~~~~



.. code:: ipython3

    from notebook_utils import device_widget
    
    device = device_widget(default="AUTO", exclude=["NPU"])
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    import openvino_genai as ov_genai
    
    ov_model = ov_genai.VLMPipeline(model_dir, device=device.value)

Run OpenVINO model inference
----------------------------



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
    
    image_path = "cat.png"
    
    
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 100
    
    
    def load_image(image_file):
        if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
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
    
    
    if not Path(image_path).exists():
        url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
        image = Image.open(requests.get(url, stream=True).raw)
        image.save(image_path)

.. code:: ipython3

    image, image_tensor = load_image(image_path)
    
    question = "What is unusual on this image?"
    
    print(f"Question:\n{question}")
    image


.. parsed-literal::

    Question:
    What is unusual on this image?
    



.. image:: minicpm-v-multimodal-chatbot-with-output_files/minicpm-v-multimodal-chatbot-with-output_12_1.png



.. code:: ipython3

    ov_model.start_chat()
    output = ov_model.generate(question, image=image_tensor, generation_config=config, streamer=streamer)


.. parsed-literal::

    The unusual aspect of this image is the cat's relaxed and vulnerable position. Typically, cats avoid exposing their bellies, which are sensitive and vulnerable areas, to potential threats. In this image, the cat is lying on its back in a cardboard box, exposing its belly and hindquarters, which is not a common sight. This behavior could indicate that the cat feels safe and comfortable in its environment, suggesting a strong bond with its owner and a sense of security in its home.

Interactive demo
----------------



.. code:: ipython3

    from gradio_helper import make_demo
    
    demo = make_demo(ov_model)
    
    try:
        demo.launch(debug=True, height=600)
    except Exception:
        demo.launch(debug=True, share=True, height=600)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
