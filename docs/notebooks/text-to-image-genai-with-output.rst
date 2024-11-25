Text to Image pipeline and OpenVINO with Generate API
=====================================================

`OpenVINO™ GenAI <https://github.com/openvinotoolkit/openvino.genai>`__
is a library of the most popular Generative AI model pipelines,
optimized execution methods, and samples that run on top of highly
performant OpenVINO Runtime.

This library is friendly to PC and laptop execution, and optimized for
resource consumption. It requires no external dependencies to run
generative models as it already includes all the core functionality
(e.g. tokenization via openvino-tokenizers).

In this notebook we will demonstrate how to use text to image models
like Stable Diffusion 1.5, 2.1, LCM using `Dreamlike Anime
1.0 <https://huggingface.co/dreamlike-art/dreamlike-anime-1.0>`__ as an
example. All it takes is two steps: 1. Export OpenVINO IR format model
using the `Hugging Face
Optimum <https://huggingface.co/docs/optimum/installation>`__ library
accelerated by OpenVINO integration. The Hugging Face Optimum Intel API
is a high-level API that enables us to convert and quantize models from
the Hugging Face Transformers library to the OpenVINO™ IR format. For
more details, refer to the `Hugging Face Optimum Intel
documentation <https://huggingface.co/docs/optimum/intel/inference>`__.
2. Run inference using the `Text to Image
pipeline <https://openvino-doc.iotg.sclab.intel.com/nightly/learn-openvino/llm_inference_guide/genai-guide/genai-use-cases.html#using-genai-for-text-to-image-generation>`__
from OpenVINO GenAI.

**Table of contents:**

-  `Prerequisites <#prerequisites>`__
-  `Convert model using Optimum-CLI
   tool <#convert-model-using-optimum-cli-tool>`__
-  `Run inference OpenVINO model with
   Text2ImagePipeline <#run-inference-openvino-model-with-text2imagepipeline>`__
-  `Run inference OpenVINO model with Text2ImagePipeline with optional
   LoRA
   adapters <#run-inference-openvino-model-with-text2imagepipeline-with-optional-lora-adapters>`__
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

    import platform
    import requests


    %pip install -q "git+https://github.com/huggingface/optimum-intel.git"
    %pip install -q -U "openvino>=2024.5" "openvino-tokenizers>=2024.5" "openvino-genai>=2024.5"
    %pip install -q Pillow "diffusers>=0.30.3" "gradio>=4.19" "typing_extensions>=4.9"
    if platform.system() == "Darwin":
        %pip install -q "numpy<2.0.0"

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/cmd_helper.py",
    )
    open("cmd_helper.py", "w").write(r.text)

Convert model using Optimum-CLI tool
------------------------------------



`Optimum Intel <https://huggingface.co/docs/optimum/intel/index>`__
is the interface between the
`Transformers <https://huggingface.co/docs/transformers/index>`__ and
`Diffusers <https://huggingface.co/docs/diffusers/index>`__ libraries
and OpenVINO to accelerate end-to-end pipelines on Intel architectures.
It provides ease-to-use cli interface for exporting models to `OpenVINO
Intermediate Representation
(IR) <https://docs.openvino.ai/2024/documentation/openvino-ir-format.html>`__
format.

The command bellow demonstrates basic command for model export with
``optimum-cli``

.. code:: bash

   optimum-cli export openvino --model <model_id_or_path> --task <task> <out_dir>

where ``--model`` argument is model id from HuggingFace Hub or local
directory with model (saved using ``.save_pretrained`` method),
``--task`` is one of `supported
task <https://huggingface.co/docs/optimum/exporters/task_manager>`__
that exported model should solve. For image generation models,
``text-to-image`` should be used. If model initialization requires to
use remote code, ``--trust-remote-code`` flag additionally should be
passed. You can also apply fp16, 8-bit or 4-bit weight compression on
the Linear, Convolutional and Embedding layers when exporting your model
with the CLI by setting ``--weight-format`` to respectively fp16, int8
or int4. This type of optimization allows to reduce the memory footprint
and inference latency.

We will use ``optimum_cli`` from our helper ``cmd_helper.py`` that is a
wrapper over cli-command.

.. code:: ipython3

    from pathlib import Path

    from cmd_helper import optimum_cli


    model_dir = Path("dreamlike_anime_1_0_ov")

    if not model_dir.exists():
        optimum_cli("dreamlike-art/dreamlike-anime-1.0", model_dir)

Run inference OpenVINO model with Text2ImagePipeline
----------------------------------------------------



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    from notebook_utils import device_widget


    device = device_widget("CPU", exclude=["NPU"])
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



And now just provide ``model_dir`` and the chosen inference device to
``openvino_genai.Text2ImagePipeline`` and call ``generate`` method for
inference. ``openvino_genai.Generator`` class wraps ``std::mt19937``
pseudo-random generator. It can be used for results reproducibility.
That’s it:)

.. code:: ipython3

    import openvino_genai as ov_genai
    import openvino as ov
    from PIL import Image
    import torch


    class Generator(ov_genai.Generator):
        def __init__(self, seed):
            ov_genai.Generator.__init__(self)
            self.generator = torch.Generator(device="cpu").manual_seed(seed)

        def next(self):
            return torch.randn(1, generator=self.generator, dtype=torch.float32).item()

        def randn_tensor(self, shape: ov.Shape):
            torch_tensor = torch.randn(list(shape), generator=self.generator, dtype=torch.float32)
            return ov.Tensor(torch_tensor.numpy())


    random_generator = Generator(42)  # openvino_genai.CppStdGenerator can be used to have same images as C++ sample
    pipe = ov_genai.Text2ImagePipeline(model_dir, device.value)
    prompt = "anime, masterpiece, high quality, a green snowman with a happy smiling face in the snows"

    image_tensor = pipe.generate(prompt, width=512, height=512, num_inference_steps=20, num_images_per_prompt=1, generator=random_generator)

    image = Image.fromarray(image_tensor.data[0])

.. code:: ipython3

    image




.. image:: text-to-image-genai-with-output_files/text-to-image-genai-with-output_9_0.png



Run inference OpenVINO model with Text2ImagePipeline with optional LoRA adapters
--------------------------------------------------------------------------------



LoRA adapters can be connected to the pipeline and modify generated
images to have certain style, details or quality. Adapters are supported
in ``Safetensors`` format and can be downloaded from public sources like
`Civitai <https://civitai.com>`__ or
`HuggingFace <https://huggingface.co/models>`__ or trained by the user.
Adapters compatible with a base model should be used only. A weighted
blend of multiple adapters can be applied by specifying multiple adapter
files with corresponding alpha parameters in command line. Check
``lora.cpp`` source code to learn how to enable adapters and specify
them in each ``generate`` call.

Here is an example how to run the sample with a single adapter. First
download adapter file from https://civitai.com/models/67927/soulcard
page manually and save it as ``soulcard.safetensors``. Or download it
from command line:

.. code:: ipython3

    r = requests.get(
        url="https://civitai.com/api/download/models/72591",
    )
    with open("soulcard.safetensors", "wb") as file:
        file.write(r.content)

.. code:: ipython3

    def prepare_adapter_config(adapters):
        adapter_config = ov_genai.AdapterConfig()

        # Multiple LoRA adapters applied simultaneously are supported, parse them all and corresponding alphas from cmd parameters:
        for i in range(int(len(adapters) / 2)):
            adapter = ov_genai.Adapter(adapters[2 * i])
            alpha = float(adapters[2 * i + 1])
            adapter_config.add(adapter, alpha)

        return adapter_config


    adapter_config = prepare_adapter_config(["soulcard.safetensors", 0.5])

    pipe = ov_genai.Text2ImagePipeline(model_dir, device.value, adapters=adapter_config)

    image_tensor = pipe.generate(prompt, generator=Generator(42), width=512, height=512, num_inference_steps=20)
    image = Image.fromarray(image_tensor.data[0])

.. code:: ipython3

    image




.. image:: text-to-image-genai-with-output_files/text-to-image-genai-with-output_13_0.png



You can find more detailed tutorial for running inference with multiple
LoRA adapters in this
`notebook <multilora-image-generation-with-output.html>`__

Interactive demo
----------------



.. code:: ipython3

    from gradio_helper import make_demo


    demo = make_demo(pipe, Generator, adapter_config)

    try:
        demo.launch(debug=True)
    except Exception:
        demo.launch(share=True, debug=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
