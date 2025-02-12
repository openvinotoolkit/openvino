Image generation with Sana and OpenVINO
=======================================

Sana is a text-to-image framework that can efficiently generate images
up to 4096 × 4096 resolution developed by NVLabs. Sana can synthesize
high-resolution, high-quality images with strong text-image alignment at
a remarkably fast speed, deployable on laptop GPU. Core designs include:
\* **Deep compression autoencoder**: unlike traditional AEs, which
compress images only 8×, we trained an AE that can compress images 32×,
effectively reducing the number of latent tokens. \* **Linear DiT**:
authors replaced all vanilla attention in DiT with linear attention,
which is more efficient at high resolutions without sacrificing quality.
\* **Decoder-only text encoder**\ *: T5 replaced by modern decoder-only
small LLM as the text encoder and designed complex human instruction
with in-context learning to enhance the image-text alignment.*
**Efficient training and sampling**: Proposed Flow-DPM-Solver to reduce
sampling steps, with efficient caption labeling and selection to
accelerate convergence.

More details about model can be found in
`paper <https://arxiv.org/abs/2410.10629>`__, `model
page <https://nvlabs.github.io/Sana/>`__ and `original
repo <https://github.com/NVlabs/Sana>`__. In this tutorial, we consider
how to optimize and run Sana model using OpenVINO.

**Table of contents:**

-  `Prerequisites <#prerequisites>`__
-  `Select model variant <#select-model-variant>`__
-  `Convert and Optimize model with
   OpenVINO <#convert-and-optimize-model-with-openvino>`__

   -  `Convert model using Optimum
      Intel <#convert-model-using-optimum-intel>`__
   -  `Compress model weights <#compress-model-weights>`__

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

    import platform

    %pip install -q "gradio>=4.19" "torch>=2.1"  "transformers" "nncf>=2.14.0" "diffusers>=0.32.0" "opencv-python" "pillow" "peft>=0.7.0" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "sentencepiece" "protobuf"
    %pip install -q "git+https://github.com/huggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -qU "openvino>=2024.6.0"

    if platform.system() == "Darwin":
        %pip install "numpy<2.0"

.. code:: ipython3

    from pathlib import Path
    import requests

    helpers = ["notebook_utils.py", "cmd_helper.py"]
    base_url = "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils"

    for helper in helpers:
        if not Path(helper).exists():
            r = requests.get(f"{base_url}/{helper}")
            with open(helper, "w") as f:
                f.write(r.text)

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/sana-image-generation/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)

    # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
    from notebook_utils import collect_telemetry

    collect_telemetry("sana-image-generation.ipynb")

Select model variant
--------------------



.. code:: ipython3

    import ipywidgets as widgets

    model_ids = [
        "Efficient-Large-Model/Sana_600M_512px_diffusers",
        "Efficient-Large-Model/Sana_600M_1024px_diffusers",
        "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        "Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers",
        "Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers",
    ]

    model_selector = widgets.Dropdown(
        options=model_ids,
        default=model_ids[0],
        description="Model:",
    )


    model_selector




.. parsed-literal::

    Dropdown(description='Model:', options=('Efficient-Large-Model/Sana_600M_512px_diffusers', 'Efficient-Large-Mo…



Convert and Optimize model with OpenVINO
----------------------------------------



Starting from 2023.0 release, OpenVINO supports PyTorch models directly
via Model Conversion API. ``ov.convert_model`` function accepts instance
of PyTorch model and example inputs for tracing and returns object of
``ov.Model`` class, ready to use or save on disk using ``ov.save_model``
function.

The pipeline consists of four important parts:

-  Gemma Text Encoder to create condition to generate an image from a
   text prompt.
-  Transformer for step-by-step denoising latent image representation.
-  Deep Compression Autoencoder (DCAE) for decoding latent space to
   image.

Convert model using Optimum Intel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



For convenience, we will use OpenVINO integration with HuggingFace
Optimum. `Optimum
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
will be auto-inferred based on the model (in case of image generation,
**text-to-image** should be selected). You can find a mapping between
tasks and model classes in Optimum TaskManager
`documentation <https://huggingface.co/docs/optimum/exporters/task_manager>`__.
Additionally, you can specify weights compression using
``--weight-format`` argument with one of following options: ``fp32``,
``fp16``, ``int8`` and ``int4``. For int8 and int4
`nncf <https://github.com/openvinotoolkit/nncf>`__ will be used for
weight compression. More details about model export provided in `Optimum
Intel
documentation <https://huggingface.co/docs/optimum/intel/openvino/export#export-your-model>`__.

.. code:: ipython3

    from pathlib import Path

    model_id = model_selector.value
    variant = "fp16" if "BF16" not in model_id else "bf16"

    model_dir = Path(model_id.split("/")[-1])

    additional_args = {"variant": variant, "weight-format": "fp16"}

.. code:: ipython3

    from cmd_helper import optimum_cli

    if not model_dir.exists():
        optimum_cli(model_id, model_dir, additional_args=additional_args)

Compress model weights
~~~~~~~~~~~~~~~~~~~~~~



For reducing model memory consumption we will use weights compression.
The `Weights
Compression <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html>`__
algorithm is aimed at compressing the weights of the models and can be
used to optimize the model footprint and performance of large models
where the size of weights is relatively larger than the size of
activations, for example, Large Language Models (LLM). Compared to INT8
compression, INT4 compression improves performance even more, but
introduces a minor drop in prediction quality. We will use
`NNCF <https://github.com/openvinotoolkit/nncf>`__ for transformer
weight compression.

.. code:: ipython3

    to_compress = widgets.Checkbox(
        value=True,
        description="Weight compression",
        disabled=False,
    )

    to_compress




.. parsed-literal::

    Checkbox(value=True, description='Weight compression')



.. code:: ipython3

    import openvino as ov
    import nncf
    import gc

    compressed_transformer = Path(model_dir) / "transformer/openvino_model_i4.xml"

    if to_compress.value and not compressed_transformer.exists():
        core = ov.Core()

        ov_model = core.read_model(model_dir / "transformer/openvino_model.xml")

        compressed_model = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT4_SYM, group_size=64, ratio=1.0)
        ov.save_model(compressed_model, compressed_transformer)
        del compressed_model
        del ov_model

        gc.collect();

















Run OpenVINO model inference
----------------------------



``OVDiffusionPipeline`` from Optimum Intel provides ready-to-use
interface for running Diffusers models using OpenVINO. It supports
various models including Stable Diffusion, Stable Diffusion XL, LCM,
Stable Diffusion v3 and Flux. Similar to original Diffusers pipeline,
for initialization, we should use ``from_preptrained`` method providing
model id from HuggingFace hub or local directory (both original PyTorch
and OpenVINO models formats supported, in the first case model class
additionally will trigger model conversion).

.. code:: ipython3

    from notebook_utils import device_widget

    device = device_widget(default="CPU", exclude=["NPU"])
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



.. code:: ipython3

    from optimum.intel.openvino import OVDiffusionPipeline

    ov_pipe = OVDiffusionPipeline.from_pretrained(model_dir, device=device.value, transformer_file_name=compressed_transformer.name if to_compress.value else None)

.. code:: ipython3

    import torch

    prompt = "Cute 🐶 Wearing 🕶 flying on the 🌈"

    image = ov_pipe(
        prompt,
        generator=torch.Generator("cpu").manual_seed(1234563),
    ).images[0]

    image



.. parsed-literal::

      0%|          | 0/20 [00:00<?, ?it/s]




.. image:: sana-image-generation-with-output_files/sana-image-generation-with-output_15_1.png



Interactive demo
----------------



.. code:: ipython3

    from gradio_helper import make_demo

    demo = make_demo(ov_pipe)

    # if you are launching remotely, specify server_name and server_port
    #  demo.launch(server_name='your server name', server_port='server port in int')
    # if you have any issue to launch on your platform, you can pass share=True to launch method:
    # demo.launch(share=True)
    # it creates a publicly shareable link for the interface. Read more in the docs: https://gradio.app/docs/
    try:
        demo.launch(debug=True)
    except Exception:
        demo.launch(debug=True, share=True)
