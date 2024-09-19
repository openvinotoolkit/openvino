Stable Diffusion v2.1 using OpenVINO TorchDynamo backend
========================================================

Stable Diffusion v2 is the next generation of Stable Diffusion model a
Text-to-Image latent diffusion model created by the researchers and
engineers from `Stability AI <https://stability.ai/>`__ and
`LAION <https://laion.ai/>`__.

General diffusion models are machine learning systems that are trained
to denoise random gaussian noise step by step, to get to a sample of
interest, such as an image. Diffusion models have shown to achieve
state-of-the-art results for generating image data. But one downside of
diffusion models is that the reverse denoising process is slow. In
addition, these models consume a lot of memory because they operate in
pixel space, which becomes unreasonably expensive when generating
high-resolution images. Therefore, it is challenging to train these
models and also use them for inference. OpenVINO brings capabilities to
run model inference on Intel hardware and opens the door to the
fantastic world of diffusion models for everyone!

This notebook demonstrates how to run stable diffusion model using
`Diffusers <https://huggingface.co/docs/diffusers/index>`__ library and
`OpenVINO TorchDynamo
backend <https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html>`__
for Text-to-Image and Image-to-Image generation tasks.

Notebook contains the following steps:

1. Create pipeline with PyTorch models.
2. Add OpenVINO optimization using OpenVINO TorchDynamo backend.
3. Run Stable Diffusion pipeline with OpenVINO.

**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Stable Diffusion with Diffusers
   library <#stable-diffusion-with-diffusers-library>`__
-  `OpenVINO TorchDynamo backend <#openvino-torchdynamo-backend>`__

   -  `Run Image generation <#run-image-generation>`__

-  `Interactive demo <#interactive-demo>`__
-  `Support for Automatic1111 Stable Diffusion
   WebUI <#support-for-automatic1111-stable-diffusion-webui>`__

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

    %pip install -q "torch>=2.2" transformers diffusers "gradio>=4.19" ipywidgets --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "openvino>=2024.1.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import torch
    
    from diffusers import StableDiffusionPipeline
    import ipywidgets as widgets


.. parsed-literal::

    2024-08-28 05:59:21.270898: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-08-28 05:59:21.305288: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-08-28 05:59:21.972104: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.


Stable Diffusion with Diffusers library
---------------------------------------



To work with Stable Diffusion v2.1, we will use Hugging Face Diffusers
library. To experiment with Stable Diffusion models, Diffusers exposes
the
`StableDiffusionPipeline <https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation>`__
and
`StableDiffusionImg2ImgPipeline <https://huggingface.co/docs/diffusers/using-diffusers/img2img>`__
similar to the other `Diffusers
pipelines <https://huggingface.co/docs/diffusers/api/pipelines/overview>`__.
The code below demonstrates how to create the
``StableDiffusionPipeline`` using ``stable-diffusion-2-1-base`` model:

.. code:: ipython3

    model_id = "stabilityai/stable-diffusion-2-1-base"
    
    # Pipeline for text-to-image generation
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)



.. parsed-literal::

    Loading pipeline components...:   0%|          | 0/6 [00:00<?, ?it/s]


OpenVINO TorchDynamo backend
----------------------------



The `OpenVINO TorchDynamo
backend <https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html>`__
lets you enable `OpenVINO <https://docs.openvino.ai/2024/home.html>`__
support for PyTorch models with minimal changes to the original PyTorch
script. It speeds up PyTorch code by JIT-compiling it into optimized
kernels. By default, Torch code runs in eager-mode, but with the use of
torch.compile it goes through the following steps: 1. Graph acquisition
- the model is rewritten as blocks of subgraphs that are either: -
compiled by TorchDynamo and “flattened”, - falling back to the
eager-mode, due to unsupported Python constructs (like control-flow
code). 2. Graph lowering - all PyTorch operations are decomposed into
their constituent kernels specific to the chosen backend. 3. Graph
compilation - the kernels call their corresponding low-level
device-specific operations.

Select device for inference and enable or disable saving the optimized
model files to a hard drive, after the first application run. This makes
them available for the following application executions, reducing the
first-inference latency. Read more about available `Environment
Variables
options <https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html#options>`__

.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="CPU",
        description="Device:",
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



.. code:: ipython3

    model_caching = widgets.Dropdown(
        options=[True, False],
        value=True,
        description="Model caching:",
        disabled=False,
    )
    
    model_caching




.. parsed-literal::

    Dropdown(description='Model caching:', options=(True, False), value=True)



To use `torch.compile()
method <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__,
you just need to add an import statement and define the OpenVINO
backend:

.. code:: ipython3

    # this import is required to activate the openvino backend for torchdynamo
    import openvino.torch  # noqa: F401
    
    pipe.unet = torch.compile(
        pipe.unet,
        backend="openvino",
        options={"device": device.value, "model_caching": model_caching.value},
    )

   **Note**: Read more about available `OpenVINO
   backends <https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html#how-to-use>`__

Run Image generation
~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]
    image


.. parsed-literal::

    Traceback (most recent call last):
      File "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-761/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/_inductor/compile_worker/__main__.py", line 45, in <module>
        main()
      File "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-761/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/_inductor/compile_worker/__main__.py", line 38, in main
        pre_fork_setup()
      File "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-761/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/_inductor/async_compile.py", line 62, in pre_fork_setup
        from triton.compiler.compiler import triton_key
    ImportError: cannot import name 'triton_key' from 'triton.compiler.compiler' (/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-761/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/triton/compiler/compiler.py)



.. parsed-literal::

      0%|          | 0/50 [00:00<?, ?it/s]




.. image:: stable-diffusion-torchdynamo-backend-with-output_files/stable-diffusion-torchdynamo-backend-with-output_14_2.png



Interactive demo
================



Now you can start the demo, choose the inference mode, define prompts
(and input image for Image-to-Image generation) and run inference
pipeline. Optionally, you can also change some input parameters.

.. code:: ipython3

    import requests
    from pathlib import Path
    
    if not Path("gradio_helper.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/stable-diffusion-torchdynamo-backend/gradio_helper.py"
        )
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    demo = make_demo(model_id)
    
    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(share=True, debug=False)
    
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.








Support for Automatic1111 Stable Diffusion WebUI
------------------------------------------------



Automatic1111 Stable Diffusion WebUI is an open-source repository that
hosts a browser-based interface for the Stable Diffusion based image
generation. It allows users to create realistic and creative images from
text prompts. Stable Diffusion WebUI is supported on Intel CPUs, Intel
integrated GPUs, and Intel discrete GPUs by leveraging OpenVINO
torch.compile capability. Detailed instructions are available
in\ `Stable Diffusion WebUI
repository <https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon>`__.
