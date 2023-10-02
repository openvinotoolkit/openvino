# PyTorch Deployment via "torch.compile" {#pytorch_2_0_torch_compile}

@sphinxdirective


The ``torch.compile`` feature enables you to use OpenVINO for PyTorch-native applications. 
It speeds up PyTorch code by JIT-compiling it into optimized kernels.
By default, Torch code runs in eager-mode, but with the use of ``torch.compile`` it goes through the following steps:

1. **Graph acquisition** - the model is rewritten as blocks of subgraphs that are either:

   * compiled by TorchDynamo and "flattened",
   * falling back to the eager-mode, due to unsupported Python constructs (like control-flow code).

2. **Graph lowering** - all PyTorch operations are decomposed into their constituent kernels specific to the chosen backend.
3. **Graph compilation** - the kernels call their corresponding low-level device-specific operations.



How to Use
#################

To use ``torch.compile``, you need to add an import statement and define one of the two available backends:

| ``openvino``
|   With this backend, Torch FX subgraphs are directly converted to OpenVINO representation without any additional PyTorch based tracing/scripting.

| ``openvino_ts``
|   With this backend, Torch FX subgraphs are first traced/scripted with PyTorch Torchscript, and then converted to OpenVINO representation.


.. tab-set::

   .. tab-item:: openvino
      :sync: backend-openvino

      .. code-block:: console

         import openvino.torch 
         ...
         model = torch.compile(model, backend='openvino')

      Execution diagram:

      .. image:: _static/images/torch_compile_backend_openvino.svg
         :width: 992px
         :height: 720px
         :scale: 60%
         :align: center

   .. tab-item:: openvino_ts
      :sync: backend-openvino-ts

      .. code-block:: console

         import openvino.torch
         ...
         model = torch.compile(model, backend='openvino_ts')

      Execution diagram:

      .. image:: _static/images/torch_compile_backend_openvino_ts.svg
         :width: 1088px
         :height: 720px
         :scale: 60%
         :align: center


Environment Variables
+++++++++++++++++++++++++++

* **OPENVINO_TORCH_BACKEND_DEVICE**: enables selecting a specific hardware device to run the application. 
  By default, the OpenVINO backend for ``torch.compile`` runs PyTorch applications using the CPU. Setting 
  this variable to GPU.0, for example, will make the application use the integrated graphics processor instead.
* **OPENVINO_TORCH_MODEL_CACHING**: enables saving the optimized model files to a hard drive, after the first application run.
  This makes them available for the following application executions, reducing the first-inference latency.
  By default, this variable is set to ``False``. Setting it to ``True`` enables caching.
* **OPENVINO_TORCH_CACHE_DIR**: enables defining a custom directory for the model files (if model caching set to ``True``).
  By default, the OpenVINO IR is saved in the ``cache`` sub-directory, created in the application's root directory. 

Windows support
++++++++++++++++++++++++++

Currently, PyTorch does not support ``torch.compile`` feature on Windows officially. However, it can be accessed by running
the below instructions:

1. Install the PyTorch nightly wheel file - `2.1.0.dev20230713 <https://download.pytorch.org/whl/nightly/cpu/torch-2.1.0.dev20230713%2Bcpu-cp38-cp38-win_amd64.whl>`__ ,
2. Update the file at ``<python_env_root>/Lib/site-packages/torch/_dynamo/eval_frames.py``
3. Find the function called ``check_if_dynamo_supported()``:

   .. code-block:: console

      def check_if_dynamo_supported():
          if sys.platform == "win32":
              raise RuntimeError("Windows not yet supported for torch.compile")
          if sys.version_info >= (3, 11):
              raise RuntimeError("Python 3.11+ not yet supported for torch.compile")

4. Put in comments the first two lines in this function, so it looks like this:

   .. code-block:: console

      def check_if_dynamo_supported():
       #if sys.platform == "win32":
       #    raise RuntimeError("Windows not yet supported for torch.compile")
       if sys.version_info >= (3, 11):
           `raise RuntimeError("Python 3.11+ not yet supported for torch.compile")


Support for Automatic1111 Stable Diffusion WebUI
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Automatic1111 Stable Diffusion WebUI is an open-source repository that hosts a browser-based interface for the Stable Diffusion 
based image generation. It allows users to create realistic and creative images from text prompts. 
Stable Diffusion WebUI is supported on Intel CPUs, Intel integrated GPUs, and Intel discrete GPUs by leveraging OpenVINO 
``torch.compile`` capability. Detailed instructions are available in 
`Stable Diffusion WebUI repository. <https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon>`__


Architecture
#################

The ``torch.compile`` feature is part of PyTorch 2.0, and is based on:

* **TorchDynamo** - a Python-level JIT that hooks into the frame evaluation API in CPython,
  (PEP 523) to dynamically modify Python bytecode right before it is executed (PyTorch operators 
  that cannot be extracted to FX graph are executed in the native Python environment). 
  It maintains the eager-mode capabilities using 
  `Guards <https://pytorch.org/docs/stable/dynamo/guards-overview.html>`__ to ensure the 
  generated graphs are valid.

* **AOTAutograd** - generates the backward graph corresponding to the forward graph captured by TorchDynamo.
* **PrimTorch** - decomposes complicated PyTorch operations into simpler and more elementary ops.
* **TorchInductor** - a deep learning compiler that generates fast code for multiple accelerators and backends.




When the PyTorch module is wrapped with ``torch.compile``, TorchDynamo traces the module and 
rewrites Python bytecode to extract sequences of PyTorch operations into an FX Graph,
which can be optimized by the OpenVINO backend. The Torch FX graphs are first converted to 
inlined FX graphs and the graph partitioning module traverses inlined FX graph to identify 
operators supported by OpenVINO. 

All the supported operators are clustered into OpenVINO submodules, converted to the OpenVINO 
graph using OpenVINO's PyTorch decoder, and executed in an optimized manner using OpenVINO runtime. 
All unsupported operators fall back to the native PyTorch runtime on CPU. If the subgraph 
fails during OpenVINO conversion, the subgraph falls back to PyTorch's default inductor backend.



Additional Resources
############################

* `PyTorch 2.0 documentation <https://pytorch.org/docs/stable/index.html>`_

@endsphinxdirective
