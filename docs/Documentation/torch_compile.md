# OpenVINO PyTorch 2.0 - torch.compile() backend {#pytorch_2_0_torch_compile}

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
    ... what this means for the user ...

| ``openvino_ts``
|   With this backend, Torch FX subgraphs are first traced/scripted with PyTorch Torchscript, and then converted to OpenVINO representation.
    ... what this means for the user ...


.. tab-set::

   .. tab-item:: openvino
      :sync: backend-openvino

      .. code-block:: console

         import torch
         from openvino.frontend.pytorch.torchdynamo import backend
         ...
         model = torch.compile(model, backend='openvino')

      Execution diagram:

      .. image:: _static/images/torch_compile_backend_openvino.svg
         :scale: 50%

   .. tab-item:: openvino_ts
      :sync: backend-openvino-ts

      .. code-block:: console

         import torch
         from openvino.frontend.pytorch.torchdynamo import backend
         ...
         model = torch.compile(model, backend='openvino_ts')

      Execution diagram:

      .. image:: _static/images/torch_compile_backend_openvino_ts.svg
         :scale: 50%


Environment Variables
+++++++++++++++++++++++++++

* **OPENVINO_TORCH_BACKEND_DEVICE**: enables selecting a speciffic hardware device to run the application. 
  By default, the OpenVINO backend for ``torch.compile`` runs PyTorch applications using the CPU. Setting 
  this variable to GPU.0, for example, will make the application use the integrated graphics processor instead.
* **OPENVINO_TORCH_MODEL_CACHING**: enables saving the optimized model files to a hard drive, after the first application run.
  This makes them available for the following application executions, redicing the first-inference latency.
  By default, this variable is set to ``False``. Setting it to ``True`` enables caching.
* **OPENVINO_TORCH_CACHE_DIR**: enables defining a custom directory for the model files (if model caching set to ``True``).
  By default, the OpenVINO IR is saved in a the ``cache`` sub-directory, created in the application's root directory. 
  

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
rewrites Python bytecode in order to extract sequences of PyTorch operations into an FX Graph,
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
