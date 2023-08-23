# OpenVINO PyTorch 2.0 - torch.compile() backend {#pytorch_2_0_torch_compile}

@sphinxdirective

PyTorch 2.0 simplifies backend (compiler) integration for developers and vendors by reducing more than 2000 Ops to a set of around 250 primitive Ops and simplyfing Op semantics. Likened to Graph Mode, PyTorch 2.0 offers the same eager-mode development experience, while adding a compiled mode via ``torch.compile`` - a new feature used for speeding up PyTorch code by JIT-compiling PyTorch code into optimized kernels. 

OpenVINO is enabled in the ``torch.compile`` to optimize generation of the graph model. This allows developers to leverage OpenVINO optimizations in PyTorch native applications on Intel CPUs, Intel integrated and discrete GPUs.

Torch code runs in eager-mode by default, but when you use ``torch.compile`` it goes through the following steps:

1. **Graph acquisition** - the model is rewritten as blocks of subgraphs. Subgraphs which can be compiled by TorchDynamo are “flattened”, while the ones that may contain unsupported Python constructs (like control-flow code) will fall back to eager-mode.
2. **Graph lowering** - all the PyTorch operations are decomposed into their constituent kernels specific to the chosen backend.
3. **Graph compilation** - the kernels call their corresponding low-level device-specific operations.

Example of usage
#################

To use it, you need to add an import statement and setting backend parameter of the ``torch.compile`` to openvino:

.. code-block:: console

   import torch
   from openvino.frontend.pytorch.torchdynamo import backend
   ...
   model = torch.compile(model, backend='openvino')

Backends
+++++++++++++++

OpenVINO has two backends available with ``torch.compile``:

1. ``openvino``
2. ``openvino_ts``

.. tab-set::

   .. tab-item:: openvino
      :sync: openvino

      In this backend, Torch FX subgraphs are directly converted to OpenVINO representation without any additional PyTorch based tracing/scripting.

      .. code-block:: console

         from openvino.frontend.pytorch.torchdynamo import backend
         ...
         model = torch.compile(model, backend='openvino')

      .. image:: _static/images/torch_compile_backend_openvino.svg

   .. tab-item:: openvino_ts
      :sync: openvino-ts

      In this backend, Torch FX subgraphs are first traced/scripted with PyTorch Torchscript, and then converted to OpenVINO representation 

      .. code-block:: console

         from openvino.frontend.pytorch.torchdynamo import backend
         ...
         model = torch.compile(model, backend='openvino_ts')

      .. image:: _static/images/torch_compile_backend_openvino_ts.svg

Architecture
++++++++++++++++++++++

``torch.compile`` was created with the new technologies – **TorchDynamo**, **AOTAutograd**, **PrimTorch** and **TorchInductor**.

* **TorchDynamo** generates FX Graphs from Python bytecode. PyTorch operators that cannot be extracted to FX graph are executed in native Python environment. It maintains the eager-mode capabilities using guards to ensure the generated graphs are valid.
* **AOTAutograd** to generate the backward graph corresponding to the forward graph captured by TorchDynamo.
* **PrimTorch** to decompose complicated PyTorch operations into simpler and more elementary ops.
* **TorchInductor** is a deep learning compiler that generates fast code for multiple accelerators and backends.

When the PyTorch module is wrapped with ``torch.compile``, TorchDynamo traces the module and rewrites Python bytecode in order to extract sequences of PyTorch operations into an FX Graph, which can be optimized by OpenVINO backend. The Torch FX graphs are first converted to inlined FX graphs. The graph partitioning module traverses inlined FX graph to identify operators supported by OpenVINO. 

All the operators that are supported are clustered into OpenVINO submodules, converted to OpenVINO graph using OpenVINO’s PyTorch decoder and executed on Intel hardware in an optimized manner using OpenVINO runtime. All unsupported operators fall back to native PyTorch runtime on CPU. If the subgraph fails during OpenVINO conversion, the subgraph falls back to PyTorch’s default inductor backend.


Environment Variables
+++++++++++++++++++++++++++

* **OPENVINO_TORCH_BACKEND_DEVICE**: By default, OpenVINO backend for ``torch.compile`` accelerates PyTorch applications on CPUs. To select another device backend (for exmaple, GPU.0), set the OPENVINO_TORCH_BACKEND_DEVICE to GPU.0
* **OPENVINO_TORCH_MODEL_CACHING**: OpenVINO graphs optimized with ``torch.compile`` can be saved to disk for subsequent application launches. By default, this variable is set to ``False``. To enable caching, it needs to be set to ``True``.
* **OPENVINO_TORCH_CACHE_DIR**: By default, the OpenVINO IR is saved in a directory ``cache`` that is created in the current directory where the application is launched. The user can set it to a different directory of their choice using this environment variable.

Additional Resources
############################

* `PyTorch 2.0 documentation <https://pytorch.org/docs/stable/index.html>`_

@endsphinxdirective
