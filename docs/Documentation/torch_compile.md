# OpenVINO PyTorch 2.0 - torch.compile() backend {#pytorch_2_0_torch_compile}

PyTorch 2.0 simplifies backend (compiler) integration for developers and vendors by reducing 2000+ Ops to a set of ~250 primitive Ops and simplyfing Op semantics. It also likened to Graph Mode, offers the same eager-mode development experience, while adding a compiled mode via torch.compile. PyTorch 2.0 is 100% backwards compatible with PyTorch 1.0. torch.compile is an opt-in feature.

PyTorch 2.0 introduced the ``torch.compile`` - a new feature used for speeding up PyTorch code by JIT-compiling PyTorch code into optimized kernels. OpenVINO is enabled in the ``torch.compile`` to optimize generation of the graph model. This allows developers to leverage OpenVINO optimizations in PyTorch native applications on Intel CPUs, Intel integrated and discrete GPUs.

By default, torch code runs in eager mode, But, when you use torch.compile:

* **Graph acquisition** - first the model is rewritten as blocks of subgraphs. Subgraphs which can be compiled by TorchDynamo are “flattened” and the other subgraphs (which might contain control-flow code or other unsupported Python constructs) will fall back to Eager-Mode.
* **Graph lowering** - all the PyTorch operations are decomposed into their constituent kernels specific to the chosen backend.
* **Graph compilation** - where the kernels call their corresponding low-level device-specific operations.


## Example Usage

Usage is as simple as adding an import statement and setting backend parameter of the torch.compile to openvino. 

.. code-block:: sh

   import torch
   from openvino.frontend.pytorch.torchdynamo import backend
   ...
   model = torch.compile(model, backend='openvino')

### torch.compile Backends

OpenVINO has the below two backends available with torch.compile:

1. ``openvino`` - In this backend, Torch FX subgraphs are directly converted to OpenVINO representation without any additional PyTorch based tracing/scripting
2. ``openvino_ts`` - In this backend, Torch FX subgraphs are first traced/scripted with PyTorch Torchscript, and then converted to OpenVINO representation 

.. code-block:: sh

   from openvino.frontend.pytorch.torchdynamo import backend
   ...
   model = torch.compile(model, backend='openvino')
   (or)
   model = torch.compile(model, backend='openvino_ts')

### Architecture

Underpinning ``torch.compile`` are new technologies – TorchDynamo, AOTAutograd, PrimTorch and TorchInductor.

* TorchDynamo generates FX Graphs from Python bytecode. It maintains the eager-mode capabilities using guards to ensure the generated graphs are valid.
* AOTAutograd to generate the backward graph corresponding to the forward graph captured by TorchDynamo.
* PrimTorch to decompose complicated PyTorch operations into simpler and more elementary ops.
* TorchInductor is a deep learning compiler that generates fast code for multiple accelerators and backends.

When the PyTorch module is wrapped with ‘torch.compile’, TorchDynamo traces the module. TorchDynamo is a Python-level JIT that hooks into the frame evaluation API in CPython (PEP 523) to dynamically modify Python bytecode right before it is executed. It rewrites Python bytecode in order to extract sequences of PyTorch operations into an FX Graph, which can be optimized by OpenVINO backend. The PyTorch operators that cannot be extracted to FX graph are executed in native Python environment.

The Torch FX graphs are first converted to inlined FX graphs. The graph partitioning module traverses inlined FX graph to identify operators supported by OpenVINO. All the operators that are supported are clustered into OpenVINO submodules, converted to OpenVINO graph using OpenVINO’s PyTorch decoder and executed on Intel hardware in an optimized manner using OpenVINO runtime. All unsupported operators fall back to native PyTorch runtime on CPU. If the subgraph fails during OpenVINO conversion, the subgraph falls back to PyTorch’s default inductor backend.

backend = “openvino”

.. image:: _static/images/torch_compile_backend_openvino.svg

backend = “openvino_ts”:

.. image:: _static/images/torch_compile_backend_openvino_ts.svg

### Environment Variables

* OPENVINO_TORCH_BACKEND_DEVICE: By default, OpenVINO backend for torch.compile accelerates PyTorch applications on CPUs. To select another device backend (GPU.0 for example), set the OPENVINO_TORCH_BACKEND_DEVICE to GPU.0
* OPENVINO_TORCH_MODEL_CACHING: OpenVINO graphs optimized with torch.compile can be saved to disk for subsequent application launches. By default, this variable to set to False. To enable caching, users can set it to True.
* OPENVINO_TORCH_CACHE_DIR: By default, the OpenVINO IR is saved in a directory “cache” that is created in the current directory where the application is launched. The user can set it to a different directory of their choice using this environment variable.


