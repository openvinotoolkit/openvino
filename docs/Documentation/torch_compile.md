# OpenVINO PyTorch 2.0 - torch.compile() backend

PyTorch 2.0 introduces a new feature ``torch.compile`` to speed up PyTorch code. It makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels, all while requiring minimal code changes. OpenVINO is enabled in the torch.compile to optimize generation of the graph model. This allows developers to leverage OpenVINO optimizations in PyTorch native applications on Intel CPUs, Intel integrated and discrete GPUs.

## Example Usage

Usage is as simple as adding an import statement and setting backend parameter of the torch.compile to openvino. 

.. code-block:: sh

   import torch
   from openvino.frontend.pytorch.torchdynamo import backend
   ...
   model = torch.compile(model, backend='openvino')

### torch.compile Backends

OpenVINO has the below two backends available with torch.compile:

1. “openvino”:  In this backend, Torch FX subgraphs are directly converted to OpenVINO representation without any additional PyTorch based tracing/scripting
2. “openvino_ts”: In this backend, Torch FX subgraphs are first traced/scripted with PyTorch Torchscript, and then converted to OpenVINO representation 

.. code-block:: sh

   from openvino.frontend.pytorch.torchdynamo import backend
   ...
   model = torch.compile(model, backend='openvino')
   (or)
   model = torch.compile(model, backend='openvino_ts')

### Architecture

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


