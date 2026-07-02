PyTorch Deployment via ExecuTorch
=================================

.. meta::
   :description: Deploy PyTorch models on edge devices using
                 OpenVINO as an ExecuTorch backend delegate.


`ExecuTorch <https://pytorch.org/executorch/>`__ is a PyTorch-native framework
for deploying models on edge devices. OpenVINO is integrated as an
ExecuTorch backend delegate, enabling optimized inference on Intel hardware while
keeping the familiar PyTorch-based workflow.

.. note::

   This feature is currently in **preview** and may change in future releases.


How It Works
############

The OpenVINO ExecuTorch integration follows three high-level steps:

1. **Export** - the PyTorch model is captured as a graph using ``torch.export``.
2. **Lowering** - the exported graph is lowered to the OpenVINO backend delegate
   via the ``OpenVINOPartitioner``, which converts supported subgraphs to
   OpenVINO IR for optimized execution.
3. **Execution** - the lowered program (``.pte`` file) is executed through the
   ExecuTorch runtime, with OpenVINO handling the delegated subgraphs on the
   target Intel device.


Installation
############

To use the OpenVINO backend with ExecuTorch you need to install both OpenVINO
and ExecuTorch with the OpenVINO backend enabled.

1. **Install OpenVINO** — download a release package from the
   `OpenVINO install guide <https://docs.openvino.ai/2026/get-started/install-openvino.html>`__
   and source ``setupvars.sh``, or
   `build OpenVINO from source <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_linux.md>`__.

2. **Set up ExecuTorch with the OpenVINO backend** — clone the ExecuTorch
   repository and run the provided build script:

   .. code-block:: console

      git clone --recurse-submodules https://github.com/pytorch/executorch.git
      cd executorch/backends/openvino/scripts
      ./openvino_build.sh            # build C++ runtime + Python package
      # or: ./openvino_build.sh --enable_python   (Python bindings only)
      # or: ./openvino_build.sh --cpp_runtime     (C++ runtime only)

3. **Install backend Python dependencies**:

   .. code-block:: console

      pip install -r executorch/backends/openvino/requirements.txt

For the full list of prerequisites and build options, see the
`Building and Running ExecuTorch with OpenVINO Backend
<https://docs.pytorch.org/executorch/1.1/build-run-openvino.html>`__.

How to Use
##########

The example below shows the basic flow: export a PyTorch model, lower it to the
OpenVINO backend, and save the resulting ``.pte`` program for execution.

.. code-block:: python

   import torch
   import torchvision
   from executorch.exir import to_edge
   from executorch.backends.openvino.partitioner import OpenVINOPartitioner

   # 1. Define the model and example input
   model = torchvision.models.resnet18(weights="DEFAULT").eval()
   example_inputs = (torch.randn(1, 3, 224, 224),)

   # 2. Export the model using torch.export
   exported_program = torch.export.export(model, example_inputs)

   # 3. Lower to the OpenVINO backend
   edge_program = to_edge(exported_program)
   edge_program = edge_program.to_backend(OpenVINOPartitioner())

   # 4. Produce the ExecuTorch program and save it
   et_program = edge_program.to_executorch()
   with open("resnet18.pte", "wb") as f:
       f.write(et_program.buffer)

The saved ``.pte`` file can then be loaded and executed through the ExecuTorch
runtime — either in Python or with the C++ ``executor_runner`` binary.

The ``executor_runner`` is built as part of the OpenVINO backend C++ runtime and
is located at ``<executorch_root>/cmake-out/executor_runner`` after running
``openvino_build.sh --cpp_runtime`` (see :ref:`Installation <Installation>`).
For details, see the
`Build C++ OpenVINO Examples <https://docs.pytorch.org/executorch/1.1/build-run-openvino.html#build-c-openvino-examples>`__
section of the ExecuTorch documentation.

.. code-block:: console

   # Run inference using the C++ runner
   ./executor_runner --model_path=resnet18.pte --num_executions=10

For comprehensive examples covering image classification, text generation
(Llama), image generation (Stable Diffusion), quantization, and more, refer to
the `ExecuTorch OpenVINO examples <https://github.com/pytorch/executorch/tree/main/examples/openvino>`__.


Model Quantization and Weight Compression
##########################################

Quantization and weight compression reduce model size and accelerate inference
with minimal accuracy loss. The ExecuTorch OpenVINO backend supports two
approaches: the native **PyTorch 2 Export (PT2E) quantization** flow and the
NNCF-based **``quantize_pt2e`` / ``compress_pt2e``** API, which wraps the native
flow with additional accuracy-preserving algorithms.

Native PyTorch 2 Export Quantization
++++++++++++++++++++++++++++++++++++

The native PT2E flow from ``torchao`` consists of three steps: annotate the
exported graph with a quantizer, calibrate with representative data, and convert
the annotations into actual quantized operations.

.. code-block:: python

   import torch
   import torchvision
   from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
   from nncf.experimental.torch.fx import OpenVINOQuantizer

   # 1. Export the model
   model = torchvision.models.resnet18(weights="DEFAULT").eval()
   example_inputs = (torch.randn(1, 3, 224, 224),)
   exported_model = torch.export.export(model, example_inputs)
   graph_module = exported_model.module()

   # 2. Annotate and prepare for calibration
   quantizer = OpenVINOQuantizer()
   prepared_model = prepare_pt2e(graph_module, quantizer)

   # 3. Calibrate with representative data
   with torch.no_grad():
       for images, _ in calibration_loader:
           prepared_model(images)

   # 4. Convert to a quantized model
   quantized_model = convert_pt2e(prepared_model, fold_quantize=False)

The resulting ``quantized_model`` can then be lowered to the OpenVINO backend and
exported as a ``.pte`` file using the same ``to_edge`` / ``to_backend`` flow
shown above.

NNCF ``quantize_pt2e`` (recommended)
+++++++++++++++++++++++++++++++++++++

The `NNCF <https://docs.openvino.ai/2026/openvino-workflow/model-optimization.html>`__
``quantize_pt2e`` function replaces the manual prepare → calibrate → convert
sequence with a single call. It additionally applies advanced accuracy-recovery
techniques such as **SmoothQuant**, **fast bias correction**, and
NNCF-optimized quantizer placement tailored for the OpenVINO runtime.

.. code-block:: python

   import torch
   import torchvision
   import nncf
   from nncf.experimental.torch.fx import quantize_pt2e, OpenVINOQuantizer

   # 1. Export the model
   model = torchvision.models.resnet18(weights="DEFAULT").eval()
   example_inputs = (torch.randn(1, 3, 224, 224),)
   exported_model = torch.export.export(model, example_inputs)

   # 2. Prepare the calibration dataset
   calibration_loader = torch.utils.data.DataLoader(...)

   def transform_fn(data_item):
       images, _ = data_item
       return images

   calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)

   # 3. Quantize in a single call
   quantizer = OpenVINOQuantizer()
   quantized_model = quantize_pt2e(
       exported_model.module(),
       quantizer,
       calibration_dataset,
       smooth_quant=True,          # enable SmoothQuant
       fast_bias_correction=True,  # enable fast bias correction
       subset_size=300,            # calibration subset size
   )

Key ``quantize_pt2e`` parameters:

* ``smooth_quant`` - enables the SmoothQuant algorithm for better accuracy on
  transformer models.
* ``fast_bias_correction`` - corrects quantization-induced bias shifts (set to
  ``None`` to disable, ``False`` for a slower but more accurate variant).
* ``subset_size`` - number of calibration samples (default: 300).
* ``fold_quantize`` - fold quantize ops after conversion (default: ``True``).

NNCF ``compress_pt2e``
++++++++++++++++++++++

For **weight-only compression** (INT4 / INT8) of large models such as LLMs, use
``compress_pt2e``. This reduces model size without requiring a calibration
dataset for basic compression, while optionally supporting data-aware algorithms
like **AWQ**, **GPTQ**, **scale estimation**, and **LoRA correction** for higher
accuracy.

.. code-block:: python

   import torch
   import nncf
   from nncf.experimental.torch.fx import compress_pt2e, OpenVINOQuantizer

   # 1. Export the model
   model = ...  # your large language model
   exported_model = torch.export.export(model, example_inputs)

   # 2. Compress weights
   quantizer = OpenVINOQuantizer()
   compressed_model = compress_pt2e(
       exported_model.module(),
       quantizer,
       dataset=nncf.Dataset(calibration_loader, transform_fn),  # optional, for data-aware methods
       awq=True,                # enable AWQ (requires dataset)
       scale_estimation=True,   # enable scale estimation (requires dataset)
       ratio=0.9,               # 90% of layers compressed to NF4, rest to INT8
       subset_size=128,         # number of calibration samples
   )

Key ``compress_pt2e`` parameters:

* ``awq`` - enables the modified AWQ algorithm (requires ``dataset``).
* ``scale_estimation`` - enables scale estimation for 4-bit layers (requires
  ``dataset``).
* ``gptq`` - enables the GPTQ algorithm (requires ``dataset``).
* ``lora_correction`` - enables LoRA Correction algorithm.
* ``ratio`` - ratio between baseline and backup precisions (e.g., 0.9 means 90%
  NF4 and 10% INT8).
* ``sensitivity_metric`` - metric for assigning per-layer precision to preserve
  accuracy.

For a complete example, see the
`NNCF TorchFX quantization tutorial <https://github.com/openvinotoolkit/nncf/tree/develop/examples/post_training_quantization/torch_fx/resnet18>`__.


Additional Resources
####################

* `ExecuTorch documentation <https://pytorch.org/executorch/>`__
* `Building and Running ExecuTorch with OpenVINO Backend <https://docs.pytorch.org/executorch/1.1/build-run-openvino.html>`__
* `PyTorch Export documentation <https://pytorch.org/docs/stable/export.html>`__
