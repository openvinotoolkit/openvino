Running Inference with OpenVINO™
==================================


.. toctree::
   :maxdepth: 1
   :hidden:

   running-inference/integrate-openvino-with-your-application
   running-inference/inference-devices-and-modes
   running-inference/changing-input-shape
   running-inference/dynamic-shapes
   running-inference/stateful-models
   running-inference/string-tensors
   Optimize Inference <running-inference/optimize-inference>

.. meta::
   :description: Learn how to run inference using OpenVINO.


OpenVINO Runtime is a set of C++ libraries with C and Python bindings providing a common API
to deploy inference on the platform of your choice. You can run any of the
:doc:`supported model formats <model-preparation>` directly or convert the model
and save it to the :doc:`OpenVINO IR <../documentation/openvino-ir-format>` format, for maximum performance.

Why is OpenVINO IR inference faster? Even if you run a supported model directly, it is
converted before inference. It may happen automatically, under the hood, for maximum convenience,
but it is not suited for the most performance-oriented use cases. For example, converting PyTorch
usually requires Python and the ``torch`` module, which take extra time and memory, on top the
conversion process itself. If OpenVINO IR is used instead, it does not require any conversion,
nor the additional dependencies, as the inference application can be written in C or C++.
OpenVINO IR provides by far the best first-inference latency scores.


.. note::

   For more detailed information on how to convert, read, and compile supported model formats
   see the :doc:`Model Preparation article <model-preparation>`.

   Note that PyTorch models can be run using the
   :doc:`torch.compile feature <torch-compile>`, as well as the standard ways of
   :doc:`converting Pytorch <model-preparation/convert-model-pytorch>`
   or running its inference.

OpenVINO Runtime uses a plugin architecture. Its plugins are software components that contain complete implementation for inference on a particular Intel® hardware device: CPU, GPU, etc. Each plugin implements the unified API and provides additional hardware-specific APIs for configuring devices or API interoperability between OpenVINO Runtime and underlying plugin backend.

The scheme below illustrates the typical workflow for deploying a trained deep learning model:


.. image:: ../assets/images/BASIC_FLOW_IE_C.svg


