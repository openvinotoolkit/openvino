.. {#openvino_docs_OV_UG_OV_Runtime_User_Guide}

Running Inference with OpenVINO™
==================================


.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_Integrate_OV_with_your_application
   openvino_docs_Runtime_Inference_Modes_Overview
   openvino_docs_OV_UG_Working_with_devices
   openvino_docs_OV_UG_ShapeInference
   openvino_docs_OV_UG_DynamicShapes
   openvino_docs_OV_UG_model_state_intro
   Optimize Inference <openvino_docs_deployment_optimization_guide_dldt_optimization_guide>

.. meta::
   :description: Learn how to run inference using OpenVINO.


OpenVINO Runtime is a set of C++ libraries with C and Python bindings providing a common API
to deploy inference on the platform of your choice. You can run any of the 
:doc:`supported model formats <Supported_Model_Formats>` directly or convert the model
and save it to the :doc:`OpenVINO IR <openvino_ir>` format, for maximum performance.

Why is OpenVINO IR inference faster? Even if you run a supported model directly, it is
converted before inference. It may happen automatically, under the hood, for maximum convenience,
but it is not suited for the most performance-oriented use cases. For example, converting PyTorch
usually requires Python and the ``torch`` module, which take extra time and memory, on top the
conversion process itself. If OpenVINO IR is used instead, it does not require any conversion,
nor the additional dependencies, as the inference application can be written in C or C++.
OpenVINO IR provides by far the best first-inference latency scores.


.. note::

   For more detailed information on how to convert, read, and compile supported model formats
   see the :doc:`Supported Formats article <Supported_Model_Formats_MO_DG>`.
   
   Note that TensorFlow models can be run using the
   :doc:`torch.compile feature <pytorch_2_0_torch_compile>`, as well as the standard ways of
   :doc:`converting TensorFlow <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_PyTorch>`
   or running its inference.

OpenVINO Runtime uses a plugin architecture. Its plugins are software components that contain complete implementation for inference on a particular Intel® hardware device: CPU, GPU, GNA, etc. Each plugin implements the unified API and provides additional hardware-specific APIs for configuring devices or API interoperability between OpenVINO Runtime and underlying plugin backend.

The scheme below illustrates the typical workflow for deploying a trained deep learning model:


.. image:: _static/images/BASIC_FLOW_IE_C.svg


