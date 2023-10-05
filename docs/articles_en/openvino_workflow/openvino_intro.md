# Inference with OpenVINO Runtime {#openvino_docs_OV_UG_OV_Runtime_User_Guide}

@sphinxdirective

.. _deep learning openvino runtime:

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
   :description: OpenVINO Runtime is an API comprised of a set of C++ libraries 
                 with C and Python bindings and it delivers inference solutions 
                 on different platforms.


OpenVINO Runtime is a set of C++ libraries with C and Python bindings providing a common API to deliver inference solutions on the platform of your choice. Use the OpenVINO Runtime API to read PyTorch, TensorFlow, TensorFlow Lite, ONNX, and PaddlePaddle models and execute them on preferred devices. OpenVINO gives you the option to use these models directly or convert them to the OpenVINO IR (Intermediate Representation) format explicitly, for maximum performance.


.. note::

   For more detailed information on how to convert, read, and compile supported model formats
   see the :doc:`Supported Formats article <Supported_Model_Formats_MO_DG>`.
   
   Note that TensorFlow models can be run using the
   :doc:`torch.compile feature <pytorch_2_0_torch_compile>`, as well as the standard ways of
   :doc:`converting TensorFlow <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_PyTorch>`
    or reading them directly.

OpenVINO Runtime uses a plugin architecture. Its plugins are software components that contain complete implementation for inference on a particular Intel® hardware device: CPU, GPU, GNA, etc. Each plugin implements the unified API and provides additional hardware-specific APIs for configuring devices or API interoperability between OpenVINO Runtime and underlying plugin backend.

The scheme below illustrates the typical workflow for deploying a trained deep learning model:


.. image:: _static/images/BASIC_FLOW_IE_C.svg


@endsphinxdirective
