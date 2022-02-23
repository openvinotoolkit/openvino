# OpenVINO™ Runtime User Guide {#openvino_docs_OV_Runtime_User_Guide}

@sphinxdirective

.. _deep learning inference engine:

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_IE_DG_Integrate_with_customer_application_new_API
   <!-- should be a part of Integrate OV in user application -->
   openvino_docs_OV_Runtime_UG_Model_Representation
   openvino_docs_IE_DG_ShapeInference
   openvino_docs_OV_UG_Working_with_devices
   openvino_docs_OV_Runtime_UG_Preprocessing_Overview
   openvino_docs_IE_DG_DynamicBatching
   openvino_docs_IE_DG_supported_plugins_AUTO
   openvino_docs_OV_UG_Running_on_multiple_devices
   openvino_docs_OV_UG_Hetero_execution
   openvino_docs_IE_DG_network_state_intro
   openvino_2_0_transition_guide
   openvino_docs_OV_Should_be_in_performance
   openvino_docs_OV_Runtime_API_Changes

@endsphinxdirective

## Introduction
OpenVINO Runtime is a set of C++ libraries with C and Python bindings providing a common API to deliver inference solutions on the platform of your choice. Use the OpenVINO Runtime API to read the Intermediate Representation (IR), ONNX, PDPD file formats and execute the model on devices.

OpenVINO runtime uses a plugin architecture. Inference plugin is a software component that contains complete implementation for inference on a certain Intel® hardware device: CPU, GPU, VPU, GNA, etc. Each plugin implements the unified API and provides additional hardware-specific APIs to configure device or interoperability API between OpenVINO Runtime and underlaying plugin backend.

The scheme below illustrates the typical workflow for deploying a trained deep learning model: 

<!-- TODO: need to update the picture below with PDPD files -->
![](img/BASIC_FLOW_IE_C.svg)


## Video

@sphinxdirective

.. list-table::

   * - .. raw:: html

           <iframe allowfullscreen mozallowfullscreen msallowfullscreen oallowfullscreen webkitallowfullscreen height="315" width="100%"
           src="https://www.youtube.com/embed/e6R13V8nbak">
           </iframe>
   * - **Inference Engine Concept**. Duration: 3:43
     
@endsphinxdirective
