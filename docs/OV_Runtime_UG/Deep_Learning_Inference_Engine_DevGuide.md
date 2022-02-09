# OpenVINO™ Runtime User Guide {#openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   openvino_2_0_transition_guide
   openvino_docs_IE_DG_Integrate_with_customer_application_new_API
   openvino_docs_OV_Runtime_UG_Model_Representation
   ngraph_transformation
   openvino_docs_deployment_optimization_guide_dldt_optimization_guide
   openvino_docs_IE_DG_Device_Plugins
   Direct ONNX Format Support <openvino_docs_IE_DG_ONNX_Support>
   openvino_docs_IE_DG_Paddle_Support
   openvino_docs_IE_DG_Int8Inference
   openvino_docs_IE_DG_Bfloat16Inference
   openvino_docs_IE_DG_DynamicBatching
   openvino_docs_IE_DG_ShapeInference
   openvino_docs_IE_DG_Model_caching_overview
   openvino_docs_IE_DG_Extensibility_DG_Intro
   openvino_docs_IE_DG_Memory_primitives
   openvino_docs_IE_DG_network_state_intro   
   openvino_docs_IE_DG_API_Changes
   openvino_docs_IE_DG_Known_Issues_Limitations
   openvino_docs_IE_DG_Glossary
      
@endsphinxdirective

## Introduction
Inference Engine is a set of C++ libraries with C and Python bindings providing a common API to deliver inference solutions on the platform of your choice. Use the Inference Engine API to read the Intermediate Representation (IR), ONNX and execute the model on devices.

Inference Engine uses a plugin architecture. Inference Engine plugin is a software component that contains complete implementation for inference on a certain Intel® hardware device: CPU, GPU, VPU, etc. Each plugin implements the unified API and provides additional hardware-specific APIs.
 
The scheme below illustrates the typical workflow for deploying a trained deep learning model: 

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
