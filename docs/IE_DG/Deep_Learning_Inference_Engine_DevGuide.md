# Inference Engine Developer Guide {#openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   openvino_docs_IE_DG_Integrate_with_customer_application_new_API
   openvino_docs_IE_DG_Intro_to_Performance
      
@endsphinxdirective

## Introduction
Inference Engine is a set of C++ libraries with C and Python bindings providing a common API to deliver inference solutions on the platform of your choice. Use the Inference Engine API to read the Intermediate Representation (IR), ONNX and execute the model on devices.

Inference Engine uses a plugin architecture. Inference Engine plugin is a software component that contains complete implementation for inference on a certain Intel® hardware device: CPU, GPU, VPU, etc. Each plugin implements the unified API and provides additional hardware-specific APIs.
 
The scheme below illustrates the typical workflow for deploying a trained deep learning model: 

![](img/ie_workflow_steps.png)

\\* _nGraph_ is the internal graph representation in the OpenVINO™ toolkit. Use it to [build a model from source code](https://docs.openvinotoolkit.org/latest/openvino_docs_nGraph_DG_build_function.html).


## Video

@sphinxdirective

.. list-table::

   * - .. youtube:: e6R13V8nbak
          :width: 560px
   * - **Inference Engine Concept**. Duration: 3:43
     
@endsphinxdirective