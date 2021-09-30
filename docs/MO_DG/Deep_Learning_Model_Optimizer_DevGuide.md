# Model Optimizer Developer Guide {#openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model
   openvino_docs_MO_DG_Additional_Optimization_Use_Cases
   openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer
   openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ
   openvino_docs_MO_DG_Known_Issues_Limitations
   openvino_docs_MO_DG_Default_Model_Optimizer_Optimizations

@endsphinxdirective

## Introduction

Model Optimizer is a cross-platform command-line tool that facilitates the transition between the training and deployment environment, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices.

Model Optimizer process assumes you have a network model trained using supported deep learning frameworks: Caffe*, TensorFlow*, Kaldi*, MXNet* or converted to the ONNX* format. Model Optimizer produces an Intermediate Representation (IR) of the network, which can be inferred with the [Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md).

> **NOTE**: Model Optimizer does not infer models. Model Optimizer is an offline tool that runs before the inference takes place.

The scheme below illustrates the typical workflow for deploying a trained deep learning model: 

![](img/BASIC_FLOW_MO_simplified.svg)

The IR is a pair of files describing the model: 

*  <code>.xml</code> - Describes the network topology

*  <code>.bin</code> - Contains the weights and biases binary data.

Below is a simple command running Model Optimizer to generate an IR for the input model:

```sh
python3 mo.py --input_model INPUT_MODEL
```
To learn about all Model Optimizer parameters and conversion technics, see the [Converting a Model to IR](prepare_model/convert_model/Converting_Model.md) page.

> **TIP**: You can quick start with the Model Optimizer inside the OpenVINO™ [Deep Learning Workbench](@ref 
> openvino_docs_get_started_get_started_dl_workbench) (DL Workbench).
> [DL Workbench](@ref workbench_docs_Workbench_DG_Introduction) is the OpenVINO™ toolkit UI that enables you to
> import a model, analyze its performance and accuracy, visualize the outputs, optimize and prepare the model for 
> deployment on various Intel® platforms.

## Videos

@sphinxdirective

.. list-table::

   * - .. raw:: html

           <iframe width="220"
           src="https://www.youtube.com/embed/Kl1ptVb7aI8">
           </iframe>
    
     - .. raw:: html

           <iframe width="220"
           src="https://www.youtube.com/embed/BBt1rseDcy0">
           </iframe>

     - .. raw:: html

           <iframe width="220"
           src="https://www.youtube.com/embed/RF8ypHyiKrY">
           </iframe>

   * - **Model Optimizer Concept.**
     - **Model Optimizer Basic Operation.**
     - **Choosing the Right Precision.**

   * - Duration: 3:56
     - Duration: 2:57
     - Duration: 4:18

@endsphinxdirective
