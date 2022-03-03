# Model Optimizer Developer Guide {#openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide}

@sphinxdirective

.. _deep learning model optimizer:

.. toctree::
   :maxdepth: 1
   :hidden:
   
   openvino_docs_MO_DG_IR_and_opsets
   openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model
   openvino_docs_MO_DG_Additional_Optimization_Use_Cases
   openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer
   openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ
   openvino_docs_MO_DG_Known_Issues_Limitations
   openvino_docs_MO_DG_Default_Model_Optimizer_Optimizations

@endsphinxdirective

## Introduction 

Model Optimizer is a cross-platform command-line tool that facilitates the transition between the training and deployment environment, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices.

Model Optimizer process assumes you have a network model trained using supported deep learning frameworks: TensorFlow*, PyTorch*, ONNX*, PaddlePaddle*, MXNet*, Caffe*, Kaldi*. Model Optimizer produces an Intermediate Representation (IR) of the network, which can be inferred with the [OpenVINO™ Runtime](../OV_Runtime_UG/openvino_intro.md).

> **NOTE**: Model Optimizer does not infer models. Model Optimizer is an offline tool that runs before the inference takes place.

The scheme below illustrates the typical workflow for deploying a trained deep learning model: 

![](img/BASIC_FLOW_MO_simplified.svg)

The IR is a pair of files describing the model: 

*  <code>.xml</code> - Describes the network topology

*  <code>.bin</code> - Contains the weights and biases binary data.

> **TIP**: You also can work with the Model Optimizer inside the OpenVINO™ [Deep Learning Workbench](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Introduction.html) (DL Workbench).
> [DL Workbench](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Introduction.html) is a web-based graphical environment that enables you to optimize, fine-tune, analyze, visualize, and compare performance of deep learning models.

## Run Model Optimizer

To convert the model to the Intermediate Representation (IR), run Model Optimizer:

```sh
mo --input_model INPUT_MODEL
```

> **NOTE**: Some models require using additional arguments to specify conversion parameters, such as `--input_shape`, `--scale`, `--scale_values`, `--mean_values`, `--mean_file`. To learn about when you need to use these parameters, refer to [Converting a Model to Intermediate Representation (IR)](prepare_model/convert_model/Converting_Model.md).

To adjust the conversion process, you may use general parameters defined in the [Converting a Model to Intermediate Representation (IR)](prepare_model/convert_model/Converting_Model.md) and 
framework-specific parameters for:
* [TensorFlow](prepare_model/convert_model/Convert_Model_From_TensorFlow.md)
* [PyTorch](prepare_model/convert_model/Convert_Model_From_ONNX.md)
* [ONNX](prepare_model/convert_model/Convert_Model_From_ONNX.md)
* [PaddlePaddle](prepare_model/convert_model/Convert_Model_From_Paddle.md)
* [MXNet](prepare_model/convert_model/Convert_Model_From_MxNet.md)
* [Caffe](prepare_model/convert_model/Convert_Model_From_Caffe.md)
* [Kaldi](prepare_model/convert_model/Convert_Model_From_Kaldi.md)

## Videos

@sphinxdirective

.. list-table::

   * - .. raw:: html

           <iframe allowfullscreen mozallowfullscreen msallowfullscreen oallowfullscreen webkitallowfullscreen width="220"
           src="https://www.youtube.com/embed/Kl1ptVb7aI8">
           </iframe>
    
     - .. raw:: html

           <iframe allowfullscreen mozallowfullscreen msallowfullscreen oallowfullscreen webkitallowfullscreen width="220"
           src="https://www.youtube.com/embed/BBt1rseDcy0">
           </iframe>

     - .. raw:: html

           <iframe allowfullscreen mozallowfullscreen msallowfullscreen oallowfullscreen webkitallowfullscreen width="220"
           src="https://www.youtube.com/embed/RF8ypHyiKrY">
           </iframe>

   * - **Model Optimizer Concept.**
     - **Model Optimizer Basic Operation.**
     - **Choosing the Right Precision.**

   * - Duration: 3:56
     - Duration: 2:57
     - Duration: 4:18

@endsphinxdirective
