# Model Optimizer Developer Guide {#openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide}

Model Optimizer is a cross-platform command-line tool that facilitates the transition between the training and deployment environment, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices.

Model Optimizer process assumes you have a network model trained using a supported deep learning framework. The scheme below illustrates the typical workflow for deploying a trained deep learning model:

![](img/workflow_steps.png)

Model Optimizer produces an Intermediate Representation (IR) of the network, which can be read, loaded, and inferred with the Inference Engine. The Inference Engine API offers a unified API across a number of supported Intel® platforms. The Intermediate Representation is a pair of files describing the model:

*  <code>.xml</code> - Describes the network topology

*  <code>.bin</code> - Contains the weights and biases binary data.

> **TIP**: You also can work with the Model Optimizer inside the OpenVINO™ [Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction) (DL Workbench).
> [DL Workbench](@ref workbench_docs_Workbench_DG_Introduction) is a platform built upon OpenVINO™ and provides a web-based graphical environment that enables you to optimize, fine-tune, analyze, visualize, and compare 
> performance of deep learning models on various Intel® architecture
> configurations. In the DL Workbench, you can use most of OpenVINO™ toolkit components.
> <br>
> Proceed to an [easy installation from Docker](@ref workbench_docs_Workbench_DG_Install_from_Docker_Hub) to get started.

## What's New in the Model Optimizer in this Release?

* Common changes:
    * Implemented several optimization transformations to replace sub-graphs of operations with HSwish, Mish, Swish and SoftPlus operations.
    * Model Optimizer generates IR keeping shape-calculating sub-graphs **by default**. Previously, this behavior was triggered if the "--keep_shape_ops" command line parameter was provided. The key is ignored in this release and will be deleted in the next release. To trigger the legacy behavior to generate an IR for a fixed input shape (folding ShapeOf operations and shape-calculating sub-graphs to Constant), use the "--static_shape" command line parameter. Changing model input shape using the Inference Engine API in runtime may fail for such an IR.
    * Fixed Model Optimizer conversion issues resulted in non-reshapeable IR using the Inference Engine reshape API.
    * Enabled transformations to fix non-reshapeable patterns in the original networks:
        * Hardcoded Reshape
            * In Reshape(2D)->MatMul pattern
            * Reshape->Transpose->Reshape when the pattern can be fused to the ShuffleChannels or DepthToSpace operation
        * Hardcoded Interpolate
            * In Interpolate->Concat pattern
        * Added a dedicated requirements file for TensorFlow 2.X as well as the dedicated install prerequisites scripts.
        * Replaced the SparseToDense operation with ScatterNDUpdate-4.
* ONNX*:
    * Enabled an ability to specify the model output **tensor** name using the "--output" command line parameter.
    * Added support for the following operations:
        * Acosh
        * Asinh
        * Atanh
        * DepthToSpace-11, 13
        * DequantizeLinear-10 (zero_point must be constant)
        * HardSigmoid-1,6
        * QuantizeLinear-10 (zero_point must be constant)
        * ReduceL1-11, 13
        * ReduceL2-11, 13
        * Resize-11, 13 (except mode="nearest" with 5D+ input, mode="tf_crop_and_resize", and attributes exclude_outside and extrapolation_value with non-zero values)
        * ScatterND-11, 13
        * SpaceToDepth-11, 13
* TensorFlow*:
    * Added support for the following operations:
        * Acosh
        * Asinh
        * Atanh
        * CTCLoss
        * EuclideanNorm
        * ExtractImagePatches
        * FloorDiv
* MXNet*:
    * Added support for the following operations:
        * Acosh
        * Asinh
        * Atanh
* Kaldi*:
    * Fixed bug with ParallelComponent support. Now it is fully supported with no restrictions.

> **NOTE:** 
> [Intel® System Studio](https://software.intel.com/en-us/system-studio) is an all-in-one, cross-platform tool suite, purpose-built to simplify system bring-up and improve system and IoT device application performance on Intel® platforms. If you are using the Intel® Distribution of OpenVINO™ with Intel® System Studio, go to [Get Started with Intel® System Studio](https://software.intel.com/en-us/articles/get-started-with-openvino-and-intel-system-studio-2019).

## Table of Content

* [Preparing and Optimizing your Trained Model with Model Optimizer](prepare_model/Prepare_Trained_Model.md)
    * [Configuring Model Optimizer](prepare_model/Config_Model_Optimizer.md)
    * [Converting a Model to Intermediate Representation (IR)](prepare_model/convert_model/Converting_Model.md)
        * [Converting a Model Using General Conversion Parameters](prepare_model/convert_model/Converting_Model_General.md)
        * [Converting Your Caffe* Model](prepare_model/convert_model/Convert_Model_From_Caffe.md)
        * [Converting Your TensorFlow* Model](prepare_model/convert_model/Convert_Model_From_TensorFlow.md)
            * [Converting BERT from TensorFlow](prepare_model/convert_model/tf_specific/Convert_BERT_From_Tensorflow.md)
            * [Converting GNMT from TensorFlow](prepare_model/convert_model/tf_specific/Convert_GNMT_From_Tensorflow.md)
            * [Converting YOLO from DarkNet to TensorFlow and then to IR](prepare_model/convert_model/tf_specific/Convert_YOLO_From_Tensorflow.md)
            * [Converting Wide and Deep Models from TensorFlow](prepare_model/convert_model/tf_specific/Convert_WideAndDeep_Family_Models.md)
            * [Converting FaceNet from TensorFlow](prepare_model/convert_model/tf_specific/Convert_FaceNet_From_Tensorflow.md)
            * [Converting DeepSpeech from TensorFlow](prepare_model/convert_model/tf_specific/Convert_DeepSpeech_From_Tensorflow.md)
            * [Converting Language Model on One Billion Word Benchmark from TensorFlow](prepare_model/convert_model/tf_specific/Convert_lm_1b_From_Tensorflow.md)
            * [Converting Neural Collaborative Filtering Model from TensorFlow*](prepare_model/convert_model/tf_specific/Convert_NCF_From_Tensorflow.md)

            * [Converting TensorFlow* Object Detection API Models](prepare_model/convert_model/tf_specific/Convert_Object_Detection_API_Models.md)
            * [Converting TensorFlow*-Slim Image Classification Model Library Models](prepare_model/convert_model/tf_specific/Convert_Slim_Library_Models.md)
            * [Converting CRNN Model from TensorFlow*](prepare_model/convert_model/tf_specific/Convert_CRNN_From_Tensorflow.md)
        * [Converting Your MXNet* Model](prepare_model/convert_model/Convert_Model_From_MxNet.md)
            * [Converting a Style Transfer Model from MXNet](prepare_model/convert_model/mxnet_specific/Convert_Style_Transfer_From_MXNet.md)
        * [Converting Your Kaldi* Model](prepare_model/convert_model/Convert_Model_From_Kaldi.md)
        * [Converting Your ONNX* Model](prepare_model/convert_model/Convert_Model_From_ONNX.md)
            * [Converting Mask-RCNN ONNX* Model](prepare_model/convert_model/onnx_specific/Convert_Mask_RCNN.md)
            * [Converting DLRM ONNX* Model](prepare_model/convert_model/onnx_specific/Convert_DLRM.md)
        * [Model Optimizations Techniques](prepare_model/Model_Optimization_Techniques.md)
        * [Cutting parts of the model](prepare_model/convert_model/Cutting_Model.md)
        * [Sub-graph Replacement in Model Optimizer](prepare_model/customize_model_optimizer/Subgraph_Replacement_Model_Optimizer.md)
            * [(Deprecated) Case-Study: Converting SSD models created with the TensorFlow* Object Detection API](prepare_model/customize_model_optimizer/TensorFlow_SSD_ObjectDetection_API.md)
            * [(Deprecated) Case-Study: Converting Faster R-CNN models created with the TensorFlow* Object Detection API](prepare_model/customize_model_optimizer/TensorFlow_Faster_RCNN_ObjectDetection_API.md)
        * [Supported Framework Layers](prepare_model/Supported_Frameworks_Layers.md)
        * [Intermediate Representation and Operation Sets](IR_and_opsets.md)
        * [Operations Specification](../ops/opset.md)
        * [Intermediate Representation suitable for INT8 inference](prepare_model/convert_model/IR_suitable_for_INT8_inference.md)

    * [Custom Layers in Model Optimizer](prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md)
        * [Extending Model Optimizer with New Primitives](prepare_model/customize_model_optimizer/Extending_Model_Optimizer_with_New_Primitives.md)
        * [Legacy Mode for Caffe* Custom Layers](prepare_model/customize_model_optimizer/Legacy_Mode_for_Caffe_Custom_Layers.md)

    * [Model Optimizer Frequently Asked Questions](prepare_model/Model_Optimizer_FAQ.md)

* [Known Issues](Known_Issues_Limitations.md)

**Typical Next Step:** [Preparing and Optimizing your Trained Model with Model Optimizer](prepare_model/Prepare_Trained_Model.md)
