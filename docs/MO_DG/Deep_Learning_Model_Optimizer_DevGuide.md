# Model Optimizer Developer Guide {#openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide}

Model Optimizer is a cross-platform command-line tool that facilitates the transition between the training and deployment environment, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices.

Model Optimizer process assumes you have a network model trained using a supported deep learning framework. The scheme below illustrates the typical workflow for deploying a trained deep learning model:

![](img/workflow_steps.png)

Model Optimizer produces an Intermediate Representation (IR) of the network, which can be read, loaded, and inferred with the Inference Engine. The Inference Engine API offers a unified API across a number of supported Intel® platforms. The Intermediate Representation is a pair of files describing the model:

*  <code>.xml</code> - Describes the network topology

*  <code>.bin</code> - Contains the weights and biases binary data.

## What's New in the Model Optimizer in this Release?

**Deprecation Notice**

<table>
  <tr>
    <td><strong>Deprecation Begins</strong></td>
    <td>June 1, 2020</td>
  </tr>
  <tr>
    <td><strong>Removal Date</strong></td>
    <td>December 1, 2020</td>
  </tr>
</table> 

*Starting with the OpenVINO™ toolkit 2020.2 release, all of the features previously available through nGraph have been merged into the OpenVINO™ toolkit. As a result, all the features previously available through ONNX RT Execution Provider for nGraph have been merged with ONNX RT Execution Provider for OpenVINO™ toolkit.*

*Therefore, ONNX RT Execution Provider for nGraph will be deprecated starting June 1, 2020 and will be completely removed on December 1, 2020. Users are recommended to migrate to the ONNX RT Execution Provider for OpenVINO™ toolkit as the unified solution for all AI inferencing on Intel® hardware.*

* Common changes:
    * Implemented generation of a compressed OpenVINO IR suitable for INT8 inference, which takes up to 4 times less disk space than an expanded one. Use the `--disable_weights_compression` Model Optimizer command-line parameter to get an expanded version.
    * Implemented an optimization transformation to replace a sub-graph with the `Erf` operation into the `GeLU` operation.
    * Implemented an optimization transformation to replace an upsamping pattern that is represented as a sequence of `Split` and `Concat` operations to a single `Interpolate` operation.
    * Fixed a number of Model Optimizer bugs to generate reshape-able IRs of many models with the command line parameter `--keep_shape_ops`.
    * Fixed a number of Model Optimizer transformations to set operations name in an IR equal to the original framework model operation name.
    * The following operations are no longer generated with `version="opset1"`: `MVN`, `ROIPooling`, `ReorgYolo`. They became a part of new `opset2` operation set and generated with `version="opset2"`. Before this fix, the operations were generated with `version="opset1"` by mistake, they were not a part of `opset1` nGraph namespace; `opset1` specification was fixed accordingly.

* ONNX*:
    * Added support for the following operations: `MeanVarianceNormalization` if normalization is performed over spatial dimensions.

* TensorFlow*:
    * Added support for the TensorFlow Object Detection models version 1.15.X.
    * Added support for the following operations: `BatchToSpaceND`, `SpaceToBatchND`, `Floor`.

* MXNet*:
    * Added support for the following operations:
        * `Reshape` with input shape values equal to -2, -3, and -4.

> **NOTE:** 
> [Intel® System Studio](https://software.intel.com/en-us/system-studio) is an all-in-one, cross-platform tool suite, purpose-built to simplify system bring-up and improve system and IoT device application performance on Intel® platforms. If you are using the Intel® Distribution of OpenVINO™ with Intel® System Studio, go to [Get Started with Intel® System Studio](https://software.intel.com/en-us/articles/get-started-with-openvino-and-intel-system-studio-2019).

## Table of Content

* [Introduction to OpenVINO™ Deep Learning Deployment Toolkit](../IE_DG/Introduction.md)

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

**Typical Next Step:** [Introduction to Intel® Deep Learning Deployment Toolkit](../IE_DG/Introduction.md)
