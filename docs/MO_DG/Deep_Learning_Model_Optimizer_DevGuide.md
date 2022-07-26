# Convert model with Model Optimizer {#openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide}

@sphinxdirective

.. _deep learning model optimizer:

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model
   openvino_docs_MO_DG_prepare_model_convert_model_Cutting_Model
   openvino_docs_MO_DG_Additional_Optimization_Use_Cases
   openvino_docs_MO_DG_FP16_Compression
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Paddle
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_MxNet
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Kaldi
   openvino_docs_MO_DG_prepare_model_convert_model_tutorials
   openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ

@endsphinxdirective

## Introduction

Model Optimizer is a cross-platform command-line tool that facilitates the transition between training and deployment environments, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices.

Using Model Optimizer tool assumes you already have a deep learning model trained using one of the supported frameworks: TensorFlow, PyTorch, PaddlePaddle, MXNet, Caffe, Kaldi, or represented in ONNX* format. Model Optimizer produces an Intermediate Representation (IR) of the model, which can be inferred with [OpenVINO™ Runtime](../OV_Runtime_UG/openvino_intro.md).

> **NOTE**: Model Optimizer does not infer models. Model Optimizer is an offline tool that converts a model into IR and optimizes before the inference takes place.

The scheme below illustrates the typical workflow for deploying a trained deep learning model:

![](img/BASIC_FLOW_MO_simplified.svg)

The IR is a pair of files describing the model:

*  <code>.xml</code> - Describes the network topology

*  <code>.bin</code> - Contains the weights and biases binary data.

> **NOTE**: The generated IR can be additionally optimized for inference by [Post-training optimization](../../tools/pot/docs/Introduction.md)
> that applies post-training quantization methods.

> **TIP**: You also can work with the Model Optimizer inside the OpenVINO™ [Deep Learning Workbench](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Introduction.html) (DL Workbench).
> [DL Workbench](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Introduction.html) is a web-based graphical environment that enables you to optimize, fine-tune, analyze, visualize, and compare performance of deep learning models.

## Run Model Optimizer

To convert the model to IR, run Model Optimizer:

```sh
mo --input_model INPUT_MODEL
```

If out-of-the-box conversion (only the `--input_model` parameter is specified) is not succeed,
try to use parameters for overriding input shapes and cutting the model, mentioned below.

To override original input shapes for model conversion, Model Optimizer provides two parameters: `--input` and `--input_shape`.
For more information about these parameters, refer to [Setting Input Shapes](prepare_model/convert_model/Converting_Model.md).

To cut off unwanted parts of a model, such as unsupported operations and training sub-graphs,
the `--input` and `--output` parameters can be used, defining new inputs and outputs of the converted model.
For a more detailed description, refer to [Cutting Off Parts of a Model](prepare_model/convert_model/Cutting_Model.md).

Also, you can insert additional input pre-processing sub-graphs into the converted model using
the `--mean_values`, `scales_values`, `--layout`, and other parameters described
in [Embedding Preprocessing Computation](prepare_model/Additional_Optimizations.md).

Model Optimizer's compression parameter `--data_type` allows to generate IR of the `FP16` data type. For more details,
please refer to [Compression of a Model to FP16](prepare_model/FP16_Compression.md).

To get the full list of conversion parameters available in Model Optimizer, run the following command:

```sh
mo --help
```

## Examples of CLI Commands

Below is a list of separate examples for different frameworks and Model Optimizer parameters.

1. Launch Model Optimizer for a TensorFlow MobileNet model in the binary protobuf format.
```sh
mo --input_model MobileNet.pb
```
Launch Model Optimizer for a TensorFlow BERT model in the SavedModel format, with three inputs. Explicitly specify input shapes
where the batch size and the sequence length equal 2 and 30 respectively.
```sh
mo --saved_model_dir BERT --input mask,word_ids,type_ids --input_shape [2,30],[2,30],[2,30]
```
For more information on TensorFlow model conversion,
refer to [Converting a TensorFlow Model](prepare_model/convert_model/Convert_Model_From_TensorFlow.md).

2. Launch Model Optimizer for an ONNX OCR model and explicitly specify new output.
```sh
mo --input_model ocr.onnx --output probabilities
```
For more information on ONNX model conversion,
please refer to [Converting an ONNX Model](prepare_model/convert_model/Convert_Model_From_ONNX.md).
Note that PyTorch models must be exported to the ONNX format before its conversion into IR.
More details can be found in [Converting a PyTorch Model](prepare_model/convert_model/Convert_Model_From_PyTorch.md).

3. Launch Model Optimizer for a PaddlePaddle UNet model and apply mean-scale normalization to the input.
```sh
mo --input_model unet.pdmodel --mean_values [123,117,104] --scale 255
```
For more information on PaddlePaddle model conversion, please refer to
[Converting a PaddlePaddle Model](prepare_model/convert_model/Convert_Model_From_Paddle.md).

4. Launch Model Optimizer for an Apache MXNet SSD Inception V3 model and specify first-channel layout for the input:
```sh
mo --input_model ssd_inception_v3-0000.params --layout NCHW
```
For more information, refer to the [Converting an Apache MXNet Model](prepare_model/convert_model/Convert_Model_From_MxNet.md) guide.

5. Launch Model Optimizer for a Caffe AlexNet model with input channels in the RGB format, which needs to be reversed.
```sh
mo --input_model alexnet.caffemodel --reverse_input_channels
```
For more information on Caffe model conversion, please refer to [Converting a Caffe Model](prepare_model/convert_model/Convert_Model_From_Caffe.md).

6. Launch Model Optimizer for a Kaldi LibriSpeech nnet2 model.
```sh
mo --input_model librispeech_nnet2.mdl --input_shape [1,140]
```
For more information on Kaldi model conversion,
refer to [Converting a Kaldi Model](prepare_model/convert_model/Convert_Model_From_Kaldi.md).

- To get conversion recipes for specific TensorFlow, ONNX, PyTorch, Apache MXNet, and Kaldi models,
refer to the [Model Conversion Tutorials](prepare_model/convert_model/Convert_Model_Tutorials.md).
- For more information about IR, see [Deep Learning Network Intermediate Representation and Operation Sets in OpenVINO™](IR_and_opsets.md).
