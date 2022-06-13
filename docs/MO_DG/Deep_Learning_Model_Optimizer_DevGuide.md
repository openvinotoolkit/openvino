# Converting Models with Model Optimizer {#openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide}

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

Model Optimizer is a cross-platform command-line tool that facilitates the transition between training and deployment environments, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices.

To use it, you need a pre-trained deep learning model in one of the supported formats: TensorFlow, PyTorch, PaddlePaddle, MXNet, Caffe, Kaldi, or ONNX. Model Optimizer converts the model to the OpenVINO Intermediate Representation format (IR), which you can infer later with [OpenVINO™ Runtime](../OV_Runtime_UG/openvino_intro.md).

Note that Model Optimizer does not infer models.

The figure below illustrates the typical workflow for deploying a trained deep learning model:

![](img/BASIC_FLOW_MO_simplified.svg)

where IR is a pair of files describing the model:

*  <code>.xml</code> - Describes the network topology.

*  <code>.bin</code> - Contains the weights and biases binary data.

The generated IR can be additionally optimized for inference by [Post-training optimization](../../tools/pot/docs/Introduction.md)
> that applies post-training quantization methods.

> **TIP**: You can also work with Model Optimizer in OpenVINO™ [Deep Learning Workbench (DL Workbench)](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Introduction.html), which is a web-based tool with GUI for optimizing, fine-tuning, analyzing, visualizing, and comparing performance of deep learning models.

## How to Run Model Optimizer

To convert a model to IR, you can run Model Optimizer by using the following command:

```sh
mo --input_model INPUT_MODEL
```

If the out-of-the-box conversion (only the `--input_model` parameter is specified) is not successful, use the parameters mentioned below to override input shapes and cut the model:

- Model Optimizer provides two parameters to override original input shapes for model conversion: `--input` and `--input_shape`.
For more information about these parameters, refer to the [Setting Input Shapes](prepare_model/convert_model/Converting_Model.md) guide.

- To cut off unwanted parts of a model (such as unsupported operations and training sub-graphs),
use the `--input` and `--output` parameters to define new inputs and outputs of the converted model.
For a more detailed description, refer to the [Cutting Off Parts of a Model](prepare_model/convert_model/Cutting_Model.md) guide.

You can also insert additional input pre-processing sub-graphs into the converted model by using
the `--mean_values`, `scales_values`, `--layout`, and other parameters described
in the [Embedding Preprocessing Computation](prepare_model/Additional_Optimizations.md) article.

The `--data_type` compression parameter in Model Optimizer allows generating IR of the `FP16` data type. For more details, refer to the [Compression of a Model to FP16](prepare_model/FP16_Compression.md) guide.

To get the full list of conversion parameters available in Model Optimizer, run the following command:

```sh
mo --help
```

## Examples of CLI Commands

Below is a list of separate examples for different frameworks and Model Optimizer parameters:

1. Launch Model Optimizer for a TensorFlow MobileNet model in the binary protobuf format:
```sh
mo --input_model MobileNet.pb
```
Launch Model Optimizer for a TensorFlow BERT model in the SavedModel format with three inputs. Specify input shapes explicitly
where the batch size and the sequence length equal 2 and 30 respectively:
```sh
mo --saved_model_dir BERT --input mask,word_ids,type_ids --input_shape [2,30],[2,30],[2,30]
```
For more information, refer to the [Converting a TensorFlow Model](prepare_model/convert_model/Convert_Model_From_TensorFlow.md) guide.

2. Launch Model Optimizer for an ONNX OCR model and specify new output explicitly:
```sh
mo --input_model ocr.onnx --output probabilities
```
For more information, refer to the [Converting an ONNX Model (prepare_model/convert_model/Convert_Model_From_ONNX.md) guide.

> **NOTE**: PyTorch models must be exported to the ONNX format before conversion into IR. More information can be found in [Converting a PyTorch Model](prepare_model/convert_model/Convert_Model_From_PyTorch.md).

3. Launch Model Optimizer for a PaddlePaddle UNet model and apply mean-scale normalization to the input:
```sh
mo --input_model unet.pdmodel --mean_values [123,117,104] --scale 255
```
For more information, refer to the [Converting a PaddlePaddle Model](prepare_model/convert_model/Convert_Model_From_Paddle.md) guide.

4. Launch Model Optimizer for an Apache MXNet SSD Inception V3 model and specify first-channel layout for the input:
```sh
mo --input_model ssd_inception_v3-0000.params --layout NCHW
```
For more information, refer to the [Converting an Apache MXNet Model](prepare_model/convert_model/Convert_Model_From_MxNet.md) guide.

5. Launch Model Optimizer for a Caffe AlexNet model with input channels in the RGB format which needs to be reversed:
```sh
mo --input_model alexnet.caffemodel --reverse_input_channels
```
For more information, refer to the [Converting a Caffe Model](prepare_model/convert_model/Convert_Model_From_Caffe.md) guide.

6. Launch Model Optimizer for a Kaldi LibriSpeech nnet2 model:
```sh
mo --input_model librispeech_nnet2.mdl --input_shape [1,140]
```
For more information, refer to the [Converting a Kaldi Model](prepare_model/convert_model/Convert_Model_From_Kaldi.md) guide.

- To get conversion recipes for specific TensorFlow, ONNX, PyTorch, Apache MXNet, and Kaldi models,
refer to the [Model Conversion Tutorials](prepare_model/convert_model/Convert_Model_Tutorials.md).
- For more information about IR, see [Deep Learning Network Intermediate Representation and Operation Sets in OpenVINO™](IR_and_opsets.md).
