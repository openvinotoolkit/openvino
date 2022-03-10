# Model Optimizer User Guide {#openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide}

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

Model Optimizer is a cross-platform command-line tool that facilitates transition between training and deployment environments, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices.

Using Model Optimizer tool assumes you already have a deep-learning model trained using one of the supported frameworks: TensorFlow, PyTorch, PaddlePaddle, MXNet, Caffe, Kaldi, or represented in ONNX* format. Model Optimizer produces an Intermediate Representation (IR) of the model, which can be inferred with [OpenVINO™ Runtime](../OV_Runtime_UG/openvino_intro.md).

> **NOTE**: Model Optimizer does not infer models. Model Optimizer is an offline tool that runs before the inference takes place.

The scheme below illustrates the typical workflow for deploying a trained deep learning model:

![](img/BASIC_FLOW_MO_simplified.svg)

The IR is a pair of files describing the model:

*  <code>.xml</code> - Describes the network topology

*  <code>.bin</code> - Contains the weights and biases binary data.

> **TIP**: You also can work with the Model Optimizer inside the OpenVINO™ [Deep Learning Workbench](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Introduction.html) (DL Workbench).
> [DL Workbench](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Introduction.html) is a web-based graphical environment that enables you to optimize, fine-tune, analyze, visualize, and compare performance of deep learning models.

## Run Model Optimizer

To convert the model to IR, run Model Optimizer:

```sh
mo --input_model INPUT_MODEL
```

To override original input shapes for model conversion, Model Optimizer provides two parameters: `--input` and `--input_shape`.
For more information about these parameters, please refer to [Setting Input Shapes](prepare_model/convert_model/Converting_Model.md).

To cut off unwanted parts of a model, such as unsupported operations and training sub-graphs,
the `--input` and `--output` parameters can be used, defining new inputs and outputs of the converted model.
For a more detailed description, refer to [Cutting Off Parts of a Model](prepare_model/convert_model/Cutting_Model.md).

Also, Model Optimizer can insert additional input pre-processing sub-graphs into the converted model.
For this, there exist `--mean_values`, `scales_values`, `--layout`, and other parameters described
in [Embedding Preprocessing Computation](prepare_model/Additional_Optimizations.md).

Model Optimizer provides compression parameter `--data_type` to generate IR of `FP16` data type. For more details,
please refer to [Compression of a Model to FP16](prepare_model/FP16_Compression.md).

To get the full list of conversion parameters available in Model Optimizer, run the following command:

```sh
mo --help
```

## Examples of CLI Commands

Launch the Model Optimizer for the TensorFlow* BERT model in binary protobuf format with three inputs and explicitly specify input shapes
where the batch size and the sequence length equal 2 and 30. For more information about TensorFlow* models conversion,
please refer to [Converting a TensorFlow* Model](prepare_model/convert_model/Convert_Model_From_TensorFlow.md).

```sh
mo --input_model bert.pb --input input_mask,input_word_ids,input_type_ids --input_shape [2,30],[2,30],[2,30]
```

Launch the Model Optimizer for the TensorFlow* MobileNet model in SavedModel format.

```sh
mo --saved_model_dir MobileNet
```

Launch the Model Optimizer for the ONNX* OCR model and explicitly specify new output. For more information about ONNX* models conversion,
please refer to [Converting a ONNX* Model](prepare_model/convert_model/Convert_Model_From_ONNX.md).
Note that PyTorch* models must be exported to ONNX* format before its conversion into IR,
follow details from [Converting a PyTorch* Model](prepare_model/convert_model/Convert_Model_From_PyTorch.md).

```sh
mo --input_model ocr.onnx --output probabilities
```

Launch the Model Optimizer for the PaddlePaddle* UNet model and apply mean-scale normalization to the input.
For more information about ONNX* models conversion, please refer to
[Converting a PaddlePaddle* Model](prepare_model/convert_model/Convert_Model_From_Paddle.md).

```sh
mo --input_model unet.pdmodel --mean_values [123,117,104] --scale 255
```

Launch the Model Optimizer for the MXNet* SSD Inception V3 model and specify input layout.
For more information about MXNet* models conversion, please refer to [Converting a MXNet* Model](prepare_model/convert_model/Convert_Model_From_MxNet.md).

```sh
mo --input_model ssd_inception_v3-0000.params --layout NCHW
```

Launch the Model Optimizer for the Caffe* AlexNet model with reversed input channels order between RGB and BGR.
For more information about Caffe* models conversion, please refer to [Converting a Caffe* Model](prepare_model/convert_model/Convert_Model_From_Caffe.md).

```sh
mo --input_model alexnet.caffemodel --reverse_input_channels
```

Launch the Model Optimizer for the Kaldi* LibriSpeech nnet2 model. For more information about Kaldi* models conversion,
please refer to [Converting a Kaldi* Model](prepare_model/convert_model/Convert_Model_From_Kaldi.md).

```sh
mo --input_model librispeech_nnet2.mdl --input_shape [1,140]
```

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
