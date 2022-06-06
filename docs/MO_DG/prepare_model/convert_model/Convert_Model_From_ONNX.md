# Converting an ONNX Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX}

## Introduction to ONNX
[ONNX](https://github.com/onnx/onnx) is a representation format for deep learning models that allows AI developers to easily transfer models between different frameworks. It is hugely popular among deep learning tools, like PyTorch, Caffe2, Apache MXNet, Microsoft Cognitive Toolkit, and many others.

## Converting an ONNX Model <a name="Convert_From_ONNX"></a>

This page provides instructions on how to convert a model from the ONNX format to the OpenVINO IR format using Model Optimizer. To use Model Optimizer, install OpenVINO Development Tools by following the [installation instructions](https://docs.openvino.ai/latest/openvino_docs_install_guides_install_dev_tools.html).

The Model Optimizer process assumes you have an ONNX model that was directly downloaded from a public repository or converted from any framework that supports exporting to the ONNX format.

To convert an ONNX model, run Model Optimizer with the path to the input model `.onnx` file:

```sh
 mo --input_model <INPUT_MODEL>.onnx
```

There are no ONNX specific parameters, so only framework-agnostic parameters are available to convert your model. For details, see the *General Conversion Parameters* section in the [Converting a Model to Intermediate Representation (IR)](Converting_Model.md) guide.

## Supported ONNX Layers
For the list of supported standard layers, refer to the [Supported Framework Layers](../Supported_Frameworks_Layers.md) page.

## Additional Resources
See the [Model Conversion Tutorials](Convert_Model_Tutorials.md) page for a set of tutorials providing step-by-step instructions for converting specific ONNX models. Here are some examples:
* [Convert ONNX* Faster R-CNN Model](onnx_specific/Convert_Faster_RCNN.md)
* [Convert ONNX* GPT-2 Model](onnx_specific/Convert_GPT2.md)
* [Convert ONNX* Mask R-CNN Model](onnx_specific/Convert_Mask_RCNN.md)

