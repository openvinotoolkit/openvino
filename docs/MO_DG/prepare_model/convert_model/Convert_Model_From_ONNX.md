# Converting an ONNX Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX}

## Introduction to ONNX
[ONNX](https://github.com/onnx/onnx) is a representation format for deep learning models. ONNX allows AI developers to easily transfer models between different frameworks that helps to choose the best combination for them. Today, PyTorch, Caffe2, Apache MXNet, Microsoft Cognitive Toolkit and other tools are developing ONNX support.

## Converting an ONNX Model <a name="Convert_From_ONNX"></a>
The Model Optimizer process assumes you have an ONNX model that was directly downloaded from a public repository or converted from any framework that supports exporting to the ONNX format.

To convert an ONNX model, run Model Optimizer with the path to the input model *`.onnx`* file:

```sh
 mo --input_model <INPUT_MODEL>.onnx
```

There are no ONNX specific parameters, so only framework-agnostic parameters are available to convert your model. For details, see the *General Conversion Parameters* section in the [Converting a Model to Intermediate Representation (IR)](Converting_Model.md) guide.

## Supported ONNX Layers
Refer to the [Supported Framework Layers](../Supported_Frameworks_Layers.md) page for the list of supported standard layers.

## See Also
[Model Conversion Tutorials](Convert_Model_Tutorials.md)
