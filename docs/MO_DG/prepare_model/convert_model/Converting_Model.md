# Setting Input Shapes {#openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model}

Model Optimizer provides additional shape definition, which results in more efficient model creation.
It is achieved with two parameters: `--input_shape` and `--static_shape`, used under certain conditions.

@anchor when_to_specify_input_shapes
## When to Specify --input_shape Command-line Parameter
Model Optimizer supports conversion of models with dynamic input shapes that contain undefined dimensions.
However, if the shape of data does not change from one inference to another,
it is recommended to set up static shapes (when all dimensions are fully defined) for the inputs.
This can have a beneficial effect on performance and memory consumption.
Model Optimizer provides the `--input_shape` parameter for setting up static shapes.
The same functionality is also available in runtime via `reshape` method. Refer to the [Changing input shapes](../../../OV_Runtime_UG/ShapeInference.md) guide.
You may also refer to the [Dynamic Shapes](../../../OV_Runtime_UG/ov_dynamic_shapes.md) guide for more information about dynamic shapes in runtime.

OpenVINO Runtime API can have limitations to infer models with undefined dimensions on some hardware (see the [Features support matrix](../../../OV_Runtime_UG/supported_plugins/Device_Plugins.md) page for reference).
In this case, the `--input_shape` parameter and the [reshape method](../../../OV_Runtime_UG/ShapeInference.md) can help to resolve undefined dimensions.

Sometimes, Model Optimizer is unable to convert models out-of-the-box (only the `--input_model` parameter is specified).
Such problem can relate to models with inputs of undefined ranks and a case of cutting off parts of a model.
In this case, user has to specify input shapes explicitly using `--input_shape` parameter.

For example, run the Model Optimizer for the TensorFlow* MobileNet model with the single input
and specify input shape `[2,300,300,3]`.

```sh
mo --input_model MobileNet.pb --input_shape [2,300,300,3]
```

If a model has multiple inputs, `--input_shape` must be used in conjunction with `--input` parameter.
The parameter `--input` contains a list of input names for which shapes in the same order are defined via `--input_shape`.
For example, launch the Model Optimizer for the ONNX OCR model with a pair of inputs `data` and `seq_len`
and specify shapes `[3,150,200,1]` and `[3]` for them.

```sh
mo --input_model ocr.onnx --input data,seq_len --input_shape [3,150,200,1],[3]
```

The alternative way to specify input shapes is to use the `--input` parameter as follows:

```sh
mo --input_model ocr.onnx --input data[3 150 200 1],seq_len[3]
```

The parameter `--input_shape` allows overriding original input shapes to the shapes compatible with a given model.
Dynamic shapes, i.e. with dynamic dimensions, in the original model can be replaced with static shapes for the converted model, and vice versa.
The dynamic dimension can be marked in Model Optimizer command-line as `-1` or `?`.
For example, launch the Model Optimizer for the ONNX OCR model and specify dynamic batch dimension for inputs.

```sh
mo --input_model ocr.onnx --input data,seq_len --input_shape [-1,150,200,1],[-1]
```

To optimize memory consumption for models with undefined dimensions in run-time, Model Optimizer provides the capability to define boundaries of dimensions.
The boundaries of undefined dimension can be specified with ellipsis.
For example, launch the Model Optimizer for the ONNX OCR model and specify a boundary for the batch dimension.

```sh
mo --input_model ocr.onnx --input data,seq_len --input_shape [1..3,150,200,1],[1..3]
```

Practically, some models are not ready for input shapes change.
In this case, a new input shape cannot be set via Model Optimizer.
Learn more about shape [inference troubleshooting](@ref troubleshooting_reshape_errors) and [ways to relax shape inference flow](@ref how-to-fix-non-reshape-able-model).

## When to Specify --static_shape Command-line Parameter
Model Optimizer provides the `--static_shape` parameter that allows evaluating shapes of all operations in the model for fixed input shapes
and to fold shape computing sub-graphs into constants. The resulting IR can be more compact in size and the loading time for such IR can be decreased.
However, the resulting IR will not be reshape-able with the help of the [reshape method](../../../OV_Runtime_UG/ShapeInference.md) from OpenVINO Runtime API.
It is worth noting that the `--input_shape` parameter does not affect reshape-ability of the model.

For example, launch the Model Optimizer for the ONNX OCR model using `--static_shape`.

```sh
mo --input_model ocr.onnx --input data[3 150 200 1],seq_len[3] --static_shape
```

## See Also
* [Introduction](../../Deep_Learning_Model_Optimizer_DevGuide.md)
* [Cutting Off Parts of a Model](Cutting_Model.md)
