# Setting Input Shapes {#openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model}

Model Optimizer provides the option of making models more efficient by providing additional shape definition.
It is achieved with two parameters, used under certain conditions: `--input_shape` and `--static_shape`.

@anchor when_to_specify_input_shapes
## Specifying --input_shape Command-line Parameter
Model Optimizer supports conversion of models with dynamic input shapes that contain undefined dimensions.
However, if the shape of data is not going to change from one inference request to another,
it is recommended to set up static shapes (when all dimensions are fully defined) for the inputs.
Doing it at this stage, instead of during inference in runtime, can be beneficial in terms of performance and memory consumption.
To set up static shapes, Model Optimizer provides the `--input_shape` parameter.
The same functionality is also available in runtime via `reshape` method. For more information refer to the [Changing input shapes](../../../OV_Runtime_UG/ShapeInference.md) guide.
To learn more about dynamic shapes in runtime, refer to the [Dynamic Shapes](../../../OV_Runtime_UG/ov_dynamic_shapes.md) guide.

The OpenVINO Runtime API may present certain limitations in inferring models with undefined dimensions on some hardware. See the [Features support matrix](../../../OV_Runtime_UG/supported_plugins/Device_Plugins.md) for reference.
In this case, the `--input_shape` parameter and the [reshape method](../../../OV_Runtime_UG/ShapeInference.md) can help resolve undefined dimensions.

Sometimes, Model Optimizer is unable to convert models out-of-the-box (only the `--input_model` parameter is specified).
Such problem can relate to models with inputs of undefined ranks and a case of cutting off parts of a model.
In this case, input shapes must be specified explicitly by `--input_shape` parameter.

For example, run Model Optimizer for the TensorFlow MobileNet model with the single input
and specify input shape `[2,300,300,3]`:

```sh
mo --input_model MobileNet.pb --input_shape [2,300,300,3]
```

If a model has multiple inputs, `--input_shape` must be used in conjunction with `--input` parameter.
The `--input` parameter contains a list of input names, for which shapes in the same order are defined via `--input_shape`.
For example, launch Model Optimizer for the ONNX OCR model with a pair of inputs `data` and `seq_len`
and specify shapes `[3,150,200,1]` and `[3]` for them:

```sh
mo --input_model ocr.onnx --input data,seq_len --input_shape [3,150,200,1],[3]
```

Alternatively, specify input shapes, using the `--input` parameter as follows:

```sh
mo --input_model ocr.onnx --input data[3 150 200 1],seq_len[3]
```

The `--input_shape` parameter allows overriding original input shapes to ones compatible with a given model.
Dynamic shapes, i.e. with dynamic dimensions, can be replaced in the original model with static shapes for the converted model, and vice versa.
The dynamic dimension can be marked in Model Optimizer command-line as `-1`* or *`?`.
For example, launch Model Optimizer for the ONNX OCR model and specify dynamic batch dimension for inputs:

```sh
mo --input_model ocr.onnx --input data,seq_len --input_shape [-1,150,200,1],[-1]
```

To optimize memory consumption for models with undefined dimensions in run-time, Model Optimizer provides the capability to define boundaries of dimensions.
The boundaries of undefined dimension can be specified with ellipsis.
For example, launch Model Optimizer for the ONNX OCR model and specify a boundary for the batch dimension:

```sh
mo --input_model ocr.onnx --input data,seq_len --input_shape [1..3,150,200,1],[1..3]
```

Practically, some models are not ready for input shapes change.
In this case, a new input shape cannot be set via Model Optimizer.
For more information about shape follow the [inference troubleshooting](@ref troubleshooting_reshape_errors) and [ways to relax shape inference flow](@ref how-to-fix-non-reshape-able-model) guides. 

## Specifying --static_shape Command-line Parameter
Model Optimizer provides the `--static_shape` parameter that allows evaluating shapes of all operations in the model for fixed input shapes
and folding shape computing sub-graphs into constants. The resulting IR may be more compact in size and the loading time for such IR may decrease.
However, the resulting IR will not be reshape-able with the help of the [reshape method](../../../OV_Runtime_UG/ShapeInference.md) from OpenVINO Runtime API.
Be aware that the `--input_shape` parameter does not affect reshape-ability of the model.

For example, launch Model Optimizer for the ONNX OCR model using `--static_shape`:

```sh
mo --input_model ocr.onnx --input data[3 150 200 1],seq_len[3] --static_shape
```

## Additional Resources
* [Introduction to converting models with Model Optimizer](../../Deep_Learning_Model_Optimizer_DevGuide.md)
* [Cutting Off Parts of a Model](Cutting_Model.md)
