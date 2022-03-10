# Setting Input Shapes {#openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model}

Paragraphs below provide details about specifing input shapes for model conversion.

## When to Specify `--input_shape` Command Line Parameter <a name="when_to_specify_input_shapes"></a>
There are situations when Model Optimizer is unable to deduce input shapes of the model. In this case, user has to specify input shapes explicitly
using `--input_shape` parameter. For example, run the Model Optimizer for the TensorFlow* MobileNet model with the single input
and specify input shape `[2,300,300,3]`.

```sh
mo --input_model MobileNet.pb --input_shape [2,300,300,3]
```

If a model has multiple inputs, `--input_shape` must be used in conjunction with `--input` parameter.
The parameter `--input` contains a list of input names for which shapes in the same order are defined via `--input_shape`.
For example, launch the Model Optimizer for the ONNX* OCR model with a pair of inputs `data` and `seq_len` 
and specify shapes `[3,150,200,1]` and `[3]` for them.

```sh
mo --input_model ocr.onnx --input data,seq_len --input_shape [3,150,200,1],[3]
```

The parameter `--input_shape` allows to override original input shapes to ones compatible for a given model.
Dynamic shapes, i.e. with dynamic dimensions, in the original model can be replaced with static shapes for the converted model, and vice versa.
The dynamic dimension can be marked in Model Optimizer command-line as `-1` or `?`.
For example, launch the Model Optimizer for the ONNX* OCR model and specify dynamic batch dimension for inputs.

```sh
mo --input_model ocr.onnx --input data,seq_len --input_shape [-1,150,200,1],[-1]
```

## When to Specify `--static_shape` Command Line Parameter
If the `--static_shape` command line parameter is specified, the Model Optimizer evaluates shapes of all operations in the model (shape propagation)
for a fixed input(s) shape(s). During the shape propagation the Model Optimizer evaluates operations *Shape* and removes them from the computation graph.
With that approach, the initial model which can consume inputs of different shapes may be converted to IR working with the input of one fixed shape only.
For example, consider the case when some blob is reshaped from 4D of a shape *[N, C, H, W]* to a shape *[N, C, H \* W]*.
During the model conversion the Model Optimize calculates output shape as a constant 1D blob with values *[N, C, H \* W]*.
So if the input shape changes to some other value *[N, C, H1, W1]* (it is possible scenario for a fully convolutional model) then the reshape layer
becomes invalid. Resulting Intermediate Representation will not be resizable with the help of OpenVINO Runtime API.

## See Also
* [Introduction](../../Deep_Learning_Model_Optimizer_DevGuide.md)
* [Cutting Off Parts of a Model](Cutting_Model.md)
* [Optimizing Preprocessing Computation](../Additional_Optimizations.md)
* [Compression of a Model to FP16](../FP16_Compression.md)
* [Converting TensorFlow Models](Convert_Model_From_TensorFlow.md)
* [Converting ONNX Models](Convert_Model_From_ONNX.md)
* [Converting PyTorch Models](Convert_Model_From_PyTorch.md)
* [Converting PaddlePaddle Models](Convert_Model_From_Paddle.md)
* [Converting MXNet Models](Convert_Model_From_MxNet.md)
* [Converting Caffe Models](Convert_Model_From_Caffe.md)
* [Converting Kaldi Models](Convert_Model_From_Kaldi.md)
