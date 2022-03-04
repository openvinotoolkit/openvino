# General Conversion Parameters {#openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model}

To get the full list of general (framework-agnostic) conversion parameters available in Model Optimizer, run the following command:

```sh
mo --help
```

Paragraphs below provide useful details on relevant parameters.

## When to Specify --input_shape Command Line Parameter <a name="when_to_specify_input_shapes"></a>
There are situations when Model Optimizer is unable to deduce input shapes of the model, for example, in case of model cutting due to unsupported operations.
The solution is to provide input shapes of a static rank explicitly.

## When to Specify --static_shape Command Line Parameter
If the `--static_shape` command line parameter is specified the Model Optimizer evaluates shapes of all operations in the model (shape propagation) for a fixed input(s) shape(s). During the shape propagation the Model Optimizer evaluates operations *Shape* and removes them from the computation graph. With that approach, the initial model which can consume inputs of different shapes may be converted to IR working with the input of one fixed shape only. For example, consider the case when some blob is reshaped from 4D of a shape *[N, C, H, W]* to a shape *[N, C, H \* W]*. During the model conversion the Model Optimize calculates output shape as a constant 1D blob with values *[N, C, H \* W]*. So if the input shape changes to some other value *[N,C,H1,W1]* (it is possible scenario for a fully convolutional model) then the reshape layer becomes invalid.
Resulting Intermediate Representation will not be resizable with the help of OpenVINO Runtime API.

## Parameters for Pre-Processing
Input data may require pre-processing such as `RGB<->BGR` conversion and mean and scale normalization. To learn about Model Optimizer parameters used for pre-processing, refer to [Optimize Preprocessing Computation](../Additional_Optimizations.md).

## See Also
* [Configuring the Model Optimizer](../../Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Cutting](Cutting_Model.md)
* [Optimize Preprocessing Computation](../Additional_Optimizations.md)
* [Convert TensorFlow Models](Convert_Model_From_TensorFlow.md)
* [Convert ONNX Models](Convert_Model_From_ONNX.md)
* [Convert PyTorch Models](Convert_Model_From_PyTorch.md)
* [Convert PaddlePaddle Models](Convert_Model_From_Paddle.md)
* [Convert MXNet Models](Convert_Model_From_MxNet.md)
* [Convert Caffe Models](Convert_Model_From_Caffe.md)
* [Convert Kaldi Models](Convert_Model_From_Kaldi.md)