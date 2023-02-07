# Converting a PyTorch Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch}

The PyTorch framework is supported through export to the ONNX format. Model Optimizer Python API allows the conversion of PyTorch models using the `convert_model()` method, which internally converts a model to ONNX.

## Converting a PyTorch model from memory with Python API (Experimental Functionality)

To convert a PyTorch model using `convert_model()`, provide the `input_shape` or `example_input`.

`input_shape` is used to set shapes of model inputs, including dynamic shapes or shapes with boundaries.
`example_input` is used to provide model input example.

PyTorch model exporting to ONNX requires providing a dummy input to `torch.onnx.export()`. 
If `example_input` is set then it is used as a dummy input, otherwise `input_shape` is used to construct a dummy input.

If `input_shape` is set and provided shape is static then zero-filled float `torch.Tensor` is created with the specified shape, and it is used as dummy input for `torch.onnx.export()`. 
If dynamic shape is specified then for creating a dummy input all fully dynamic dimensions are replaced with ones, dimensions with boundaries are replaced with lower bound or upper bound if lower bound is not set.
After exporting to ONNX the model is converted with original dynamic shape from `input_shape` parameter.

If both `input_shape` and `example_input` are set then `example_input` is used as dummy input for exporting model to ONNX and then the model is converted with the shape from `input_shape` parameter.


```sh
import torchvision
import torch
from openvino.tools.mo import convert_model

model = torchvision.models.resnet50(pretrained=True)
ov_model = convert_model(model, example_input=torch.zeros(1, 3, 100, 100))
```

'example_input' accepts the following formats:

* `openvino.runtime.Tensor`
* `torch.Tensor`
* `np.ndarray`
* `list` or `tuple` with tensors (`openvino.runtime.Tensor` / `torch.Tensor` / `np.ndarray`)
* `dictionary` where key is the input name, value is the tensor (`openvino.runtime.Tensor` / `torch.Tensor` / `np.ndarray`)

ONNX opset version can be set using an optional `onnx_opset_version` parameter.
If the `onnx_opset_version` is not set, the default opset from `torch.onnx.export()` is used.

```sh
import torchvision

model = torchvision.models.resnet50(pretrained=True)
ov_model = convert_model(model, input_shape=[1,3,100,100], onnx_opset_version=13)
```

## Exporting a PyTorch Model to ONNX Format <a name="export-to-onnx"></a>

When `convert_model()` fails to convert the model you can convert it to ONNX first with specific `torch.onnx.export()` parameters which are not available in `convert_model()`, explore the model and apply model cutting techniques (with `input`/`output` parameters) if there are ONNX operators that are not supported by OpenVINO.

1. [Export a PyTorch model to ONNX](#export-to-onnx).
2. [Convert the ONNX model](Convert_Model_From_ONNX.md) to produce an optimized [Intermediate Representation](@ref openvino_docs_MO_DG_IR_and_opsets) of the model based on the trained network topology, weights, and biases values.

PyTorch models are defined in Python. To export them, use the `torch.onnx.export()` method. The code to
evaluate or test the model is usually provided with its code and can be used for its initialization and export.
The export to ONNX is crucial for this process, but it is covered by PyTorch framework, therefore, It will not be covered here in detail. 
For more information, refer to the [Exporting PyTorch models to ONNX format](https://pytorch.org/docs/stable/onnx.html) guide.

To export a PyTorch model, you need to obtain the model as an instance of `torch.nn.Module` class and call the `export` function.

```python
import torch

# Instantiate your model. This is just a regular PyTorch model that will be exported in the following steps.
model = SomeModel()
# Evaluate the model to switch some operations from training mode to inference.
model.eval()
# Create dummy input for the model. It will be used to run the model inside export function.
dummy_input = torch.randn(1, 3, 224, 224)
# Call the export function
torch.onnx.export(model, (dummy_input, ), 'model.onnx')
```

## Known Issues

* As of version 1.8.1, not all PyTorch operations can be exported to ONNX opset 9 which is used by default.
It is recommended to export models to opset 11 or higher when export to default opset 9 is not working. In that case, use `opset_version`
option of the `torch.onnx.export`. For more information about ONNX opset, refer to the [Operator Schemas](https://github.com/onnx/onnx/blob/master/docs/Operators.md) page.

## Additional Resources
See the [Model Conversion Tutorials](@ref openvino_docs_MO_DG_prepare_model_convert_model_tutorials) page for a set of tutorials providing step-by-step instructions for converting specific PyTorch models. Here are some examples:
* [Convert PyTorch BERT-NER Model](@ref openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_Bert_ner)
* [Convert PyTorch RCAN Model](@ref openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_RCAN)
* [Convert PyTorch YOLACT Model](@ref openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_YOLACT)
