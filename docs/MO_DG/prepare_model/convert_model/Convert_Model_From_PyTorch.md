# Converting a PyTorch Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch}

The PyTorch framework is supported through export to the ONNX format. Model Optimizer Python API allows to convert PyTorch models through usage of `convert_model()` method which internally converts a model to ONNX.

## Conversion of PyTorch model from the memory using Python API (Experimental Functionality)

Converting a PyTorch model using `convert_model()` requires providing of `input_shape` or `example_input`.

```sh
import torchvision
import torch
from openvino.tools.mo import convert_model

model = torchvision.models.resnet50(pretrained=True)
ov_model = convert_model(model, example_input=torch.zeros(1, 3, 100, 100))
```

'example_input' accepts following formats:

* `openvino.runtime.Tensor`
* `torch.Tensor`
* `np.ndarray`
* `list` or `tuple` with tensors (`openvino.runtime.Tensor` / `torch.Tensor` / `np.ndarray`)
* `dictionary` where key is input name, value is tensor (`openvino.runtime.Tensor` / `torch.Tensor` / `np.ndarray`)

ONNX opset version can be set using optional `onnx_opset_version` parameter.
If `onnx_opset_version` is not set default opset from `torch.onnx.export()` is used.

```sh
import torchvision

model = torchvision.models.resnet50(pretrained=True)
ov_model = convert_model(model, input_shape=[1,3,100,100], onnx_opset_version=13)
```

## Exporting a PyTorch Model to ONNX Format <a name="export-to-onnx"></a>

For complicated cases PyTorch models can be first exported to ONNX prior to MO conversion:

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
