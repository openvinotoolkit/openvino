# Converting a PyTorch Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch}

The PyTorch framework is supported through export to the ONNX format. In order to optimize and deploy a model that was trained with it:

1. [Export a PyTorch model to ONNX](#export-to-onnx).
2. [Convert the ONNX model](Convert_Model_From_ONNX.md) to produce an optimized [Intermediate Representation](../../IR_and_opsets.md) of the model based on the trained network topology, weights, and biases values.

## Exporting a PyTorch Model to ONNX Format <a name="export-to-onnx"></a>
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

## See Also
[Model Conversion Tutorials](Convert_Model_Tutorials.md)
