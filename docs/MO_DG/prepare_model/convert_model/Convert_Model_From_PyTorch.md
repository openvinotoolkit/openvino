# Converting a PyTorch* Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch}

PyTorch* framework is supported through export to ONNX\* format. A summary of the steps for optimizing and deploying a model that was trained with the PyTorch\* framework:

1. [Export PyTorch model to ONNX\*](#export-to-onnx).
2. [Configure the Model Optimizer](../Config_Model_Optimizer.md) for ONNX\*.
3. [Convert an ONNX\* model](Convert_Model_From_ONNX.md) to produce an optimized [Intermediate Representation (IR)](../../IR_and_opsets.md) of the model based on the trained network topology, weights, and biases values.
4. Test the model in the Intermediate Representation format using the [Inference Engine](../../../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) in the target environment via provided [sample applications](../../../IE_DG/Samples_Overview.md).
5. [Integrate](../../../IE_DG/Samples_Overview.md) the Inference Engine in your application to deploy the model in the target environment.

## Supported Topologies

Here is the list of models that were tested and are guaranteed to be supported.
It is not a full list of models that can be converted to ONNX\* and to IR.

|Package Name|Supported Models|
|:----|:----|
| [Torchvision Models](https://pytorch.org/docs/stable/torchvision/index.html) | alexnet, densenet121, densenet161, densenet169, densenet201, resnet101, resnet152, resnet18, resnet34, resnet50, vgg11, vgg13, vgg16, vgg19 |
| [Pretrained Models](https://github.com/Cadene/pretrained-models.pytorch) | alexnet, fbresnet152, resnet101, resnet152, resnet18, resnet34, resnet152, resnet18, resnet34, resnet50, resnext101_32x4d, resnext101_64x4d, vgg11 |

**Other supported topologies**

* [ESPNet Models](https://github.com/sacmehta/ESPNet/tree/master/pretrained)
* [MobileNetV3](https://github.com/d-li14/mobilenetv3.pytorch)
* F3Net topology can be converted using [Convert PyTorch\* F3Net to the IR](pytorch_specific/Convert_F3Net.md) instruction.
* QuartzNet topologies from [NeMo project](https://github.com/NVIDIA/NeMo) can be converted using [Convert PyTorch\* QuartzNet to the IR](pytorch_specific/Convert_QuartzNet.md) instruction.
* YOLACT topology can be converted using [Convert PyTorch\* YOLACT to the IR](pytorch_specific/Convert_YOLACT.md) instruction.
* [RCAN](https://github.com/yulunzhang/RCAN) topologies can be converted using [Convert PyTorch\* RCAN to the IR](pytorch_specific/Convert_RCAN.md) instruction.
* [BERT_NER](https://github.com/kamalkraj/BERT-NER) can be converted using [Convert PyTorch* BERT-NER to the IR](pytorch_specific/Convert_Bert_ner.md) instruction.

## Export PyTorch\* Model to ONNX\* Format <a name="export-to-onnx"></a>

PyTorch models are defined in a Python\* code, to export such models use `torch.onnx.export()` method.
Only the basics will be covered here, the step to export to ONNX\* is crucial but it is covered by PyTorch\* framework.
For more information, please refer to [PyTorch\* documentation](https://pytorch.org/docs/stable/onnx.html).

To export a PyTorch\* model you need to obtain the model as an instance of `torch.nn.Module` class and call the `export` function.
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

* Not all PyTorch\* operations can be exported to ONNX\* opset 9 which is used by default, as of version 1.8.1.
It is recommended to export models to opset 11 or higher when export to default opset 9 is not working. In that case, use `opset_version`
option of the `torch.onnx.export`. For more information about ONNX* opset, refer to the [Operator Schemas](https://github.com/onnx/onnx/blob/master/docs/Operators.md).
