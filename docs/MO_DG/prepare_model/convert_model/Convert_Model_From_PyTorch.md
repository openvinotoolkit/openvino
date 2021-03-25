# Converting a PyTorch* Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch}

PyTorch* framework is supported through export to ONNX\* format. A summary of the steps for optimizing and deploying a model that was trained with the PyTorch\* framework:

1. [Configure the Model Optimizer](../Config_Model_Optimizer.md) for ONNX\*.
2. [Export PyTorch model to ONNX\*](#export-to-onnx) if your model is not already frozen or skip this step and use the [instruction](#loading-nonfrozen-models) to a convert a non-frozen model.
3. [Convert an ONNX\* model](Convert_Model_From_ONNX.md) to produce an optimized [Intermediate Representation (IR)](../../IR_and_opsets.md) of the model based on the trained network topology, weights, and biases values.
4. Test the model in the Intermediate Representation format using the [Inference Engine](../../../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) in the target environment via provided [sample applications](../../../IE_DG/Samples_Overview.md).
5. [Integrate](../../../IE_DG/Samples_Overview.md) the Inference Engine in your application to deploy the model in the target environment.

## Supported Topologies

|Package Name|Supported Models|
|:----|:----|
| [Torchvision Models](https://pytorch.org/docs/stable/torchvision/index.html) | alexnet, densenet121, densenet161, densenet169, densenet201, resnet101, resnet152, resnet18, resnet34, resnet50, vgg11, vgg13, vgg16, vgg19 |
| [Pretrained Models](https://github.com/Cadene/pretrained-models.pytorch) | alexnet, fbresnet152, resnet101, resnet152, resnet18, resnet34, resnet152, resnet18, resnet34, resnet50, resnext101_32x4d, resnext101_64x4d, vgg11 |

**Other supported topologies**

* [ESPNet Models](https://github.com/sacmehta/ESPNet/tree/master/pretrained)
* [MobileNetV3](https://github.com/d-li14/mobilenetv3.pytorch)
* F3Net topology can be converted using [instruction](pytorch_specific/Convert_F3Net.md)
* QuartzNet topologies from [NeMo project](https://github.com/NVIDIA/NeMo) can be converted using [instruction](pytorch_specific/Convert_QuartzNet.md)
* YOLACT topology can be converted using [instruction](pytorch_specific/Convert_YOLACT.md)

## Export PyTorch\* model to ONNX\* format <a name="export-to-onnx"></a>

PyTorch models are defined in a Python\* code, to export such models `torch.onnx.export()` method should be used.
We will cover only the basics here, for more information, please refer to [pytorch documentation](https://pytorch.org/docs/stable/onnx.html).

To export a PyTorch model you basically need to obtain the model as an instance of `torch.nn.Module` class and call the export function.
```python
import torch
import torchvision

# Obtain your model
model = SomeModel()
# Evaluate the model to switch some operations from training mode to inference
model.eval()
# Create dummy input for the model. It will be used to run the model inside export function. 
dummy_input = torch.randn(2, 3, 224, 224)
# Call the export function
torch.onnx.export(model, (dummy_input, ), 'model.onnx')
```

