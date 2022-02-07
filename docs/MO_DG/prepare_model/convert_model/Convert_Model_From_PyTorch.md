# Converting a PyTorch* Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_F3Net
   openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_QuartzNet
   openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_RNNT
   openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_YOLACT
   openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_Bert_ner
   openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_RCAN

@endsphinxdirective

## Supported Topologies

Here is the list of models that are tested and guaranteed to be supported. However, you can also use these instructions to convert PyTorch\* models that are not presented in the list.

* [Torchvision Models](https://pytorch.org/docs/stable/torchvision/index.html):  alexnet, densenet121, densenet161,
  densenet169, densenet201, resnet101, resnet152, resnet18, resnet34, resnet50, vgg11, vgg13, vgg16, vgg19.
  The models can be converted using [regular instructions](#typical-pytorch).
* [Cadene Pretrained Models](https://github.com/Cadene/pretrained-models.pytorch): alexnet, fbresnet152, resnet101,
  resnet152, resnet18, resnet34, resnet152, resnet18, resnet34, resnet50, resnext101_32x4d, resnext101_64x4d, vgg11.
  The models can be converted using [regular instructions](#typical-pytorch).
* [ESPNet Models](https://github.com/sacmehta/ESPNet/tree/master/pretrained) can be converted using [regular instructions](#typical-pytorch).
* [MobileNetV3](https://github.com/d-li14/mobilenetv3.pytorch) can be converted using [regular instructions](#typical-pytorch).
* [iSeeBetter](https://github.com/amanchadha/iSeeBetter) can be converted using [regular instructions](#typical-pytorch).
  Please refer to [`iSeeBetterTest.py`](https://github.com/amanchadha/iSeeBetter/blob/master/iSeeBetterTest.py) script for code to initialize the model.
* F3Net topology can be converted using steps described in [Convert PyTorch\* F3Net to the IR](pytorch_specific/Convert_F3Net.md)
  instruction which is used instead of steps 2 and 3 of [regular instructions](#typical-pytorch).
* QuartzNet topologies from [NeMo project](https://github.com/NVIDIA/NeMo) can be converted using steps described in
  [Convert PyTorch\* QuartzNet to the IR](pytorch_specific/Convert_QuartzNet.md) instruction which is used instead of
  steps 2 and 3 of [regular instructions](#typical-pytorch).
* YOLACT topology can be converted using steps described in [Convert PyTorch\* YOLACT to the IR](pytorch_specific/Convert_YOLACT.md)
  instruction which is used instead of steps 2 and 3 of [regular instructions](#typical-pytorch).
* [RCAN](https://github.com/yulunzhang/RCAN) topology can be converted using steps described in [Convert PyTorch\* RCAN to the IR](pytorch_specific/Convert_RCAN.md)
  instruction which is used instead of steps 2 and 3 of [regular instructions](#typical-pytorch).
* [BERT_NER](https://github.com/kamalkraj/BERT-NER) topology can be converted using steps described in [Convert PyTorch* BERT-NER to the IR](pytorch_specific/Convert_Bert_ner.md)
  instruction which is used instead of steps 2 and 3 of [regular instructions](#typical-pytorch).

## Typical steps to convert PyTorch\* model <a name="typical-pytorch"></a>

PyTorch* framework is supported through export to ONNX\* format. A summary of the steps for optimizing and deploying a model that was trained with the PyTorch\* framework:

1. [Configure the Model Optimizer](../../Deep_Learning_Model_Optimizer_DevGuide.md) for ONNX\*.
2. [Export PyTorch model to ONNX\*](#export-to-onnx).
3. [Convert an ONNX\* model](Convert_Model_From_ONNX.md) to produce an optimized [Intermediate Representation (IR)](../../IR_and_opsets.md) of the model based on the trained network topology, weights, and biases values.
4. Test the model in the Intermediate Representation format using the [Inference Engine](../../../OV_Runtime_UG/Deep_Learning_Inference_Engine_DevGuide.md) in the target environment via provided [sample applications](../../../OV_Runtime_UG/Samples_Overview.md).
5. [Integrate](../../../OV_Runtime_UG/Samples_Overview.md) the Inference Engine in your application to deploy the model in the target environment.

## Export PyTorch\* Model to ONNX\* Format <a name="export-to-onnx"></a>

PyTorch models are defined in a Python\* code, to export such models use `torch.onnx.export()` method. Usually code to
evaluate or test the model is provided with the model code and can be used to initialize and export model.
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
