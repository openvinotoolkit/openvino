# Supported PyTorch Topologies {#openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_supported_topologies}

Here is the list of models that are tested and guaranteed to be supported. However, you can also use these instructions to convert PyTorch\* models that are not presented in the list.

* [Torchvision Models](https://pytorch.org/docs/stable/torchvision/index.html):  alexnet, densenet121, densenet161,
  densenet169, densenet201, resnet101, resnet152, resnet18, resnet34, resnet50, vgg11, vgg13, vgg16, vgg19.
  The models can be converted using the [regular instructions](../Convert_Model_From_PyTorch.md).
* [Cadene Pretrained Models](https://github.com/Cadene/pretrained-models.pytorch): alexnet, fbresnet152, resnet101,
  resnet152, resnet18, resnet34, resnet152, resnet18, resnet34, resnet50, resnext101_32x4d, resnext101_64x4d, vgg11.
  The models can be converted using [regular instructions](../Convert_Model_From_PyTorch.md).
* [ESPNet Models](https://github.com/sacmehta/ESPNet/tree/master/pretrained) can be converted using [regular instructions](../Convert_Model_From_PyTorch.md).
* [MobileNetV3](https://github.com/d-li14/mobilenetv3.pytorch) can be converted using [regular instructions](../Convert_Model_From_PyTorch.md).
* [iSeeBetter](https://github.com/amanchadha/iSeeBetter) can be converted using [regular instructions](../Convert_Model_From_PyTorch.md).
  Please refer to [`iSeeBetterTest.py`](https://github.com/amanchadha/iSeeBetter/blob/master/iSeeBetterTest.py) script for code to initialize the model.
* F3Net topology can be converted using steps described in [Convert PyTorch\* F3Net to the IR](Convert_F3Net.md)
  instruction which is used instead of steps 2 and 3 of [regular instructions](../Convert_Model_From_PyTorch.md).
* QuartzNet topologies from [NeMo project](https://github.com/NVIDIA/NeMo) can be converted using steps described in
  [Convert PyTorch\* QuartzNet to the IR](Convert_QuartzNet.md) instruction which is used instead of
  steps 2 and 3 of [regular instructions](../Convert_Model_From_PyTorch.md).
* YOLACT topology can be converted using steps described in [Convert PyTorch\* YOLACT to the IR](Convert_YOLACT.md)
  instruction which is used instead of steps 2 and 3 of [regular instructions](../Convert_Model_From_PyTorch.md).
* [RCAN](https://github.com/yulunzhang/RCAN) topology can be converted using steps described in [Convert PyTorch\* RCAN to the IR](Convert_RCAN.md)
  instruction which is used instead of steps 2 and 3 of [regular instructions](../Convert_Model_From_PyTorch.md).
* [BERT_NER](https://github.com/kamalkraj/BERT-NER) topology can be converted using steps described in [Convert PyTorch* BERT-NER to the IR](Convert_Bert_ner.md)
  instruction which is used instead of steps 2 and 3 of [regular instructions](../Convert_Model_From_PyTorch.md).