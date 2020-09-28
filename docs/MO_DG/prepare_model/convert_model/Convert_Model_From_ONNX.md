# Converting a ONNX* Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX}

## Introduction to ONNX

[ONNX*](https://github.com/onnx/onnx) is a representation format for deep learning models. ONNX allows AI developers easily transfer models between different frameworks that helps to choose the best combination for them. Today, PyTorch\*, Caffe2\*, Apache MXNet\*, Microsoft Cognitive Toolkit\* and other tools are developing ONNX support.

## Supported Public ONNX Topologies
| Model Name | Path to <a href="https://github.com/onnx/models">Public Models</a> master branch|
|:----|:----|
| bert_large | [model archive](https://github.com/mlperf/inference/tree/master/v0.7/language/bert) |
| bvlc_alexnet | [model archive](https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_alexnet.tar.gz) |
| bvlc_googlenet | [model archive](https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_googlenet.tar.gz) |
| bvlc_reference_caffenet | [model archive](https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_reference_caffenet.tar.gz) |
| bvlc_reference_rcnn_ilsvrc13 | [model archive](https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_reference_rcnn_ilsvrc13.tar.gz) |
| inception_v1 | [model archive](https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v1.tar.gz) |
| inception_v2 | [model archive](https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v2.tar.gz) |
| resnet50 | [model archive](https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz) |
| squeezenet | [model archive](https://s3.amazonaws.com/download.onnx/models/opset_8/squeezenet.tar.gz) |
| densenet121 | [model archive](https://s3.amazonaws.com/download.onnx/models/opset_8/densenet121.tar.gz) |
| emotion_ferplus | [model archive](https://www.cntk.ai/OnnxModels/emotion_ferplus/opset_2/emotion_ferplus.tar.gz) |
| mnist | [model archive](https://www.cntk.ai/OnnxModels/mnist/opset_1/mnist.tar.gz) |
| shufflenet | [model archive](https://s3.amazonaws.com/download.onnx/models/opset_8/shufflenet.tar.gz) |
| VGG19 | [model archive](https://s3.amazonaws.com/download.onnx/models/opset_8/vgg19.tar.gz) |
| zfnet512 | [model archive](https://s3.amazonaws.com/download.onnx/models/opset_8/zfnet512.tar.gz) |
| GPT-2 | [model archive](https://github.com/onnx/models/blob/master/text/machine_comprehension/gpt-2/model/gpt2-10.tar.gz) |

Listed models are built with the operation set version 8 except the GPT-2 model. Models that are upgraded to higher operation set versions may not be supported.

## Supported Pytorch* Models via ONNX Conversion
Starting from the 2019R4 release, the OpenVINO™ toolkit officially supports public Pytorch* models (from `torchvision` 0.2.1 and `pretrainedmodels` 0.7.4 packages) via ONNX conversion.
The list of supported topologies is presented below:

|Package Name|Supported Models|
|:----|:----|
| [Torchvision Models](https://pytorch.org/docs/stable/torchvision/index.html) | alexnet, densenet121, densenet161, densenet169, densenet201, resnet101, resnet152, resnet18, resnet34, resnet50, vgg11, vgg13, vgg16, vgg19 |
| [Pretrained Models](https://github.com/Cadene/pretrained-models.pytorch) | alexnet, fbresnet152, resnet101, resnet152, resnet18, resnet34, resnet152, resnet18, resnet34, resnet50, resnext101_32x4d, resnext101_64x4d, vgg11 |
| [ESPNet Models](https://github.com/sacmehta/ESPNet/tree/master/pretrained) | |

## Supported PaddlePaddle* Models via ONNX Conversion
Starting from the R5 release, the OpenVINO™ toolkit officially supports public PaddlePaddle* models via ONNX conversion.
The list of supported topologies downloadable from PaddleHub is presented below:

| Model Name | Command to download the model from PaddleHub |
|:----|:----|
| [MobileNetV2](https://www.paddlepaddle.org.cn/hubdetail?name=mobilenet_v2_imagenet) | `hub install mobilenet_v2_imagenet==1.0.1` |
| [ResNet18](https://www.paddlepaddle.org.cn/hubdetail?name=resnet_v2_18_imagenet) | `hub install resnet_v2_18_imagenet==1.0.0` |
| [ResNet34](https://www.paddlepaddle.org.cn/hubdetail?name=resnet_v2_34_imagenet) | `hub install resnet_v2_34_imagenet==1.0.0` |
| [ResNet50](https://www.paddlepaddle.org.cn/hubdetail?name=resnet_v2_50_imagenet) | `hub install resnet_v2_50_imagenet==1.0.1` |
| [ResNet101](https://www.paddlepaddle.org.cn/hubdetail?name=resnet_v2_101_imagenet) | `hub install resnet_v2_101_imagenet==1.0.1` |
| [ResNet152](https://www.paddlepaddle.org.cn/hubdetail?name=resnet_v2_152_imagenet) | `hub install resnet_v2_152_imagenet==1.0.1` |
> **NOTE**: To convert a model downloaded from PaddleHub use [paddle2onnx](https://github.com/PaddlePaddle/paddle2onnx) converter.

The list of supported topologies from the [models v1.5](https://github.com/PaddlePaddle/models/tree/release/1.5) package:
* [MobileNetV1](https://github.com/PaddlePaddle/models/blob/release/1.5/PaddleCV/image_classification/models/mobilenet.py)
* [MobileNetV2](https://github.com/PaddlePaddle/models/blob/release/1.5/PaddleCV/image_classification/models/mobilenet_v2.py)
* [ResNet](https://github.com/PaddlePaddle/models/blob/release/1.5/PaddleCV/image_classification/models/resnet.py)
* [ResNet_vc](https://github.com/PaddlePaddle/models/blob/release/1.5/PaddleCV/image_classification/models/resnet_vc.py)
* [ResNet_vd](https://github.com/PaddlePaddle/models/blob/release/1.5/PaddleCV/image_classification/models/resnet_vd.py)
* [ResNeXt](https://github.com/PaddlePaddle/models/blob/release/1.5/PaddleCV/image_classification/models/resnext.py)
* [ResNeXt_vd](https://github.com/PaddlePaddle/models/blob/release/1.5/PaddleCV/image_classification/models/resnext_vd.py)

> **NOTE**: To convert these topologies one should first serialize the model by calling `paddle.fluid.io.save_inference_model`
 ([description](https://www.paddlepaddle.org.cn/documentation/docs/en/1.3/api/io.html#save-inference-model)) command and
  after that use [paddle2onnx](https://github.com/PaddlePaddle/paddle2onnx) converter.

## Convert an ONNX* Model <a name="Convert_From_ONNX"></a>
The Model Optimizer process assumes you have an ONNX model that was directly downloaded from a public repository or converted from any framework that supports exporting to the ONNX format.

To convert an ONNX\* model:

1. Go to the `<INSTALL_DIR>/deployment_tools/model_optimizer` directory.
2. Use the `mo.py` script to simply convert a model with the path to the input model `.nnet` file:
```sh
python3 mo.py --input_model <INPUT_MODEL>.onnx
```

There are no ONNX\* specific parameters, so only [framework-agnostic parameters](Converting_Model_General.md) are available to convert your model.

## Supported ONNX\* Layers
Refer to [Supported Framework Layers](../Supported_Frameworks_Layers.md) for the list of supported standard layers.
