# ******************************************************************************
# Copyright 2018-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
# ## Prepare a list of models from the ONNX Model Zoo
#
# from pathlib import Path
# from operator import itemgetter
# import re
#
# MODELS_ROOT_DIR = '/path/to/onnx/models'
# zoo_models = []
# for path in Path(MODELS_ROOT_DIR).rglob('*.tar.gz'):
#     match = re.search('.*onnx\/models\/(.*\/model\/(.+)-(\d+)\.tar\.gz)', str(path))
#     url = match.group(1)
#     model_name = match.group(2)
#     opset = match.group(3)
#     zoo_models.append({'model_name': '{}_opset{}'.format(model_name.replace('-', '_'), opset), 'url': url})
#
# sorted(zoo_models, key=itemgetter('model_name'))
from tests.test_onnx.utils import OpenVinoOnnxBackend
from tests.test_onnx.utils.model_zoo_tester import ModelZooTestRunner
from tests import (BACKEND_NAME,
                   xfail_issue_36533,
                   xfail_issue_36534,
                   xfail_issue_35926,
                   xfail_issue_36535,
                   xfail_issue_36537,
                   xfail_issue_36538)

_GITHUB_MODELS_LTS = "https://media.githubusercontent.com/media/onnx/models/master/"

zoo_models = [
    {
        "model_name": "FasterRCNN_opset10",
        "url": _GITHUB_MODELS_LTS
        + "vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.tar.gz",
    },
    {
        "model_name": "MaskRCNN_opset10",
        "url": _GITHUB_MODELS_LTS + "vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.tar.gz",
    },
    {
        "model_name": "ResNet101_DUC_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/object_detection_segmentation/duc/model/ResNet101-DUC-7.tar.gz",
    },
    {
        "model_name": "arcfaceresnet100_opset8",
        "url": _GITHUB_MODELS_LTS + "vision/body_analysis/arcface/model/arcfaceresnet100-8.tar.gz",
    },
    {
        "model_name": "bertsquad_opset10",
        "url": _GITHUB_MODELS_LTS + "text/machine_comprehension/bert-squad/model/bertsquad-10.tar.gz",
    },
    {
        "model_name": "bertsquad_opset8",
        "url": _GITHUB_MODELS_LTS + "text/machine_comprehension/bert-squad/model/bertsquad-8.tar.gz",
    },
    {
        "model_name": "bidaf_opset9",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS
        + "text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.tar.gz",
    },
    {
        "model_name": "bvlcalexnet_opset3",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/alexnet/model/bvlcalexnet-3.tar.gz",
    },
    {
        "model_name": "bvlcalexnet_opset6",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/alexnet/model/bvlcalexnet-6.tar.gz",
    },
    {
        "model_name": "bvlcalexnet_opset7",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/alexnet/model/bvlcalexnet-7.tar.gz",
    },
    {
        "model_name": "bvlcalexnet_opset8",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/alexnet/model/bvlcalexnet-8.tar.gz",
    },
    {
        "model_name": "bvlcalexnet_opset9",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/alexnet/model/bvlcalexnet-9.tar.gz",
    },
    {
        "model_name": "caffenet_opset3",
        "url": _GITHUB_MODELS_LTS + "vision/classification/caffenet/model/caffenet-3.tar.gz",
    },
    {
        "model_name": "caffenet_opset6",
        "url": _GITHUB_MODELS_LTS + "vision/classification/caffenet/model/caffenet-6.tar.gz",
    },
    {
        "model_name": "caffenet_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/caffenet/model/caffenet-7.tar.gz",
    },
    {
        "model_name": "caffenet_opset8",
        "url": _GITHUB_MODELS_LTS + "vision/classification/caffenet/model/caffenet-8.tar.gz",
    },
    {
        "model_name": "caffenet_opset9",
        "url": _GITHUB_MODELS_LTS + "vision/classification/caffenet/model/caffenet-9.tar.gz",
    },
    {
        "model_name": "candy_opset8",
        "url": _GITHUB_MODELS_LTS + "vision/style_transfer/fast_neural_style/model/candy-8.tar.gz",
    },
    {
        "model_name": "candy_opset9",
        "url": _GITHUB_MODELS_LTS + "vision/style_transfer/fast_neural_style/model/candy-9.tar.gz",
    },
    {
        "model_name": "densenet_opset3",
        "atol": 1e-07,
        "rtol": 0.002,
        "url": _GITHUB_MODELS_LTS + "vision/classification/densenet-121/model/densenet-3.tar.gz",
    },
    {
        "model_name": "densenet_opset6",
        "atol": 1e-07,
        "rtol": 0.002,
        "url": _GITHUB_MODELS_LTS + "vision/classification/densenet-121/model/densenet-6.tar.gz",
    },
    {
        "model_name": "densenet_opset7",
        "atol": 1e-07,
        "rtol": 0.002,
        "url": _GITHUB_MODELS_LTS + "vision/classification/densenet-121/model/densenet-7.tar.gz",
    },
    {
        "model_name": "densenet_opset8",
        "atol": 1e-07,
        "rtol": 0.002,
        "url": _GITHUB_MODELS_LTS + "vision/classification/densenet-121/model/densenet-8.tar.gz",
    },
    {
        "model_name": "densenet_opset9",
        "atol": 1e-07,
        "rtol": 0.002,
        "url": _GITHUB_MODELS_LTS + "vision/classification/densenet-121/model/densenet-9.tar.gz",
    },
    {
        "model_name": "emotion_ferplus_opset2",
        "url": _GITHUB_MODELS_LTS + "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-2.tar.gz",
    },
    {
        "model_name": "emotion_ferplus_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-7.tar.gz",
    },
    {
        "model_name": "emotion_ferplus_opset8",
        "url": _GITHUB_MODELS_LTS + "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.tar.gz",
    },
    {
        "model_name": "googlenet_opset3",
        "url": _GITHUB_MODELS_LTS
        + "vision/classification/inception_and_googlenet/googlenet/model/googlenet-3.tar.gz",
    },
    {
        "model_name": "googlenet_opset6",
        "url": _GITHUB_MODELS_LTS
        + "vision/classification/inception_and_googlenet/googlenet/model/googlenet-6.tar.gz",
    },
    {
        "model_name": "googlenet_opset7",
        "url": _GITHUB_MODELS_LTS
        + "vision/classification/inception_and_googlenet/googlenet/model/googlenet-7.tar.gz",
    },
    {
        "model_name": "googlenet_opset8",
        "url": _GITHUB_MODELS_LTS
        + "vision/classification/inception_and_googlenet/googlenet/model/googlenet-8.tar.gz",
    },
    {
        "model_name": "googlenet_opset9",
        "url": _GITHUB_MODELS_LTS
        + "vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.tar.gz",
    },
    {
        "model_name": "gpt2_opset10",
        "url": _GITHUB_MODELS_LTS + "text/machine_comprehension/gpt-2/model/gpt2-10.tar.gz",
    },
    {
        "model_name": "inception_v1_opset3",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS
        + "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-3.tar.gz",
    },
    {
        "model_name": "inception_v1_opset6",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS
        + "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-6.tar.gz",
    },
    {
        "model_name": "inception_v1_opset7",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS
        + "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-7.tar.gz",
    },
    {
        "model_name": "inception_v1_opset8",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS
        + "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-8.tar.gz",
    },
    {
        "model_name": "inception_v1_opset9",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS
        + "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-9.tar.gz",
    },
    {
        "model_name": "inception_v2_opset3",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS
        + "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-3.tar.gz",
    },
    {
        "model_name": "inception_v2_opset6",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS
        + "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-6.tar.gz",
    },
    {
        "model_name": "inception_v2_opset7",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS
        + "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-7.tar.gz",
    },
    {
        "model_name": "inception_v2_opset8",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS
        + "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-8.tar.gz",
    },
    {
        "model_name": "inception_v2_opset9",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS
        + "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.tar.gz",
    },
    {
        "model_name": "mnist_opset1",
        "url": _GITHUB_MODELS_LTS + "vision/classification/mnist/model/mnist-1.tar.gz",
    },
    {
        "model_name": "mnist_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/mnist/model/mnist-7.tar.gz",
    },
    {
        "model_name": "mnist_opset8",
        "url": _GITHUB_MODELS_LTS + "vision/classification/mnist/model/mnist-8.tar.gz",
    },
    {
        "model_name": "mobilenetv2_opset7",
        "atol": 1e-07,
        "rtol": 0.002,
        "url": _GITHUB_MODELS_LTS + "vision/classification/mobilenet/model/mobilenetv2-7.tar.gz",
    },
    {
        "model_name": "mosaic_opset8",
        "url": _GITHUB_MODELS_LTS + "vision/style_transfer/fast_neural_style/model/mosaic-8.tar.gz",
    },
    {
        "model_name": "mosaic_opset9",
        "url": _GITHUB_MODELS_LTS + "vision/style_transfer/fast_neural_style/model/mosaic-9.tar.gz",
    },
    {
        "model_name": "pointilism_opset8",
        "url": _GITHUB_MODELS_LTS + "vision/style_transfer/fast_neural_style/model/pointilism-8.tar.gz",
    },
    {
        "model_name": "pointilism_opset9",
        "url": _GITHUB_MODELS_LTS + "vision/style_transfer/fast_neural_style/model/pointilism-9.tar.gz",
    },
    {
        "model_name": "rain_princess_opset8",
        "url": _GITHUB_MODELS_LTS + "vision/style_transfer/fast_neural_style/model/rain-princess-8.tar.gz",
    },
    {
        "model_name": "rain_princess_opset9",
        "url": _GITHUB_MODELS_LTS + "vision/style_transfer/fast_neural_style/model/rain-princess-9.tar.gz",
    },
    {
        "model_name": "rcnn_ilsvrc13_opset3",
        "url": _GITHUB_MODELS_LTS + "vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-3.tar.gz",
    },
    {
        "model_name": "rcnn_ilsvrc13_opset6",
        "url": _GITHUB_MODELS_LTS + "vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-6.tar.gz",
    },
    {
        "model_name": "rcnn_ilsvrc13_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-7.tar.gz",
    },
    {
        "model_name": "rcnn_ilsvrc13_opset8",
        "url": _GITHUB_MODELS_LTS + "vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-8.tar.gz",
    },
    {
        "model_name": "rcnn_ilsvrc13_opset9",
        "url": _GITHUB_MODELS_LTS + "vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-9.tar.gz",
    },
    {
        "model_name": "resnet101_v1_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/resnet/model/resnet101-v1-7.tar.gz",
    },
    {
        "model_name": "resnet101_v2_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/resnet/model/resnet101-v2-7.tar.gz",
    },
    {
        "model_name": "resnet152_v1_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/resnet/model/resnet152-v1-7.tar.gz",
    },
    {
        "model_name": "resnet152_v2_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/resnet/model/resnet152-v2-7.tar.gz",
    },
    {
        "model_name": "resnet18_v1_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/resnet/model/resnet18-v1-7.tar.gz",
    },
    {
        "model_name": "resnet18_v2_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/resnet/model/resnet18-v2-7.tar.gz",
    },
    {
        "model_name": "resnet34_v1_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/resnet/model/resnet34-v1-7.tar.gz",
    },
    {
        "model_name": "resnet34_v2_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/resnet/model/resnet34-v2-7.tar.gz",
    },
    {
        "model_name": "resnet50_caffe2_v1_opset3",
        "url": _GITHUB_MODELS_LTS + "vision/classification/resnet/model/resnet50-caffe2-v1-3.tar.gz",
    },
    {
        "model_name": "resnet50_caffe2_v1_opset6",
        "url": _GITHUB_MODELS_LTS + "vision/classification/resnet/model/resnet50-caffe2-v1-6.tar.gz",
    },
    {
        "model_name": "resnet50_caffe2_v1_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/resnet/model/resnet50-caffe2-v1-7.tar.gz",
    },
    {
        "model_name": "resnet50_caffe2_v1_opset8",
        "url": _GITHUB_MODELS_LTS + "vision/classification/resnet/model/resnet50-caffe2-v1-8.tar.gz",
    },
    {
        "model_name": "resnet50_caffe2_v1_opset9",
        "url": _GITHUB_MODELS_LTS + "vision/classification/resnet/model/resnet50-caffe2-v1-9.tar.gz",
    },
    {
        "model_name": "resnet50_v1_opset7",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/resnet/model/resnet50-v1-7.tar.gz",
    },
    {
        "model_name": "resnet50_v2_opset7",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/resnet/model/resnet50-v2-7.tar.gz",
    },
    {
        "model_name": "shufflenet_opset3",
        "url": _GITHUB_MODELS_LTS + "vision/classification/shufflenet/model/shufflenet-3.tar.gz",
    },
    {
        "model_name": "shufflenet_opset6",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/shufflenet/model/shufflenet-6.tar.gz",
    },
    {
        "model_name": "shufflenet_opset7",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/shufflenet/model/shufflenet-7.tar.gz",
    },
    {
        "model_name": "shufflenet_opset8",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/shufflenet/model/shufflenet-8.tar.gz",
    },
    {
        "model_name": "shufflenet_opset9",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/shufflenet/model/shufflenet-9.tar.gz",
    },
    {
        "model_name": "shufflenet_v2_opset10",
        "url": _GITHUB_MODELS_LTS + "vision/classification/shufflenet/model/shufflenet-v2-10.tar.gz",
    },
    {
        "model_name": "squeezenet1.0_opset3",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/squeezenet/model/squeezenet1.0-3.tar.gz",
    },
    {
        "model_name": "squeezenet1.0_opset6",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/squeezenet/model/squeezenet1.0-6.tar.gz",
    },
    {
        "model_name": "squeezenet1.0_opset7",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/squeezenet/model/squeezenet1.0-7.tar.gz",
    },
    {
        "model_name": "squeezenet1.0_opset8",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/squeezenet/model/squeezenet1.0-8.tar.gz",
    },
    {
        "model_name": "squeezenet1.0_opset9",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/squeezenet/model/squeezenet1.0-9.tar.gz",
    },
    {
        "model_name": "squeezenet1.1_opset7",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/classification/squeezenet/model/squeezenet1.1-7.tar.gz",
    },
    {
        "model_name": "ssd_opset10",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/object_detection_segmentation/ssd/model/ssd-10.tar.gz",
    },
    {
        "model_name": "super_resolution_opset10",
        "url": _GITHUB_MODELS_LTS
        + "vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.tar.gz",
    },
    {
        "model_name": "tiny_yolov3_opset11",
        "url": _GITHUB_MODELS_LTS
        + "vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.tar.gz",
    },
    {
        "model_name": "tinyyolov2_opset1",
        "url": _GITHUB_MODELS_LTS
        + "vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-1.tar.gz",
    },
    {
        "model_name": "tinyyolov2_opset7",
        "url": _GITHUB_MODELS_LTS
        + "vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-7.tar.gz",
    },
    {
        "model_name": "tinyyolov2_opset8",
        "url": _GITHUB_MODELS_LTS
        + "vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.tar.gz",
    },
    {
        "model_name": "udnie_opset8",
        "url": _GITHUB_MODELS_LTS + "vision/style_transfer/fast_neural_style/model/udnie-8.tar.gz",
    },
    {
        "model_name": "udnie_opset9",
        "url": _GITHUB_MODELS_LTS + "vision/style_transfer/fast_neural_style/model/udnie-9.tar.gz",
    },
    {
        "model_name": "vgg16_bn_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/vgg/model/vgg16-bn-7.tar.gz",
    },
    {
        "model_name": "vgg16_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/vgg/model/vgg16-7.tar.gz",
    },
    {
        "model_name": "vgg19_bn_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/vgg/model/vgg19-bn-7.tar.gz",
    },
    {
        "model_name": "vgg19_caffe2_opset3",
        "url": _GITHUB_MODELS_LTS + "vision/classification/vgg/model/vgg19-caffe2-3.tar.gz",
    },
    {
        "model_name": "vgg19_caffe2_opset6",
        "url": _GITHUB_MODELS_LTS + "vision/classification/vgg/model/vgg19-caffe2-6.tar.gz",
    },
    {
        "model_name": "vgg19_caffe2_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/vgg/model/vgg19-caffe2-7.tar.gz",
    },
    {
        "model_name": "vgg19_caffe2_opset8",
        "url": _GITHUB_MODELS_LTS + "vision/classification/vgg/model/vgg19-caffe2-8.tar.gz",
    },
    {
        "model_name": "vgg19_caffe2_opset9",
        "url": _GITHUB_MODELS_LTS + "vision/classification/vgg/model/vgg19-caffe2-9.tar.gz",
    },
    {
        "model_name": "vgg19_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/vgg/model/vgg19-7.tar.gz",
    },
    {
        "model_name": "yolov3_opset10",
        "atol": 1e-07,
        "rtol": 0.001,
        "url": _GITHUB_MODELS_LTS + "vision/object_detection_segmentation/yolov3/model/yolov3-10.tar.gz",
    },
    {
        "model_name": "zfnet512_opset3",
        "url": _GITHUB_MODELS_LTS + "vision/classification/zfnet-512/model/zfnet512-3.tar.gz",
    },
    {
        "model_name": "zfnet512_opset6",
        "url": _GITHUB_MODELS_LTS + "vision/classification/zfnet-512/model/zfnet512-6.tar.gz",
    },
    {
        "model_name": "zfnet512_opset7",
        "url": _GITHUB_MODELS_LTS + "vision/classification/zfnet-512/model/zfnet512-7.tar.gz",
    },
    {
        "model_name": "zfnet512_opset8",
        "url": _GITHUB_MODELS_LTS + "vision/classification/zfnet-512/model/zfnet512-8.tar.gz",
    },
    {
        "model_name": "zfnet512_opset9",
        "url": _GITHUB_MODELS_LTS + "vision/classification/zfnet-512/model/zfnet512-9.tar.gz",
    },
]

# Set backend device name to be used instead of hardcoded by ONNX BackendTest class ones.
OpenVinoOnnxBackend.backend_name = BACKEND_NAME

# import all test cases at global scope to make them visible to pytest
backend_test = ModelZooTestRunner(OpenVinoOnnxBackend, zoo_models, __name__)
test_cases = backend_test.test_cases["OnnxBackendZooModelTest"]

test_cases_list = [
    test_cases.test_udnie_opset8_cpu,
    test_cases.test_udnie_opset8_cpu,
    test_cases.test_udnie_opset9_cpu,
    test_cases.test_mosaic_opset8_cpu,
    test_cases.test_vgg16_opset7_cpu,
    test_cases.test_pointilism_opset9_cpu,
    test_cases.test_vgg19_bn_opset7_cpu,
    test_cases.test_candy_opset9_cpu,
    test_cases.test_rain_princess_opset8_cpu,
    test_cases.test_mosaic_opset9_cpu,
    test_cases.test_pointilism_opset8_cpu,
    test_cases.test_rain_princess_opset9_cpu,
    test_cases.test_ssd_opset10_cpu,
    test_cases.test_resnet152_v2_opset7_cpu,
    test_cases.test_resnet18_v1_opset7_cpu,
    test_cases.test_resnet18_v2_opset7_cpu,
    test_cases.test_resnet34_v2_opset7_cpu,
    test_cases.test_resnet101_v2_opset7_cpu,
    test_cases.test_resnet101_v1_opset7_cpu,
    test_cases.test_ResNet101_DUC_opset7_cpu,
    test_cases.test_candy_opset8_cpu,
    test_cases.test_resnet152_v1_opset7_cpu
]

xfail_issue_36534(test_cases.test_FasterRCNN_opset10_cpu)
xfail_issue_36534(test_cases.test_MaskRCNN_opset10_cpu)

xfail_issue_35926(test_cases.test_bertsquad_opset8_cpu)
xfail_issue_35926(test_cases.test_bertsquad_opset10_cpu)

xfail_issue_35926(test_cases.test_gpt2_opset10_cpu)

xfail_issue_36535(test_cases.test_super_resolution_opset10_cpu)
xfail_issue_36535(test_cases.test_tinyyolov2_opset7_cpu)
xfail_issue_36535(test_cases.test_tinyyolov2_opset8_cpu)

xfail_issue_36537(test_cases.test_shufflenet_v2_opset10_cpu)
xfail_issue_36538(test_cases.test_yolov3_opset10_cpu)
xfail_issue_36538(test_cases.test_tiny_yolov3_opset11_cpu)

for test_case in test_cases_list:
    xfail_issue_36533(test_case)

del test_cases
globals().update(backend_test.enable_report().test_cases)
