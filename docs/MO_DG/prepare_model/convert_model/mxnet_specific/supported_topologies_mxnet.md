# Supported MXNet Topologies {#openvino_docs_MO_DG_prepare_model_convert_model_mxnet_specific_supported_topologies}

## Supported Topologies

> **NOTE**: SSD models from the table require converting to the deploy mode. For details, see the [Conversion Instructions](https://github.com/zhreshold/mxnet-ssd/#convert-model-to-deploy-mode) in the GitHub MXNet-SSD repository.

| Model Name| Model File |
| ------------- |:-------------:|
|VGG-16|	[Repo](https://github.com/dmlc/mxnet-model-gallery/tree/master), [Symbol](http://data.mxnet.io/models/imagenet/vgg/vgg16-symbol.json), [Params](http://data.mxnet.io/models/imagenet/vgg/vgg16-0000.params)|
|VGG-19|	[Repo](https://github.com/dmlc/mxnet-model-gallery/tree/master), [Symbol](http://data.mxnet.io/models/imagenet/vgg/vgg19-symbol.json), [Params](http://data.mxnet.io/models/imagenet/vgg/vgg19-0000.params)|
|ResNet-152 v1|	[Repo](https://github.com/dmlc/mxnet-model-gallery/tree/master), [Symbol](http://data.mxnet.io/models/imagenet/resnet/152-layers/resnet-152-symbol.json), [Params](http://data.mxnet.io/models/imagenet/resnet/152-layers/resnet-152-0000.params)|
|SqueezeNet_v1.1|	[Repo](https://github.com/dmlc/mxnet-model-gallery/tree/master), [Symbol](http://data.mxnet.io/models/imagenet/squeezenet/squeezenet_v1.1-symbol.json), [Params](http://data.mxnet.io/models/imagenet/squeezenet/squeezenet_v1.1-0000.params)|
|Inception BN|	[Repo](https://github.com/dmlc/mxnet-model-gallery/tree/master), [Symbol](http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN-symbol.json), [Params](http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN-0126.params)|
|CaffeNet|	[Repo](https://github.com/dmlc/mxnet-model-gallery/tree/master), [Symbol](http://data.mxnet.io/mxnet/models/imagenet/caffenet/caffenet-symbol.json), [Params](http://data.mxnet.io/models/imagenet/caffenet/caffenet-0000.params)|
|DenseNet-121|	[Repo](https://github.com/miraclewkf/DenseNet), [Symbol](https://raw.githubusercontent.com/miraclewkf/DenseNet/master/model/densenet-121-symbol.json), [Params](https://drive.google.com/file/d/0ByXcv9gLjrVcb3NGb1JPa3ZFQUk/view?usp=drive_web)|
|DenseNet-161|	[Repo](https://github.com/miraclewkf/DenseNet), [Symbol](https://raw.githubusercontent.com/miraclewkf/DenseNet/master/model/densenet-161-symbol.json), [Params](https://drive.google.com/file/d/0ByXcv9gLjrVcS0FwZ082SEtiUjQ/view)|
|DenseNet-169| 	[Repo](https://github.com/miraclewkf/DenseNet), [Symbol](https://raw.githubusercontent.com/miraclewkf/DenseNet/master/model/densenet-169-symbol.json), [Params](https://drive.google.com/file/d/0ByXcv9gLjrVcOWZJejlMOWZvZmc/view)|
|DenseNet-201|	[Repo](https://github.com/miraclewkf/DenseNet), [Symbol](https://raw.githubusercontent.com/miraclewkf/DenseNet/master/model/densenet-201-symbol.json), [Params](https://drive.google.com/file/d/0ByXcv9gLjrVcUjF4MDBwZ3FQbkU/view)|
|MobileNet|	[Repo](https://github.com/KeyKy/mobilenet-mxnet), [Symbol](https://github.com/KeyKy/mobilenet-mxnet/blob/master/mobilenet.py), [Params](https://github.com/KeyKy/mobilenet-mxnet/blob/master/mobilenet-0000.params)|
|SSD-ResNet-50|	[Repo](https://github.com/zhreshold/mxnet-ssd), [Symbol + Params](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.6/resnet50_ssd_512_voc0712_trainval.zip)|
|SSD-VGG-16-300|	[Repo](https://github.com/zhreshold/mxnet-ssd), [Symbol + Params](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.5-beta/vgg16_ssd_300_voc0712_trainval.zip)|
|SSD-Inception v3|	[Repo](https://github.com/zhreshold/mxnet-ssd), [Symbol + Params](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.7-alpha/ssd_inceptionv3_512_voc0712trainval.zip)|
|FCN8 (Semantic Segmentation)|	[Repo](https://github.com/apache/incubator-mxnet/tree/master/example/fcn-xs), [Symbol](https://www.dropbox.com/sh/578n5cxej7ofd6m/AAA9SFCBN8R_uL2CnAd3WQ5ia/FCN8s_VGG16-symbol.json?dl=0), [Params](https://www.dropbox.com/sh/578n5cxej7ofd6m/AABHWZHCtA2P6iR6LUflkxb_a/FCN8s_VGG16-0019-cpu.params?dl=0)|
|MTCNN part 1 (Face Detection)| [Repo](https://github.com/pangyupo/mxnet_mtcnn_face_detection), [Symbol](https://github.com/pangyupo/mxnet_mtcnn_face_detection/blob/master/model/det1-symbol.json), [Params](https://github.com/pangyupo/mxnet_mtcnn_face_detection/blob/master/model/det1-0001.params)|
|MTCNN part 2 (Face Detection)| [Repo](https://github.com/pangyupo/mxnet_mtcnn_face_detection), [Symbol](https://github.com/pangyupo/mxnet_mtcnn_face_detection/blob/master/model/det2-symbol.json), [Params](https://github.com/pangyupo/mxnet_mtcnn_face_detection/blob/master/model/det2-0001.params)|
|MTCNN part 3 (Face Detection)| [Repo](https://github.com/pangyupo/mxnet_mtcnn_face_detection), [Symbol](https://github.com/pangyupo/mxnet_mtcnn_face_detection/blob/master/model/det3-symbol.json), [Params](https://github.com/pangyupo/mxnet_mtcnn_face_detection/blob/master/model/det3-0001.params)|
|MTCNN part 4 (Face Detection)| [Repo](https://github.com/pangyupo/mxnet_mtcnn_face_detection), [Symbol](https://github.com/pangyupo/mxnet_mtcnn_face_detection/blob/master/model/det4-symbol.json), [Params](https://github.com/pangyupo/mxnet_mtcnn_face_detection/blob/master/model/det4-0001.params)|
|Lightened_moon| [Repo](https://github.com/tornadomeet/mxnet-face/tree/master/model/lightened_moon), [Symbol](https://github.com/tornadomeet/mxnet-face/blob/master/model/lightened_moon/lightened_moon_fuse-symbol.json), [Params](https://github.com/tornadomeet/mxnet-face/blob/master/model/lightened_moon/lightened_moon_fuse-0082.params)|
|RNN-Transducer| [Repo](https://github.com/HawkAaron/mxnet-transducer) |
|word_lm| [Repo](https://github.com/apache/incubator-mxnet/tree/master/example/rnn/word_lm) |

**Other supported topologies**

* [GluonCV SSD and YOLO-v3 models](https://gluon-cv.mxnet.io/model_zoo/detection.html) can be converted using the following [instructions](Convert_GluonCV_Models.md).
* [Style transfer model](https://github.com/zhaw/neural_style) can be converted using the following [instructions](Convert_Style_Transfer_From_MXNet.md).