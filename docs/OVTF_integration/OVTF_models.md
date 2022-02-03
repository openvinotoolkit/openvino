# Supported Models {#ovtf_models}


This document provides a list of all validated models that are supported by **OpenVINO™ integration with TensorFlow**. This list is continuously evolving as we optimize models and enable more operators.
The performance gain depends on various factors such as model architecture, Intel® Platform (e.g. Xeon® or CoreTM), Backend device (e.g. CPU, GPU or VPU), etc.

## TensorFlow-Slim Image Classification Library
| Model Name | Supported Devices |
|---|---|
| [Inception V3](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)  | CPU, iGPU, MYRIAD, VAD-M |
| [Inception_V4](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz)  | CPU, iGPU, MYRIAD, VAD-M |
| [Resnet V1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)  | CPU, iGPU, MYRIAD, VAD-M |
| [Resnet V2 152](http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [Resnet V2 50](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz)  | CPU, iGPU, MYRIAD, VAD-M |
| [VGG 16](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)              | CPU, iGPU, MYRIAD, VAD-M |
| [VGG 19](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz)              | CPU, iGPU, MYRIAD, VAD-M |
| [MobileNet_v1_1.0_224](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)| CPU, iGPU, MYRIAD, VAD-M |
| [MobileNet_v2_1.4_224](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz)| CPU, iGPU, MYRIAD, VAD-M |
| [CifarNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/cifarnet.py)| CPU, iGPU, MYRIAD, VAD-M |
| [LeNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/lenet.py)| CPU, iGPU, MYRIAD, VAD-M |

The links to the TensorFlow-Slim models include the pre-trained checkpoint files only. You should refer to TensorFlow-Slim models [instructions page](https://github.com/tensorflow/models/tree/master/research/slim) to run inference or freeze the models. (No pre-trained checkpoint files provided for CifarNet and LeNet.)

## TensorFlow Object Detection Model Zoo
| Model Name | Supported Devices |
|---|---|
| [faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [faster_rcnn_resnet50_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [faster_rcnn_resnet50_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [ssd_inception_v2](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [ssd_mobilenet_v1](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [ssd_mobilenet_v1_fpn](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [ssd_mobilenet_v2](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [ssd_resnet_50_fpn](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [ssdlite_mobilenet_v2](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [mask_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [mask_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |

Pre-trained frozen models are provided for these models.

## TensorFlow Keras Applications
| Model Name | Supported Devices |
|---|---|
| [DenseNet121](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet121)| CPU, iGPU, MYRIAD, VAD-M |
| [DenseNet169](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet169)| CPU, iGPU, MYRIAD, VAD-M |
| [DenseNet201](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet201)| CPU, iGPU, MYRIAD, VAD-M |
| [EfficientnetB0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)| CPU, iGPU, MYRIAD, VAD-M |
| [EfficientnetB1](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB1)| CPU, iGPU, MYRIAD, VAD-M |
| [EfficientnetB2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB2)| CPU, iGPU, MYRIAD, VAD-M |
| [EfficientnetB3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB3)| CPU, iGPU, MYRIAD, VAD-M |
| [EfficientnetB4](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB4)| CPU, iGPU, MYRIAD, VAD-M |
| [EfficientnetB5](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB5)| CPU, iGPU, MYRIAD, VAD-M |
| [EfficientnetB6](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB6)| CPU, iGPU, MYRIAD, VAD-M |
| [EfficientnetB7](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB7)| CPU, iGPU, MYRIAD, VAD-M |
| [InceptionV3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3)| CPU, iGPU, MYRIAD, VAD-M |
| [NASNetLarge](https://www.tensorflow.org/api_docs/python/tf/keras/applications/NASNetLarge)| CPU, iGPU, MYRIAD, VAD-M |
| [NASNetMobile](https://www.tensorflow.org/api_docs/python/tf/keras/applications/NASNetMobile)| CPU, iGPU, MYRIAD, VAD-M |
| [ResNet50v2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50V2)| CPU, iGPU, MYRIAD, VAD-M |

Please follow the instructions on [Keras Applications](https://keras.io/api/applications/) page for further information about using these pre-trained models.

## TensorFlow-Hub Models
| Model Name | Supported Devices |
|---|---|
| [aiy_vision_classifier_birds_V1](https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [aiy_vision_classifier_food_V1](https://tfhub.dev/google/aiy/vision/classifier/food_V1/1?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [albert_en_base](https://tfhub.dev/tensorflow/albert_en_base/3)| CPU, iGPU, MYRIAD |
| [albert_en_preprocess](https://tfhub.dev/tensorflow/albert_en_preprocess/3)| CPU, iGPU, MYRIAD |
| [albert_en_xxlarge](https://tfhub.dev/tensorflow/albert_en_xxlarge/3?tf-hub-format=compressed)| CPU, iGPU, MYRIAD |
| [bert_en_cased_L-12_H-768_A-12](https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2)| CPU, iGPU, MYRIAD |
| [bert_en_cased_L-24_H-1024_A-16](https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/1)| CPU, iGPU, MYRIAD |
| [bert_en_cased_preprocess](https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [bert_en_uncased_L-12_H-768_A-12](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/)| CPU, iGPU, MYRIAD |
| [bert_en_uncased_L-24_H-1024_A-16](https://tfhub.dev/tensorflow/brrt_en_uncased_L-24_H-1024_A-16/)| CPU, iGPU, MYRIAD |
| [bert_en_uncased_preprocess](https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3)| CPU, iGPU, MYRIAD |
| [bert_en_wwm_uncased_L-24_H-1024_A-16](https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/4)| CPU, iGPU, MYRIAD |
| [bert_multi_cased_preprocess](https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/2)| CPU, iGPU, MYRIAD |
| [bert_zh_L-12_H-768_A-12](https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/1)| CPU, iGPU, MYRIAD |
| [bert_zh_preprocess](https://tfhub.dev/tensorflow/bert_zh_preprocess/3)| CPU, iGPU, MYRIAD |
| [bit​/m-r101x1](https://tfhub.dev/google/bit/m-r101x1/1)| CPU, iGPU, MYRIAD |
| [bit​/m-r101x3](https://tfhub.dev/google/bit/m-r101x3/1)| CPU, iGPU, MYRIAD |
| [bit​/m-r50x1​/ilsvrc2012_classification](https://tfhub.dev/google/bit/m-r50x1/ilsvrc2012_classification/1)| CPU, iGPU, MYRIAD |
| [bit​/m-r50x1](https://tfhub.dev/google/bit/m-r50x1/1)| CPU, iGPU, MYRIAD |
| [bit​/m-r50x3](https://tfhub.dev/google/bit/m-r50x3/1)| CPU, iGPU, MYRIAD |
| [bit​/s-r50x1](https://tfhub.dev/google/bit/s-r50x1/1)| CPU, iGPU, MYRIAD |
| [centernet​/hourglass_512x512](https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1)| iGPU, MYRIAD |
| [centernet​/hourglass_512x512_kpts](https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1)| CPU, iGPU, MYRIAD |
| [centernet​/resnet50v1_fpn_512x512](https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1)| iGPU, MYRIAD |
| [centernet_hourglass_1024x1024](https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1)| CPU, iGPU, MYRIAD |
| [centernet_resnet101v1_fpn_512x512](https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1)| CPU, iGPU, MYRIAD |
| [centernet_resnet50v1_fpn_512x512_kpts](https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1)| CPU, iGPU, MYRIAD |
| [centernet_resnet50v2_512x512](https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1)| CPU, iGPU, MYRIAD |
| [cord-19_swivel-128d](https://tfhub.dev/tensorflow/cord-19/swivel-128d/3?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [cropnet​/classifier​/cassava_disease_V1](https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2)| CPU, iGPU, MYRIAD |
| [efficientdet​/d0](https://tfhub.dev/tensorflow/efficientdet/d0/1)| MYRIAD |
| [efficientdet​/d4](https://tfhub.dev/tensorflow/efficientdet/d4/1)| MYRIAD |
| [efficientdet​/d7](https://tfhub.dev/tensorflow/efficientdet/d7/1)|CPU, iGPU, MYRIAD |
| [efficientdet​/lite0​/detection](https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1)| CPU, MYRIAD |
| [efficientdet​/lite0​/feature-vector](https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1)| CPU, iGPU, MYRIAD |
| [efficientdet_d1 ](https://tfhub.dev/tensorflow/efficientdet/d1/1?tf-hub-format=compressed)  | iGPU, MYRIAD |
| [efficientdet_d2](https://tfhub.dev/tensorflow/efficientdet/d2/1?tf-hub-format=compressed)  | iGPU, MYRIAD |
| [efficientdet_d6](https://tfhub.dev/tensorflow/efficientdet/d6/1?tf-hub-format=compressed)  | CPU, MYRIAD |
| [efficientdet_lite1_detection](https://tfhub.dev/tensorflow/efficientdet/lite1/detection/1?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [efficientdet_lite1_feature-vector](https://tfhub.dev/tensorflow/efficientdet/lite1/feature-vector/1?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [efficientdet_lite2_feature-vector](https://tfhub.dev/tensorflow/efficientdet/lite2/feature-vector/1?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [efficientdet_lite3_detection](https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [efficientdet_lite3x_detection](https://tfhub.dev/tensorflow/efficientdet/lite3x/detection/1?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [efficientdet_lite4_detection](https://tfhub.dev/tensorflow/efficientdet/lite4/detection/2?tf-hub-format=compressed)  | CPU, MYRIAD |
| [efficientdet_lite4_feature-vector](https://tfhub.dev/tensorflow/efficientdet/lite4/feature-vector/2?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [efficientnet​/b0​/classification](https://tfhub.dev/google/efficientnet/b0/classification/1)| CPU, iGPU, MYRIAD |
| [efficientnet​/b0​/feature-vector](https://tfhub.dev/google/efficientnet/b0/feature-vector/1)| CPU, iGPU, MYRIAD |
| [efficientnet​/b3​/feature-vector](https://tfhub.dev/google/efficientnet/b3/feature-vector/1)| CPU, iGPU, MYRIAD |
| [efficientnet​/b4​/feature-vector](https://tfhub.dev/google/efficientnet/b4/feature-vector/1)| CPU, iGPU, MYRIAD |
| [efficientnet​/b7​/classification](https://tfhub.dev/google/efficientnet/b7/classification/1)| CPU, iGPU, MYRIAD |
| [efficientnet​/b7​/feature-vector](https://tfhub.dev/google/efficientnet/b7/feature-vector/1)| CPU, iGPU, MYRIAD |
| [efficientnet_b2_feature-vector](https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [electra_base](https://tfhub.dev/google/electra_base/2?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [electra_small](https://tfhub.dev/google/electra_small/2)| CPU, iGPU, MYRIAD |
| [elmo](https://tfhub.dev/google/elmo/3?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [esrgan-tf2](https://tfhub.dev/captain-pool/esrgan-tf2/1)| CPU, iGPU, MYRIAD |
| [experts​/bert​/wiki_books](https://tfhub.dev/google/experts/bert/wiki_books/2)| CPU, iGPU, MYRIAD |
| [experts_bert_pubmed](https://tfhub.dev/google/experts/bert/pubmed/2?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [faster_rcnn​/inception_resnet_v2_1024x1024](https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1)| CPU, iGPU, MYRIAD |
| [faster_rcnn​/inception_resnet_v2_640x640](https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1)| CPU, iGPU, MYRIAD |
| [faster_rcnn​/resnet50_v1_640x640](https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1)| CPU, iGPU, MYRIAD |
| [faster_rcnn_resnet101_v1_640x640](https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [imagenet​/inception_resnet_v2​/classification](https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/5)| CPU, iGPU, MYRIAD |
| [imagenet​/inception_resnet_v2​/feature_vector](https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5)| CPU, iGPU, MYRIAD |
| [imagenet​/inception_v1​/classification](https://tfhub.dev/google/imagenet/inception_v1/classification/5)| CPU, iGPU, MYRIAD |
| [imagenet​/inception_v1​/feature_vector](https://tfhub.dev/google/imagenet/inception_v1/feature_vector/5)| CPU, iGPU, MYRIAD |
| [imagenet​/inception_v2​/feature_vector](https://tfhub.dev/google/imagenet/inception_v2/feature_vector/5)| CPU, iGPU, MYRIAD |
| [imagenet​/inception_v3​/classification](https://tfhub.dev/google/imagenet/inception_v3/classification/5)| CPU, iGPU, MYRIAD |
| [imagenet​/inception_v3​/feature_vector](https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5)| CPU, iGPU, MYRIAD |
| [imagenet​/mobilenet_v1_025_224​/classification](https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/classification/5)| CPU, iGPU, MYRIAD |
| [imagenet​/mobilenet_v1_050_160​/classification](https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/5)| CPU, iGPU, MYRIAD |
| [imagenet​/mobilenet_v1_100_224​/classification](https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5)| CPU, iGPU, MYRIAD |
| [imagenet​/mobilenet_v1_100_224​/feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/5)| CPU, iGPU, MYRIAD |
| [imagenet​/mobilenet_v2_035_128​/feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/5)| CPU, iGPU, MYRIAD |
| [imagenet​/mobilenet_v2_035_224​/classification](https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/classification/5)| CPU, iGPU, MYRIAD |
| [imagenet​/mobilenet_v2_035_96​/classification](https://tfhub.dev/google/imagenet/mobilenet_v2_035_96/classification/5)| CPU, iGPU, MYRIAD |
| [imagenet​/mobilenet_v2_050_224​/classification](https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/classification/5)| CPU, iGPU, MYRIAD |
| [imagenet​/mobilenet_v2_075_224​/classification](https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/classification/5)| CPU, iGPU, MYRIAD |
| [imagenet​/mobilenet_v2_100_224​/classification](https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5)| CPU, iGPU, MYRIAD |
| [imagenet​/mobilenet_v2_100_224​/feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5)| CPU, iGPU, MYRIAD |
| [imagenet​/mobilenet_v2_130_224​/classification](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5)| CPU, iGPU, MYRIAD |
| [imagenet​/mobilenet_v2_140_224​/classification](https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5)| CPU, iGPU, MYRIAD |
| [imagenet​/mobilenet_v2_140_224​/feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/5)| CPU, iGPU, MYRIAD |
| [imagenet​/mobilenet_v3_small_100_224​/feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5)| CPU, iGPU, MYRIAD |
| [imagenet​/pnasnet_large​/feature_vector](https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/5)| CPU, iGPU, MYRIAD |
| [imagenet​/resnet_v1_50​/feature_vector](https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/5)| CPU, iGPU, MYRIAD |
| [imagenet​/resnet_v2_152​/feature_vector](https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5)| CPU, iGPU, MYRIAD |
| [imagenet​/resnet_v2_50​/classification](https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5)| CPU, iGPU, MYRIAD |
| [imagenet​/resnet_v2_50​/feature_vector](https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5)| CPU, iGPU, MYRIAD |
| [imagenet_efficientnet_v2_imagenet21k_xl_feature_vector](https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [imagenet_mobilenet_v2_100_96_feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/5?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [imagenet_mobilenet_v2_130_224_feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/5?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [imagenet_mobilenet_v3_large_100_224_classification](https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [imagenet_mobilenet_v3_large_100_224_feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [imagenet_mobilenet_v3_small_075_224_classification](https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/classification/5?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [imagenet_mobilenet_v3_small_100_224_classification](https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [imagenet_nasnet_large_classification](https://tfhub.dev/google/imagenet/nasnet_large/classification/5?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [imagenet_nasnet_large_feature_vector](https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/5?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [imagenet_nasnet_mobile_feature_vector](https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/5?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [imagenet_resnet_v1_50_classification](https://tfhub.dev/google/imagenet/resnet_v1_50/classification/5?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [imagenet_resnet_v2_101_classification](https://tfhub.dev/google/imagenet/resnet_v2_101/classification/5?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [imagenet_resnet_v2_152_classification](https://tfhub.dev/google/imagenet/resnet_v2_152/classification/5?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [inaturalist​/inception_v3​/feature_vector](https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5)| CPU, iGPU, MYRIAD |
| [LaBSE](https://tfhub.dev/google/LaBSE/2)| CPU, iGPU, MYRIAD |
| [magenta_arbitrary-image-stylization-v1-256](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [mask_rcnn​/inception_resnet_v2_1024x1024](https://hub.tensorflow.google.cn/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1)| CPU, iGPU, MYRIAD |
| [mask_rcnn_inception_resnet_v2_1024x1024](https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [movenet​/singlepose​/lightning](https://tfhub.dev/google/movenet/singlepose/lightning/3)| CPU, iGPU, MYRIAD |
| [movenet_multipose_lightning](https://tfhub.dev/google/movenet/multipose/lightning/1?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [movenet_singlepose_thunder](https://tfhub.dev/google/movenet/singlepose/thunder/4?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [MuRIL](https://tfhub.dev/google/MuRIL/1)| CPU, iGPU, MYRIAD |
| [MuRIL_preprocess](https://tfhub.dev/google/MuRIL_preprocess/1?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [nnlm-de-dim50](https://tfhub.dev/google/nnlm-de-dim50/2)| CPU, iGPU, MYRIAD |
| [nnlm-de-dim50-with-normalization](https://tfhub.dev/google/nnlm-de-dim50-with-normalization/2)| CPU, iGPU, MYRIAD |
| [nnlm-en-dim128](https://tfhub.dev/google/nnlm-en-dim128/2)| CPU, iGPU, MYRIAD |
| [nnlm-en-dim128-with-normalization](https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2)| CPU, iGPU, MYRIAD |
| [nnlm-en-dim50](https://tfhub.dev/google/nnlm-en-dim50/2)| CPU, iGPU, MYRIAD |
| [nnlm-en-dim50-with-normalization](https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2)| CPU, iGPU, MYRIAD |
| [nnlm-es-dim128](https://tfhub.dev/google/nnlm-es-dim128/2)| CPU, iGPU, MYRIAD |
| [nnlm-es-dim128-with-normalization](https://tfhub.dev/google/nnlm-es-dim128-with-normalization/2?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [nnlm-ja-dim128](https://tfhub.dev/google/nnlm-ja-dim128/2)| CPU, iGPU, MYRIAD |
| [nnlm-ja-dim128-with-normalization](https://tfhub.dev/google/nnlm-ja-dim128-with-normalization/2)| CPU, iGPU, MYRIAD |
| [nnlm-ja-dim50](https://tfhub.dev/google/nnlm-ja-dim50/2)| CPU, iGPU, MYRIAD |
| [nonsemantic-speech-benchmark​/trill](https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3)| CPU, iGPU, MYRIAD |
| [nonsemantic-speech-benchmark​/trill-distilled](https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3)| CPU, iGPU, MYRIAD |
| [resnet_50​/classification](https://tfhub.dev/tensorflow/resnet_50/classification/1)| CPU, iGPU, MYRIAD |
| [resnet_50​/feature_vector](https://tfhub.dev/tensorflow/resnet_50/feature_vector/1?tf-hub-format=compressed)| CPU, iGPU, MYRIAD |
| [retinanet_resnet50_v1_fpn_640x640](https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [small_bert​/bert_en_uncased_L-2_H-128_A-2](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2)| CPU, iGPU, MYRIAD |
| [small_bert​/bert_en_uncased_L-4_H-512_A-8](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2)| CPU, iGPU, MYRIAD |
| [small_bert_bert_en_uncased_L-12_H-128_A-2](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/2?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [small_bert_bert_en_uncased_L-12_H-768_A-12](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/2?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [small_bert_bert_en_uncased_L-2_H-256_A-4](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/2?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [small_bert_bert_en_uncased_L-4_H-128_A-2](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [small_bert_bert_en_uncased_L-4_H-256_A-4](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [small_bert_bert_en_uncased_L-8_H-512_A-8](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/2?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [spice](https://tfhub.dev/google/spice/2?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [ssd_mobilenet_v1_fpn_640x640](https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [ssd_mobilenet_v2​/fpnlite_320x320](https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1)| CPU, iGPU, MYRIAD |
| [ssd_mobilenet_v2​/fpnlite_640x640](https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1)| CPU, iGPU, MYRIAD |
| [ssd_mobilenet_v2](https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2)| CPU, iGPU, MYRIAD |
| [talkheads_ggelu_bert_en_base](https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/2?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [tf2-preview​/gnews-swivel-20dim](https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1)| CPU, iGPU, MYRIAD |
| [tf2-preview​/gnews-swivel-20dim-with-oov](https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1)| CPU, iGPU, MYRIAD |
| [tf2-preview​/inception_v3​/classification](https://tfhub.dev/google/tf2-preview/inception_v3/classification/4)| CPU, iGPU, MYRIAD |
| [tf2-preview​/inception_v3​/feature_vector](https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4)| CPU, iGPU, MYRIAD |
| [tf2-preview​/mobilenet_v2​/classification](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4)| CPU, iGPU, MYRIAD |
| [tf2-preview​/mobilenet_v2​/feature_vector](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4)| CPU, iGPU, MYRIAD |
| [tf2-preview​/nnlm-en-dim128](https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1)| CPU, iGPU, MYRIAD |
| [tf2-preview​/nnlm-en-dim128-with-normalization](https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1)| CPU, iGPU, MYRIAD |
| [tf2-preview​/nnlm-en-dim50](https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1)| CPU, iGPU, MYRIAD |
| [tf2-preview​/nnlm-es-dim50-with-normalization](https://tfhub.dev/google/tf2-preview/nnlm-es-dim50-with-normalization/1)| CPU, iGPU, MYRIAD |
| [universal-sentence-encoder](https://tfhub.dev/google/universal-sentence-encoder/4)| CPU, iGPU, MYRIAD |
| [universal-sentence-encoder-cmlm​/multilingual-preprocess](https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/)| CPU, iGPU, MYRIAD |
| [universal-sentence-encoder-cmlm_multilingual-base](https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [universal-sentence-encoder-cmlm_multilingual-base-br](https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [universal-sentence-encoder-large](https://tfhub.dev/google/universal-sentence-encoder-large/5)| CPU, iGPU, MYRIAD |
| [universal-sentence-encoder-multilingual](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3)| CPU, iGPU, MYRIAD |
| [universal-sentence-encoder-multilingual-large](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3)| CPU, iGPU, MYRIAD |
| [universal-sentence-encoder-multilingual-qa](https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3)| CPU, iGPU, MYRIAD |
| [universal-sentence-encoder-qa](https://tfhub.dev/google/universal-sentence-encoder-qa/3)| CPU, iGPU, MYRIAD |
| [vggish](https://tfhub.dev/google/vggish/1)| CPU, iGPU, MYRIAD |
| [Wiki-words-250](https://tfhub.dev/google/Wiki-words-250/2)| CPU, iGPU, MYRIAD |
| [Wiki-words-250-with-normalization](https://tfhub.dev/google/Wiki-words-250-with-normalization/2)| CPU, iGPU, MYRIAD |
| [Wiki-words-500](https://tfhub.dev/google/Wiki-words-500/2?tf-hub-format=compressed)  | CPU, iGPU, MYRIAD |
| [Wiki-words-500-with-normalization](https://tfhub.dev/google/Wiki-words-500-with-normalization/2)| CPU, iGPU, MYRIAD |
| [yamnet](https://tfhub.dev/google/yamnet/1)| CPU, iGPU, MYRIAD |



## Other Models
| Model Name | Supported Devices |
|---|---|
| [3d-pose-baseline](https://drive.google.com/file/d/0BxWzojlLp259MF9qSFpiVjl0cU0/view?usp=sharing)| CPU, iGPU, MYRIAD, VAD-M |
| [ACGAN](https://github.com/hwalsuklee/tensorflow-generative-model-collections) | CPU |
| [adv_inception_v3](http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz) | CPU |
| [ALBERT](https://storage.googleapis.com/albert_models/albert_base_v2.tar.gz) | CPU |
| [arttrack-coco-multi](https://github.com/eldar/pose-tensorflow/tree/master/models/coco) | CPU, iGPU, MYRIAD |
| [arttrack-mpii-single](https://github.com/eldar/pose-tensorflow/tree/master/models/mpii) | CPU, iGPU, MYRIAD |
| [BERT_LARGE](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT) | CPU |
| [context_rcnn_resnet101_snapshot_serenget](http://download.tensorflow.org/models/object_detection/context_rcnn_resnet101_snapshot_serengeti_2020_06_10.tar.gz) | CPU |
| [cpm-person](https://github.com/CMU-Perceptual-Computing-Lab/convolutional-pose-machines-release)| CPU, iGPU, MYRIAD, VAD-M |
| [cpm-pose](https://github.com/CMU-Perceptual-Computing-Lab/convolutional-pose-machines-release)| CPU, iGPU, VAD-M |
| [CRNN](https://github.com/MaybeShewill-CV/CRNN_Tensorflow)| CPU, iGPU, MYRIAD, VAD-M |
| [ctpn](https://github.com/eragonruan/text-detection-ctpn/releases/download/untagged-48d74c6337a71b6b5f87/ctpn.pb)| CPU, iGPU, MYRIAD, VAD-M |
| [deeplab](http://download.tensorflow.org/models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz) | CPU |
| [deeplabv3](http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz)| CPU, MYRIAD, VAD-M |
| [densenet161](https://drive.google.com/file/d/0B_fUSpodN0t0NmZvTnZZa2plaHc/view)| CPU, iGPU, MYRIAD, VAD-M |
| [dilation](https://github.com/fyu/dilation)| CPU, iGPU |
| [east_resnet_v1_50](https://github.com/argman/EAST#download)| CPU, iGPU, MYRIAD, VAD-M |
| [efficientdet-d0_frozen](https://github.com/google/automl/tree/aa6480fe7e07bd99030e56b7f05c75e5291db357/efficientdett)| CPU,  iGPU,  MYRIAD |
| [EfficientDet-D0-512x512](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz) | CPU |
| [efficientdet-d1_frozen](https://github.com/google/automl/tree/aa6480fe7e07bd99030e56b7f05c75e5291db357/efficientdet)| CPU,  iGPU,  MYRIAD |
| [EfficientDet-D1-640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz) | CPU |
| [EfficientDet-D2-768x768](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz) | CPU |
| [EfficientDet-D3-896x896](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz) | CPU |
| [EfficientDet-D4-1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz) | CPU |
| [EfficientDet-D5-1280x1280](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz) | CPU |
| [EfficientDet-D6-1280x1280](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz) | CPU |
| [EfficientDet-D7-1536x1536](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz) | CPU |
| [efficientnet-b0_auto_aug](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b0.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [efficientnet-b7_auto_aug](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b7.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [ens3_adv_inception_v3](http://download.tensorflow.org/models/ens3_adv_inception_v3_2017_08_18.tar.gz) | CPU |
| [facenet-20180408-102900](https://docs.openvinotoolkit.org/latest/omz_models_model_facenet_20180408_102900.html) | CPU, iGPU, MYRIAD |
| [faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz) | CPU |
| [faster_rcnn_inception_resnet_v2_atrous_oid](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28.tar.gz) | CPU |
| [faster_rcnn_nas_coco](https://github.com/shaoqingren/faster_rcnn)| CPU |
| [faster_rcnn_nas_coco_2018_01_28](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz) | CPU, VAD-M |
| [faster_rcnn_nas_lowproposals_coco](https://github.com/rbgirshick/py-faster-rcnn)| CPU, MYRIAD, VAD-M |
| [faster_rcnn_resnet101_ava_v2.1](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_ava_v2.1_2018_04_30.tar.gz)| CPU|
| [faster_rcnn_resnet101_fgvc](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_fgvc_2018_07_19.tar.gz) | CPU|
| [faster_rcnn_resnet101_kitti](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_kitti_2018_01_28.tar.gz) | CPU |
| [faster_rcnn_resnet101_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz) | CPU |
| [faster_rcnn_resnet101_snapshot_serengeti](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_snapshot_serengeti_2020_06_10.tar.gz) | CPU |
| [faster_rcnn_resnet50_fgvc](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_fgvc_2018_07_19.tar.gz) | CPU |
| [fc_densenet_103](https://github.com/AI-slam/FC-DenseNet-Tiramisu)| CPU, iGPU, MYRIAD, VAD-M |
| [fcrn-dp-nyu-depth-v2-tf](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_FCRN-checkpoint.zip)| CPU, iGPU, MYRIAD, VAD-M |
| [googlenet-v1](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [googlenet-v2](http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [googlenet-v3](https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py)| CPU, iGPU, MYRIAD, VAD-M |
| [googlenet-v4](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [handwritten-score-recognition-0003](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/handwritten-score-recognition-0003/README.md) | CPU |
| [i3d-flow](https://github.com/deepmind/kinetics-i3d)| CPU, iGPU, MYRIAD, VAD-M |
| [i3d-rgb](https://download.01.org/opencv/public_models/032020/i3d-rgb/rgb.frozen.pb)| CPU, iGPU, MYRIAD, VAD-M |
| [icnet-camvid-ava-0001](https://docs.openvinotoolkit.org/latest/omz_models_model_icnet_camvid_ava_0001.html)| CPU |
| [icnet-camvid-ava-sparse-30-0001](https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/2/icnet-camvid-ava-sparse-30-0001/)| CPU|
| [icnet-camvid-ava-sparse-60-0001](https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/2/icnet-camvid-ava-sparse-60-0001/)| CPU|
| [image-retrieval-0001](https://download.01.org/opencv/openvino_training_extensions/models/image_retrieval/image-retrieval-0001.tar.gz)| CPU |
| [inception_resnet_v2](https://github.com/openvinotoolkit/open_model_zoo/blob/2021.2/models/public/inception-resnet-v2-tf/model.yml)| CPU, iGPU, MYRIAD, VAD-M |
| [license-plate-recognition-barrier-0007](https://download.01.org/openvinotoolkit/training_toolbox_tensorflow/models/lpr/chinese_lp/license-plate-recognition-barrier-0007.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [lm_1b](http://download.tensorflow.org/models/LM_LSTM_CNN/graph-2016-09-10.pbtxt)| CPU, iGPU, MYRIAD |
| [mask_rcnn_resnet101_atrous_coco](https://github.com/facebookresearch/Detectron)| CPU, iGPU, MYRIAD, VAD-M |
| [mask_rcnn_resnet50_atrous](https://github.com/facebookresearch/Detectron)| CPU, iGPU, MYRIAD |
| [mask_rcnn_resnet50_atrous_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz) | CPU, iGPU, MYRIAD, VAD-M |
| [mobilenet_v1_0.25_128](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128.tgz)| CPU, iGPU, MYRIAD, VAD-M |
| [mobilenet_v1_0.50_160](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_160.tgz)| CPU, iGPU, MYRIAD, VAD-M |
| [mobilenet_v1_0.50_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224.tgz)| CPU, iGPU, MYRIAD, VAD-M |
| [mobilenet_v1_1.0_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz)| CPU, iGPU, MYRIAD, VAD-M |
| [mobilenet_v2_1.0_224](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz)| CPU, iGPU, MYRIAD, VAD-M |
| [mobilenet_v2_fpn_ssdlite_crossroad](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)| CPU, iGPU, MYRIAD, VAD-M |
| [mobilenet-v3-large-1.0-224-tf](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz)| CPU, iGPU, MYRIAD, VAD-M |
| [mobilenet-v3-small-1.0-224-tf](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-small_224_1.0_float.tgz)| CPU, iGPU, MYRIAD, VAD-M |
| [mozilla-deepspeech-0.6.1](https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/deepspeech-0.6.1-models.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [mozilla-deepspeech-0.7.1](https://github.com/mozilla/DeepSpeech/archive/v0.7.1.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [mozilla-deepspeech-0.8.2](https://github.com/mozilla/DeepSpeech)| CPU, MYRIAD, VAD-M |
| [NCF-1B](https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_5/ncf_fp32_pretrained_model.tar.gz) | CPU |
| [netvlad](http://rpg.ifi.uzh.ch/datasets/netvlad/vd16_pitts30k_conv5_3_vlad_preL2_intra_white.zip)| CPU, iGPU, MYRIAD, VAD-M |
| [NiftyNet](https://github.com/NifTK/NiftyNetModelZoo/tree/5-reorganising-with-lfs/mr_ct_regression)| CPU, iGPU, MYRIAD, VAD-M |
| [openpose-pose](http://www.mediafire.com/file/qlzzr20mpocnpa3/graph_opt.pb)| CPU, iGPU, MYRIAD, VAD-M |
| [person-vehicle-bike-detection-crossroad-yolov3-1020](https://docs.openvinotoolkit.org/latest/omz_models_model_person_vehicle_bike_detection_crossroad_yolov3_1020.html) | CPU |
| [pose-ae-multiperson](https://github.com/umich-vl/pose-ae-demo)| CPU, iGPU, MYRIAD, VAD-M |
| [pose-ae-refinement](https://github.com/umich-vl/pose-ae-demo)| CPU, iGPU, MYRIAD, VAD-M |
| [PRNet](https://github.com/YadiraF/PRNet)| CPU, iGPU, MYRIAD, VAD-M |
| [resnet_50](https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/resnet50_fp32_pretrained_model.pb)| CPU, iGPU, MYRIAD, VAD-M |
| [resnet_50_v1.5](https://zenodo.org/record/2535873/files/resnet50_v1.pb)| CPU, iGPU, MYRIAD, VAD-M |
| [resnet_v2_101](http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [resnet-101](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [resnet-152](http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [ResNet-50_v1.5](https://zenodo.org/record/2535873/files/resnet50_v1.pb) | CPU |
| [resnet-50-tf](http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v1_fp32_savedmodel_NHWC_jpg.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [ResNeXt_101](https://drive.google.com/uc?id=1AEYDWTWEGh6xGN-fSFB_f94FujdTJyKS) | CPU |
| [ResNext_50](https://github.com/HiKapok/TF-SENet) | CPU |
| [resnext50v2](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz) | CPU, iGPU, MYRIAD, VAD-M |
| [R-FCN](https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/rfcn_resnet101_fp32_coco_pretrained_model.tar.gz) | CPU |
| [rfcn_resnet101_coco](https://download.pytorch.org/models/resnet50-19c8e357.pth)| CPU, iGPU, MYRIAD, VAD-M |
| [se-resnext-50](https://drive.google.com/drive/folders/1k5MtfqbNRA8ziE3f18vu00Q1FQCzk4__)| CPU, iGPU, MYRIAD, VAD-M |
| [SqueezeNet](https://github.com/Dawars/SqueezeNet-tf.git) | CPU |
| [squeezenet1.1](https://github.com/forresti/SqueezeNet)| CPU, iGPU, MYRIAD, VAD-M |
| [srgan](https://github.com/tensorlayer/srgan)| CPU, iGPU, MYRIAD, VAD-M |
| [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz) | CPU |
| [ssd_mobilenet_v1_0.75_depth_300x300_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz) | CPU |
| [ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz) | CPU |
| [ssd_resnet_101_fpn_oidv4](http://download.tensorflow.org/models/object_detection/ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync_2019_01_20.tar.gz) | CPU |
| [ssd_resnet34_1200x1200](https://zenodo.org/record/3345892/files/tf_ssd_resnet34_22.1.zip?download=1) | CPU |
| [ssd_resnet34_300x300](https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/ssd_resnet34_fp32_bs1_pretrained_model.pb) | CPU |
| [ssd_resnet34_fp32_1200x1200_pretrained_model](https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_fp32_1200x1200_pretrained_model.pb) | CPU |
| [SSD_ResNet50_V1_FPN_640x640_RetinaNet50](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) | CPU |
| [ssd_resnet50_v1_fpn_coco](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)| CPU |
| [STN](https://github.com/oarriaga/STN.keras)| CPU, iGPU, MYRIAD, VAD-M |
| [text-recognition-0012](https://docs.openvinotoolkit.org/latest/omz_models_model_text_recognition_0012.html) | CPU |
| [Transformer-LT](https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/transformer_lt_official_fp32_pretrained_model.tar.gz) | CPU |
| [vehicle-attributes-barrier-0103](https://download.01.org/opencv/openvino_training_extensions/models/vehicle_attributes/vehicle-attributes-barrier-0103.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [vehicle-license-plate-detection-barrier-0106](https://docs.openvinotoolkit.org/latest/omz_models_model_vehicle_license_plate_detection_barrier_0106.html) | CPU |
| [vehicle-license-plate-detection-barrier-0123](https://download.01.org/opencv/openvino_training_extensions/models/ssd_detector/ssd-mobilenet-v2-0.35.1-barrier-256x256-0123.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [vggvox](https://github.com/linhdvu14/vggvox-speaker-identification)| CPU, iGPU, MYRIAD, VAD-M |
| [wavenet](https://storage.googleapis.com/intel-optimized-tensorflow/models/wavenet_fp32_pretrained_model.tar.gz) | CPU |
| [wide_deep](https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/wide_deep_fp32_pretrained_model.pb) | CPU |
| [yolo_v2](https://github.com/david8862/keras-YOLOv3-model-set.git)| CPU, iGPU, MYRIAD, VAD-M |
| [yolo_v2_tiny](https://github.com/david8862/keras-YOLOv3-model-set.git)| CPU, iGPU, MYRIAD, VAD-M |
| [yolo_v3_darknet](https://github.com/mystic123/tensorflow-yolo-v3.git)| CPU, iGPU, MYRIAD, VAD-M |
| [yolo-v1-tiny-tf](https://www.npmjs.com/package/tfjs-yolo)| CPU, iGPU, MYRIAD |
| [yolo-v3](https://download.01.org/opencv/public_models/022020/yolo_v3/yolov3.pb)| CPU, iGPU, MYRIAD, VAD-M |
| [yolo-v3-tiny-tf](https://download.01.org/opencv/public_models/082020/yolo-v3-tiny-tf/yolo-v3-tiny-tf.zip)| CPU, iGPU, MYRIAD, VAD-M |
| [yolo-v4](https://github.com/david8862/keras-YOLOv3-model-set)| CPU, iGPU, MYRIAD, VAD-M |



Pre-trained frozen model files are provided for only some of these models. For the rest, please refer to the links provided.

## Quantization Support (Preview)

 **OpenVINO™ integration with TensorFlow** now supports INT8 models quantized using Quantization-Aware Training (QAT) tools such as OpenVINO™ [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf) and [TensorFlow Model Optimization ToolKit (TFMOT)](https://www.tensorflow.org/model_optimization). This support is currently in a preview state and performance optimizations are in progress.
 
Some examples of NNCF usage to produce quantized models can be found [here](https://github.com/openvinotoolkit/nncf/tree/develop/examples/tensorflow/).

Some quantized models are shown to provide more optimized performance by setting the environment variable 'OPENVINO_TF_CONSTANT_FOLDING' to 1 before running inference.
 
**Note:** The latest supported TensorFlow versions for NNCF and **OpenVINO™ integration with TensorFlow** may be different. It is advised that the users create a separate virtual environment for quantizing the models with NNCF to avoid any TensorFlow version incompatability issues. The quantized models can then be run in the environment that is compatible with **OpenVINO™ integration with TensorFlow**. NNCF compatible with TensorFlow version 2.4.2 is validated with **OpenVINO™ integration with TensorFlow** compatible with TensorFlow version 2.7.0.