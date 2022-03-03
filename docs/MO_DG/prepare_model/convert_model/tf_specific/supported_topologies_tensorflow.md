# Supported TensorFlow Topologies {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_supported_topologies}

**Supported Non-Frozen Topologies with Links to the Associated Slim Model Classification Download Files**

Detailed information on how to convert models from the <a href="https://github.com/tensorflow/models/tree/master/research/slim/README.md">TensorFlow\*-Slim Image Classification Model Library</a> is available in the [Converting TensorFlow*-Slim Image Classification Model Library Models](Convert_Slim_Library_Models.md) chapter. The table below contains list of supported TensorFlow\*-Slim Image Classification Model Library models and required mean/scale values. The mean values are specified as if the input image is read in BGR channels order layout like OpenVINO classification sample does.

| Model Name| Slim Model Checkpoint File| \-\-mean_values | \-\-scale|
| ------------- | ------------ | ------------- | -----:|
|Inception v1| [inception_v1_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz)| [127.5,127.5,127.5]| 127.5|
|Inception v2| [inception_v1_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz)| [127.5,127.5,127.5]| 127.5|
|Inception v3| [inception_v3_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)| [127.5,127.5,127.5]| 127.5|
|Inception V4| [inception_v4_2016_09_09.tar.gz](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz)| [127.5,127.5,127.5]| 127.5|
|Inception ResNet v2| [inception_resnet_v2_2016_08_30.tar.gz](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz)| [127.5,127.5,127.5]| 127.5|
|MobileNet v1 128| [mobilenet_v1_0.25_128.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz)| [127.5,127.5,127.5]| 127.5|
|MobileNet v1 160| [mobilenet_v1_0.5_160.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_160.tgz)| [127.5,127.5,127.5]| 127.5|
|MobileNet v1 224| [mobilenet_v1_1.0_224.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)| [127.5,127.5,127.5]| 127.5|
|NasNet Large| [nasnet-a_large_04_10_2017.tar.gz](https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz)| [127.5,127.5,127.5]| 127.5|
|NasNet Mobile| [nasnet-a_mobile_04_10_2017.tar.gz](https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_mobile_04_10_2017.tar.gz)| [127.5,127.5,127.5]| 127.5|
|ResidualNet-50 v1| [resnet_v1_50_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)| [103.94,116.78,123.68] | 1 |
|ResidualNet-50 v2| [resnet_v2_50_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz)| [103.94,116.78,123.68] | 1 |
|ResidualNet-101 v1| [resnet_v1_101_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz)| [103.94,116.78,123.68] | 1 |
|ResidualNet-101 v2| [resnet_v2_101_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz)| [103.94,116.78,123.68] | 1 |
|ResidualNet-152 v1| [resnet_v1_152_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz)| [103.94,116.78,123.68] | 1 |
|ResidualNet-152 v2| [resnet_v2_152_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz)| [103.94,116.78,123.68] | 1 |
|VGG-16| [vgg_16_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)| [103.94,116.78,123.68] | 1 |
|VGG-19| [vgg_19_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz)| [103.94,116.78,123.68] | 1 |

**Supported Pre-Trained Topologies from TensorFlow 1 Detection Model Zoo**

Detailed information on how to convert models from the <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md">TensorFlow 1 Detection Model Zoo</a> is available in the [Converting TensorFlow Object Detection API Models](Convert_Object_Detection_API_Models.md) chapter. The table below contains models from the Object Detection Models zoo that are supported.

| Model Name| TensorFlow 1 Object Detection API Models|
| :------------- | -----:|
|SSD MobileNet V1 COCO\*| [ssd_mobilenet_v1_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)|
|SSD MobileNet V1 0.75 Depth COCO|  [ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz)|
|SSD MobileNet V1 PPN COCO|  [ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz)|
|SSD MobileNet V1 FPN COCO|  [ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)|
|SSD ResNet50 FPN COCO|  [ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)|
|SSD MobileNet V2 COCO|  [ssd_mobilenet_v2_coco_2018_03_29.tar.gz](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)|
|SSD Lite MobileNet V2 COCO|  [ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)|
|SSD Inception V2 COCO|	[ssd_inception_v2_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)|
|RFCN ResNet 101 COCO|  [rfcn_resnet101_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_2018_01_28.tar.gz)|
|Faster R-CNN Inception V2 COCO|  [faster_rcnn_inception_v2_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)|
|Faster R-CNN ResNet 50 COCO|  [faster_rcnn_resnet50_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz)|
|Faster R-CNN ResNet 50 Low Proposals COCO|  [faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz)|
|Faster R-CNN ResNet 101 COCO|  [faster_rcnn_resnet101_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz)|
|Faster R-CNN ResNet 101 Low Proposals COCO|  [faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz)|
|Faster R-CNN Inception ResNet V2 COCO|  [faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)|
|Faster R-CNN Inception ResNet V2 Low Proposals COCO|  [faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz)|
|Faster R-CNN NasNet COCO|  [faster_rcnn_nas_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz)|
|Faster R-CNN NasNet Low Proposals COCO|  [faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz)|
|Mask R-CNN Inception ResNet V2 COCO|  [mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)|
|Mask R-CNN Inception V2 COCO|  [mask_rcnn_inception_v2_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz)|
|Mask R-CNN ResNet 101 COCO|  [mask_rcnn_resnet101_atrous_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet101_atrous_coco_2018_01_28.tar.gz)|
|Mask R-CNN ResNet 50 COCO|  [mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz)|
|Faster R-CNN ResNet 101 Kitti\*|  [faster_rcnn_resnet101_kitti_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_kitti_2018_01_28.tar.gz)|
|Faster R-CNN Inception ResNet V2 Open Images\*|  [faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28.tar.gz)|
|Faster R-CNN Inception ResNet V2 Low Proposals Open Images\*|  [faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28.tar.gz)|
|Faster R-CNN ResNet 101 AVA v2.1\*|  [faster_rcnn_resnet101_ava_v2.1_2018_04_30.tar.gz](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_ava_v2.1_2018_04_30.tar.gz)|

**Supported Pre-Trained Topologies from TensorFlow 2 Detection Model Zoo**

Detailed information on how to convert models from the <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md">TensorFlow 2 Detection Model Zoo</a> is available in the [Converting TensorFlow Object Detection API Models](Convert_Object_Detection_API_Models.md) chapter. The table below contains models from the Object Detection Models zoo that are supported.

| Model Name| TensorFlow 2 Object Detection API Models|
| :------------- | -----:|
| EfficientDet D0 512x512 |  [efficientdet_d0_coco17_tpu-32.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz)|
| EfficientDet D1 640x640 |  [efficientdet_d1_coco17_tpu-32.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz)|
| EfficientDet D2 768x768 |  [efficientdet_d2_coco17_tpu-32.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz)|
| EfficientDet D3 896x896 |  [efficientdet_d3_coco17_tpu-32.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz)|
| EfficientDet D4 1024x1024 |  [efficientdet_d4_coco17_tpu-32.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz)|
| EfficientDet D5 1280x1280 |  [efficientdet_d5_coco17_tpu-32.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz)|
| EfficientDet D6 1280x1280 |  [efficientdet_d6_coco17_tpu-32.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz)|
| EfficientDet D7 1536x1536 |  [efficientdet_d7_coco17_tpu-32.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz)|
| SSD MobileNet v2 320x320 |  [ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz)|
| SSD MobileNet V1 FPN 640x640 |  [ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz)|
| SSD MobileNet V2 FPNLite 320x320 |  [ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz)|
| SSD MobileNet V2 FPNLite 640x640 |  [ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz)|
| SSD ResNet50 V1 FPN 640x640 (RetinaNet50) |  [ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)|
| SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50) |  [ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz)|
| SSD ResNet101 V1 FPN 640x640 (RetinaNet101) |  [ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz)|
| SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101) |  [ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz)|
| SSD ResNet152 V1 FPN 640x640 (RetinaNet152) |  [ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz)|
| SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152) |  [ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz)|
| Faster R-CNN ResNet50 V1 640x640 |  [faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz)|
| Faster R-CNN ResNet50 V1 1024x1024 |  [faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz)|
| Faster R-CNN ResNet50 V1 800x1333 |  [faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.tar.gz)|
| Faster R-CNN ResNet101 V1 640x640 |  [faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz)|
| Faster R-CNN ResNet101 V1 1024x1024 |  [faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz)|
| Faster R-CNN ResNet101 V1 800x1333 |  [faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz)|
| Faster R-CNN ResNet152 V1 640x640 |  [faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz)|
| Faster R-CNN ResNet152 V1 1024x1024 |  [faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz)|
| Faster R-CNN ResNet152 V1 800x1333 |  [faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.tar.gz)|
| Faster R-CNN Inception ResNet V2 640x640 |  [faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz)|
| Faster R-CNN Inception ResNet V2 1024x1024 |  [faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz)|
| Mask R-CNN Inception ResNet V2 1024x1024 |  [mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz](http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz)|

**Supported Frozen Quantized Topologies**

The topologies hosted on the TensorFlow\* Lite [site](https://www.tensorflow.org/lite/guide/hosted_models). The frozen model file (`.pb` file) should be fed to the Model Optimizer.

| Model Name            |                                                                                                                Frozen Model File |
|:----------------------|---------------------------------------------------------------------------------------------------------------------------------:|
| Mobilenet V1 0.25 128 | [mobilenet_v1_0.25_128_quant.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128_quant.tgz) |
| Mobilenet V1 0.25 160 | [mobilenet_v1_0.25_160_quant.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_160_quant.tgz) |
| Mobilenet V1 0.25 192 | [mobilenet_v1_0.25_192_quant.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_192_quant.tgz) |
| Mobilenet V1 0.25 224 | [mobilenet_v1_0.25_224_quant.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224_quant.tgz) |
| Mobilenet V1 0.50 128 |   [mobilenet_v1_0.5_128_quant.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_128_quant.tgz) |
| Mobilenet V1 0.50 160 |   [mobilenet_v1_0.5_160_quant.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_160_quant.tgz) |
| Mobilenet V1 0.50 192 |   [mobilenet_v1_0.5_192_quant.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_192_quant.tgz) |
| Mobilenet V1 0.50 224 |   [mobilenet_v1_0.5_224_quant.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224_quant.tgz) |
| Mobilenet V1 0.75 128 | [mobilenet_v1_0.75_128_quant.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_128_quant.tgz) |
| Mobilenet V1 0.75 160 | [mobilenet_v1_0.75_160_quant.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_160_quant.tgz) |
| Mobilenet V1 0.75 192 | [mobilenet_v1_0.75_192_quant.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_192_quant.tgz) |
| Mobilenet V1 0.75 224 | [mobilenet_v1_0.75_224_quant.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_224_quant.tgz) |
| Mobilenet V1 1.0 128  |   [mobilenet_v1_1.0_128_quant.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_128_quant.tgz) |
| Mobilenet V1 1.0 160  |   [mobilenet_v1_1.0_160_quant.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_160_quant.tgz) |
| Mobilenet V1 1.0 192  |   [mobilenet_v1_1.0_192_quant.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_192_quant.tgz) |
| Mobilenet V1 1.0 224  |   [mobilenet_v1_1.0_224_quant.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) |
| Mobilenet V2 1.0 224  |           [mobilenet_v2_1.0_224_quant.tgz](http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz) |
| Inception V1          |                 [inception_v1_224_quant_20181026.tgz](http://download.tensorflow.org/models/inception_v1_224_quant_20181026.tgz) |
| Inception V2          |                 [inception_v2_224_quant_20181026.tgz](http://download.tensorflow.org/models/inception_v2_224_quant_20181026.tgz) |
| Inception V3          |                           [inception_v3_quant.tgz](http://download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz) |
| Inception V4          |                 [inception_v4_299_quant_20181026.tgz](http://download.tensorflow.org/models/inception_v4_299_quant_20181026.tgz) |

It is necessary to specify the following command line parameters for the Model Optimizer to convert some of the models from the list above: `--input input --input_shape [1,HEIGHT,WIDTH,3]`.
Where `HEIGHT` and `WIDTH` are the input images height and width for which the model was trained.

**Other supported topologies**

| Model Name| Repository |
| :------------- | -----:|
| ResNext | [Repo](https://github.com/taki0112/ResNeXt-Tensorflow)|
| DenseNet | [Repo](https://github.com/taki0112/Densenet-Tensorflow)|
| CRNN | [Repo](https://github.com/MaybeShewill-CV/CRNN_Tensorflow) |
| NCF | [Repo](https://github.com/tensorflow/models/tree/master/official/recommendation) |
| lm_1b | [Repo](https://github.com/tensorflow/models/tree/master/research/lm_1b) |
| DeepSpeech | [Repo](https://github.com/mozilla/DeepSpeech) |
| A3C | [Repo](https://github.com/miyosuda/async_deep_reinforce) |
| VDCNN | [Repo](https://github.com/WenchenLi/VDCNN) |
| Unet | [Repo](https://github.com/kkweon/UNet-in-Tensorflow) |
| Keras-TCN | [Repo](https://github.com/philipperemy/keras-tcn) |
| PRNet | [Repo](https://github.com/YadiraF/PRNet) |
| YOLOv4 | [Repo](https://github.com/Ma-Dan/keras-yolo4) |
| STN | [Repo](https://github.com/oarriaga/STN.keras) |

* YOLO topologies from DarkNet* can be converted using [these instructions](Convert_YOLO_From_Tensorflow.md).
* FaceNet topologies can be converted using [these instructions](Convert_FaceNet_From_Tensorflow.md).
* CRNN topologies can be converted using [these instructions](Convert_CRNN_From_Tensorflow.md).
* NCF topologies can be converted using [these instructions](Convert_NCF_From_Tensorflow.md).
* [GNMT](https://github.com/tensorflow/nmt) topology can be converted using [these instructions](Convert_GNMT_From_Tensorflow.md).
* [BERT](https://github.com/google-research/bert) topology can be converted using [these instructions](Convert_BERT_From_Tensorflow.md).
* [XLNet](https://github.com/zihangdai/xlnet) topology can be converted using [these instructions](Convert_XLNet_From_Tensorflow.md).
* [Attention OCR](https://github.com/emedvedev/attention-ocr) topology can be converted using [these instructions](Convert_AttentionOCR_From_Tensorflow.md).