# Converting a TensorFlow* Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow}

A summary of the steps for optimizing and deploying a model that was trained with the TensorFlow\* framework:

1. [Configure the Model Optimizer](../Config_Model_Optimizer.md) for TensorFlow\* (TensorFlow was used to train your model).
2. [Freeze the TensorFlow model](#freeze-the-tensorflow-model) if your model is not already frozen or skip this step and use the [instruction](#loading-nonfrozen-models) to a convert a non-frozen model.
3. [Convert a TensorFlow\* model](#Convert_From_TF) to produce an optimized [Intermediate Representation (IR)](../../IR_and_opsets.md) of the model based on the trained network topology, weights, and biases values.
4. Test the model in the Intermediate Representation format using the [Inference Engine](../../../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) in the target environment via provided [sample applications](../../../IE_DG/Samples_Overview.md).
5. [Integrate](../../../IE_DG/Samples_Overview.md) the Inference Engine in your application to deploy the model in the target environment.

## Supported Topologies

**Supported Non-Frozen Topologies with Links to the Associated Slim Model Classification Download Files**

Detailed information on how to convert models from the <a href="https://github.com/tensorflow/models/tree/master/research/slim/README.md">TensorFlow\*-Slim Image Classification Model Library</a> is available in the [Converting TensorFlow*-Slim Image Classification Model Library Models](tf_specific/Convert_Slim_Library_Models.md) chapter. The table below contains list of supported TensorFlow\*-Slim Image Classification Model Library models and required mean/scale values. The mean values are specified as if the input image is read in BGR channels order layout like Inference Engine classification sample does.

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

**Supported Frozen Topologies from TensorFlow Object Detection Models Zoo**

Detailed information on how to convert models from the <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md">Object Detection Models Zoo</a> is available in the [Converting TensorFlow Object Detection API Models](tf_specific/Convert_Object_Detection_API_Models.md) chapter. The table below contains models from the Object Detection Models zoo that are supported.

| Model Name| TensorFlow Object Detection API Models (Frozen)|
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

* YOLO topologies from DarkNet* can be converted using [instruction](tf_specific/Convert_YOLO_From_Tensorflow.md),
* FaceNet topologies can be converted using [instruction](tf_specific/Convert_FaceNet_From_Tensorflow.md).
* CRNN topologies can be converted using [instruction](tf_specific/Convert_CRNN_From_Tensorflow.md).
* NCF topologies can be converted using [instruction](tf_specific/Convert_NCF_From_Tensorflow.md)
* [GNMT](https://github.com/tensorflow/nmt) topology can be converted using [instruction](tf_specific/Convert_GNMT_From_Tensorflow.md)
* [BERT](https://github.com/google-research/bert) topology can be converted using [this instruction](tf_specific/Convert_BERT_From_Tensorflow.md).
* [XLNet](https://github.com/zihangdai/xlnet) topology can be converted using [this instruction](tf_specific/Convert_XLNet_From_Tensorflow.md).

  

## Loading Non-Frozen Models to the Model Optimizer <a name="loading-nonfrozen-models"></a>

There are three ways to store non-frozen TensorFlow models and load them to the Model Optimizer:

1. Checkpoint:

    In this case, a model consists of two files:
    - `inference_graph.pb` or `inference_graph.pbtxt`
    - `checkpoint_file.ckpt`

    If you do not have an inference graph file, refer to [Freezing Custom Models in Python](#freeze-the-tensorflow-model).

    To convert such TensorFlow model:

    1. Go to the `<INSTALL_DIR>/deployment_tools/model_optimizer` directory
    2. Run the `mo_tf.py` script with the path to the checkpoint file to convert a model:

    * If input model is in `.pb` format:<br>
```sh
python3 mo_tf.py --input_model <INFERENCE_GRAPH>.pb --input_checkpoint <INPUT_CHECKPOINT>
```
    * If input model is in `.pbtxt` format:<br>
```sh
python3 mo_tf.py --input_model <INFERENCE_GRAPH>.pbtxt --input_checkpoint <INPUT_CHECKPOINT> --input_model_is_text
```

2. MetaGraph:

    In this case, a model consists of three or four files stored in the same directory:
    - `model_name.meta`
    - `model_name.index`
    - `model_name.data-00000-of-00001` (digit part may vary)
    - `checkpoint` (optional)

    To convert such TensorFlow model:

    1. Go to the `<INSTALL_DIR>/deployment_tools/model_optimizer` directory
    2. Run the `mo_tf.py` script with a path to the MetaGraph `.meta` file to convert a model:<br>
```sh
python3 mo_tf.py --input_meta_graph <INPUT_META_GRAPH>.meta
```

3. SavedModel format of TensorFlow 1.x and 2.x versions:

    In this case, a model consists of a special directory with a `.pb` file and several subfolders: `variables`, `assets`, and `assets.extra`. For more information about the SavedModel directory, refer to the [README](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/saved_model#components) file in the TensorFlow repository.

    To convert such TensorFlow model:

    1. Go to the `<INSTALL_DIR>/deployment_tools/model_optimizer` directory
    2. Run the `mo_tf.py` script with a path to the SavedModel directory to convert a model:<br>
```sh
python3 mo_tf.py --saved_model_dir <SAVED_MODEL_DIRECTORY>
```

You can convert TensorFlow 1.x SavedModel format in the environment that has a 1.x or 2.x version of TensorFlow. However, TensorFlow 2.x SavedModel format strictly requires the 2.x version of TensorFlow.
If a model contains operations currently unsupported by OpenVINO, prune these operations by explicit specification of input nodes using the `--input` option.
To determine custom input nodes, display a graph of the model in TensorBoard. To generate TensorBoard logs of the graph, use the `--tensorboard_logs` option.
TensorFlow 2.x SavedModel format has a specific graph due to eager execution. In case of pruning, find custom input nodes in the `StatefulPartitionedCall/*` subgraph of TensorFlow 2.x SavedModel format.

## Freezing Custom Models in Python\* <a name="freeze-the-tensorflow-model"></a>

When a network is defined in Python\* code, you have to create an inference graph file. Usually graphs are built in a form
that allows model training. That means that all trainable parameters are represented as variables in the graph.
To be able to use such graph with Model Optimizer such graph should be frozen.
The graph is frozen and dumped to a file with the following code:
```python
import tensorflow as tf
from tensorflow.python.framework import graph_io
frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["name_of_the_output_node"])
graph_io.write_graph(frozen, './', 'inference_graph.pb', as_text=False)
```

Where:

* `sess` is the instance of the TensorFlow\* Session object where the network topology is defined.
* `["name_of_the_output_node"]` is the list of output node names in the graph; `frozen` graph will
    include only those nodes from the original `sess.graph_def` that are directly or indirectly used
    to compute given output nodes. `'name_of_the_output_node'` here is an example of possible output
    node name. You should derive the names based on your own graph.
* `./` is the directory where the inference graph file should be generated.
* `inference_graph.pb` is the name of the generated inference graph file.
* `as_text` specifies whether the generated file should be in human readable text format or binary.

## Convert a TensorFlow* Model <a name="Convert_From_TF"></a>

To convert a TensorFlow model:

1. Go to the `<INSTALL_DIR>/deployment_tools/model_optimizer` directory
2. Use the `mo_tf.py` script to simply convert a model with the path to the input model `.pb` file:
```sh
python3 mo_tf.py --input_model <INPUT_MODEL>.pb
```

Two groups of parameters are available to convert your model:

* [Framework-agnostic parameters](Converting_Model_General.md): These parameters are used to convert any model trained in any supported framework.
* [TensorFlow-specific parameters](#tensorflow_specific_conversion_params): Parameters used to convert only TensorFlow models.

> **NOTE:** The color channel order (RGB or BGR) of an input data should match the channel order of the model training dataset. If they are different, perform the `RGB<->BGR` conversion specifying the command-line parameter: `--reverse_input_channels`. Otherwise, inference results may be incorrect. For more information about the parameter, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](Converting_Model_General.md).

### Using TensorFlow\*-Specific Conversion Parameters  <a name="tensorflow_specific_conversion_params"></a>
The following list provides the TensorFlow\*-specific parameters.

```
TensorFlow*-specific parameters:
  --input_model_is_text
                        TensorFlow*: treat the input model file as a text
                        protobuf format. If not specified, the Model Optimizer
                        treats it as a binary file by default.
  --input_checkpoint INPUT_CHECKPOINT
                        TensorFlow*: variables file to load.
  --input_meta_graph INPUT_META_GRAPH
                        Tensorflow*: a file with a meta-graph of the model
                        before freezing
  --saved_model_dir SAVED_MODEL_DIR
                        TensorFlow*: directory with a model in SavedModel format
                        of TensorFlow 1.x or 2.x version
  --saved_model_tags SAVED_MODEL_TAGS
                        Group of tag(s) of the MetaGraphDef to load, in string
                        format, separated by ','. For tag-set contains
                        multiple tags, all tags must be passed in.
  --tensorflow_custom_operations_config_update TENSORFLOW_CUSTOM_OPERATIONS_CONFIG_UPDATE
                        TensorFlow*: update the configuration file with node
                        name patterns with input/output nodes information.
  --tensorflow_object_detection_api_pipeline_config TENSORFLOW_OBJECT_DETECTION_API_PIPELINE_CONFIG
                        TensorFlow*: path to the pipeline configuration file
                        used to generate model created with help of Object
                        Detection API.
  --tensorboard_logdir TENSORBOARD_LOGDIR
                        TensorFlow*: dump the input graph to a given directory
                        that should be used with TensorBoard.
  --tensorflow_custom_layer_libraries TENSORFLOW_CUSTOM_LAYER_LIBRARIES
                        TensorFlow*: comma separated list of shared libraries
                        with TensorFlow* custom operations implementation.
  --disable_nhwc_to_nchw
                        Disables default translation from NHWC to NCHW
```

> **NOTE:** Models produces with TensorFlow\* usually have not fully defined shapes (contain `-1` in some dimensions). It is necessary to pass explicit shape for the input using command line parameter `--input_shape` or `-b` to override just batch dimension. If the shape is fully defined, then there is no need to specify either `-b` or `--input_shape` options.

#### Command-Line Interface (CLI) Examples Using TensorFlow\*-Specific Parameters

* Launching the Model Optimizer for Inception V1 frozen model when model file is a plain text protobuf:
```sh
python3 mo_tf.py --input_model inception_v1.pbtxt --input_model_is_text -b 1
```

* Launching the Model Optimizer for Inception V1 frozen model and update custom sub-graph replacement file `transform.json` with information about input and output nodes of the matched sub-graph. For more information about this feature, refer to [Sub-Graph Replacement in the Model Optimizer](../customize_model_optimizer/Subgraph_Replacement_Model_Optimizer.md).
```sh
python3 mo_tf.py --input_model inception_v1.pb -b 1 --tensorflow_custom_operations_config_update transform.json
```

* Launching the Model Optimizer for Inception V1 frozen model and use custom sub-graph replacement file `transform.json` for model conversion. For more information about this feature, refer to [Sub-Graph Replacement in the Model Optimizer](../customize_model_optimizer/Subgraph_Replacement_Model_Optimizer.md).
```sh
python3 mo_tf.py --input_model inception_v1.pb -b 1 --tensorflow_use_custom_operations_config transform.json
```

* Launching the Model Optimizer for Inception V1 frozen model and dump information about the graph to TensorBoard log dir `/tmp/log_dir`
```sh
python3 mo_tf.py --input_model inception_v1.pb -b 1 --tensorboard_logdir /tmp/log_dir
```

* Launching the Model Optimizer for a model with custom TensorFlow operations (refer to the [TensorFlow* documentation](https://www.tensorflow.org/extend/adding_an_op)) implemented in C++ and compiled into the shared library `my_custom_op.so`. Model Optimizer falls back to TensorFlow to infer output shape of operations implemented in the library if a custom TensorFlow operation library is provided. If it is not provided, a custom operation with an inference function is needed. For more information about custom operations, refer to the [Extending the Model Optimizer with New Primitives](../customize_model_optimizer/Extending_Model_Optimizer_with_New_Primitives.md).
```sh
python3 mo_tf.py --input_model custom_model.pb --tensorflow_custom_layer_libraries ./my_custom_op.so
```


## Convert TensorFlow* 2 Models <a name="Convert_From_TF2X"></a>

In order to convert TensorFlow* 2 models, installation of dependencies from `requirements_tf2.txt` is required.
TensorFlow* 2.X officially supports two model formats: SavedModel and Keras H5 (or HDF5).    
Below are the instructions on how to convert each of them.

### SavedModel Format     

A model in the SavedModel format consists of a directory with a `saved_model.pb` file and two subfolders: `variables` and `assets`. 
To convert such a model:
1. Go to the `<INSTALL_DIR>/deployment_tools/model_optimizer` directory.
2. Run the `mo_tf.py` script with a path to the SavedModel directory:
```sh
python3 mo_tf.py --saved_model_dir <SAVED_MODEL_DIRECTORY>
```

TensorFlow* 2 SavedModel format strictly requires the 2.x version of TensorFlow installed in the
environment for conversion to the Intermediate Representation (IR). 

If a model contains operations currently unsupported by OpenVINO™,
prune these operations by explicit specification of input nodes using the `--input` or `--output`
options. To determine custom input nodes, visualize a model graph in the TensorBoard.   

To generate TensorBoard logs of the graph, use the Model Optimizer `--tensorboard_logs` command-line
option.      

TensorFlow* 2 SavedModel format has a specific graph structure due to eager execution. In case of
pruning, find custom input nodes in the `StatefulPartitionedCall/*` subgraph.

### Keras H5        

If you have a model in the HDF5 format, load the model using TensorFlow* 2 and serialize it in the
SavedModel format. Here is an example of how to do it:
```python
import tensorflow as tf
model = tf.keras.models.load_model('model.h5')
tf.saved_model.save(model,'model')
```

The Keras H5 model with a custom layer has specifics to be converted into SavedModel format.
For example, the model with a custom layer `CustomLayer` from `custom_layer.py` is converted as follows:
```python
import tensorflow as tf
from custom_layer import CustomLayer
model = tf.keras.models.load_model('model.h5', custom_objects={'CustomLayer': CustomLayer})
tf.saved_model.save(model,'model')
```

Then follow the above instructions for the SavedModel format. 

> **NOTE:** Do not use other hacks to resave TensorFlow* 2 models into TensorFlow* 1 formats.    

> **NOTE**: Currently, OpenVINO™ support for TensorFlow* 2 models is in preview (aka Beta), which means limited and not of production quality yet. OpenVINO™ does not support models with Keras RNN and Embedding layers.


## Custom Layer Definition

Internally, when you run the Model Optimizer, it loads the model, goes through the topology, and tries to find each layer type in a list of known layers. Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in this list of known layers, the Model Optimizer classifies them as custom.

See [Custom Layers in the Model Optimizer](../customize_model_optimizer/Customize_Model_Optimizer.md) for information about:

* Model Optimizer internal procedure for working with custom layers
* How to convert a TensorFlow model that has custom layers
* Custom layer implementation details


## Supported TensorFlow\* Layers
Refer to [Supported Framework Layers ](../Supported_Frameworks_Layers.md) for the list of supported standard layers.


## Frequently Asked Questions (FAQ)

The Model Optimizer provides explanatory messages if it is unable to run to completion due to issues like typographical errors, incorrectly used options, or other issues. The message describes the potential cause of the problem and gives a link to the [Model Optimizer FAQ](../Model_Optimizer_FAQ.md). The FAQ has instructions on how to resolve most issues. The FAQ also includes links to relevant sections in the Model Optimizer Developer Guide to help you understand what went wrong.


## Summary
In this document, you learned:

* Basic information about how the Model Optimizer works with TensorFlow\* models
* Which TensorFlow models are supported
* How to freeze a TensorFlow model
* How to convert a trained TensorFlow model using the Model Optimizer with both framework-agnostic and TensorFlow-specific command-line options
