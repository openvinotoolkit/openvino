# Converting a MXNet* Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_MxNet}

A summary of the steps for optimizing and deploying a model that was trained with the MXNet\* framework:

1. [Configure the Model Optimizer](../Config_Model_Optimizer.md) for MXNet* (MXNet was used to train your model)
2. [Convert a MXNet model](#ConvertMxNet) to produce an optimized [Intermediate Representation (IR)](../../IR_and_opsets.md) of the model based on the trained network topology, weights, and biases values
3. Test the model in the Intermediate Representation format using the [Inference Engine](../../../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) in the target environment via provided Inference Engine [sample applications](../../../IE_DG/Samples_Overview.md)
4. [Integrate](../../../IE_DG/Samples_Overview.md) the [Inference Engine](../../../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) in your application to deploy the model in the target environment

## Supported Topologies

> **NOTE:** SSD models from the table require converting to the deploy mode. For details, see the [Conversion Instructions](https://github.com/zhreshold/mxnet-ssd/#convert-model-to-deploy-mode) in the GitHub MXNet-SSD repository.

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

* Style transfer [model](https://github.com/zhaw/neural_style) can be converted using [instruction](mxnet_specific/Convert_Style_Transfer_From_MXNet.md),

## Convert an MXNet* Model <a name="ConvertMxNet"></a>

To convert an MXNet\* model:

1. Go to the `<INSTALL_DIR>/deployment_tools/model_optimizer` directory.
2. To convert an MXNet\* model contained in a `model-file-symbol.json` and `model-file-0000.params`, run the Model Optimizer launch script `mo.py`, specifying a path to the input model file:
```sh
python3 mo_mxnet.py --input_model model-file-0000.params
```

Two groups of parameters are available to convert your model:

* [Framework-agnostic parameters](Converting_Model_General.md): These parameters are used to convert any model trained in any supported framework.
* [MXNet-specific parameters](#mxnet_specific_conversion_params): Parameters used to convert only MXNet models


### Using MXNet\*-Specific Conversion Parameters <a name="mxnet_specific_conversion_params"></a>
The following list provides the MXNet\*-specific parameters.

```
MXNet-specific parameters:
  --input_symbol <SYMBOL_FILE_NAME>
            Symbol file (for example, "model-symbol.json") that contains a topology structure and layer attributes
  --nd_prefix_name <ND_PREFIX_NAME>
            Prefix name for args.nd and argx.nd files
  --pretrained_model_name <PRETRAINED_MODEL_NAME>
            Name of a pretrained MXNet model without extension and epoch
            number. This model will be merged with args.nd and argx.nd
            files
  --save_params_from_nd
            Enable saving built parameters file from .nd files
  --legacy_mxnet_model
            Enable MXNet loader to make a model compatible with the latest MXNet version.
            Use only if your model was trained with MXNet version lower than 1.0.0
  --enable_ssd_gluoncv
            Enable transformation for converting the gluoncv ssd topologies.
            Use only if your topology is one of ssd gluoncv topologies
```

> **NOTE:** By default, the Model Optimizer does not use the MXNet loader, as it transforms the topology to another format, which is compatible with the latest
> version of MXNet, but it is required for models trained with lower version of MXNet. If your model was trained with MXNet version lower than 1.0.0, specify the
> `--legacy_mxnet_model` key to enable the MXNet loader. However, the loader does not support models with custom layers. In this case, you must manually
> recompile MXNet with custom layers and install it to your environment.

## Custom Layer Definition

Internally, when you run the Model Optimizer, it loads the model, goes through the topology, and tries to find each layer type in a list of known layers. Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in this list of known layers, the Model Optimizer classifies them as custom.

## Supported MXNet\* Layers
Refer to [Supported Framework Layers ](../Supported_Frameworks_Layers.md) for the list of supported standard layers.

## Frequently Asked Questions (FAQ)

The Model Optimizer provides explanatory messages if it is unable to run to completion due to issues like typographical errors, incorrectly used options, or other issues. The message describes the potential cause of the problem and gives a link to the [Model Optimizer FAQ](../Model_Optimizer_FAQ.md). The FAQ has instructions on how to resolve most issues. The FAQ also includes links to relevant sections in the Model Optimizer Developer Guide to help you understand what went wrong.

## Summary

In this document, you learned:

* Basic information about how the Model Optimizer works with MXNet\* models
* Which MXNet\* models are supported
* How to convert a trained MXNet\* model using the Model Optimizer with both framework-agnostic and MXNet-specific command-line options
