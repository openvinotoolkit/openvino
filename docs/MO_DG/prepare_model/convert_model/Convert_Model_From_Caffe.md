# Converting a Caffe* Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe}

A summary of the steps for optimizing and deploying a model that was trained with Caffe\*:

1. [Configure the Model Optimizer](../Config_Model_Optimizer.md) for Caffe\*.
2. [Convert a Caffe\* Model](#Convert_From_Caffe) to produce an optimized [Intermediate Representation (IR)](../../IR_and_opsets.md) of the model based on the trained network topology, weights, and biases values
3. Test the model in the Intermediate Representation format using the [Inference Engine](../../../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) in the target environment via provided Inference Engine [sample applications](../../../IE_DG/Samples_Overview.md)
4. [Integrate](../../../IE_DG/Samples_Overview.md) the [Inference Engine](../../../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) in your application to deploy the model in the target environment

## Supported Topologies

* **Classification models:**
	* AlexNet
	* VGG-16, VGG-19
	* SqueezeNet v1.0, SqueezeNet v1.1
	* ResNet-50, ResNet-101, Res-Net-152
	* Inception v1, Inception v2, Inception v3, Inception v4
	* CaffeNet
	* MobileNet
	* Squeeze-and-Excitation Networks: SE-BN-Inception, SE-Resnet-101, SE-ResNet-152, SE-ResNet-50, SE-ResNeXt-101, SE-ResNeXt-50
	* ShuffleNet v2

* **Object detection models:**
	* SSD300-VGG16, SSD500-VGG16
	* Faster-RCNN
	* RefineDet (Myriad plugin only)

* **Face detection models:**
	* VGG Face
    * SSH: Single Stage Headless Face Detector

* **Semantic segmentation models:**
	* FCN8

> **NOTE:** It is necessary to specify mean and scale values for most of the Caffe\* models to convert them with the Model Optimizer. The exact values should be determined separately for each model. For example, for Caffe\* models trained on ImageNet, the mean values usually are `123.68`, `116.779`, `103.939` for blue, green and red channels respectively. The scale value is usually `127.5`. Refer to [Framework-agnostic parameters](Converting_Model_General.md) for the information on how to specify mean and scale values.

## Convert a Caffe* Model <a name="Convert_From_Caffe"></a>

To convert a Caffe\* model:

1. Go to the `<INSTALL_DIR>/deployment_tools/model_optimizer` directory.
2. Use the `mo.py` script to simply convert a model with the path to the input model `.caffemodel` file:
```sh
python3 mo.py --input_model <INPUT_MODEL>.caffemodel
```

Two groups of parameters are available to convert your model:

* [Framework-agnostic parameters](Converting_Model_General.md): These parameters are used to convert a model trained with any supported framework.
* [Caffe-specific parameters](#caffe_specific_conversion_params): Parameters used to convert only Caffe\* models

### Using Caffe\*-Specific Conversion Parameters <a name="caffe_specific_conversion_params"></a>

The following list provides the Caffe\*-specific parameters.

```
Caffe*-specific parameters:
  --input_proto INPUT_PROTO, -d INPUT_PROTO
                        Deploy-ready prototxt file that contains a topology
                        structure and layer attributes
  --caffe_parser_path CAFFE_PARSER_PATH
                        Path to python Caffe parser generated from caffe.proto
  -k K                  Path to CustomLayersMapping.xml to register custom
                        layers
  --mean_file MEAN_FILE, -mf MEAN_FILE
                        Mean image to be used for the input. Should be a
                        binaryproto file
  --mean_file_offsets MEAN_FILE_OFFSETS, -mo MEAN_FILE_OFFSETS
                        Mean image offsets to be used for the input
                        binaryproto file. When the mean image is bigger than
                        the expected input, it is cropped. By default, centers
                        of the input image and the mean image are the same and
                        the mean image is cropped by dimensions of the input
                        image. The format to pass this option is the
                        following: "-mo (x,y)". In this case, the mean file is
                        cropped by dimensions of the input image with offset
                        (x,y) from the upper left corner of the mean image
  --disable_omitting_optional
                        Disable omitting optional attributes to be used for
                        custom layers. Use this option if you want to transfer
                        all attributes of a custom layer to IR. Default
                        behavior is to transfer the attributes with default
                        values and the attributes defined by the user to IR.
  --enable_flattening_nested_params
                        Enable flattening optional params to be used for
                        custom layers. Use this option if you want to transfer
                        attributes of a custom layer to IR with flattened
                        nested parameters. Default behavior is to transfer the
                        attributes without flattening nested parameters.
```

#### Command-Line Interface (CLI) Examples Using Caffe\*-Specific Parameters

* Launching the Model Optimizer for the [bvlc_alexnet.caffemodel](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) with a specified `prototxt` file. This is needed when the name of the Caffe\* model and the `.prototxt` file are different or are placed in different directories. Otherwise, it is enough to provide only the path to the input `model.caffemodel` file.
```sh
python3 mo.py --input_model bvlc_alexnet.caffemodel --input_proto bvlc_alexnet.prototxt
```

* Launching the Model Optimizer for the [bvlc_alexnet.caffemodel](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) with a specified `CustomLayersMapping` file. This is the legacy method of quickly enabling model conversion if your model has custom layers. This requires system Caffe\* on the computer. To read more about this, see [Legacy Mode for Caffe* Custom Layers](../customize_model_optimizer/Legacy_Mode_for_Caffe_Custom_Layers.md).
Optional parameters without default values and not specified by the user in the `.prototxt` file are removed from the Intermediate Representation, and nested parameters are flattened:
```sh
python3 mo.py --input_model bvlc_alexnet.caffemodel -k CustomLayersMapping.xml --disable_omitting_optional --enable_flattening_nested_params
```
		This example shows a multi-input model with input layers: `data`, `rois`
```
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape { dim: 1 dim: 3 dim: 224 dim: 224 }
  }
}
layer {
  name: "rois"
  type: "Input"
  top: "rois"
  input_param {
    shape { dim: 1 dim: 5 dim: 1 dim: 1 }
  }
}
```

* Launching the Model Optimizer for a multi-input model with two inputs and providing a new shape for each input in the order they are passed to the Model Optimizer. In particular, for data, set the shape to `1,3,227,227`. For rois, set the shape to `1,6,1,1`:
```sh
python3 mo.py --input_model /path-to/your-model.caffemodel --input data,rois --input_shape (1,3,227,227),[1,6,1,1]
```

## Custom Layer Definition

Internally, when you run the Model Optimizer, it loads the model, goes through the topology, and tries to find each layer type in a list of known layers. Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in this list of known layers, the Model Optimizer classifies them as custom.

## Supported Caffe\* Layers
Refer to [Supported Framework Layers](../Supported_Frameworks_Layers.md) for the list of supported standard layers.

## Frequently Asked Questions (FAQ)

The Model Optimizer provides explanatory messages if it is unable to run to completion due to issues like typographical errors, incorrectly used options, or other issues. The message describes the potential cause of the problem and gives a link to the [Model Optimizer FAQ](../Model_Optimizer_FAQ.md). The FAQ has instructions on how to resolve most issues. The FAQ also includes links to relevant sections in the Model Optimizer Developer Guide to help you understand what went wrong.

## Summary

In this document, you learned:

* Basic information about how the Model Optimizer works with Caffe\* models
* Which Caffe\* models are supported
* How to convert a trained Caffe\* model using the Model Optimizer with both framework-agnostic and Caffe-specific command-line options
