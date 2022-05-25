# Converting a Caffe Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe}

<a name="Convert_From_Caffe"></a>To convert a Caffe model, run Model Optimizer with the path to the input model `.caffemodel` file:

```sh
 mo --input_model <INPUT_MODEL>.caffemodel
```

The following list provides the Caffe-specific parameters.

```
Caffe-specific parameters:
  --input_proto INPUT_PROTO, -d INPUT_PROTO
                        Deploy-ready prototxt file that contains a topology
                        structure and layer attributes
  --caffe_parser_path CAFFE_PARSER_PATH
                        Path to python Caffe parser generated from caffe.proto
  -k K                  Path to CustomLayersMapping.xml to register custom
                        layers
  --mean_file MEAN_FILE, -mf MEAN_FILE
                        [DEPRECATED] Mean image to be used for the input. Should be a
                        binaryproto file
  --mean_file_offsets MEAN_FILE_OFFSETS, -mo MEAN_FILE_OFFSETS
                        [DEPRECATED] Mean image offsets to be used for the input
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

### CLI Examples Using Caffe-Specific Parameters

* Launching the Model Optimizer for the [bvlc_alexnet.caffemodel](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) with a specified `prototxt` file. This is needed when the name of the Caffe model and the `.prototxt` file are different or are placed in different directories. Otherwise, it is enough to provide only a path to the input `model.caffemodel` file.
   ```sh
   mo --input_model bvlc_alexnet.caffemodel --input_proto bvlc_alexnet.prototxt
   ```
* Launching the Model Optimizer for the [bvlc_alexnet.caffemodel](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) with a specified `CustomLayersMapping` file. This is the legacy method of quickly enabling model conversion if your model has custom layers. This requires the Caffe system on the computer.
Optional parameters without default values and not specified by the user in the `.prototxt` file are removed from the Intermediate Representation, and nested parameters are flattened:
   ```sh
   mo --input_model bvlc_alexnet.caffemodel -k CustomLayersMapping.xml --disable_omitting_optional --enable_flattening_nested_params
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
   mo --input_model /path-to/your-model.caffemodel --input data,rois --input_shape (1,3,227,227),[1,6,1,1]
   ```
## Custom Layer Definition

Internally, when you run the Model Optimizer, it loads the model, goes through the topology, and tries to find each layer type in a list of known layers. Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in this list of known layers, the Model Optimizer classifies them as custom.

## Supported Caffe Layers
For the list of supported standard layers, refer to the [Supported Framework Layers](../Supported_Frameworks_Layers.md) page.

## Frequently Asked Questions (FAQ)

The Model Optimizer provides explanatory messages if it is unable to run to completion due to typographical errors, incorrectly used options, or other issues. The message describes the potential cause of the problem and gives a link to the [Model Optimizer FAQ](../Model_Optimizer_FAQ.md). The FAQ provides instructions on how to resolve most issues. The FAQ also includes links to relevant sections in the Model Optimizer Developer Guide to help you understand what went wrong.

## Summary

In this document, you learned:

* Basic information about how the Model Optimizer works with Caffe models
* Which Caffe models are supported
* How to convert a trained Caffe model using the Model Optimizer with both framework-agnostic and Caffe-specific command-line options

## See Also
[Model Conversion Tutorials](Convert_Model_Tutorials.md)
