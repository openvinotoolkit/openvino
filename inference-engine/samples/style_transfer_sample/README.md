# Neural Style Transfer C++ Sample

This topic demonstrates how to run the Neural Style Transfer sample application, which performs
inference of style transfer models.

> **NOTE**: The OpenVINO™ toolkit does not include a pre-trained model to run the Neural Style Transfer sample. A public model from the [Zhaw's Neural Style Transfer repository](https://github.com/zhaw/neural_style) can be used. Read the [Converting a Style Transfer Model from MXNet*](./docs/MO_DG/prepare_model/convert_model/mxnet_specific/Convert_Style_Transfer_From_MXNet.md) topic from the [Model Optimizer Developer Guide](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) to learn about how to get the trained model and how to convert it to the Inference Engine format (\*.xml + \*.bin).

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](./docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
./style_transfer_sample --help
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

style_transfer_sample [OPTION]
Options:

    -h                      Print a usage message
    -i "<path>"             Required. Path to a .bmp image file or a sequence of paths separated by spaces.
    -m "<path>"             Required. Path to an .xml file with a trained model.
    -d "<device>"           The target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The sample looks for a suitable plugin for the device specified.
    -mean_val_r,
    -mean_val_g,
    -mean_val_b             Mean values. Required if the model needs mean values for preprocessing and postprocessing

```

Running the application with the empty list of options yields the usage message given above and an error message.

To perform inference of an image using a trained model of NST network on Intel® CPUs, use the following command:
```sh
./style_transfer_sample -i <path_to_image>/cat.bmp -m <path_to_model>/1_decoder_FP32.xml
```

## Sample Output

The application outputs an image (`out1.bmp`) or a sequence of images (`out1.bmp`, ..., `out<N>.bmp`) which are redrawn in style of the style transfer model used for sample.

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
* [Model Optimizer](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader)
