# Image Classification C++ Sample

This topic demonstrates how to run the Image Classification sample application, which performs
inference using image classification networks such as AlexNet and GoogLeNet.

> **NOTE:** This topic describes usage of C++ implementation of the Image Classification Sample. For the Python* implementation, refer to [Image Classification Python* Sample](./inference-engine/ie_bridges/python/sample/classification_sample/README.md).

## How It Works

Upon the start-up, the sample application reads command line parameters and loads a network and an image to the Inference
Engine plugin. When inference is done, the application creates an
output image and outputs data to the standard output stream.

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Specify Input Shapes** section of [Converting a Model Using General Conversion Parameters](./docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

## Running
Running the application with the `-h` option yields the following usage message:
```sh
./classification_sample -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

classification_sample [OPTION]
Options:

    -h                        Print a usage message.
    -i "<path1>" "<path2>"    Required. Path to a folder with images or path to an image files: a .ubyte file for LeNet
                              and a .bmp file for the other networks.
    -m "<path>"               Required. Path to an .xml file with a trained model.
        -l "<absolute_path>"  Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.
        Or
        -c "<absolute_path>"  Required for GPU custom kernels. Absolute path to the .xml file with the kernels descriptions.
    -pp "<path>"              Path to a plugin folder.
    -d "<device>"             Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified
    -nt "<integer>"           Number of top results. Default value is 10
    -ni "<integer>"           Number of iterations. Default value is 1
    -pc                       Enables per-layer performance report
    -p_msg                    Enables messages from a plugin

```

Running the application with the empty list of options yields the usage message given above.

To run the sample, you can use AlexNet and GoogLeNet or other public or pre-trained image classification models. To download the pre-trained models, use the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

For example, to perform inference of an AlexNet model on CPU, use the following command:

```sh
./classification_sample -i <path_to_image>/cat.bmp -m <path_to_model>/alexnet_fp32.xml
```

## Demo Output

By default the application outputs top-10 inference results.
Add the `-nt` option to the previous command to modify the number of top output results.

For example, to get the top-5 results on GPU, use the following commands:
```sh
./classification_sample -i <path_to_image>/cat.bmp -m <path_to_model>/alexnet_fp32.xml -nt 5 -d GPU
```

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
* [Model Optimizer](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader)
