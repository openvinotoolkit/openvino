# Image Classification Sample

This topic demonstrates how to run the Image Classification sample application, which performs 
inference using image classification networks such as AlexNet and GoogLeNet.

## Running

Running the application with the <code>-h</code> option yields the following usage message:
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
        -l "<absolute_path>"  Optional. Absolute path to library with MKL-DNN (CPU) custom layers (*.so).
        Or
        -c "<absolute_path>"  Optional. Absolute path to clDNN (GPU) custom layers config (*.xml).
    -pp "<path>"              Path to a plugin folder.
    -d "<device>"             Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified
    -nt "<integer>"           Number of top results (default 10)
    -ni "<integer>"           Number of iterations (default 1)
    -pc                       Enables per-layer performance report
    -p_msg                    Enables messages from a plugin

```

Running the application with the empty list of options yields the usage message given above.

To run the sample you can use AlexNet and GoogLeNet models that can be downloaded with the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader) or other image classification models. 

> **IMPORTANT**: To run the sample, the model should be first converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

For example, to perform inference of an AlexNet model (previously converted to the Inference Engine format) on CPU, use the following command:

```sh
./classification_sample -i <path_to_image>/cat.bmp -m <path_to_model>/alexnet_fp32.xml
```

### Outputs

By default the application outputs top-10 inference results. 
Add the <code>-nt</code> option to the previous command to modify the number of top output results.
<br>For example, to get the top-5 results on Intel&reg; HD Graphics, use the following commands:
```sh
./classification_sample -i <path_to_image>/cat.bmp -m <path_to_model>/alexnet_fp32.xml -nt 5 -d GPU
```

### How it works

Upon the start-up the sample application reads command line parameters and loads a network and an image to the Inference 
Engine plugin. When inference is done, the application creates an 
output image and outputs data to the standard output stream.

## See Also 
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
* [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader)