# Image Classification C++ Sample Async

This sample demonstrates how to run the Image Classification sample application with inference executed in the asynchronous mode.

> **NOTE:** This topic describes usage of C++ implementation of the Image Classification Sample Async. For the Python* implementation, refer to [Image Classification Python* Sample Async](./inference-engine/ie_bridges/python/sample/classification_sample_async/README.md).

The sample demonstrates how to use the new Infer Request API of Inference Engine in applications.
Refer to [Integrate the Inference Engine New Request API with Your Application](./docs/IE_DG/Integrate_with_customer_application_new_API.md) for details.
The sample demonstrates how to build and execute an inference request 10 times in the asynchronous mode on example of classifications networks.
The asynchronous mode might increase the throughput of the pictures.

The batch mode is an independent attribute on the asynchronous mode. Asynchronous mode works efficiently with any batch size.

## How It Works

Upon the start-up, the sample application reads command line parameters and loads specified network and input images (or a
folder with images) to the Inference Engine plugin. The batch size of the network is set according to the number of read images.

Then, the sample creates an inference request object and assigns completion callback for it. In scope of the completion callback
handling the inference request is executed again.

After that, the application starts inference for the first infer request and waits of 10th inference request execution being completed.

When inference is done, the application outputs data to the standard output stream.

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](./docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

## Running

Running the application with the `-h` option yields the following usage message:
```sh
./classification_sample_async -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

classification_sample_async [OPTION]
Options:

    -h                      Print a usage message.
    -i "<path>"             Required. Path to a folder with images or path to an image files: a .ubyte file for LeNetand a .bmp file for the other networks.
    -m "<path>"             Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"  Required for CPU custom layers.Absolute path to a shared library with the kernels implementation
          Or
      -c "<absolute_path>"  Required for GPU custom kernels.Absolute path to the .xml file with kernels description
    -d "<device>"           Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Sample will look for a suitable plugin for device specified.
    -nt "<integer>"         Optional. Number of top results. Default value is 10.
    -p_msg                  Optional. Enables messages from a plugin

```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the sample, use AlexNet and GoogLeNet or other public or pre-trained image classification models. To download the pre-trained models, use the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

You can do inference of an image using a trained AlexNet network on FPGA with fallback to CPU using the following command:
```sh
./classification_sample_async -i <path_to_image>/cat.bmp -m <path_to_model>/alexnet_fp32.xml -nt 5 -d HETERO:FPGA,CPU
```

## Sample Output

By default the application outputs top-10 inference results for each infer request.

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader)
* [Model Optimizer](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
