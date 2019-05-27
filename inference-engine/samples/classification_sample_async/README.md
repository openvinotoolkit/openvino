# Image Classification C++ Sample Async

This sample demonstrates how to build and execute inference in pipelined mode on example of classifications networks.

> **NOTE:** This topic describes usage of C++ implementation of the Image Classification Sample Async. For the Python* implementation, refer to [Image Classification Python* Sample Async](./inference-engine/ie_bridges/python/sample/classification_sample_async/README.md).

The pipelined mode might increase the throughput of the pictures. The latency of one inference will be the same as for synchronous execution.

The throughput increases due to follow reasons:
* Some plugins have heterogeneity inside themselves. Data transferring, execution on remote device, pre-processing and post-processing on the host
* Using of explicit heterogeneous plugin with execution of different parts of network on different devices

When two or more devices process one image, creating several infer requests and starting asynchronous inference allow for using devices in the most efficient way.
If two devices are involved in execution, the most optimal value for `-nireq` option is 2.

To process infer requests more efficiently, Classification Sample Async uses round-robin algorithm. It starts execution of the current infer request and switches to waiting for results of the previous one. After finishing of waiting, it switches infer requests and repeat the procedure.

Another required aspect of good throughput is a number of iterations. Only with big number of iterations you can emulate the real application work and get good performance.

The batch mode is an independent attribute on the pipelined mode. Pipelined mode works efficiently with any batch size.

## How It Works

Upon the start-up, the sample application reads command line parameters and loads a network and an image to the Inference
Engine plugin.
Then application creates several infer requests pointed in `-nireq` parameter and loads images for inference.

Then in a loop it starts inference for the current infer request and switches to waiting for the previous one. When results are ready, it swaps infer requests.

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

    -h                      
                            Print a usage message.
    -i "<path1>" "<path2>"
                            Required. Path to a folder with images or path to an image files: a .ubyte file for LeNet
                            and a .bmp file for the other networks.
    -m "<path>"             
                            Required. Path to an .xml file with a trained model.
        -l "<absolute_path>"
                            Required for CPU. Absolute path to a shared library with the kernel implementations
        Or
        -c "<absolute_path>"
                            Required for GPU custom kernels. Absolute path to the .xml file with kernel descriptions
    -pp "<path>"            
                            Optional. Path to a plugin folder.
    -d "<device>"           
                            Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified. Default value is "CPU".
    -nt "<integer>"         
                            Optional. Number of top results. Default value is 10.
    -ni "<integer>"         
                            Optional. Number of iterations. Default value is 1.
    -pc                     
                            Optional. Enables per-layer performance report
    -nireq "<integer>"
                            Optional. Number of infer request for pipelined mode. Default value is 1.
    -p_msg                  
                            Optional. Enables messages from a plugin
    -nthreads "<integer>"
                            Optional. Number of threads to use for inference on the CPU (including HETERO cases)
    -pin "YES"/"NO"
                            Optional. Enable ("YES", default) or disable ("NO") CPU threads pinning for CPU-involved inference
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the sample, use AlexNet and GoogLeNet or other public or pre-trained image classification models. To download the pre-trained models, use the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

You can do inference on an image using a trained AlexNet network on FPGA with fallback to CPU using the following command:
```sh
./classification_sample_async -i <path_to_image>/cat.bmp -m <path_to_model>/alexnet_fp32.xml -nt 5 -d HETERO:FPGA,CPU -nireq 2 -ni 200
```

## Sample Output

By default the application outputs top-10 inference results for each infer request.
In addition to this information it will provide throughput value measured in frames per seconds.

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader)
* [Model Optimizer](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
