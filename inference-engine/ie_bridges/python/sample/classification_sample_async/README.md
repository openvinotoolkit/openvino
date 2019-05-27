# Image Classification Python* Sample Async

This sample demonstrates how to build and execute inference in pipelined mode on example of classifications networks.

The pipelined mode might increase the throughput of the pictures. The latency of one inference will be the same as for synchronous execution.
<br>
The throughput increases due to follow reasons:
* Some plugins have heterogeneity inside themselves: data transferring, execution on remote device, pre-processing and post-processing on the host.
* Using of explicit heterogeneous plugin with execution of different parts of network on different devices, for example HETERO:CPU,GPU.

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

Running the application with the <code>-h</code> option yields the following usage message:
```
python3 classification_sample_async.py -h
```
The command yields the following usage message:
```
usage: classification_sample_async.py [-h] -m MODEL -i INPUT [INPUT ...]
                                      [-l CPU_EXTENSION] [-pp PLUGIN_DIR]
                                      [-d DEVICE] [--labels LABELS]
                                      [-nt NUMBER_TOP] [-ni NUMBER_ITER] [-pc]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Required. Path to a folder with images or path to an
                        image files
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. Required for CPU custom layers. Absolute
                        path to a shared library with the kernels
                        implementations.
  -pp PLUGIN_DIR, --plugin_dir PLUGIN_DIR
                        Optional. Path to a plugin folder
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The sample
                        will look for a suitable plugin for device specified.
                        Default value is CPU
  --labels LABELS       Optional. Labels mapping file
  -nt NUMBER_TOP, --number_top NUMBER_TOP
                        Optional. Number of top results
  -ni NUMBER_ITER, --number_iter NUMBER_ITER
                        Optional. Number of inference iterations
  -pc, --perf_counts    Optional. Report performance counters

```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the sample, you can use AlexNet and GoogLeNet or other image classification models. You can download the pre-trained models with the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader) or from [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).


You can do inference on an image using a trained AlexNet network on FPGA with fallback to CPU using the following command:
```
    python3 classification_sample_async.py -i <path_to_image>/cat.bmp -m <path_to_model>/alexnet_fp32.xml -nt 5 -d HETERO:FPGA,CPU -nireq 2 -ni 200
```

## Sample Output

By default, the application outputs top-10 inference results for each infer request.
It also provides throughput value measured in frames per seconds.

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
