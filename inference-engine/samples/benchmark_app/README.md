# Benchmark Application Demo

This topic demonstrates how to use the Benchmark Application to estimate deep learning inference performance on supported devices. Performance can be measured for two inference modes: synchronous and asynchronous. 

> **NOTE:** This topic describes usage of C++ implementation of the Benchmark Application. For the Python* implementation, refer to [Benchmark Application (Python*)](./samples/python_samples/benchmark_app/README.md)


## How It Works

> **NOTE:** To achieve benchmark results similar to the official published results, set CPU frequency to 2.9GHz and GPU frequency to 1GHz.

Upon the start-up, the application reads command-line parameters and loads a network and images to the Inference Engine plugin. The number of infer requests and execution approach depend on a mode defined with the `-api` command-line parameter.


### Synchronous API
For synchronous mode, the primary metric is latency. The application creates one infer request and executes the `Infer` method. A number of executions is defined by one of the two values:
* Number of iterations defined with the `-niter` command-line argument
* Predefined duration if `-niter` is skipped. Predefined duration value depends on device.

During the execution, the application collects two types of metrics:
* Latency for each infer request executed with `Infer` method
* Duration of all executions

Reported latency value is calculated as mean value of all collected latencies. Reported throughput value is a derivative from reported latency and additionally depends on batch size.

### Asynchronous API
For asynchronous mode, the primary metric is throughput in frames per second (FPS). The application creates a certain number of infer requests and executes the `StartAsync` method. A number of infer is specified with the `-nireq` command-line parameter. A number of executions is defined by one of the two values:
* Number of iterations defined with the `-niter` command-line argument
* Predefined duration if `-niter` is skipped. Predefined duration value depends on device.

The infer requests are executed asynchronously. `Wait` method is used to wait for previous execution to complete. The application measures all infer requests executions and reports the throughput metric based on batch size and total execution duration.

## Running

Running the application with the `-h` option yields the following usage message:
```sh
./benchmark_app -h
InferenceEngine:
        API version ............ <version>
        Build .................. <number>
[ INFO ] Parsing input parameters

benchmark_app [OPTION]
Options:

    -h                      Print a usage message
    -i "<path>"             Required. Path to a folder with images or to image files.
    -m "<path>"             Required. Path to an .xml file with a trained model.
    -pp "<path>"            Path to a plugin folder.
    -api "<sync/async>"     Required. Enable using sync/async API.
    -d "<device>"           Specify a target device to infer on: CPU, GPU, FPGA or MYRIAD. Use "-d HETERO:<comma separated devices list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -niter "<integer>"      Optional. Number of iterations. If not specified, the number of iterations is calculated depending on a device.
    -nireq "<integer>"      Optional. Number of infer requests (default value is 2).
    -l "<absolute_path>"    Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.
          Or
    -c "<absolute_path>"    Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.
    -b "<integer>"          Optional. Batch size value. If not specified, the batch size value is determined from IR.
```

Running the application with the empty list of options yields the usage message given above and an error message.

You can run the application for one input layer four-dimensional models that support images as input, for example, public 
AlexNet and GoogLeNet models that can be downloaded 
with the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader).

> **NOTE**: To run the application, the model should be first converted to the Inference Engine format (\*.xml + \*.bin) 
using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

For example, to perform inference on CPU in the synchronous mode and get estimated performance metrics for AlexNet model, run the following command:

```sh
./benchmark_app -i <path_to_image>/inputImage.bmp -m <path_to_model>/alexnet_fp32.xml -d CPU -api sync
```

For the asynchronous mode:
```sh
./benchmark_app -i <path_to_image>/inputImage.bmp -m <path_to_model>/alexnet_fp32.xml -d CPU -api async
```


## Demo Output

Application output depends on a used API. For synchronous API, the application outputs latency and throughput:
```
[ INFO ] Start inference synchronously (60000 ms duration)

[ INFO ] Latency: 37.91 ms
[ INFO ] Throughput: 52.7566 FPS
```

For asynchronous API, the application outputs only throughput:
```
[ INFO ] Start inference asynchronously (60000 ms duration, 2 inference requests in parallel)

[ INFO ] Throughput: 48.2031 FPS
```

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
* [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader)
