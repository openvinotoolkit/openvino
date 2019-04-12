# Benchmark Application Python* Demo

This topic demonstrates how to run the Benchmark Application demo, which performs inference using convolutional networks.

## How It Works

> **NOTE:** To achieve benchmark results similar to the official published results, set CPU frequency to 2.9GHz and GPU frequency to 1GHz.

Upon the start-up, the application reads command-line parameters and loads a network and images to the Inference Engine plugin. The number of infer requests and execution approach depend on a mode defined with the `-api` command-line parameter.

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Specify Input Shapes** section of [Converting a Model Using General Conversion Parameters](./docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

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

Running the application with the `-h` or `--help`' option yields the following usage message:
```python3 benchmark_app.py -h```

The command yields the following usage message:
```
   usage: benchmark_app.py [-h] -i PATH_TO_IMAGES -m PATH_TO_MODEL
                        [-c PATH_TO_CLDNN_CONFIG] [-l PATH_TO_EXTENSION]
                        [-api {sync,async}] [-d TARGET_DEVICE]
                        [-niter NUMBER_ITERATIONS]
                        [-nireq NUMBER_INFER_REQUESTS]
                        [-nthreads NUMBER_THREADS] [-b BATCH_SIZE]
                        [-pin {YES,NO}]

Options:
  -h, --help            Show this help message and exit.
  -i PATH_TO_IMAGES, --path_to_images PATH_TO_IMAGES
                        Required. Path to a folder with images or to image
                        files.
  -m PATH_TO_MODEL, --path_to_model PATH_TO_MODEL
                        Required. Path to an .xml file with a trained model.
  -c PATH_TO_CLDNN_CONFIG, --path_to_cldnn_config PATH_TO_CLDNN_CONFIG
                        Optional. Required for GPU custom kernels. Absolute
                        path to an .xml file with the kernels description.
  -l PATH_TO_EXTENSION, --path_to_extension PATH_TO_EXTENSION
                        Optional. Required for GPU custom kernels. Absolute
                        path to an .xml file with the kernels description.
  -api {sync,async}, --api_type {sync,async}
                        Optional. Enable using sync/async API. Default value
                        is sync
  -d TARGET_DEVICE, --target_device TARGET_DEVICE
                        Optional. Specify a target device to infer on: CPU,
                        GPU, FPGA, HDDL or MYRIAD. Use "-d HETERO:<comma
                        separated devices list>" format to specify HETERO
                        plugin. The application looks for a suitable plugin
                        for the specified device.
  -niter NUMBER_ITERATIONS, --number_iterations NUMBER_ITERATIONS
                        Optional. Number of iterations. If not specified, the
                        number of iterations is calculated depending on a
                        device.
  -nireq NUMBER_INFER_REQUESTS, --number_infer_requests NUMBER_INFER_REQUESTS
                        Optional. Number of infer requests (default value is
                        2).
  -nthreads NUMBER_THREADS, --number_threads NUMBER_THREADS
                        Number of threads to use for inference on the CPU
                        (including Hetero cases).
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Optional. Batch size value. If not specified, the
                        batch size value is determined from IR
  -pin {YES,NO}, --infer_threads_pinning {YES,NO}
                        Optional. Enable ("YES" is default value) or disable
                        ("NO")CPU threads pinning for CPU-involved inference.
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

For example, to do inference on an image using a trained network with multiple outputs on CPU, run the following command:

```
python3 benchmark_app.py -i <path_to_image>/inputImage.bmp -m <path_to_model>/multiple-output.xml -d CPU
```

## Demo Output

Application output depends on a used API. For synchronous API, the application outputs latency and throughput:
```
[ INFO ] Start inference synchronously (10 s duration)
[BENCHMARK RESULT] Latency is 15.5520 msec
[BENCHMARK RESULT] Throughput is 1286.0082 FPS
```

For asynchronous API, the application outputs only throughput:
```
[ INFO ] Start inference asynchronously (10 s duration, 8 inference requests in parallel)
[BENCHMARK RESULT] Throughput is 1444.2591 FPS
```

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
* [Model Optimizer](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader)
