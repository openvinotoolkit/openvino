# Benchmark C++ Application

This topic demonstrates how to use the Benchmark Application to estimate deep learning inference performance on
supported devices. Performance can be measured for two inference modes: synchronous (latency-oriented) and asynchronous (throughput-oriented).

> **NOTE:** This topic describes usage of C++ implementation of the Benchmark Application. For the Python* implementation, refer to [Benchmark Application (Python*)](./inference-engine/ie_bridges/python/sample/benchmark_app/README.md).


## How It Works

Upon start-up, the application reads command-line parameters and loads a network and images/binary files to the Inference Engine
plugin, which is chosen depending on a specified device. The number of infer requests and execution approach depend
on the mode defined with the `-api` command-line parameter.

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](./docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

If you run the application in the synchronous mode, it creates one infer request and executes the `Infer` method.
If you run the application in the asynchronous mode, it creates as many infer requests as specified in the `-nireq` command-line parameter and executes the `StartAsync` method for each of them. If `-nireq` is not set, the demo will use the default value for specified device.

A number of execution steps is defined by one of the following parameters:
* Number of iterations specified with the `-niter` command-line argument
* Time duration specified with the `-t` command-line argument
* Both of them (execution will continue until both conditions are met)
* Predefined duration if `-niter` and `-t` are not specified. Predefined duration value depends on a device.

During the execution, the application collects latency for each executed infer request.

Reported latency value is calculated as a median value of all collected latencies. Reported throughput value is reported
in frames per second (FPS) and calculated as a derivative from:
* Reported latency in the Sync mode
* The total execution time in the Async mode

Throughput value also depends on batch size.

The application also collects per-layer Performance Measurement (PM) counters for each executed infer request if you
enable statistics dumping by setting the `-report_type` parameter to one of the possible values:
* `no_counters` report includes configuration options specified, resulting FPS and latency.
* `average_counters` report extends the `no_counters` report and additionally includes average PM counters values for each layer from the network.
* `detailed_counters` report extends the `average_counters` report and additionally includes per-layer PM counters and latency for each executed infer request.

Depending on the type, the report is stored to `benchmark_no_counters_report.csv`, `benchmark_average_counters_report.csv`,
or `benchmark_detailed_counters_report.csv` file located in the path specified in `-report_folder`.

The application also saves executable graph information serialized to a XML file if you specify a path to it with the
`-exec_graph_path` parameter.


## Running
Notice that the benchmark_app usually produces optimal performance for any device out of the box.

**So in most cases you don't need to play the app options explicitly and the plain device name is enough**, e.g.:
```
$benchmark_app -m <model> -i <input> -d CPU
```
As explained in the  [Introduction to Performance Topics](./docs/IE_DG/Intro_to_Performance.md) section, it is preferable to use the FP16 IR for the model.

Running the application with the `-h` option yields the following usage message:
```
./benchmark_app -h
InferenceEngine:
        API version ............ <version>
        Build .................. <number>
[ INFO ] Parsing input parameters

benchmark_app [OPTION]
Options:

    -h, --help                Print a usage message
    -i "<path>"               Optional. Path to a folder with images and/or binaries or to specific image or binary file.
    -m "<path>"               Required. Path to an .xml file with a trained model.
    -d "<device>"             Optional. Specify a target device to infer on (the list of available devices is shown below). Default value is CPU.
                              Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin.
    The application looks for a suitable plugin for the specified device.
    -l "<absolute_path>"      Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.
          Or
    -c "<absolute_path>"      Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.
    -api "<sync/async>"       Optional. Enable Sync/Async API. Default value is "async".
    -niter "<integer>"        Optional. Number of iterations. If not specified, the number of iterations is calculated depending on a device.
    -nireq "<integer>"        Optional. Number of infer requests. Default value is determined automatically for a device.
    -b "<integer>"            Optional. Batch size value. If not specified, the batch size value is determined from Intermediate Representation.
    -stream_output            Optional. Print progress as a plain text. When specified, an interactive progress bar is replaced with a multiline output.
    -t                        Optional. Time in seconds to execute topology.
    -progress                 Optional. Show progress bar (can affect performance measurement). Default values is "false".

  CPU-specific performance options:
    -nstreams "<integer>"     Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode
                              (for HETERO device case use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).
    -nthreads "<integer>"     Optional. Number of threads to use for inference on the CPU (including HETERO case).
    -pin "YES"/"NO"           Optional. Enable ("YES" is default value) or disable ("NO") CPU threads pinning for CPU-involved inference.

  Statistics dumping options:
    -report_type "<type>"     Optional. Enable collecting statistics report. "no_counters" report contains configuration options specified, resulting FPS and latency. "average_counters" report extends "no_counters" report and additionally includes average PM counters values for each layer from the network. "detailed_counters" report extends "average_counters" report and additionally includes per-layer PM counters and latency for each executed infer request.
    -report_folder            Optional. Path to a folder where statistics report is stored.
    -exec_graph_path          Optional. Path to a file where to store executable graph information serialized.
    -pc                       Optional. Report performance counters.
```

Running the application with the empty list of options yields the usage message given above and an error message.

Application supports topologies with one or more inputs. If a topology is not data sensitive, you can skip the input parameter. In this case, inputs are filled with random values.
If a model has only image input(s), please a provide folder with images or a path to an image as input.
If a model has some specific input(s) (not images), please prepare a binary file(s), which is filled with data of appropriate precision and provide a path to them as input.
If a model has mixed input types, input folder should contain all required files. Image inputs are filled with image files one by one. Binary inputs are filled with binary inputs one by one.

To download the pre-trained models, use the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

For example, to perform inference on CPU in the synchronous mode and get estimated performance metrics for AlexNet model,
run the following command:

```sh
./benchmark_app -i <path_to_image>/inputImage.bmp -m <path_to_model>/alexnet_fp32.xml -d CPU -api sync
```

For the asynchronous mode:
```sh
./benchmark_app -i <path_to_image>/inputImage.bmp -m <path_to_model>/alexnet_fp32.xml -d CPU -api async
```

## Demo Output

The application outputs the number of executed iterations, total duration of execution, latency and throughput.
Additionally, if you set the `-report_type` parameter, the application outputs statistics report.
If you set the `-pc` parameter, the application outputs performance counters.
If you set `-exec_graph_path`, the application reports executable graph information serialized.

```
[Step 8/9] Measuring performance (Start inference asyncronously, 60000 ms duration, 4 inference requests in parallel using 4 streams)
Progress: [....................] 100.00% done

[Step 9/9] Dumping statistics report
[ INFO ] Statistics collecting was not requested. No reports are dumped.
Progress: [....................] 100.00% done

Count:      4612 iterations
Duration:   60110.04 ms
Latency:    50.99 ms
Throughput: 76.73 FPS

```

All measurements including per-layer PM counters are reported in milliseconds.


## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
* [Model Optimizer](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader)
