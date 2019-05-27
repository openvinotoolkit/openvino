# Benchmark Application C++ Demo

This topic demonstrates how to use the Benchmark Application to estimate deep learning inference performance on
supported devices. Performance can be measured for two inference modes: synchronous and asynchronous.

> **NOTE:** This topic describes usage of C++ implementation of the Benchmark Application. For the Python* implementation, refer to [Benchmark Application (Python*)](./inference-engine/ie_bridges/python/sample/benchmark_app/README.md).


## How It Works

> **NOTE:** To achieve benchmark results similar to the official published results, set CPU frequency to 2.9 GHz and GPU frequency to 1 GHz.

Upon start-up, the application reads command-line parameters and loads a network and images to the Inference Engine
plugin, which is chosen depending on a specified device. The number of infer requests and execution approach depend
on the mode defined with the `-api` command-line parameter.

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](./docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

If you run the application in the synchronous mode, it creates one infer request and executes the `Infer` method.
If you run the application in the asynchronous mode, it creates as many infer requests as specified in the `-nireq`
command-line parameter and executes the `StartAsync` method for each of them.

The `Wait` method is used to wait for a previous execution of an infer request to complete. A number of execution steps
is defined by one of the two values:
* Number of iterations specified with the `-niter` command-line argument
* Predefined duration if `-niter` is not specified. Predefined duration value depends on device.

During the execution, the application collects latency for each executed infer request.

Reported latency value is calculated as a median value of all collected latencies. Reported throughput value is reported
in frames per second (FPS) and calculated as a derivative from:
* Reported latency in the Sync mode
* The total execution time in the Async mode

Throughput value also depends on batch size.

The application also collects per-layer Performance Measurement (PM) counters for each executed infer request if you
enable statistics dumping by setting the `-report_type` parameter to one of the possible values:
* `no_counters` report includes configuration options specified, resulting FPS and latency.
* `median_counters` report extends the `no_counters` report and additionally includes median PM counters values for each layer from the network.
* `detailed_counters` report extends the `median_counters` report and additionally includes per-layer PM counters and latency for each executed infer request.

Depending on the type, the report is stored to `benchmark_no_counters_report.csv`, `benchmark_median_counters_report.csv`,
or `benchmark_detailed_counters_report.csv` file located in the path specified in `-report_folder`.

The application also saves executable graph information serialized to a XML file if you specify a path to it with the
`-exec_graph_path` parameter.


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

    -h                        Print a usage message
    -i "<path>"               Required. Path to a folder with images or to image files.
    -m "<path>"               Required. Path to an .xml file with a trained model.
    -pp "<path>"              Optional. Path to a plugin folder.
    -d "<device>"             Optional. Specify a target device to infer on: CPU, GPU, FPGA, HDDL or MYRIAD. Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -l "<absolute_path>"      Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.
          Or
    -c "<absolute_path>"      Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.
    -api "<sync/async>"       Optional. Enable Sync/Async API. Default value is "async".
    -niter "<integer>"        Optional. Number of iterations. If not specified, the number of iterations is calculated depending on a device.
    -nireq "<integer>"        Optional. Number of infer requests. Default value is 2.
    -b "<integer>"            Optional. Batch size value. If not specified, the batch size value is determined from Intermediate Representation.
    -stream_output            Optional. Print progress as a plain text. When specified, an interactive progress bar is replaced with a multiline output.

  CPU-specific performance options:
    -nthreads "<integer>"     Optional. Number of threads to use for inference on the CPU (including HETERO cases).
    -pin "YES"/"NO"           Optional. Enable ("YES" is default value) or disable ("NO") CPU threads pinning for CPU-involved inference.

  Statistics dumping options:
    -report_type "<type>"     Optional. Enable collecting statistics report. "no_counters" report contains configuration options specified, resulting FPS and latency. "median_counters" report extends "no_counters" report and additionally includes median PM counters values for each layer from the network. "detailed_counters" report extends "median_counters" report and additionally includes per-layer PM counters and latency for each executed infer request.
    -report_folder            Optional. Path to a folder where statistics report is stored.
    -exec_graph_path          Optional. Path to a file where to store executable graph information serialized.
```

Running the application with the empty list of options yields the usage message given above and an error message.

You can run the application for one input layer four-dimensional models that support images as input, for example, public
AlexNet and GoogLeNet models. To download the pre-trained models, use the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

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

The application outputs latency and throughput. Additionally, if you set the `-report_type` parameter, the application
outputs statistics report. If you set `-exec_graph_path`, the application reports executable graph information serialized.
Progress bar shows the progress of each execution step:

```
[Step 7/8] Start inference asynchronously (100 async inference executions, 4 inference requests in parallel)
Progress: [....................] 100.00% done

[Step 8/8] Dump statistics report
[ INFO ] statistics report is stored to benchmark_detailed_counters_report.csv
Progress: [....................] 100.00% done

Latency: 73.33 ms
Throughput: 53.28 FPS
```

All measurements including per-layer PM counters are reported in milliseconds.


## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
* [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader)
