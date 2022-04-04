# Benchmark C++ Tool {#openvino_inference_engine_samples_benchmark_app_README}

This topic demonstrates how to use the Benchmark C++ Tool to estimate deep learning inference performance on supported devices.
Performance can be measured for two inference modes: latency- and throughput-oriented.

> **NOTE:** This topic describes usage of C++ implementation of the Benchmark Tool. For the Python implementation, refer to [Benchmark Python Tool](../../../tools/benchmark_tool/README.md).


## How It Works

Upon start-up, the application reads command-line parameters and loads a network and inputs (images/binary files) to the specified device.

> **NOTE**: By default, OpenVINO™ Toolkit Samples, Tools and Demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application, or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Embedding Preprocessing Computation](../../../docs/MO_DG/prepare_model/convert_model/Converting_Model.md).

Device-specific execution parameters (number of streams, threads, and so on) can be either explicitly specified through the command line
or left default. In the last case, the sample logic will select the values for the optimal throughput.
Experimenting with individual parameters allows to find the performance sweet spot. However, the parameters are not very performance-portable,
so the values from one machine or device are not necessarily optimal for another.
From this perspective, the most portable way is experimenting only with the performance hints. To learn more, refer to the section on the command-line parameters below.

A number of execution steps is defined by one of the following parameters:
* Number of iterations specified with the `-niter` command-line argument
* Time duration specified with the `-t` command-line argument
* Both of them (execution will continue until both conditions are met)
* Predefined duration if `-niter` and `-t` are not specified. Predefined duration value depends on a device.

During the execution, the application calculates latency (if applicable) and overall throughput:
* By default, the median latency value is reported
* Throughput is calculated as overall_inference_time/number_of_processed_requests. The throughput value also depends on batch size.

The application also collects per-layer Performance Measurement (PM) counters for each executed infer request if you
enable statistics dumping by setting the `-report_type` parameter to one of the possible values:
* `no_counters` report includes configuration options specified, resulting FPS and latency.
* `average_counters` report extends the `no_counters` report and additionally includes average PM counters values for each layer from the network.
* `detailed_counters` report extends the `average_counters` report and additionally includes per-layer PM counters and latency for each executed infer request.

Depending on the type, the report is stored to `benchmark_no_counters_report.csv`, `benchmark_average_counters_report.csv`,
or `benchmark_detailed_counters_report.csv` file located in the path specified in `-report_folder`.

The application also saves executable graph information serialized to an XML file if you specify a path to it with the
`-exec_graph_path` parameter.


## Run the Tool

The benchmark_app usually produces optimal performance for any device out of the box.

**In most cases, you don't need to play the app options explicitly, and the plain device name is enough.** For example, for CPU:
```sh
./benchmark_app -m <model> -i <input> -d CPU
```

But it still may be sub-optimal for some cases, especially for very small networks. You will find more details in [Performance Optimization Guide](../../../docs/optimization_guide/dldt_optimization_guide.md).

As explained in the  [Performance Optimization Guide](../../../docs/optimization_guide/dldt_optimization_guide.md) section, for all devices, including new [MULTI device](../../../docs/OV_Runtime_UG/multi_device.md), it is preferable to use the FP16 IR for the model.
Also if latency of the CPU inference on the multi-socket machines is of concern, you might refer to the same
[Performance Optimization Guide](../../../docs/optimization_guide/dldt_optimization_guide.md).

Running the application with the `-h` option yields the following usage message:
```
./benchmark_app -h

benchmark_app [OPTION]
Options:

    -h, --help                Print a usage message
    -m "<path>"               Required. Path to an .xml/.onnx file with a trained model or to a .blob files with a trained compiled model.
    -i "<path>"               Optional. Path to a folder with images and/or binaries or to specific image or binary file.
                              In case of dynamic shapes networks with several inputs, provide the same number of files for each input (except cases with single file for any input):"input1:1.jpg input2:1.bin", "input1:1.bin,2.bin input2:3.bin input3:4.bin,5.bin ". Also you can pass specific keys for inputs: "random" - for fillling input with random data, "image_info" - for filling input with image size.
                              You should specify either one files set to be used for all inputs (without providing input names) or separate files sets for every input of model (providing inputs names).
    -d "<device>"             Optional. Specify a target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. Use "-d MULTI:<comma-separated_devices_list>" format to specify MULTI plugin. The application looks for a suitable plugin for the specified device.
    -l "<absolute_path>"      Required for CPU custom layers. Absolute path to a shared library with the implementations of kernels.
          Or
    -c "<absolute_path>"      Required for GPU custom kernels. Absolute path to an .xml file with the description of kernels.
    -hint "performance hint (latency or throughput or none)"   Optional. Performance hint allows the OpenVINO device to select the right network-specific settings.
                               'throughput' or 'tput': device performance mode will be set to THROUGHPUT.
                               'latency': device performance mode will be set to LATENCY.
                               'none': no device performance mode will be set.
                              Using explicit 'nstreams' or other device-specific options, please set hint to 'none'
    -api "<sync/async>"       Optional (deprecated). Enable Sync/Async API. Default value is "async".
    -niter "<integer>"        Optional. Number of iterations. If not specified, the number of iterations is calculated depending on a device.
    -nireq "<integer>"        Optional. Number of infer requests. Default value is determined automatically for device.
    -b "<integer>"            Optional. Batch size value. If not specified, the batch size value is determined from Intermediate Representation.
    -stream_output            Optional. Print progress as a plain text. When specified, an interactive progress bar is replaced with a multiline output.
    -t                        Optional. Time in seconds to execute topology.
    -progress                 Optional. Show progress bar (can affect performance measurement). Default values is "false".
    -shape                    Optional. Set shape for network input. For example, "input1[1,3,224,224],input2[1,4]" or "[1,3,224,224]" in case of one input size. This parameter affect model input shape and can be dynamic. For dynamic dimensions use symbol `?` or '-1'. Ex. [?,3,?,?]. For bounded dimensions specify range 'min..max'. Ex. [1..10,3,?,?].
    -data_shape               Required for networks with dynamic shapes. Set shape for input blobs. In case of one input size: "[1,3,224,224]" or "input1[1,3,224,224],input2[1,4]". In case of several input sizes provide the same number for each input (except cases with single shape for any input): "[1,3,128,128][3,3,128,128][1,3,320,320]", "input1[1,1,128,128][1,1,256,256],input2[80,1]" or "input1[1,192][1,384],input2[1,192][1,384],input3[1,192][1,384],input4[1,192][1,384]". If network shapes are all static, specifying the option will cause an exception.
    -layout                   Optional. Prompts how network layouts should be treated by application. For example, "input1[NCHW],input2[NC]" or "[NCHW]" in case of one input size.
    -cache_dir "<path>"       Optional. Enables caching of loaded models to specified directory. List of devices which support caching is shown at the end of this message.
    -load_from_file           Optional. Loads model from file directly without ReadNetwork. All CNNNetwork options (like re-shape) will be ignored
    -latency_percentile       Optional. Defines the percentile to be reported in latency metric. The valid range is [1, 100]. The default value is 50 (median).

  Device-specific performance options:
    -nstreams "<integer>"     Optional. Number of streams to use for inference on the CPU, GPU or MYRIAD devices (for HETERO and MULTI device cases use format <dev1>:<nstreams1>,<dev2>:<nstreams2> or just <nstreams>). Default value is determined automatically for a device. Note that although the automatic selection usually provides a reasonable performance, it still may be non - optimal for some cases, especially for very small networks. See sample's README for more details. Also, using nstreams>1 is inherently throughput-oriented option, while for the best-latency estimations the number of streams should be set to 1.
    -nthreads "<integer>"     Optional. Number of threads to use for inference on the CPU (including HETERO and MULTI cases).
    -pin ("YES"|"CORE")/"HYBRID_AWARE"/("NO"|"NONE")/"NUMA"   Optional. Explicit inference threads binding options (leave empty to let the OpenVINO to make a choice):
                                enabling threads->cores pinning("YES", which is already default for any conventional CPU),
                                letting the runtime to decide on the threads->different core types("HYBRID_AWARE", which is default on the hybrid CPUs)
                                threads->(NUMA)nodes("NUMA") or
                                completely disable("NO") CPU inference threads pinning

  Statistics dumping options:
    -report_type "<type>"       Optional. Enable collecting statistics report. "no_counters" report contains configuration options specified, resulting FPS and latency.
                                "average_counters" report extends "no_counters" report and additionally includes average PM counters values for each layer from the network.
                                "detailed_counters" report extends "average_counters" report and additionally includes per-layer PM counters
                                and latency for each executed infer request.
    -report_folder              Optional. Path to a folder where statistics report is stored.
    -exec_graph_path            Optional. Path to a file where to store executable graph information serialized.
    -pc                         Optional. Report performance counters.
    -dump_config                Optional. Path to JSON file to dump IE parameters, which were set by application.
    -load_config                Optional. Path to JSON file to load custom IE parameters. Command line parameters have higher priority than parameters from configuration file.

   Statistics dumping options:
    -report_type "<type>"     Optional. Enable collecting statistics report. "no_counters" report contains configuration options specified, resulting FPS and latency. "average_counters" report extends "no_counters" report and additionally includes average PM counters values for each layer from the network. "detailed_counters" report extends "average_counters" report and additionally includes per-layer PM counters and latency for each executed infer request.
    -report_folder            Optional. Path to a folder where statistics report is stored.
    -json_stats               Optional. Enables JSON-based statistics output (by default reporting system will use CSV format). Should be used together with -report_folder option.    -exec_graph_path          Optional. Path to a file where to store executable graph information serialized.
    -pc                       Optional. Report performance counters.
    -pcseq                    Optional. Report latencies for each shape in -data_shape sequence.
    -dump_config              Optional. Path to JSON file to dump IE parameters, which were set by application.
    -load_config              Optional. Path to JSON file to load custom IE parameters. Command line parameters have higher priority then parameters from configuration file.
    -infer_precision "<element type>"Optional. Inference precission
    -ip                          <value>     Optional. Specifies precision for all input layers of the network.
    -op                          <value>     Optional. Specifies precision for all output layers of the network.
    -iop                        "<value>"    Optional. Specifies precision for input and output layers by name.
                                             Example: -iop "input:FP16, output:FP16".
                                             Notice that quotes are required.
                                             Overwrites precision from ip and op options for specified layers.
    -iscale                    Optional. Scale values to be used for the input image per channel.
Values to be provided in the [R, G, B] format. Can be defined for desired input of the model.
Example: -iscale data[255,255,255],info[255,255,255]

    -imean                     Optional. Mean values to be used for the input image per channel.
Values to be provided in the [R, G, B] format. Can be defined for desired input of the model,
Example: -imean data[255,255,255],info[255,255,255]

    -inference_only              Optional. Measure only inference stage. Default option for static models. Dynamic models are measured in full mode which includes inputs setup stage. Inference only mode is available for them with single input data shape only. To enable full mode for static models pass "false" value to this argument: ex. "-inference_only=false".
```

Running the application with an empty list of the options yields the usage message given above and an error message.

Application supports topologies with one or more inputs. If a topology is not data-sensitive, you may skip the input parameter. In this case, inputs are filled with random values.
If a model has only image input(s), provide a folder with images or a path to an image as input.
If a model has some specific input(s) (not images), prepare a binary file(s) that is filled with data of appropriate precision and provide a path to them as input.
If a model has mixed input types, the input folder should contain all required files. Image inputs are filled with image files one by one. Binary inputs are filled with binary inputs one by one.

To run the tool, you may use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader).

> **NOTE:** Before running the tool with a trained model, make sure the model is converted to the OpenVINO IR (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

## Examples of Running the Tool

This section provides step-by-step instructions on how to run the Benchmark Tool with the `googlenet-v1` public model on CPU or GPU devices.  The [dog.bmp](https://storage.openvinotoolkit.org/data/test_data/images/224x224/dog.bmp) file is used as an input.

> **NOTE**: The Internet access is required to execute the following steps successfully. If you have access to the Internet through the proxy server only, make sure that it is configured in your OS environment.

1. Install OpenVINO Development Tools to work with Caffe models:

   ``` sh
   pip install openvino-dev[caffe]
   ```

2. Download the model. Go to the Model Downloader directory and run the `omz_downloader` script with specifying the model name and directory to download the model to:

   ```sh
   omz_downloader --name googlenet-v1 -o <models_dir>
   ```
3. Convert the model to the OpenVINO IR format. Run the Model Optimizer using the `mo` command with the path to the model, model format and output directory to generate the IR files:

   ```sh
   mo --input_model <models_dir>/public/googlenet-v1/googlenet-v1.caffemodel --data_type FP32 --output_dir <ir_dir>
   ```

4. Run the tool with specifying the `dog.bmp` file as an input image, the IR of the `googlenet-v1` model and a device to perform inference on. The following commands demonstrate running the Benchmark Tool in the asynchronous mode on CPU and GPU devices:

   * On CPU:
   ```sh
   ./benchmark_app -m <ir_dir>/googlenet-v1.xml -i dog.bmp  -d CPU -api async -progress
   ```
   * On GPU:
   ```sh
   ./benchmark_app -m <ir_dir>/googlenet-v1.xml -i dog.bmp -d GPU -api async -progress
   ```

The application outputs the number of executed iterations, total duration of execution, latency, and throughput.
Additionally, if you set the `-report_type` parameter, the application outputs statistics report. If you set the `-pc` parameter, the application outputs performance counters. If you set `-exec_graph_path`, the application reports executable graph information serialized. All measurements including per-layer PM counters are reported in milliseconds.

Below are fragments of sample output static and dynamic networks:

* For static network:
   ```
   [Step 10/11] Measuring performance (Start inference asynchronously, 4 inference requests using 4 streams for CPU, limits: 60000 ms duration)
   [ INFO ] BENCHMARK IS IN INFERENCE ONLY MODE.
   [ INFO ] Input blobs will be filled once before performance measurements.
   [ INFO ] First inference took 26.26 ms
   Progress: [................... ]  99% done

   [Step 11/11] Dumping statistics report
   [ INFO ] Count:      6640 iterations
   [ INFO ] Duration:   60039.70 ms
   [ INFO ] Latency:
   [ INFO ]        Median:  35.36 ms
   [ INFO ]        Avg:    36.12 ms
   [ INFO ]        Min:    18.55 ms
   [ INFO ]        Max:    88.96 ms
   [ INFO ] Throughput: 110.59 FPS
   ```

* For dynamic network:
   ```
   [Step 10/11] Measuring performance (Start inference asynchronously, 4 inference requests using 4 streams for CPU, limits: 60000 ms duration)
   [ INFO ] BENCHMARK IS IN FULL MODE.
   [ INFO ] Inputs setup stage will be included in performance measurements.
   [ INFO ] First inference took 26.80 ms
   Progress: [................... ]  99% done

   [Step 11/11] Dumping statistics report
   [ INFO ] Count:      5199 iterations
   [ INFO ] Duration:   60043.34 ms
   [ INFO ] Latency:
   [ INFO ]        Median:  41.58 ms
   [ INFO ]        Avg:    46.07 ms
   [ INFO ]        Min:    8.44 ms
   [ INFO ]        Max:    115.65 ms
   [ INFO ] Latency for each data shape group:
   [ INFO ] 1. data : [1, 3, 224, 224]
   [ INFO ]        Median:  38.37 ms
   [ INFO ]        Avg:    30.29 ms
   [ INFO ]        Min:    8.44 ms
   [ INFO ]        Max:    61.30 ms
   [ INFO ] 2. data : [1, 3, 448, 448]
   [ INFO ]        Median:  68.21 ms
   [ INFO ]        Avg:    61.85 ms
   [ INFO ]        Min:    29.58 ms
   [ INFO ]        Max:    115.65 ms
   [ INFO ] Throughput: 86.59 FPS
   ```

## See Also
* [Using OpenVINO Runtime Samples](../../../docs/OV_Runtime_UG/Samples_Overview.md)
* [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](@ref omz_tools_downloader)
